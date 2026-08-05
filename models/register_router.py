"""Register the five risk models as ONE MLflow pyfunc "router" model.

Why a router rather than five ``mlflow models serve`` processes:
  * one port, one process, one lifecycle to manage (GUIDE.md §4 recommends this);
  * the five models share an identical call contract, so routing on a ``model``
    param keeps the backend's client code trivial;
  * crucially, it lets us override ``predict`` to return ``predict_proba(X)[:, 1]``.
    MLflow's default sklearn pyfunc flavour calls ``.predict()``, which returns
    class LABELS (0/1) — that silently destroys the entire risk story, so the
    override is the whole point of this file.

The router is the single source of truth for each model's feature list: it
reindexes the incoming frame to ``model.feature_names_in_`` before predicting, so
column ORDER is always correct regardless of what order the caller sent.

It also produces EXPLANATIONS (``params={"explain": true}``). See
``_contributions`` for the maths and why it costs essentially nothing here.

Run:
    uv run python models/register_router.py        # writes models/mlflow_risk_router/
    uv run mlflow models serve -m models/mlflow_risk_router -p 5001 --env-manager local
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import shutil
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import DataType, ParamSchema, ParamSpec

MODELS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = MODELS_DIR / "mlflow_risk_router"

# artifact key -> pickle filename. The artifact key is the value callers pass as
# params={"model": ...}.
MODEL_FILES = {
    "prevent_cvd": "prevent_cvd.pkl",
    "ada_t2dm": "ada_t2dm.pkl",
    "framingham_ckd": "framingham_ckd.pkl",
    "clivd_cld": "clivd_cld.pkl",
    "caide_dementia": "caide_dementia.pkl",
}

REFERENCE_ARTIFACT = "reference_vectors"
# Bumped whenever the reference population changes, so an explanation can always
# be traced to the baseline it was measured against.
REFERENCE_ID = "healthy-anchor-v1"


def _load_generator_spec() -> dict[str, dict]:
    """Read the reference ("healthy") patient for each model from its generator.

    The pickles carry coefficients and metadata but not the calibration anchors,
    and re-typing those anchors here would let them drift. Importing the spec
    means the reference vector is definitionally the same one the model was
    calibrated against.
    """
    path = MODELS_DIR / "generate_models.py"
    spec = importlib.util.spec_from_file_location("generate_models", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {entry["metadata"]["model_name"]: entry["healthy"] for entry in module.MODELS}


class RiskRouter(mlflow.pyfunc.PythonModel):
    """Serves all five risk models behind a single ``/invocations`` endpoint.

    Call shape::

        {"dataframe_split": {"columns": [...], "data": [[...]]},
         "params": {"model": "framingham_ckd"}}
        -> {"predictions": [0.50...]}

    The response is the POSITIVE-CLASS PROBABILITY, not a class label.

    With ``params={"model": ..., "explain": true}`` each row becomes::

        {"probability": 0.50, "base_value": -3.89, "reference_id": "...",
         "contributions": {"age_years": 1.85, "egfr": 1.44, ...}}
    """

    def load_context(self, context) -> None:
        self.models = {}
        for name, path in context.artifacts.items():
            if name == REFERENCE_ARTIFACT:
                continue
            with open(path, "rb") as fh:
                self.models[name] = pickle.load(fh)

        with open(context.artifacts[REFERENCE_ARTIFACT]) as fh:
            payload = json.load(fh)
        self.reference_id = payload["reference_id"]
        self.references = payload["vectors"]

    def _contributions(self, name: str, X: pd.DataFrame) -> list[dict]:
        """Exact SHAP values, in closed form.

        For a linear model the Shapley value of feature j is not approximated —
        it is analytic::

            log-odds(x) = b + sum_j w_j * x_j
            phi_j(x)    = w_j * (x_j - x_ref_j)
            sum_j phi_j + base = log-odds(x)      where base = log-odds(x_ref)

        So there is no sampling, no KernelExplainer, and no latency cliff: the
        whole explanation is one vector subtraction and a multiply. That is why
        explaining every risk on every request is affordable here.
        (``backend/tests/test_explanations.py`` proves this equals the true
        Shapley value by brute-force subset enumeration, and proves the additive
        identity holds for all five models across all eight patients.)

        Contributions are in LOG-ODDS. They are additive there and NOT in
        probability space — turning one into "this feature adds N% of risk" is
        wrong, and the tool docstrings say so explicitly.
        """
        model = self.models[name]
        features = list(model.feature_names_in_)
        coef = np.asarray(model.coef_[0], dtype="float64")
        intercept = float(model.intercept_[0])

        reference = self.references[name]
        ref_vec = np.array([float(reference[f]) for f in features], dtype="float64")
        base_value = intercept + float(coef @ ref_vec)

        probabilities = model.predict_proba(X)[:, 1]
        rows: list[dict] = []
        for position, (_, row) in enumerate(X.iterrows()):
            x = row.to_numpy(dtype="float64")
            phi = coef * (x - ref_vec)
            rows.append(
                {
                    "probability": float(probabilities[position]),
                    "base_value": base_value,
                    "reference_id": self.reference_id,
                    "model_name": name,
                    "contributions": {f: float(v) for f, v in zip(features, phi)},
                    "reference_values": {f: float(v) for f, v in zip(features, ref_vec)},
                    "feature_values": {f: float(v) for f, v in zip(features, x)},
                }
            )
        return rows

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None):
        params = params or {}
        name = params.get("model")
        if not name:
            raise ValueError(
                "Missing required param 'model'. "
                f"Expected one of: {sorted(self.models)}"
            )
        if name not in self.models:
            raise ValueError(
                f"Unknown model {name!r}. Expected one of: {sorted(self.models)}"
            )

        model = self.models[name]
        expected = list(model.feature_names_in_)

        missing = [c for c in expected if c not in model_input.columns]
        if missing:
            raise ValueError(
                f"Model {name!r} requires features {expected}; missing {missing}. "
                f"Received columns: {list(model_input.columns)}"
            )

        # Reindex to the model's own feature order, then coerce to float — sklearn
        # matches on position, so a correct-names/wrong-order frame would score
        # silently and wrongly.
        X = model_input[expected].astype("float64")

        if params.get("explain"):
            return self._contributions(name, X)

        # Default shape is unchanged, so the GUIDE's curl and every existing
        # caller keep working.
        return model.predict_proba(X)[:, 1]


def _build_signature() -> ModelSignature:
    """Declare the params only — deliberately NO input schema.

    Two dead ends ruled this out. A union-of-all-features schema typed ``double``
    makes MLflow reject the integer JSON in GUIDE.md §4's curl ("Can not safely
    convert int64 to float64"), and typing it ``long`` would then reject the
    genuinely fractional features (``bmi``, ``waist_hip_ratio``). Since callers
    legitimately send different column subsets with mixed int/float types, schema
    enforcement buys nothing here.

    Validation is not lost — it moves into ``RiskRouter.predict``, which checks
    against each model's own ``feature_names_in_`` and raises a message naming the
    model and the missing columns. That is strictly more useful than MLflow's
    generic dtype error.
    """
    params = ParamSchema(
        [
            ParamSpec("model", DataType.string, default="framingham_ckd"),
            ParamSpec("explain", DataType.boolean, default=False),
        ]
    )
    return ModelSignature(inputs=None, params=params)


def main() -> None:
    artifacts = {name: str(MODELS_DIR / fn) for name, fn in MODEL_FILES.items()}
    for name, path in artifacts.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing pickle for {name!r}: {path}")

    # The reference population travels WITH the model, so an explanation can
    # always be reproduced against the baseline it was actually measured from.
    references = _load_generator_spec()
    missing_refs = set(MODEL_FILES) - set(references)
    if missing_refs:
        raise RuntimeError(f"generate_models.py has no reference vector for {missing_refs}")

    staging = Path(tempfile.mkdtemp(prefix="risk-router-"))
    reference_path = staging / "reference_vectors.json"
    reference_path.write_text(
        json.dumps({"reference_id": REFERENCE_ID, "vectors": references}, indent=2),
        encoding="utf-8",
    )
    artifacts[REFERENCE_ARTIFACT] = str(reference_path)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    mlflow.pyfunc.save_model(
        path=str(OUTPUT_DIR),
        python_model=RiskRouter(),
        artifacts=artifacts,
        signature=_build_signature(),
        pip_requirements=["scikit-learn", "pandas", "numpy"],
    )
    shutil.rmtree(staging, ignore_errors=True)
    print(f"Saved router model -> {OUTPUT_DIR}")

    # Round-trip check: load it back the way the server will and score the P004
    # CKD payload from GUIDE.md §4. Expected ~0.50 (it is the model's own
    # calibration anchor), so this is a real oracle, not a smoke test.
    loaded = mlflow.pyfunc.load_model(str(OUTPUT_DIR))
    p004_ckd = pd.DataFrame(
        [[72, 1, 1, 1, 52]],
        columns=["age_years", "diabetes", "hypertension", "proteinuria_trace_plus", "egfr"],
    )
    prob = loaded.predict(p004_ckd, params={"model": "framingham_ckd"})[0]
    print(f"  P004 CKD probability: {prob:.4f}  (expected ~0.50)")

    # And the explanation must reconstruct that same probability exactly.
    explained = loaded.predict(
        p004_ckd, params={"model": "framingham_ckd", "explain": True}
    )[0]
    total = explained["base_value"] + sum(explained["contributions"].values())
    rebuilt = 1.0 / (1.0 + np.exp(-total))
    print(f"  explanation rebuilds probability: {rebuilt:.4f}  (identity check)")
    if abs(rebuilt - prob) > 1e-9:
        raise RuntimeError(
            f"SHAP additivity broken: contributions rebuild {rebuilt}, model says {prob}"
        )

    top = sorted(
        explained["contributions"].items(), key=lambda kv: abs(kv[1]), reverse=True
    )[:3]
    print("  top drivers:", ", ".join(f"{f} {v:+.3f}" for f, v in top))

    for name in MODEL_FILES:
        print(f"  routed model available: {name}")

    print("\nServe with:")
    print(f"  uv run mlflow models serve -m {OUTPUT_DIR.as_posix()} -p 5001 --env-manager local")


if __name__ == "__main__":
    main()

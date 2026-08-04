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

Run:
    uv run python models/register_router.py        # writes models/mlflow_risk_router/
    uv run mlflow models serve -m models/mlflow_risk_router -p 5001 --env-manager local
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import mlflow
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


class RiskRouter(mlflow.pyfunc.PythonModel):
    """Serves all five risk models behind a single ``/invocations`` endpoint.

    Call shape::

        {"dataframe_split": {"columns": [...], "data": [[...]]},
         "params": {"model": "framingham_ckd"}}
        -> {"predictions": [0.50...]}

    The response is the POSITIVE-CLASS PROBABILITY, not a class label.
    """

    def load_context(self, context) -> None:
        self.models = {}
        for name, path in context.artifacts.items():
            with open(path, "rb") as fh:
                self.models[name] = pickle.load(fh)

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None):
        name = (params or {}).get("model")
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
        return model.predict_proba(X)[:, 1]


def _build_signature() -> ModelSignature:
    """Declare the ``model`` param only — deliberately NO input schema.

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
        [ParamSpec("model", DataType.string, default="framingham_ckd")]
    )
    return ModelSignature(inputs=None, params=params)


def main() -> None:
    artifacts = {name: str(MODELS_DIR / fn) for name, fn in MODEL_FILES.items()}
    for name, path in artifacts.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing pickle for {name!r}: {path}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    mlflow.pyfunc.save_model(
        path=str(OUTPUT_DIR),
        python_model=RiskRouter(),
        artifacts=artifacts,
        signature=_build_signature(),
        pip_requirements=["scikit-learn", "pandas", "numpy"],
    )
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

    for name in MODEL_FILES:
        print(f"  routed model available: {name}")

    print("\nServe with:")
    print(f"  uv run mlflow models serve -m {OUTPUT_DIR.as_posix()} -p 5001 --env-manager local")


if __name__ == "__main__":
    main()

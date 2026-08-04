"""Domain errors.

These are raised by the service layer and translated to HTTP status codes in
``app/api/v1/endpoints.py``. Keeping them HTTP-free means the service can be unit
tested — and later reused by a non-HTTP caller — without importing FastAPI.
"""

from __future__ import annotations


class PatientNotFoundError(LookupError):
    """No such patient. -> HTTP 404."""

    def __init__(self, patient_id: str) -> None:
        self.patient_id = patient_id
        super().__init__(f"Unknown patient: {patient_id}")


class ModelServerError(RuntimeError):
    """The MLflow model server was unreachable or returned something unusable.

    -> HTTP 502. Deliberately NOT swallowed into a default probability: a
    fabricated risk is the worst failure mode this system has, so an outage must
    surface as an outage.
    """


class IncompletePatientDataError(ValueError):
    """A feature a model requires is missing from the record. -> HTTP 422.

    Distinct from ``PatientNotFoundError``: the patient exists, but scoring them
    would mean inventing an input. We refuse rather than impute — except for the
    one documented, clinically-justified default in ``features.OPTIONAL_DEFAULTS``.
    """

    def __init__(self, patient_id: str, missing: list[str]) -> None:
        self.patient_id = patient_id
        self.missing = missing
        super().__init__(
            f"Cannot score patient {patient_id}: missing required inputs {sorted(missing)}"
        )

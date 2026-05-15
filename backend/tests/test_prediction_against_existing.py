from __future__ import annotations

import math

import pandas as pd
import pytest

from app.core.config import get_settings
from app.db.database import Base, SessionLocal, engine
from app.services.import_service import reset_and_import_seed_data
from app.services.prediction_service import get_prediction_service


def test_prediction_matches_existing_row() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        reset_and_import_seed_data(db)
        expected_df = pd.read_csv(get_settings().model_outputs_dir / "prediction_test_rows.csv")
        expected = expected_df.iloc[0]
        result = get_prediction_service().predict_invoice(
            db,
            factura_id=str(expected["factura_id"]),
            fecha_corte=pd.to_datetime(expected["fecha_corte"]).date(),
            persist=False,
        )

    assert result.feature_source == "prepared_features_snapshot"
    assert result.predicted_class_tecnica == expected["predicted_class"]
    assert math.isclose(result.priority_score_0_100, float(expected["priority_score_0_100"]), abs_tol=0.01)
    assert math.isclose(result.prob_atraso_leve, float(expected["prob_plus_30"]), abs_tol=1e-10)
    assert math.isclose(result.prob_atraso_alto, float(expected["prob_plus_60"]), abs_tol=1e-10)
    assert math.isclose(result.prob_atraso_critico, float(expected["prob_plus_90"]), abs_tol=1e-10)
    assert result.accion_sugerida.codigo == "SIN_ACCION"


def test_rejects_cutoff_before_invoice_issue_date() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        reset_and_import_seed_data(db)
        with pytest.raises(ValueError, match="fecha_corte"):
            get_prediction_service().predict_invoice(
                db,
                factura_id="FAC000001",
                fecha_corte=pd.to_datetime("2022-12-01").date(),
                persist=False,
            )

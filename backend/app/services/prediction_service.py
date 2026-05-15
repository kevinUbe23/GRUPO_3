from __future__ import annotations

from datetime import date

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Factura, PrediccionFactura, SegmentoCliente
from app.schemas import ActionOut, PredictionOut
from app.services.feature_builder import FeatureBuilder
from app.services.recommendation_service import recommend_action


TECHNICAL_TO_USER_LABEL = {
    "on_time": "Pago esperado dentro del plazo",
    "+30": "Atraso leve probable",
    "+60": "Atraso alto probable",
    "+90": "Atraso critico probable",
}
LATE_WEIGHTS = {"+30": 0.40, "+60": 0.70, "+90": 1.00}
HIGH_RISK_CLASSES = {"+60", "+90"}


def _decode_labels(values, target_encoder) -> np.ndarray:
    values_array = np.asarray(values)
    if target_encoder is None:
        return values_array.astype(str)
    return target_encoder.inverse_transform(values_array.astype(int)).astype(str)


class PredictionService:
    def __init__(self) -> None:
        settings = get_settings()
        self.artifact = joblib.load(settings.model_path)
        self.model = self.artifact["model"]
        self.preprocessor = self.artifact["preprocessor"]
        self.feature_cols = self.artifact["feature_cols"]
        self.target_encoder = self.artifact.get("target_encoder")
        self.model_version = "fase4_logistic_regression"
        self.feature_builder = FeatureBuilder()
        self.class_names = self._class_names()

    def _class_names(self) -> list[str]:
        classes = getattr(self.model, "classes_", None)
        if classes is None and self.target_encoder is not None:
            return [str(c) for c in self.target_encoder.classes_]
        return _decode_labels(classes, self.target_encoder).tolist()

    def predict_invoice(
        self,
        db: Session,
        factura_id: str,
        fecha_corte: date,
        persist: bool = True,
        use_prepared_snapshot: bool = True,
    ) -> PredictionOut:
        feature_row = self.feature_builder.build(
            db,
            factura_id,
            fecha_corte,
            use_prepared_snapshot=use_prepared_snapshot,
        )
        factura = db.get(Factura, factura_id)
        if factura is None:
            raise ValueError(f"No existe la factura {factura_id}")
        estado_al_corte = self._estado_factura_al_corte(factura, fecha_corte, feature_row.data)
        recommendation_features = {
            **feature_row.data,
            "estado_factura": estado_al_corte,
            "saldo_pendiente": factura.saldo_pendiente,
        }
        missing = [col for col in self.feature_cols if col not in feature_row.data]
        if missing:
            raise ValueError(f"Faltan features para predecir: {missing}")

        x_df = pd.DataFrame([{col: feature_row.data[col] for col in self.feature_cols}])
        x_matrix = self.preprocessor.transform(x_df)
        raw_pred = self.model.predict(x_matrix)
        pred_class = str(_decode_labels(raw_pred, self.target_encoder)[0])
        proba = self.model.predict_proba(x_matrix)[0]
        proba_by_class = {class_name: float(proba[idx]) for idx, class_name in enumerate(self.class_names)}

        prob_pago_plazo = proba_by_class.get("on_time", 0.0)
        prob_atraso_leve = proba_by_class.get("+30", 0.0)
        prob_atraso_alto = proba_by_class.get("+60", 0.0)
        prob_atraso_critico = proba_by_class.get("+90", 0.0)
        any_late = prob_atraso_leve + prob_atraso_alto + prob_atraso_critico
        high_risk = prob_atraso_alto + prob_atraso_critico
        priority = round(
            100
            * (
                LATE_WEIGHTS["+30"] * prob_atraso_leve
                + LATE_WEIGHTS["+60"] * prob_atraso_alto
                + LATE_WEIGHTS["+90"] * prob_atraso_critico
            ),
            2,
        )
        user_label = TECHNICAL_TO_USER_LABEL.get(pred_class, pred_class)
        cliente_id = factura.cliente_id
        segmento = db.get(SegmentoCliente, cliente_id) if cliente_id else None
        action = recommend_action(
            features=recommendation_features,
            predicted_label_usuario=user_label,
            any_late_probability=any_late,
            high_risk_probability=high_risk,
            priority_score_0_100=priority,
            segmento=segmento,
        )

        if persist:
            db.add(
                PrediccionFactura(
                    factura_id=factura_id,
                    cliente_id=cliente_id,
                    fecha_corte=fecha_corte,
                    modelo_version=self.model_version,
                    predicted_class_tecnica=pred_class,
                    predicted_label_usuario=user_label,
                    prob_pago_plazo=prob_pago_plazo,
                    prob_atraso_leve=prob_atraso_leve,
                    prob_atraso_alto=prob_atraso_alto,
                    prob_atraso_critico=prob_atraso_critico,
                    any_late_probability=any_late,
                    high_risk_probability=high_risk,
                    priority_score_0_100=priority,
                    accion_sugerida_codigo=action.codigo,
                    accion_sugerida_nombre=action.nombre,
                    motivo_accion=action.motivo,
                )
            )
            db.commit()

        return PredictionOut(
            factura_id=factura_id,
            cliente_id=cliente_id,
            fecha_corte=fecha_corte,
            modelo_version=self.model_version,
            predicted_class_tecnica=pred_class,
            predicted_label_usuario=user_label,
            prob_pago_plazo=prob_pago_plazo,
            prob_atraso_leve=prob_atraso_leve,
            prob_atraso_alto=prob_atraso_alto,
            prob_atraso_critico=prob_atraso_critico,
            any_late_probability=any_late,
            high_risk_probability=high_risk,
            priority_score_0_100=priority,
            accion_sugerida=ActionOut(
                codigo=action.codigo,
                nombre=action.nombre,
                canal_recomendado=action.canal_recomendado,
                severidad=action.severidad,
                motivo=action.motivo,
                regla=action.regla,
            ),
            feature_source=feature_row.source,
        )

    def _cliente_id_for_factura(self, db: Session, factura_id: str) -> str:
        factura = db.get(Factura, factura_id)
        if factura is None:
            raise ValueError(f"No existe la factura {factura_id}")
        return factura.cliente_id

    @staticmethod
    def _estado_factura_al_corte(factura: Factura, fecha_corte: date, features: dict) -> str:
        if factura.estado_factura in {"anulada", "castigada"}:
            return factura.estado_factura
        if factura.fecha_pago_real and factura.fecha_pago_real <= fecha_corte:
            return "pagada"
        if int(features.get("tiene_disputa_activa", 0)) == 1:
            return "en_disputa"
        return "abierta"


_prediction_service: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service

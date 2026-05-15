from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Cliente, Factura, GestionCobranza, PromesaPago


SECTOR_COLUMNS = [
    "sector_retail",
    "sector_manufactura",
    "sector_servicios",
    "sector_construccion",
    "sector_agro",
    "sector_tecnologia",
    "sector_salud",
    "sector_transporte",
]
RESULT_TO_CODE = {
    "pagado": "cod_0",
    "confirma_pago": "cod_1",
    "promesa_de_pago": "cod_2",
    "en_proceso_interno": "cod_3",
    "disputa_monto": "cod_4",
    "cliente_ausente": "cod_5",
    "no_contesta": "cod_6",
    "numero_invalido": "cod_7",
    "rechazo_pago": "cod_8",
}


@dataclass
class FeatureRow:
    data: dict
    source: str


class FeatureBuilder:
    def __init__(self) -> None:
        settings = get_settings()
        self.feature_schema = pd.read_csv(settings.feature_schema_path)["feature"].tolist()
        self.prepared_path = settings.prepared_features_path
        self._prepared_df: pd.DataFrame | None = None

    def build(
        self,
        db: Session,
        factura_id: str,
        fecha_corte: date,
        use_prepared_snapshot: bool = True,
    ) -> FeatureRow:
        factura = db.get(Factura, factura_id)
        if factura is None:
            raise ValueError(f"No existe la factura {factura_id}")
        if fecha_corte < factura.fecha_emision:
            raise ValueError("fecha_corte no puede ser anterior a fecha_emision")
        if use_prepared_snapshot:
            exact = self._from_prepared_snapshot(factura_id, fecha_corte)
            if exact is not None:
                return FeatureRow(exact, "prepared_features_snapshot")
        return FeatureRow(self._from_operational_db(db, factura_id, fecha_corte), "operational_db")

    def _load_prepared(self) -> pd.DataFrame:
        if self._prepared_df is None:
            df = pd.read_csv(self.prepared_path, parse_dates=["fecha_corte"])
            self._prepared_df = df
        return self._prepared_df

    def _from_prepared_snapshot(self, factura_id: str, fecha_corte: date) -> dict | None:
        df = self._load_prepared()
        mask = (df["factura_id"].astype(str) == factura_id) & (
            df["fecha_corte"].dt.date == fecha_corte
        )
        if not mask.any():
            return None
        row = df.loc[mask].sort_values("fecha_corte").iloc[-1]
        return {feature: row[feature] for feature in self.feature_schema}

    def _from_operational_db(self, db: Session, factura_id: str, fecha_corte: date) -> dict:
        factura = db.get(Factura, factura_id)
        if factura is None:
            raise ValueError(f"No existe la factura {factura_id}")
        cliente = db.get(Cliente, factura.cliente_id)
        if cliente is None:
            raise ValueError(f"No existe el cliente {factura.cliente_id}")

        prev_facturas = db.scalars(
            select(Factura)
            .where(Factura.cliente_id == cliente.cliente_id)
            .where(Factura.fecha_emision < factura.fecha_emision)
            .where(Factura.fecha_pago_real.is_not(None))
            .where(Factura.fecha_pago_real <= fecha_corte)
            .order_by(Factura.fecha_emision)
        ).all()
        gestiones = db.scalars(
            select(GestionCobranza)
            .where(GestionCobranza.factura_id == factura_id)
            .where(GestionCobranza.fecha_gestion <= fecha_corte)
            .order_by(GestionCobranza.fecha_gestion)
        ).all()
        cliente_gestiones = db.scalars(
            select(GestionCobranza)
            .where(GestionCobranza.cliente_id == cliente.cliente_id)
            .where(GestionCobranza.fecha_gestion <= fecha_corte)
        ).all()
        promesas = db.scalars(
            select(PromesaPago)
            .where(PromesaPago.cliente_id == cliente.cliente_id)
            .where(PromesaPago.fecha_promesa <= fecha_corte)
        ).all()

        monto_promedio_hist = (
            sum(f.monto for f in prev_facturas) / len(prev_facturas)
            if prev_facturas
            else factura.monto
        )
        mora_values = [f.dias_mora_real or 0 for f in prev_facturas if f.fecha_pago_real]
        mora_promedio_hist = sum(mora_values) / len(mora_values) if mora_values else 0.0
        mora_ultimo_tramo = float(mora_values[-1]) if mora_values else 0.0
        tasa_cumplimiento = (
            sum(1 for value in mora_values if value == 0) / len(mora_values)
            if mora_values
            else 1.0
        )
        moras_consecutivas = 0
        for value in reversed(mora_values):
            if value > 0:
                moras_consecutivas += 1
            else:
                break

        dias_transcurridos = max((fecha_corte - factura.fecha_emision).days, 0)
        dias_hasta_vence = (factura.fecha_vencimiento - fecha_corte).days
        dias_mora_observable = max((fecha_corte - factura.fecha_vencimiento).days, 0)
        ultima_gestion = gestiones[-1] if gestiones else None
        dias_desde_ultima = (
            (fecha_corte - ultima_gestion.fecha_gestion).days if ultima_gestion else -1
        )
        contacto_total = len(cliente_gestiones)
        tasa_contacto = (
            sum(1 for g in cliente_gestiones if g.contacto_exitoso) / contacto_total
            if contacto_total
            else 0.5
        )
        no_contacto_results = {"no_contesta", "numero_invalido", "cliente_ausente"}
        num_no_contesta_cons = 0
        for gestion in reversed(gestiones):
            if gestion.resultado in no_contacto_results:
                num_no_contesta_cons += 1
            else:
                break
        promesas_vencidas_al_corte = [p for p in promesas if p.fecha_compromiso <= fecha_corte]
        promesas_total = len(promesas)
        num_promesas_rotas = sum(1 for p in promesas_vencidas_al_corte if not p.se_cumplio)
        tasa_cumpl_promesas = (
            sum(1 for p in promesas_vencidas_al_corte if p.se_cumplio)
            / len(promesas_vencidas_al_corte)
            if promesas_vencidas_al_corte
            else 1.0
        )
        tiene_promesa_activa = any(
            p.estado_promesa == "activa"
            and p.fecha_promesa <= fecha_corte < p.fecha_compromiso
            and not p.se_cumplio
            for p in promesas
            if p.factura_id == factura_id
        )
        tiene_disputa = any(g.resultado == "disputa_monto" for g in gestiones)

        row = {
            "monto": factura.monto,
            "monto_promedio_hist": monto_promedio_hist,
            "ratio_monto": factura.monto / max(monto_promedio_hist, 1.0),
            "mora_promedio_hist": mora_promedio_hist,
            "mora_ultimo_tramo": mora_ultimo_tramo,
            "num_gestiones_factura": len(gestiones),
            "dias_hasta_vence_pos": max(dias_hasta_vence, 0),
            "dias_mora_observable": dias_mora_observable,
            "num_no_contesta_cons": num_no_contesta_cons,
            "num_promesas_rotas": num_promesas_rotas,
            "promesas_total": promesas_total,
            "dias_transcurridos_corte": dias_transcurridos,
            "condicion_dias": factura.condicion_dias,
            "antiguedad_meses": cliente.antiguedad_meses,
            "num_facturas_prev": len(prev_facturas),
            "tasa_cumplimiento": tasa_cumplimiento,
            "moras_consecutivas": moras_consecutivas,
            "dias_desde_ultima_gestion": dias_desde_ultima,
            "dias_hasta_vence": dias_hasta_vence,
            "tasa_contacto_cliente": tasa_contacto,
            "tasa_cumpl_promesas": tasa_cumpl_promesas,
            "intensidad_gestion": len(gestiones) / (dias_transcurridos + 1),
            "friccion_contacto": num_no_contesta_cons / max(len(gestiones), 1),
            "ratio_promesas_rotas": num_promesas_rotas / max(promesas_total, 1),
            "tiene_garantia": int(cliente.tiene_garantia),
            "tiene_disputa_activa": int(tiene_disputa),
            "tiene_promesa_activa": int(tiene_promesa_activa),
            "sin_gestion_previa": int(len(gestiones) == 0),
            "esta_vencida_al_corte": int(fecha_corte > factura.fecha_vencimiento),
            "cliente_nuevo": int(len(prev_facturas) == 0),
            "ultimo_resultado_enc": RESULT_TO_CODE.get(ultima_gestion.resultado, "cod_nan")
            if ultima_gestion
            else "cod_nan",
        }
        for sector_col in SECTOR_COLUMNS:
            row[sector_col] = int(sector_col == f"sector_{cliente.sector}")

        missing = [col for col in self.feature_schema if col not in row]
        if missing:
            raise ValueError(f"Feature builder incompleto. Faltan: {missing}")
        return {feature: row[feature] for feature in self.feature_schema}

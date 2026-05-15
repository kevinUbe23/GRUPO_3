from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClienteOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    cliente_id: str
    nombre: str
    sector: str
    antiguedad_meses: int
    tiene_garantia: bool


class FacturaOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    factura_id: str
    cliente_id: str
    fecha_emision: date
    fecha_vencimiento: date
    fecha_pago_real: date | None
    condicion_dias: int
    monto: float
    saldo_pendiente: float
    estado_factura: str
    target_mora_simulado: str | None
    dias_mora_real: int | None


class FacturaCreate(BaseModel):
    factura_id: str | None = None
    cliente_id: str
    fecha_emision: date
    fecha_vencimiento: date
    fecha_pago_real: date | None = None
    condicion_dias: int | None = Field(default=None, ge=0)
    monto: float = Field(gt=0)
    saldo_pendiente: float | None = Field(default=None, ge=0)
    estado_factura: str = "abierta"
    target_mora_simulado: str | None = None
    dias_mora_real: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_dates(self) -> "FacturaCreate":
        if self.fecha_vencimiento < self.fecha_emision:
            raise ValueError("fecha_vencimiento debe ser mayor o igual a fecha_emision")
        if self.fecha_pago_real and self.fecha_pago_real < self.fecha_emision:
            raise ValueError("fecha_pago_real no puede ser anterior a fecha_emision")
        return self


class FacturaUpdate(BaseModel):
    cliente_id: str | None = None
    fecha_emision: date | None = None
    fecha_vencimiento: date | None = None
    fecha_pago_real: date | None = None
    condicion_dias: int | None = Field(default=None, ge=0)
    monto: float | None = Field(default=None, gt=0)
    saldo_pendiente: float | None = Field(default=None, ge=0)
    estado_factura: str | None = None
    target_mora_simulado: str | None = None
    dias_mora_real: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def reject_null_required_fields(cls, data):
        if not isinstance(data, dict):
            return data
        non_nullable_fields = {
            "cliente_id",
            "fecha_emision",
            "fecha_vencimiento",
            "condicion_dias",
            "monto",
            "saldo_pendiente",
            "estado_factura",
        }
        null_fields = sorted(field for field in non_nullable_fields if field in data and data[field] is None)
        if null_fields:
            raise ValueError(f"Campos no anulables en PATCH: {null_fields}")
        return data


class SegmentoClienteOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    cliente_id: str
    cluster: int
    tipo_cliente: str
    riesgo_0_100: float
    rating_estrellas: int
    rating_label: str
    por_que_rating: str
    por_que_cluster: str
    sector_dominante_modal: str | None
    n_facturas_total: int | None
    n_cortes_total: int | None


class ScoreRequest(BaseModel):
    fecha_corte: date | None = Field(default=None, description="Fecha de scoring. Si se omite, usa hoy.")
    persist: bool = Field(default=True, description="Guarda la prediccion en base de datos.")


class RecalculateRequest(BaseModel):
    fecha_corte: date
    limit: int | None = Field(default=None, ge=1, le=10000)
    persist: bool = True


class RecalculateResult(BaseModel):
    fecha_corte: date
    total_evaluadas: int
    total_con_error: int
    errores: list[dict]


class ActionOut(BaseModel):
    codigo: str
    nombre: str
    canal_recomendado: str
    severidad: int
    motivo: str
    regla: str


class PredictionOut(BaseModel):
    factura_id: str
    cliente_id: str
    fecha_corte: date
    modelo_version: str
    predicted_class_tecnica: str
    predicted_label_usuario: str
    prob_pago_plazo: float
    prob_atraso_leve: float
    prob_atraso_alto: float
    prob_atraso_critico: float
    any_late_probability: float
    high_risk_probability: float
    priority_score_0_100: float
    accion_sugerida: ActionOut
    feature_source: str


class PredictionHistoryOut(BaseModel):
    prediccion_id: int
    factura_id: str
    cliente_id: str
    fecha_corte: date
    modelo_version: str
    predicted_class_tecnica: str
    predicted_label_usuario: str
    prob_pago_plazo: float
    prob_atraso_leve: float
    prob_atraso_alto: float
    prob_atraso_critico: float
    any_late_probability: float
    high_risk_probability: float
    priority_score_0_100: float
    accion_sugerida_codigo: str
    accion_sugerida_nombre: str
    motivo_accion: str
    fecha_pago_real: date | None
    dias_mora_real: int | None
    target_mora_simulado: str | None


class DashboardSummary(BaseModel):
    total_facturas: int
    facturas_activas: int
    monto_pendiente: float
    monto_vencido: float
    promesas_activas: int
    facturas_en_disputa: int


class InitDbResult(BaseModel):
    clientes: int
    facturas: int
    gestiones: int
    promesas: int
    segmentos: int
    acciones: int = 0
    reglas: int = 0


class GestionCreate(BaseModel):
    factura_id: str
    fecha_gestion: date
    canal: str
    contacto_exitoso: bool
    resultado: str
    motivo_no_pago: str | None = None
    observacion: str | None = None
    recalculate: bool = True


class GestionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    gestion_id: str
    factura_id: str
    cliente_id: str
    fecha_gestion: date
    canal: str
    contacto_exitoso: bool
    resultado: str
    motivo_no_pago: str | None
    dias_mora_en_gestion: int
    observacion: str | None


class PromesaCreate(BaseModel):
    gestion_id: str
    fecha_compromiso: date
    recalculate: bool = True


class PromesaUpdate(BaseModel):
    estado_promesa: str
    se_cumplio: bool | None = None


class PromesaOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    promesa_id: str
    gestion_id: str
    factura_id: str
    cliente_id: str
    fecha_promesa: date
    fecha_compromiso: date
    se_cumplio: bool
    estado_promesa: str


class PaymentCreate(BaseModel):
    factura_id: str
    fecha_pago: date


class ActionCatalogOut(BaseModel):
    codigo: str
    nombre: str
    descripcion: str
    canal_recomendado: str
    nivel_severidad: int
    activa: bool

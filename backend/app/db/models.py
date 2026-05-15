from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.time import utc_now
from app.db.database import Base


class Cliente(Base):
    __tablename__ = "clientes"

    cliente_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    nombre: Mapped[str] = mapped_column(String(255), nullable=False)
    sector: Mapped[str] = mapped_column(String(80), nullable=False)
    antiguedad_meses: Mapped[int] = mapped_column(Integer, nullable=False)
    tiene_garantia: Mapped[bool] = mapped_column(Boolean, nullable=False)
    perfil_pago_simulado: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    facturas: Mapped[list["Factura"]] = relationship(back_populates="cliente")


class Factura(Base):
    __tablename__ = "facturas"

    factura_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    cliente_id: Mapped[str] = mapped_column(ForeignKey("clientes.cliente_id"), index=True)
    fecha_emision: Mapped[date] = mapped_column(Date, nullable=False)
    fecha_vencimiento: Mapped[date] = mapped_column(Date, nullable=False)
    fecha_pago_real: Mapped[date | None] = mapped_column(Date)
    condicion_dias: Mapped[int] = mapped_column(Integer, nullable=False)
    monto: Mapped[float] = mapped_column(Float, nullable=False)
    saldo_pendiente: Mapped[float] = mapped_column(Float, nullable=False)
    estado_factura: Mapped[str] = mapped_column(String(40), nullable=False, default="abierta")
    target_mora_simulado: Mapped[str | None] = mapped_column(String(40))
    dias_mora_real: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    cliente: Mapped[Cliente] = relationship(back_populates="facturas")
    gestiones: Mapped[list["GestionCobranza"]] = relationship(back_populates="factura")
    promesas: Mapped[list["PromesaPago"]] = relationship(back_populates="factura")


class GestionCobranza(Base):
    __tablename__ = "gestiones_cobranza"

    gestion_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    factura_id: Mapped[str] = mapped_column(ForeignKey("facturas.factura_id"), index=True)
    cliente_id: Mapped[str] = mapped_column(ForeignKey("clientes.cliente_id"), index=True)
    fecha_gestion: Mapped[date] = mapped_column(Date, nullable=False)
    canal: Mapped[str] = mapped_column(String(40), nullable=False)
    contacto_exitoso: Mapped[bool] = mapped_column(Boolean, nullable=False)
    resultado: Mapped[str] = mapped_column(String(80), nullable=False)
    motivo_no_pago: Mapped[str | None] = mapped_column(String(120))
    dias_mora_en_gestion: Mapped[int] = mapped_column(Integer, nullable=False)
    observacion: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    factura: Mapped[Factura] = relationship(back_populates="gestiones")


class PromesaPago(Base):
    __tablename__ = "promesas_pago"

    promesa_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    gestion_id: Mapped[str] = mapped_column(ForeignKey("gestiones_cobranza.gestion_id"), index=True)
    factura_id: Mapped[str] = mapped_column(ForeignKey("facturas.factura_id"), index=True)
    cliente_id: Mapped[str] = mapped_column(ForeignKey("clientes.cliente_id"), index=True)
    fecha_promesa: Mapped[date] = mapped_column(Date, nullable=False)
    fecha_compromiso: Mapped[date] = mapped_column(Date, nullable=False)
    se_cumplio: Mapped[bool] = mapped_column(Boolean, nullable=False)
    estado_promesa: Mapped[str] = mapped_column(String(40), nullable=False, default="activa")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    factura: Mapped[Factura] = relationship(back_populates="promesas")


class PrediccionFactura(Base):
    __tablename__ = "predicciones_factura"

    prediccion_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    factura_id: Mapped[str] = mapped_column(ForeignKey("facturas.factura_id"), index=True)
    cliente_id: Mapped[str] = mapped_column(ForeignKey("clientes.cliente_id"), index=True)
    fecha_corte: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    modelo_version: Mapped[str] = mapped_column(String(80), nullable=False)
    predicted_class_tecnica: Mapped[str] = mapped_column(String(40), nullable=False)
    predicted_label_usuario: Mapped[str] = mapped_column(String(80), nullable=False)
    prob_pago_plazo: Mapped[float] = mapped_column(Float, nullable=False)
    prob_atraso_leve: Mapped[float] = mapped_column(Float, nullable=False)
    prob_atraso_alto: Mapped[float] = mapped_column(Float, nullable=False)
    prob_atraso_critico: Mapped[float] = mapped_column(Float, nullable=False)
    any_late_probability: Mapped[float] = mapped_column(Float, nullable=False)
    high_risk_probability: Mapped[float] = mapped_column(Float, nullable=False)
    priority_score_0_100: Mapped[float] = mapped_column(Float, nullable=False)
    accion_sugerida_codigo: Mapped[str] = mapped_column(String(80), nullable=False)
    accion_sugerida_nombre: Mapped[str] = mapped_column(String(160), nullable=False)
    motivo_accion: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SegmentoCliente(Base):
    __tablename__ = "segmentos_cliente"

    cliente_id: Mapped[str] = mapped_column(ForeignKey("clientes.cliente_id"), primary_key=True)
    cluster: Mapped[int] = mapped_column(Integer, nullable=False)
    tipo_cliente: Mapped[str] = mapped_column(String(160), nullable=False)
    riesgo_0_100: Mapped[float] = mapped_column(Float, nullable=False)
    rating_estrellas: Mapped[int] = mapped_column(Integer, nullable=False)
    rating_label: Mapped[str] = mapped_column(String(80), nullable=False)
    por_que_rating: Mapped[str] = mapped_column(Text, nullable=False)
    por_que_cluster: Mapped[str] = mapped_column(Text, nullable=False)
    sector_dominante_modal: Mapped[str | None] = mapped_column(String(80))
    n_facturas_total: Mapped[int | None] = mapped_column(Integer)
    n_cortes_total: Mapped[int | None] = mapped_column(Integer)
    fecha_calculo: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    modelo_version: Mapped[str] = mapped_column(String(80), nullable=False, default="fase4_kmeans_k3")


class AccionSugerida(Base):
    __tablename__ = "acciones_sugeridas"

    codigo: Mapped[str] = mapped_column(String(80), primary_key=True)
    nombre: Mapped[str] = mapped_column(String(160), nullable=False)
    descripcion: Mapped[str] = mapped_column(Text, nullable=False)
    canal_recomendado: Mapped[str] = mapped_column(String(80), nullable=False)
    nivel_severidad: Mapped[int] = mapped_column(Integer, nullable=False)
    activa: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class ReglaAccionSugerida(Base):
    __tablename__ = "reglas_accion_sugerida"

    regla_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    accion_codigo: Mapped[str] = mapped_column(ForeignKey("acciones_sugeridas.codigo"), index=True)
    nombre_regla: Mapped[str] = mapped_column(String(160), nullable=False)
    prioridad_regla: Mapped[int] = mapped_column(Integer, nullable=False)
    condiciones_json: Mapped[str] = mapped_column(Text, nullable=False)
    motivo_template: Mapped[str] = mapped_column(Text, nullable=False)
    activa: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

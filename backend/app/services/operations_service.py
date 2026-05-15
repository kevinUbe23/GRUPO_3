from __future__ import annotations

from datetime import date

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.time import utc_now
from app.db.models import Cliente, Factura, GestionCobranza, PrediccionFactura, PromesaPago
from app.schemas import FacturaCreate, FacturaUpdate, GestionCreate, PaymentCreate, PromesaCreate, PromesaUpdate


VALID_CHANNELS = {"whatsapp", "email", "llamada", "visita", "carta_notarial"}
VALID_INVOICE_STATES = {"abierta", "pagada", "en_disputa", "anulada", "castigada"}
NO_CONTACT_RESULTS = {"no_contesta", "numero_invalido", "cliente_ausente"}
CONTACT_RESULTS = {
    "pagado",
    "promesa_de_pago",
    "disputa_monto",
    "rechazo_pago",
    "en_proceso_interno",
    "confirma_pago",
}
PROMISE_STATES = {"activa", "cumplida", "incumplida", "reemplazada", "cancelada"}


def _next_id(db: Session, model, id_col: str, prefix: str) -> str:
    count = db.scalar(select(func.count(getattr(model, id_col)))) or 0
    return f"{prefix}{count + 1:06d}"


def _validate_invoice_state(state: str) -> None:
    if state not in VALID_INVOICE_STATES:
        raise ValueError(f"estado_factura invalido: {state}")


def _validate_invoice_dates(factura: Factura) -> None:
    if factura.fecha_vencimiento < factura.fecha_emision:
        raise ValueError("fecha_vencimiento debe ser mayor o igual a fecha_emision")
    if factura.fecha_pago_real and factura.fecha_pago_real < factura.fecha_emision:
        raise ValueError("fecha_pago_real no puede ser anterior a fecha_emision")


def _validate_invoice_amounts(factura: Factura) -> None:
    if factura.saldo_pendiente > factura.monto:
        raise ValueError("saldo_pendiente no puede ser mayor al monto")


def _apply_payment_state(factura: Factura) -> None:
    if factura.fecha_pago_real:
        factura.estado_factura = "pagada"
        factura.saldo_pendiente = 0.0
        factura.dias_mora_real = max((factura.fecha_pago_real - factura.fecha_vencimiento).days, 0)
    elif factura.estado_factura == "pagada":
        raise ValueError("Una factura pagada debe tener fecha_pago_real")


def create_invoice(db: Session, payload: FacturaCreate) -> Factura:
    if db.get(Cliente, payload.cliente_id) is None:
        raise ValueError(f"No existe el cliente {payload.cliente_id}")
    factura_id = payload.factura_id or _next_id(db, Factura, "factura_id", "FACAPP")
    if db.get(Factura, factura_id) is not None:
        raise ValueError(f"Ya existe la factura {factura_id}")

    estado = payload.estado_factura
    _validate_invoice_state(estado)
    condicion_dias = payload.condicion_dias
    if condicion_dias is None:
        condicion_dias = (payload.fecha_vencimiento - payload.fecha_emision).days
    saldo_pendiente = payload.saldo_pendiente
    if saldo_pendiente is None:
        saldo_pendiente = 0.0 if estado in {"anulada", "castigada"} else payload.monto

    factura = Factura(
        factura_id=factura_id,
        cliente_id=payload.cliente_id,
        fecha_emision=payload.fecha_emision,
        fecha_vencimiento=payload.fecha_vencimiento,
        fecha_pago_real=payload.fecha_pago_real,
        condicion_dias=condicion_dias,
        monto=payload.monto,
        saldo_pendiente=saldo_pendiente,
        estado_factura=estado,
        target_mora_simulado=payload.target_mora_simulado,
        dias_mora_real=payload.dias_mora_real,
    )
    _validate_invoice_dates(factura)
    _apply_payment_state(factura)
    _validate_invoice_amounts(factura)
    db.add(factura)
    db.commit()
    db.refresh(factura)
    return factura


def update_invoice(db: Session, factura_id: str, payload: FacturaUpdate) -> Factura:
    factura = db.get(Factura, factura_id)
    if factura is None:
        raise ValueError(f"No existe la factura {factura_id}")

    updates = payload.model_dump(exclude_unset=True)
    cliente_id = updates.get("cliente_id")
    if cliente_id is not None and cliente_id != factura.cliente_id:
        raise ValueError("cliente_id no puede cambiarse en una factura existente")
    if cliente_id is not None and db.get(Cliente, cliente_id) is None:
        raise ValueError(f"No existe el cliente {cliente_id}")
    estado = updates.get("estado_factura")
    if estado is not None:
        _validate_invoice_state(estado)

    for field, value in updates.items():
        setattr(factura, field, value)
    _validate_invoice_dates(factura)
    _apply_payment_state(factura)
    _validate_invoice_amounts(factura)
    factura.updated_at = utc_now()
    if updates:
        db.execute(delete(PrediccionFactura).where(PrediccionFactura.factura_id == factura.factura_id))
    db.commit()
    db.refresh(factura)
    return factura


def _validate_interaction(payload: GestionCreate, factura: Factura) -> None:
    if payload.canal not in VALID_CHANNELS:
        raise ValueError(f"Canal invalido: {payload.canal}")
    valid_results = CONTACT_RESULTS if payload.contacto_exitoso else NO_CONTACT_RESULTS
    if payload.resultado not in valid_results:
        raise ValueError("resultado no es coherente con contacto_exitoso")
    if payload.fecha_gestion < factura.fecha_emision:
        raise ValueError("fecha_gestion no puede ser anterior a fecha_emision")
    if factura.fecha_pago_real and payload.fecha_gestion > factura.fecha_pago_real:
        raise ValueError("No se puede registrar gestion posterior al pago real")


def create_interaction(db: Session, payload: GestionCreate) -> GestionCobranza:
    factura = db.get(Factura, payload.factura_id)
    if factura is None:
        raise ValueError(f"No existe la factura {payload.factura_id}")
    _validate_interaction(payload, factura)
    dias_mora = max((payload.fecha_gestion - factura.fecha_vencimiento).days, 0)
    gestion = GestionCobranza(
        gestion_id=_next_id(db, GestionCobranza, "gestion_id", "GESAPP"),
        factura_id=factura.factura_id,
        cliente_id=factura.cliente_id,
        fecha_gestion=payload.fecha_gestion,
        canal=payload.canal,
        contacto_exitoso=payload.contacto_exitoso,
        resultado=payload.resultado,
        motivo_no_pago=payload.motivo_no_pago,
        dias_mora_en_gestion=dias_mora,
        observacion=payload.observacion,
    )
    if payload.resultado == "disputa_monto" and factura.estado_factura != "pagada":
        factura.estado_factura = "en_disputa"
        factura.updated_at = utc_now()
    db.add(gestion)
    db.commit()
    db.refresh(gestion)
    return gestion


def create_payment_promise(db: Session, payload: PromesaCreate) -> PromesaPago:
    gestion = db.get(GestionCobranza, payload.gestion_id)
    if gestion is None:
        raise ValueError(f"No existe la gestion {payload.gestion_id}")
    if gestion.resultado != "promesa_de_pago":
        raise ValueError("Solo una gestion con resultado promesa_de_pago puede originar una promesa")
    existing = db.scalar(
        select(func.count(PromesaPago.promesa_id)).where(PromesaPago.gestion_id == payload.gestion_id)
    )
    if existing:
        raise ValueError("Ya existe una promesa asociada a esta gestion")
    if payload.fecha_compromiso < gestion.fecha_gestion:
        raise ValueError("fecha_compromiso no puede ser anterior a fecha_promesa")
    promesa = PromesaPago(
        promesa_id=_next_id(db, PromesaPago, "promesa_id", "PROAPP"),
        gestion_id=gestion.gestion_id,
        factura_id=gestion.factura_id,
        cliente_id=gestion.cliente_id,
        fecha_promesa=gestion.fecha_gestion,
        fecha_compromiso=payload.fecha_compromiso,
        se_cumplio=False,
        estado_promesa="activa",
    )
    db.add(promesa)
    db.commit()
    db.refresh(promesa)
    return promesa


def update_payment_promise(db: Session, promesa_id: str, payload: PromesaUpdate) -> PromesaPago:
    promesa = db.get(PromesaPago, promesa_id)
    if promesa is None:
        raise ValueError(f"No existe la promesa {promesa_id}")
    if payload.estado_promesa not in PROMISE_STATES:
        raise ValueError(f"estado_promesa invalido: {payload.estado_promesa}")
    promesa.estado_promesa = payload.estado_promesa
    if payload.se_cumplio is not None:
        promesa.se_cumplio = payload.se_cumplio
    elif payload.estado_promesa == "cumplida":
        promesa.se_cumplio = True
    elif payload.estado_promesa == "incumplida":
        promesa.se_cumplio = False
    promesa.updated_at = utc_now()
    db.commit()
    db.refresh(promesa)
    return promesa


def register_payment(db: Session, payload: PaymentCreate) -> Factura:
    factura = db.get(Factura, payload.factura_id)
    if factura is None:
        raise ValueError(f"No existe la factura {payload.factura_id}")
    if payload.fecha_pago < factura.fecha_emision:
        raise ValueError("fecha_pago no puede ser anterior a fecha_emision")
    factura.fecha_pago_real = payload.fecha_pago
    factura.saldo_pendiente = 0.0
    factura.estado_factura = "pagada"
    factura.dias_mora_real = max((payload.fecha_pago - factura.fecha_vencimiento).days, 0)
    factura.updated_at = utc_now()

    promesas = db.scalars(
        select(PromesaPago)
        .where(PromesaPago.factura_id == factura.factura_id)
        .where(PromesaPago.estado_promesa == "activa")
    ).all()
    for promesa in promesas:
        promesa.se_cumplio = payload.fecha_pago <= promesa.fecha_compromiso
        promesa.estado_promesa = "cumplida" if promesa.se_cumplio else "incumplida"
        promesa.updated_at = utc_now()
    db.commit()
    db.refresh(factura)
    return factura


def active_invoice_ids_at_cutoff(db: Session, fecha_corte: date, limit: int | None = None) -> list[str]:
    stmt = (
        select(Factura.factura_id)
        .where(Factura.fecha_emision <= fecha_corte)
        .where(Factura.estado_factura.notin_(["anulada", "castigada"]))
        .where((Factura.fecha_pago_real.is_(None)) | (Factura.fecha_pago_real > fecha_corte))
        .order_by(Factura.fecha_vencimiento, Factura.factura_id)
    )
    if limit:
        stmt = stmt.limit(limit)
    return list(db.scalars(stmt))

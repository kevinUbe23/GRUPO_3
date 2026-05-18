from __future__ import annotations

from datetime import date

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func, or_, select
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.core.config import get_settings
from app.db.models import (
    AccionSugerida,
    Cliente,
    Factura,
    GestionCobranza,
    PrediccionFactura,
    PromesaPago,
    SegmentoCliente,
)
from app.schemas import (
    ActionCatalogOut,
    ClienteOut,
    DashboardSummary,
    FacturaCreate,
    FacturaListOut,
    FacturaOut,
    FacturaUpdate,
    GestionCreate,
    InitDbResult,
    GestionOut,
    InteractionOut,
    PaymentCreate,
    PredictionDailyOut,
    PredictionOut,
    PredictionHistoryOut,
    PrioritizedInvoiceOut,
    PromesaCreate,
    PromesaOut,
    PromesaUpdate,
    RecalculateRequest,
    RecalculateResult,
    ScoreRequest,
    SegmentoClienteOut,
)
from app.services.import_service import reset_and_import_seed_data
from app.services.operations_service import (
    active_invoice_ids_at_cutoff,
    create_invoice,
    create_interaction,
    create_payment_promise,
    cutoff_invoice_state,
    interaction_payload,
    invoice_status_at_cutoff,
    observable_days_late,
    register_payment,
    update_invoice,
    update_payment_promise,
)
from app.services.prediction_service import get_prediction_service


router = APIRouter()


def _invoice_list_payload(factura: Factura, fecha_corte: date | None) -> dict:
    fecha_pago_visible = (
        factura.fecha_pago_real
        if fecha_corte is None or (factura.fecha_pago_real and factura.fecha_pago_real <= fecha_corte)
        else None
    )
    dias_mora_real_visible = factura.dias_mora_real if fecha_pago_visible else None
    saldo_pendiente_visible = factura.saldo_pendiente if fecha_corte is None or fecha_pago_visible else factura.monto
    target_mora_visible = factura.target_mora_simulado if fecha_corte is None or fecha_pago_visible else None
    payload = {
        "factura_id": factura.factura_id,
        "cliente_id": factura.cliente_id,
        "fecha_emision": factura.fecha_emision,
        "fecha_vencimiento": factura.fecha_vencimiento,
        "fecha_pago_real": fecha_pago_visible,
        "condicion_dias": factura.condicion_dias,
        "monto": factura.monto,
        "saldo_pendiente": saldo_pendiente_visible,
        "estado_factura": invoice_status_at_cutoff(factura, fecha_corte) if fecha_corte else factura.estado_factura,
        "target_mora_simulado": target_mora_visible,
        "dias_mora_real": dias_mora_real_visible,
        "fecha_corte": fecha_corte,
        "estado_corte": None,
        "dias_mora_observable": None,
    }
    if fecha_corte:
        payload["estado_corte"] = cutoff_invoice_state(factura, fecha_corte)
        payload["dias_mora_observable"] = observable_days_late(factura, fecha_corte)
    return payload


def _promise_payload(promesa: PromesaPago, fecha_corte: date | None = None) -> dict:
    estado_promesa = promesa.estado_promesa
    se_cumplio = promesa.se_cumplio
    if fecha_corte and promesa.updated_at and promesa.updated_at.date() > fecha_corte:
        estado_promesa = "activa"
        se_cumplio = False
    return {
        "promesa_id": promesa.promesa_id,
        "gestion_id": promesa.gestion_id,
        "factura_id": promesa.factura_id,
        "cliente_id": promesa.cliente_id,
        "fecha_promesa": promesa.fecha_promesa,
        "fecha_compromiso": promesa.fecha_compromiso,
        "se_cumplio": se_cumplio,
        "estado_promesa": estado_promesa,
    }


@router.post("/admin/init-db", response_model=InitDbResult, tags=["admin"])
def init_db(db: Session = Depends(get_db)) -> InitDbResult:
    return InitDbResult(**reset_and_import_seed_data(db))


@router.get("/dashboard/summary", response_model=DashboardSummary, tags=["dashboard"])
def dashboard_summary(
    fecha_corte: date | None = None,
    db: Session = Depends(get_db),
) -> DashboardSummary:
    total_facturas = db.scalar(select(func.count(Factura.factura_id))) or 0
    if fecha_corte:
        active_stmt = (
            select(Factura)
            .where(Factura.fecha_emision <= fecha_corte)
            .where(Factura.estado_factura.notin_(["anulada", "castigada"]))
            .where(or_(Factura.fecha_pago_real.is_(None), Factura.fecha_pago_real > fecha_corte))
        )
        active_facturas = list(db.scalars(active_stmt))
        facturas_activas = len(active_facturas)
        monto_pendiente = sum(f.monto for f in active_facturas)
        monto_vencido = sum(f.monto for f in active_facturas if f.fecha_vencimiento < fecha_corte)
        clientes_con_monto_vencido = len(
            {f.cliente_id for f in active_facturas if f.fecha_vencimiento < fecha_corte}
        )
    else:
        facturas_activas = db.scalar(
            select(func.count(Factura.factura_id)).where(Factura.estado_factura == "abierta")
        ) or 0
        monto_pendiente = db.scalar(select(func.coalesce(func.sum(Factura.saldo_pendiente), 0.0))) or 0.0
        today = date.today()
        monto_vencido = db.scalar(
            select(func.coalesce(func.sum(Factura.saldo_pendiente), 0.0))
            .where(Factura.estado_factura == "abierta")
            .where(Factura.fecha_vencimiento < today)
        ) or 0.0
        clientes_con_monto_vencido = db.scalar(
            select(func.count(func.distinct(Factura.cliente_id)))
            .where(Factura.estado_factura == "abierta")
            .where(Factura.fecha_vencimiento < today)
        ) or 0
    promesas_activas = db.scalar(
        select(func.count(PromesaPago.promesa_id)).where(PromesaPago.estado_promesa == "activa")
    ) or 0
    facturas_en_disputa = db.scalar(
        select(func.count(Factura.factura_id)).where(Factura.estado_factura == "en_disputa")
    ) or 0
    if fecha_corte:
        promesas_corte = db.scalars(
            select(PromesaPago)
            .where(PromesaPago.fecha_promesa <= fecha_corte)
            .where(PromesaPago.fecha_compromiso >= fecha_corte)
        )
        promesas_activas = sum(
            1 for promesa in promesas_corte if _promise_payload(promesa, fecha_corte)["estado_promesa"] == "activa"
        )
        facturas_en_disputa = sum(
            1
            for factura in active_facturas
            if factura.estado_factura == "en_disputa"
            and (factura.updated_at is None or factura.updated_at.date() <= fecha_corte)
        )
    return DashboardSummary(
        total_facturas=total_facturas,
        facturas_activas=facturas_activas,
        monto_pendiente=float(monto_pendiente),
        monto_vencido=float(monto_vencido),
        clientes_con_monto_vencido=clientes_con_monto_vencido,
        promesas_activas=promesas_activas,
        facturas_en_disputa=facturas_en_disputa,
    )


@router.get("/customers", response_model=list[ClienteOut], tags=["customers"])
def list_customers(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    sector: str | None = None,
    q: str | None = Query(default=None, min_length=1, max_length=120),
    db: Session = Depends(get_db),
) -> list[ClienteOut]:
    stmt = select(Cliente)
    if sector:
        stmt = stmt.where(Cliente.sector == sector)
    if q:
        search = f"%{q.strip()}%"
        stmt = stmt.where(Cliente.nombre.ilike(search))
    stmt = stmt.order_by(Cliente.nombre, Cliente.cliente_id).limit(limit).offset(offset)
    return list(db.scalars(stmt))


@router.get("/customers/{cliente_id}", response_model=ClienteOut, tags=["customers"])
def get_customer(cliente_id: str, db: Session = Depends(get_db)) -> ClienteOut:
    cliente = db.get(Cliente, cliente_id)
    if cliente is None:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    return cliente


@router.get("/customers/{cliente_id}/segment", response_model=SegmentoClienteOut, tags=["customers"])
def get_customer_segment(cliente_id: str, db: Session = Depends(get_db)) -> SegmentoClienteOut:
    segmento = db.get(SegmentoCliente, cliente_id)
    if segmento is None:
        raise HTTPException(status_code=404, detail="Segmento no encontrado")
    return segmento


@router.get("/invoices", response_model=list[FacturaListOut], tags=["invoices"])
def list_invoices(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    active_only: bool = False,
    cliente_id: str | None = None,
    fecha_corte: date | None = None,
    db: Session = Depends(get_db),
) -> list[FacturaListOut]:
    stmt = select(Factura).order_by(Factura.factura_id).limit(limit).offset(offset)
    if fecha_corte:
        stmt = stmt.where(Factura.fecha_emision <= fecha_corte)
    if active_only and fecha_corte:
        stmt = (
            stmt.where(Factura.estado_factura.notin_(["anulada", "castigada"]))
            .where(or_(Factura.fecha_pago_real.is_(None), Factura.fecha_pago_real > fecha_corte))
        )
    elif active_only:
        stmt = stmt.where(Factura.estado_factura == "abierta")
    if cliente_id:
        stmt = stmt.where(Factura.cliente_id == cliente_id)
    return [_invoice_list_payload(factura, fecha_corte) for factura in db.scalars(stmt)]


@router.post("/invoices", response_model=FacturaOut, status_code=201, tags=["invoices"])
def post_invoice(payload: FacturaCreate, db: Session = Depends(get_db)) -> FacturaOut:
    try:
        return create_invoice(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/invoices/prioritized", response_model=list[PrioritizedInvoiceOut], tags=["invoices"])
def prioritized_invoices(
    limit: int = Query(default=50, ge=1, le=500),
    fecha_corte: date | None = None,
    estado_corte: str = Query(default="preventive", pattern="^(preventive|overdue|paid)$"),
    db: Session = Depends(get_db),
) -> list[PrioritizedInvoiceOut]:
    if fecha_corte is None:
        fecha_corte = date.today()
    latest_stmt = select(
        PrediccionFactura.factura_id,
        func.max(PrediccionFactura.prediccion_id).label("max_prediccion_id"),
    ).where(PrediccionFactura.fecha_corte == fecha_corte)
    latest_subq = latest_stmt.group_by(PrediccionFactura.factura_id).subquery()

    stmt = (
        select(Factura, Cliente, PrediccionFactura, SegmentoCliente)
        .join(Cliente, Factura.cliente_id == Cliente.cliente_id)
        .where(Factura.fecha_emision <= fecha_corte)
        .where(Factura.estado_factura.notin_(["anulada", "castigada"]))
        .outerjoin(latest_subq, latest_subq.c.factura_id == Factura.factura_id)
        .outerjoin(
            PrediccionFactura,
            PrediccionFactura.prediccion_id == latest_subq.c.max_prediccion_id,
        )
        .outerjoin(SegmentoCliente, SegmentoCliente.cliente_id == Factura.cliente_id)
        .order_by(desc(func.coalesce(PrediccionFactura.priority_score_0_100, 0)), Factura.factura_id)
        .limit(limit)
    )
    if estado_corte == "paid":
        stmt = stmt.where(Factura.fecha_pago_real.is_not(None)).where(Factura.fecha_pago_real <= fecha_corte)
    else:
        stmt = stmt.where(or_(Factura.fecha_pago_real.is_(None), Factura.fecha_pago_real > fecha_corte))
        if estado_corte == "overdue":
            stmt = stmt.where(Factura.fecha_vencimiento < fecha_corte)
        else:
            stmt = stmt.where(Factura.fecha_vencimiento >= fecha_corte)

    rows = []
    for factura, cliente, pred, segmento in db.execute(stmt).all():
        estado_al_corte = invoice_status_at_cutoff(factura, fecha_corte)
        fecha_pago_visible = (
            factura.fecha_pago_real
            if factura.fecha_pago_real and factura.fecha_pago_real <= fecha_corte
            else None
        )
        rows.append(
            {
                "factura_id": factura.factura_id,
                "cliente_id": cliente.cliente_id,
                "cliente_nombre": cliente.nombre,
                "sector": cliente.sector,
                "monto": factura.monto,
                "fecha_emision": factura.fecha_emision,
                "fecha_vencimiento": factura.fecha_vencimiento,
                "fecha_pago_real": fecha_pago_visible,
                "dias_mora_real": factura.dias_mora_real if fecha_pago_visible else None,
                "dias_mora_observable": observable_days_late(factura, fecha_corte),
                "estado_factura": estado_al_corte,
                "estado_factura_actual": factura.estado_factura,
                "estado_corte": cutoff_invoice_state(factura, fecha_corte),
                "fecha_corte": fecha_corte,
                "predicted_label_usuario": pred.predicted_label_usuario if pred else None,
                "prob_pago_plazo": pred.prob_pago_plazo if pred else None,
                "prob_atraso_leve": pred.prob_atraso_leve if pred else None,
                "prob_atraso_alto": pred.prob_atraso_alto if pred else None,
                "prob_atraso_critico": pred.prob_atraso_critico if pred else None,
                "any_late_probability": pred.any_late_probability if pred else None,
                "high_risk_probability": pred.high_risk_probability if pred else None,
                "priority_score_0_100": pred.priority_score_0_100 if pred else None,
                "accion_sugerida": pred.accion_sugerida_nombre if pred else None,
                "rating_estrellas": segmento.rating_estrellas if segmento else None,
            }
        )
    return rows


@router.post("/scoring/recalculate", response_model=RecalculateResult, tags=["prediction"])
def recalculate_scoring(
    payload: RecalculateRequest,
    db: Session = Depends(get_db),
) -> RecalculateResult:
    invoice_ids = active_invoice_ids_at_cutoff(db, payload.fecha_corte, payload.limit)
    errores: list[dict] = []
    evaluated = 0
    service = get_prediction_service()
    for factura_id in invoice_ids:
        try:
            service.predict_invoice(
                db,
                factura_id,
                payload.fecha_corte,
                persist=payload.persist,
                use_prepared_snapshot=False,
            )
            evaluated += 1
        except ValueError as exc:
            errores.append({"factura_id": factura_id, "error": str(exc)})
    return RecalculateResult(
        fecha_corte=payload.fecha_corte,
        total_evaluadas=evaluated,
        total_con_error=len(errores),
        errores=errores[:20],
    )


@router.get("/invoices/{factura_id}", response_model=FacturaOut, tags=["invoices"])
def get_invoice(
    factura_id: str,
    fecha_corte: date | None = None,
    db: Session = Depends(get_db),
) -> FacturaOut:
    factura = db.get(Factura, factura_id)
    if factura is None:
        raise HTTPException(status_code=404, detail="Factura no encontrada")
    if fecha_corte:
        return _invoice_list_payload(factura, fecha_corte)
    return factura


@router.patch("/invoices/{factura_id}", response_model=FacturaOut, tags=["invoices"])
def patch_invoice(
    factura_id: str,
    payload: FacturaUpdate,
    db: Session = Depends(get_db),
) -> FacturaOut:
    try:
        return update_invoice(db, factura_id, payload)
    except ValueError as exc:
        status_code = 404 if str(exc).startswith("No existe la factura") else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@router.get("/invoices/{factura_id}/interactions", response_model=list[InteractionOut], tags=["invoices"])
def get_invoice_interactions(
    factura_id: str,
    fecha_corte: date | None = None,
    db: Session = Depends(get_db),
) -> list[InteractionOut]:
    factura = db.get(Factura, factura_id)
    if factura is None:
        raise HTTPException(status_code=404, detail="Factura no encontrada")
    stmt = (
        select(GestionCobranza)
        .where(GestionCobranza.factura_id == factura_id)
        .order_by(GestionCobranza.fecha_gestion, GestionCobranza.gestion_id)
    )
    if fecha_corte:
        stmt = stmt.where(GestionCobranza.fecha_gestion <= fecha_corte)
    return [interaction_payload(row) for row in db.scalars(stmt)]


@router.get("/invoices/{factura_id}/promises", response_model=list[PromesaOut], tags=["promises"])
def get_invoice_promises(
    factura_id: str,
    fecha_corte: date | None = None,
    db: Session = Depends(get_db),
) -> list[PromesaOut]:
    factura = db.get(Factura, factura_id)
    if factura is None:
        raise HTTPException(status_code=404, detail="Factura no encontrada")
    stmt = (
        select(PromesaPago)
        .where(PromesaPago.factura_id == factura_id)
        .order_by(PromesaPago.fecha_promesa, PromesaPago.promesa_id)
    )
    if fecha_corte:
        stmt = stmt.where(PromesaPago.fecha_promesa <= fecha_corte)
    return list(db.scalars(stmt))


@router.get("/invoices/{factura_id}/prediction-daily", response_model=list[PredictionDailyOut], tags=["prediction"])
def get_invoice_prediction_daily(
    factura_id: str,
    fecha_corte: date = Query(...),
    db: Session = Depends(get_db),
) -> list[PredictionDailyOut]:
    try:
        return get_prediction_service().predict_invoice_daily(
            db,
            factura_id=factura_id,
            fecha_corte=fecha_corte,
        )
    except ValueError as exc:
        status_code = 404 if str(exc).startswith("No existe la factura") else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@router.get("/invoices/{factura_id}/predictions", response_model=list[PredictionHistoryOut], tags=["prediction"])
def get_invoice_prediction_history(
    factura_id: str,
    fecha_corte: date | None = None,
    db: Session = Depends(get_db),
) -> list[PredictionHistoryOut]:
    factura = db.get(Factura, factura_id)
    if factura is None:
        raise HTTPException(status_code=404, detail="Factura no encontrada")
    fecha_pago_visible = (
        factura.fecha_pago_real
        if fecha_corte is None or (factura.fecha_pago_real and factura.fecha_pago_real <= fecha_corte)
        else None
    )
    dias_mora_real_visible = factura.dias_mora_real if fecha_pago_visible else None
    target_mora_visible = factura.target_mora_simulado if fecha_pago_visible else None

    stmt = (
        select(PrediccionFactura)
        .where(PrediccionFactura.factura_id == factura_id)
        .order_by(PrediccionFactura.fecha_corte, PrediccionFactura.created_at, PrediccionFactura.prediccion_id)
    )
    if fecha_corte:
        stmt = stmt.where(PrediccionFactura.fecha_corte <= fecha_corte)
    return [
        PredictionHistoryOut(
            prediccion_id=pred.prediccion_id,
            factura_id=pred.factura_id,
            cliente_id=pred.cliente_id,
            fecha_corte=pred.fecha_corte,
            modelo_version=pred.modelo_version,
            predicted_class_tecnica=pred.predicted_class_tecnica,
            predicted_label_usuario=pred.predicted_label_usuario,
            prob_pago_plazo=pred.prob_pago_plazo,
            prob_atraso_leve=pred.prob_atraso_leve,
            prob_atraso_alto=pred.prob_atraso_alto,
            prob_atraso_critico=pred.prob_atraso_critico,
            any_late_probability=pred.any_late_probability,
            high_risk_probability=pred.high_risk_probability,
            priority_score_0_100=pred.priority_score_0_100,
            accion_sugerida_codigo=pred.accion_sugerida_codigo,
            accion_sugerida_nombre=pred.accion_sugerida_nombre,
            motivo_accion=pred.motivo_accion,
            fecha_pago_real=fecha_pago_visible,
            dias_mora_real=dias_mora_real_visible,
            target_mora_simulado=target_mora_visible,
        )
        for pred in db.scalars(stmt)
    ]


@router.post("/collections/interactions", response_model=GestionOut, tags=["collections"])
def create_collection_interaction(
    payload: GestionCreate,
    db: Session = Depends(get_db),
) -> GestionOut:
    try:
        gestion = create_interaction(db, payload)
        if payload.recalculate:
            get_prediction_service().predict_invoice(
                db,
                factura_id=gestion.factura_id,
                fecha_corte=gestion.fecha_gestion,
                persist=True,
                use_prepared_snapshot=False,
            )
        return gestion
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/payment-promises", response_model=PromesaOut, tags=["promises"])
def create_promise(payload: PromesaCreate, db: Session = Depends(get_db)) -> PromesaOut:
    try:
        promesa = create_payment_promise(db, payload)
        if payload.recalculate:
            get_prediction_service().predict_invoice(
                db,
                factura_id=promesa.factura_id,
                fecha_corte=promesa.fecha_promesa,
                persist=True,
                use_prepared_snapshot=False,
            )
        return promesa
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/payment-promises/{promesa_id}", response_model=PromesaOut, tags=["promises"])
def patch_promise(
    promesa_id: str,
    payload: PromesaUpdate,
    db: Session = Depends(get_db),
) -> PromesaOut:
    try:
        return update_payment_promise(db, promesa_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/payments", response_model=FacturaOut, tags=["payments"])
def post_payment(payload: PaymentCreate, db: Session = Depends(get_db)) -> FacturaOut:
    try:
        return register_payment(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/invoices/{factura_id}/score", response_model=PredictionOut, tags=["prediction"])
def score_invoice(
    factura_id: str,
    payload: ScoreRequest,
    db: Session = Depends(get_db),
) -> PredictionOut:
    fecha_corte = payload.fecha_corte or date.today()
    try:
        return get_prediction_service().predict_invoice(
            db,
            factura_id=factura_id,
            fecha_corte=fecha_corte,
            persist=payload.persist,
            use_prepared_snapshot=payload.use_prepared_snapshot,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/actions/recommend", response_model=PredictionOut, tags=["actions"])
def recommend_action_for_invoice(
    factura_id: str = Query(...),
    fecha_corte: date = Query(...),
    db: Session = Depends(get_db),
) -> PredictionOut:
    try:
        return get_prediction_service().predict_invoice(
            db,
            factura_id=factura_id,
            fecha_corte=fecha_corte,
            persist=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/actions", response_model=list[ActionCatalogOut], tags=["actions"])
def list_actions(db: Session = Depends(get_db)) -> list[ActionCatalogOut]:
    return list(db.scalars(select(AccionSugerida).order_by(AccionSugerida.nivel_severidad, AccionSugerida.codigo)))


@router.get("/model/status", tags=["model"])
def model_status() -> dict:
    service = get_prediction_service()
    return {
        "modelo_version": service.model_version,
        "feature_count": len(service.feature_cols),
        "class_names": service.class_names,
    }


@router.get("/model/metrics", tags=["model"])
def model_metrics() -> dict:
    metrics_path = get_settings().model_outputs_dir / "benchmark_metrics.csv"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metricas del modelo no encontradas")
    df = pd.read_csv(metrics_path)
    selected = df[df["model"].astype(str).eq("Logistic Regression")]
    if selected.empty:
        selected = df.head(1)
    return selected.iloc[0].to_dict()

from __future__ import annotations

from datetime import date, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.db.database import Base, SessionLocal, engine
from app.db.models import PrediccionFactura
from app.main import app
from app.services.import_service import reset_and_import_seed_data


def _seed_db() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        reset_and_import_seed_data(db)


def _prediction(
    factura_id: str,
    cliente_id: str,
    fecha_corte: date,
    priority: float,
) -> PrediccionFactura:
    return PrediccionFactura(
        factura_id=factura_id,
        cliente_id=cliente_id,
        fecha_corte=fecha_corte,
        modelo_version="test",
        predicted_class_tecnica="+30",
        predicted_label_usuario="Atraso leve probable",
        prob_pago_plazo=0.5,
        prob_atraso_leve=0.4,
        prob_atraso_alto=0.07,
        prob_atraso_critico=0.03,
        any_late_probability=0.5,
        high_risk_probability=0.1,
        priority_score_0_100=priority,
        accion_sugerida_codigo="RECORDATORIO_PREVENTIVO",
        accion_sugerida_nombre="Enviar recordatorio preventivo",
        motivo_accion="test",
    )


def test_actions_catalog_and_batch_recalculate() -> None:
    _seed_db()
    client = TestClient(app)

    actions = client.get("/api/v1/actions")
    assert actions.status_code == 200
    assert any(action["codigo"] == "LLAMADA_URGENTE" for action in actions.json())

    response = client.post(
        "/api/v1/scoring/recalculate",
        json={"fecha_corte": "2023-01-30", "limit": 5, "persist": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_evaluadas"] > 0
    assert body["total_con_error"] == 0

    prioritized = client.get("/api/v1/invoices/prioritized?fecha_corte=2023-01-30&limit=5")
    assert prioritized.status_code == 200
    assert len(prioritized.json()) > 0
    assert prioritized.json()[0]["estado_factura"] == "abierta"
    assert all(row["fecha_corte"] == "2023-01-30" for row in prioritized.json())

    dashboard = client.get("/api/v1/dashboard/summary?fecha_corte=2023-01-30")
    assert dashboard.status_code == 200
    assert dashboard.json()["facturas_activas"] > 0

    active_invoices = client.get("/api/v1/invoices?active_only=true&fecha_corte=2023-01-30&limit=5")
    assert active_invoices.status_code == 200
    assert len(active_invoices.json()) > 0


def test_invoice_cutoff_classification_in_list_and_prioritized_queue() -> None:
    _seed_db()
    client = TestClient(app)
    cutoff = date(2024, 1, 20)
    invoices = [
        {
            "factura_id": "FACAPPCUT001",
            "cliente_id": "CLI0001",
            "fecha_emision": "2024-01-01",
            "fecha_vencimiento": "2024-02-01",
            "monto": 1000.0,
        },
        {
            "factura_id": "FACAPPCUT002",
            "cliente_id": "CLI0001",
            "fecha_emision": "2024-01-01",
            "fecha_vencimiento": "2024-01-10",
            "monto": 1000.0,
        },
        {
            "factura_id": "FACAPPCUT003",
            "cliente_id": "CLI0001",
            "fecha_emision": "2024-01-01",
            "fecha_vencimiento": "2024-01-10",
            "fecha_pago_real": "2024-01-15",
            "monto": 1000.0,
        },
        {
            "factura_id": "FACAPPCUT004",
            "cliente_id": "CLI0001",
            "fecha_emision": "2024-01-01",
            "fecha_vencimiento": "2024-01-10",
            "fecha_pago_real": "2024-02-15",
            "monto": 1000.0,
        },
        {
            "factura_id": "FACAPPCUT005",
            "cliente_id": "CLI0001",
            "fecha_emision": "2024-02-01",
            "fecha_vencimiento": "2024-03-01",
            "monto": 1000.0,
        },
    ]
    for payload in invoices:
        created = client.post("/api/v1/invoices", json=payload)
        assert created.status_code == 201

    with SessionLocal() as db:
        db.add_all(
            [
                _prediction("FACAPPCUT001", "CLI0001", cutoff, 10.0),
                _prediction("FACAPPCUT002", "CLI0001", cutoff, 20.0),
                _prediction("FACAPPCUT003", "CLI0001", cutoff, 30.0),
                _prediction("FACAPPCUT004", "CLI0001", cutoff, 40.0),
                _prediction("FACAPPCUT005", "CLI0001", cutoff, 50.0),
            ]
        )
        db.commit()

    listed = client.get("/api/v1/invoices?cliente_id=CLI0001&fecha_corte=2024-01-20&limit=500")
    assert listed.status_code == 200
    listed_by_id = {row["factura_id"]: row for row in listed.json() if row["factura_id"].startswith("FACAPPCUT")}
    assert listed_by_id["FACAPPCUT001"]["estado_corte"] == "preventive"
    assert listed_by_id["FACAPPCUT001"]["dias_mora_observable"] == 0
    assert listed_by_id["FACAPPCUT002"]["estado_corte"] == "overdue"
    assert listed_by_id["FACAPPCUT002"]["dias_mora_observable"] == 10
    assert listed_by_id["FACAPPCUT003"]["estado_corte"] == "paid"
    assert listed_by_id["FACAPPCUT003"]["dias_mora_observable"] == 5
    assert listed_by_id["FACAPPCUT004"]["estado_corte"] == "overdue"
    assert listed_by_id["FACAPPCUT004"]["fecha_pago_real"] is None
    assert listed_by_id["FACAPPCUT004"]["dias_mora_real"] is None
    assert "FACAPPCUT005" not in listed_by_id

    detail_before_payment = client.get("/api/v1/invoices/FACAPPCUT004?fecha_corte=2024-01-20")
    assert detail_before_payment.status_code == 200
    assert detail_before_payment.json()["estado_factura"] == "abierta"
    assert detail_before_payment.json()["saldo_pendiente"] == 1000.0
    assert detail_before_payment.json()["fecha_pago_real"] is None
    assert detail_before_payment.json()["dias_mora_real"] is None

    history_before_payment = client.get("/api/v1/invoices/FACAPPCUT004/predictions?fecha_corte=2024-01-20")
    assert history_before_payment.status_code == 200
    assert history_before_payment.json()[0]["fecha_pago_real"] is None
    assert history_before_payment.json()[0]["dias_mora_real"] is None
    assert history_before_payment.json()[0]["target_mora_simulado"] is None

    preventive = client.get(
        "/api/v1/invoices/prioritized?fecha_corte=2024-01-20&estado_corte=preventive&limit=500"
    )
    overdue = client.get(
        "/api/v1/invoices/prioritized?fecha_corte=2024-01-20&estado_corte=overdue&limit=500"
    )
    paid = client.get(
        "/api/v1/invoices/prioritized?fecha_corte=2024-01-20&estado_corte=paid&limit=500"
    )
    assert preventive.status_code == 200
    assert overdue.status_code == 200
    assert paid.status_code == 200
    assert "FACAPPCUT001" in {row["factura_id"] for row in preventive.json()}
    overdue_by_id = {row["factura_id"]: row for row in overdue.json() if row["factura_id"].startswith("FACAPPCUT")}
    assert "FACAPPCUT002" in overdue_by_id
    assert overdue_by_id["FACAPPCUT004"]["fecha_pago_real"] is None
    assert "FACAPPCUT005" not in overdue_by_id
    paid_by_id = {row["factura_id"]: row for row in paid.json() if row["factura_id"].startswith("FACAPPCUT")}
    assert paid_by_id["FACAPPCUT003"]["estado_corte"] == "paid"
    assert paid_by_id["FACAPPCUT003"]["estado_factura"] == "pagada"


def test_cors_preflight_allows_local_dev_ports() -> None:
    client = TestClient(app)

    response = client.options(
        "/api/v1/dashboard/summary?fecha_corte=2023-01-30",
        headers={
            "Origin": "http://localhost:3001",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3001"


def test_create_and_patch_invoice() -> None:
    _seed_db()
    client = TestClient(app)

    created = client.post(
        "/api/v1/invoices",
        json={
            "factura_id": "FACAPPTEST001",
            "cliente_id": "CLI0001",
            "fecha_emision": "2024-01-10",
            "fecha_vencimiento": "2024-02-09",
            "monto": 1500.0,
        },
    )
    assert created.status_code == 201
    body = created.json()
    assert body["factura_id"] == "FACAPPTEST001"
    assert body["condicion_dias"] == 30
    assert body["saldo_pendiente"] == 1500.0
    assert body["estado_factura"] == "abierta"

    with SessionLocal() as db:
        db.add_all(
            [
                PrediccionFactura(
                    factura_id="FACAPPTEST001",
                    cliente_id="CLI0001",
                    fecha_corte=date(2024, 1, 20),
                    modelo_version="test",
                    predicted_class_tecnica="on_time",
                    predicted_label_usuario="Pago esperado dentro del plazo",
                    prob_pago_plazo=0.9,
                    prob_atraso_leve=0.05,
                    prob_atraso_alto=0.03,
                    prob_atraso_critico=0.02,
                    any_late_probability=0.1,
                    high_risk_probability=0.05,
                    priority_score_0_100=5.0,
                    accion_sugerida_codigo="SIN_ACCION",
                    accion_sugerida_nombre="Monitorear sin gestion inmediata",
                    motivo_accion="test",
                ),
                PrediccionFactura(
                    factura_id="FACAPPTEST001",
                    cliente_id="CLI0001",
                    fecha_corte=date(2024, 1, 20),
                    modelo_version="test",
                    predicted_class_tecnica="+30",
                    predicted_label_usuario="Atraso leve probable",
                    prob_pago_plazo=0.5,
                    prob_atraso_leve=0.4,
                    prob_atraso_alto=0.07,
                    prob_atraso_critico=0.03,
                    any_late_probability=0.5,
                    high_risk_probability=0.1,
                    priority_score_0_100=25.0,
                    accion_sugerida_codigo="RECORDATORIO_PREVENTIVO",
                    accion_sugerida_nombre="Enviar recordatorio preventivo",
                    motivo_accion="test ultima del mismo corte",
                ),
                PrediccionFactura(
                    factura_id="FACAPPTEST001",
                    cliente_id="CLI0001",
                    fecha_corte=date(2024, 1, 21),
                    modelo_version="test",
                    predicted_class_tecnica="+90",
                    predicted_label_usuario="Atraso critico probable",
                    prob_pago_plazo=0.01,
                    prob_atraso_leve=0.04,
                    prob_atraso_alto=0.05,
                    prob_atraso_critico=0.9,
                    any_late_probability=0.99,
                    high_risk_probability=0.95,
                    priority_score_0_100=99.0,
                    accion_sugerida_codigo="LLAMADA_URGENTE",
                    accion_sugerida_nombre="Realizar llamada urgente",
                    motivo_accion="test otro corte",
                ),
            ]
        )
        db.commit()

    prioritized_same_cutoff = client.get("/api/v1/invoices/prioritized?fecha_corte=2024-01-20&limit=500")
    assert prioritized_same_cutoff.status_code == 200
    same_cutoff_row = next(
        row for row in prioritized_same_cutoff.json() if row["factura_id"] == "FACAPPTEST001"
    )
    assert same_cutoff_row["fecha_corte"] == "2024-01-20"
    assert same_cutoff_row["priority_score_0_100"] == 25.0
    assert same_cutoff_row["prob_atraso_leve"] == 0.4

    prioritized_other_cutoff = client.get("/api/v1/invoices/prioritized?fecha_corte=2024-01-21&limit=500")
    assert prioritized_other_cutoff.status_code == 200
    other_cutoff_row = next(
        row for row in prioritized_other_cutoff.json() if row["factura_id"] == "FACAPPTEST001"
    )
    assert other_cutoff_row["fecha_corte"] == "2024-01-21"
    assert other_cutoff_row["priority_score_0_100"] == 99.0

    with SessionLocal() as db:
        db.add(
            PrediccionFactura(
                factura_id="FACAPPTEST001",
                cliente_id="CLI0001",
                fecha_corte=date(2024, 1, 20),
                modelo_version="test",
                predicted_class_tecnica="on_time",
                predicted_label_usuario="Pago esperado dentro del plazo",
                prob_pago_plazo=0.9,
                prob_atraso_leve=0.05,
                prob_atraso_alto=0.03,
                prob_atraso_critico=0.02,
                any_late_probability=0.1,
                high_risk_probability=0.05,
                priority_score_0_100=5.0,
                accion_sugerida_codigo="SIN_ACCION",
                accion_sugerida_nombre="Monitorear sin gestion inmediata",
                motivo_accion="test",
            )
        )
        db.commit()

    paid = client.patch(
        "/api/v1/invoices/FACAPPTEST001",
        json={"fecha_pago_real": "2024-02-15"},
    )
    assert paid.status_code == 200
    assert paid.json()["estado_factura"] == "pagada"
    assert paid.json()["saldo_pendiente"] == 0.0
    assert paid.json()["dias_mora_real"] == 6

    with SessionLocal() as db:
        remaining_predictions = db.scalar(
            select(func.count(PrediccionFactura.prediccion_id)).where(
                PrediccionFactura.factura_id == "FACAPPTEST001"
            )
        )
    assert remaining_predictions == 0

    invalid_dates = client.post(
        "/api/v1/invoices",
        json={
            "factura_id": "FACAPPTEST002",
            "cliente_id": "CLI0001",
            "fecha_emision": "2024-02-10",
            "fecha_vencimiento": "2024-02-09",
            "monto": 1500.0,
        },
    )
    assert invalid_dates.status_code == 422

    missing = client.patch("/api/v1/invoices/FAC_NO_EXISTE", json={"estado_factura": "anulada"})
    assert missing.status_code == 404

    null_required = client.patch("/api/v1/invoices/FACAPPTEST001", json={"fecha_vencimiento": None})
    assert null_required.status_code == 422

    client.post(
        "/api/v1/invoices",
        json={
            "factura_id": "FACAPPTEST003",
            "cliente_id": "CLI0001",
            "fecha_emision": "2024-01-10",
            "fecha_vencimiento": "2024-02-09",
            "monto": 1500.0,
        },
    )
    invalid_amount = client.patch("/api/v1/invoices/FACAPPTEST003", json={"monto": 100.0})
    assert invalid_amount.status_code == 400

    changed_customer = client.patch("/api/v1/invoices/FACAPPTEST001", json={"cliente_id": "CLI0002"})
    assert changed_customer.status_code == 400


def test_create_interaction_promise_and_payment_flow() -> None:
    _seed_db()
    client = TestClient(app)

    invoice = client.get("/api/v1/invoices/FAC000002").json()
    fecha_gestion = date.fromisoformat(invoice["fecha_vencimiento"]) + timedelta(days=1)

    interaction = client.post(
        "/api/v1/collections/interactions",
        json={
            "factura_id": "FAC000002",
            "fecha_gestion": fecha_gestion.isoformat(),
            "canal": "llamada",
            "contacto_exitoso": True,
            "resultado": "promesa_de_pago",
            "motivo_no_pago": "flujo_caja",
            "observacion": "Cliente confirma pago la proxima semana.",
            "recalculate": False,
        },
    )
    assert interaction.status_code == 200
    gestion_id = interaction.json()["gestion_id"]

    promise = client.post(
        "/api/v1/payment-promises",
        json={
            "gestion_id": gestion_id,
            "fecha_compromiso": (fecha_gestion + timedelta(days=7)).isoformat(),
            "recalculate": False,
        },
    )
    assert promise.status_code == 200
    promesa_id = promise.json()["promesa_id"]

    duplicated = client.post(
        "/api/v1/payment-promises",
        json={
            "gestion_id": gestion_id,
            "fecha_compromiso": (fecha_gestion + timedelta(days=8)).isoformat(),
            "recalculate": False,
        },
    )
    assert duplicated.status_code == 400

    patched = client.patch(
        f"/api/v1/payment-promises/{promesa_id}",
        json={"estado_promesa": "cumplida"},
    )
    assert patched.status_code == 200
    assert patched.json()["se_cumplio"] is True

    paid = client.post(
        "/api/v1/payments",
        json={"factura_id": "FAC000002", "fecha_pago": (fecha_gestion + timedelta(days=3)).isoformat()},
    )
    assert paid.status_code == 200
    assert paid.json()["estado_factura"] == "pagada"
    assert paid.json()["saldo_pendiente"] == 0.0


def test_invoice_interactions_filter_future_gestiones_and_add_labels() -> None:
    _seed_db()
    client = TestClient(app)

    created = client.post(
        "/api/v1/invoices",
        json={
            "factura_id": "FACAPPINT001",
            "cliente_id": "CLI0001",
            "fecha_emision": "2024-01-01",
            "fecha_vencimiento": "2024-01-10",
            "monto": 1200.0,
        },
    )
    assert created.status_code == 201

    first = client.post(
        "/api/v1/collections/interactions",
        json={
            "factura_id": "FACAPPINT001",
            "fecha_gestion": "2024-01-12",
            "canal": "llamada",
            "contacto_exitoso": True,
            "resultado": "promesa_de_pago",
            "motivo_no_pago": "flujo_caja",
            "recalculate": False,
        },
    )
    assert first.status_code == 200
    future = client.post(
        "/api/v1/collections/interactions",
        json={
            "factura_id": "FACAPPINT001",
            "fecha_gestion": "2024-01-25",
            "canal": "email",
            "contacto_exitoso": True,
            "resultado": "confirma_pago",
            "recalculate": False,
        },
    )
    assert future.status_code == 200

    filtered = client.get("/api/v1/invoices/FACAPPINT001/interactions?fecha_corte=2024-01-20")
    assert filtered.status_code == 200
    assert [row["fecha_gestion"] for row in filtered.json()] == ["2024-01-12"]
    interaction = filtered.json()[0]
    assert interaction["canal_label"] == "Llamada telefonica"
    assert interaction["resultado_label"] == "Promesa de pago"
    assert interaction["motivo_no_pago_label"] == "Restriccion de flujo de caja"
    assert "Contacto exitoso" in interaction["interpretacion"]

    unfiltered = client.get("/api/v1/invoices/FACAPPINT001/interactions")
    assert unfiltered.status_code == 200
    assert len(unfiltered.json()) == 2


def test_prediction_daily_is_on_demand_and_stops_when_paid_at_cutoff() -> None:
    _seed_db()
    client = TestClient(app)

    created = client.post(
        "/api/v1/invoices",
        json={
            "factura_id": "FACAPPDAILY001",
            "cliente_id": "CLI0001",
            "fecha_emision": "2024-01-01",
            "fecha_vencimiento": "2024-01-04",
            "fecha_pago_real": "2024-01-03",
            "monto": 900.0,
        },
    )
    assert created.status_code == 201

    with SessionLocal() as db:
        before_count = db.scalar(
            select(func.count(PrediccionFactura.prediccion_id)).where(
                PrediccionFactura.factura_id == "FACAPPDAILY001"
            )
        )

    response = client.get("/api/v1/invoices/FACAPPDAILY001/prediction-daily?fecha_corte=2024-01-10")
    assert response.status_code == 200
    body = response.json()
    assert [row["fecha_corte"] for row in body] == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert all(row["factura_id"] == "FACAPPDAILY001" for row in body)

    with SessionLocal() as db:
        after_count = db.scalar(
            select(func.count(PrediccionFactura.prediccion_id)).where(
                PrediccionFactura.factura_id == "FACAPPDAILY001"
            )
        )
    assert after_count == before_count == 0

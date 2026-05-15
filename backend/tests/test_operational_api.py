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

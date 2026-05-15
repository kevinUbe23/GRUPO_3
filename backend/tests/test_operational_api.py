from __future__ import annotations

from datetime import date, timedelta

from fastapi.testclient import TestClient

from app.db.database import Base, SessionLocal, engine
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

    prioritized = client.get("/api/v1/invoices/prioritized?limit=5")
    assert prioritized.status_code == 200
    assert len(prioritized.json()) > 0
    assert prioritized.json()[0]["estado_factura"] == "abierta"

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

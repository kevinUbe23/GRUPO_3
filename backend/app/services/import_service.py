from __future__ import annotations

import pandas as pd
from sqlalchemy import delete, func
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.time import utc_now
from app.db.models import (
    AccionSugerida,
    Cliente,
    Factura,
    GestionCobranza,
    PrediccionFactura,
    PromesaPago,
    ReglaAccionSugerida,
    SegmentoCliente,
)


ACTION_CATALOG = [
    ("SIN_ACCION", "Monitorear sin gestion inmediata", "No requiere cobranza activa inmediata.", "Sistema", 0),
    ("RECORDATORIO_SUAVE", "Enviar recordatorio suave", "Recordatorio de baja presion para casos preventivos.", "WhatsApp o email", 1),
    ("RECORDATORIO_PREVENTIVO", "Enviar recordatorio preventivo", "Contacto preventivo antes o al inicio del atraso.", "WhatsApp", 2),
    ("CONFIRMAR_FECHA_PAGO", "Confirmar fecha estimada de pago", "Confirmar intencion y fecha esperada de pago.", "WhatsApp o llamada", 2),
    ("LLAMADA_SEGUIMIENTO", "Realizar llamada de seguimiento", "Contacto telefonico para seguimiento del caso.", "Llamada", 3),
    ("LLAMADA_URGENTE", "Realizar llamada urgente", "Contacto urgente por mora o riesgo relevante.", "Llamada", 4),
    ("SOLICITAR_PROMESA", "Solicitar promesa formal de pago", "Formalizar compromiso de pago.", "Llamada o WhatsApp", 4),
    ("SEGUIMIENTO_PROMESA", "Dar seguimiento a promesa activa", "Confirmar cumplimiento de una promesa vigente.", "WhatsApp o llamada", 3),
    ("ESCALAR_PROMESA_VENCIDA", "Escalar promesa incumplida", "Escalar tras incumplimiento de promesa.", "Llamada", 5),
    ("REVISAR_DISPUTA", "Revisar disputa antes de cobrar", "Resolver disputa antes de escalar cobranza.", "Interno", 5),
    ("VISITA_CLIENTE", "Programar visita de cobranza", "Gestion presencial por mora alta o contacto dificil.", "Visita", 6),
    ("CARTA_FORMAL", "Enviar comunicacion formal", "Comunicacion formal por mora severa.", "Carta o email formal", 7),
    ("FORMALIZAR_GARANTIA", "Formalizar o revisar garantia", "Revision de garantia ante mora severa.", "Interno/legal", 8),
    ("ESCALAMIENTO_LEGAL", "Escalar a gestion legal", "Escalamiento legal para casos criticos.", "Legal", 9),
    ("REVISAR_DATOS_CONTACTO", "Revisar datos de contacto", "Validar canales alternativos por baja contactabilidad.", "Interno", 4),
    ("REVISION_MANUAL_ALTO_MONTO", "Revision manual por exposicion alta", "Revision humana por alto monto y alto riesgo.", "Interno", 6),
]

RULE_CATALOG = [
    ("SIN_ACCION", "factura_pagada", 100, {"estado_factura": "pagada"}, "La factura ya fue pagada y no requiere cobranza activa."),
    ("REVISAR_DISPUTA", "disputa_activa", 95, {"tiene_disputa_activa": 1}, "Existe una disputa activa; se recomienda revisar el caso antes de cobrar."),
    ("ESCALAMIENTO_LEGAL", "mora_60_alto_riesgo_rating_bajo", 88, {"dias_mora_observable_gte": 60, "high_risk_probability_gte": 0.70, "rating_estrellas_lte": 2}, "Mora 60+ con riesgo alto y cliente de bajo rating."),
    ("FORMALIZAR_GARANTIA", "mora_60_con_garantia", 86, {"dias_mora_observable_gte": 60, "tiene_garantia": 1}, "La mora supera 60 dias y existe garantia."),
    ("CARTA_FORMAL", "mora_60", 84, {"dias_mora_observable_gte": 60}, "La factura tiene mora mayor a 60 dias."),
    ("VISITA_CLIENTE", "mora_31_riesgo_o_no_contacto", 82, {"dias_mora_observable_gte": 31}, "La factura tiene mora alta y senales de riesgo operativo."),
    ("LLAMADA_URGENTE", "mora_15_riesgo", 76, {"dias_mora_observable_gte": 15}, "La factura ya acumula mora relevante y requiere contacto urgente."),
    ("LLAMADA_SEGUIMIENTO", "mora_8", 70, {"dias_mora_observable_gte": 8}, "La factura lleva mas de una semana vencida."),
    ("RECORDATORIO_PREVENTIVO", "mora_inicial", 66, {"dias_mora_observable_gte": 1}, "La factura ya vencio; se recomienda recordatorio inmediato."),
    ("SEGUIMIENTO_PROMESA", "promesa_activa", 62, {"tiene_promesa_activa": 1}, "Existe una promesa activa; se recomienda seguimiento."),
    ("CONFIRMAR_FECHA_PAGO", "vence_3_riesgo", 60, {"dias_hasta_vence_lte": 3, "any_late_probability_gte": 0.60}, "La factura esta por vencer y el riesgo de atraso es alto."),
    ("RECORDATORIO_PREVENTIVO", "vence_7_riesgo", 56, {"dias_hasta_vence_lte": 7}, "Queda una semana o menos para el vencimiento y hay riesgo de atraso."),
    ("REVISAR_DATOS_CONTACTO", "contactabilidad_baja", 44, {"num_no_contesta_cons_gte": 3}, "Hay dificultad de contacto; conviene validar medios alternativos."),
    ("REVISION_MANUAL_ALTO_MONTO", "monto_alto_score", 36, {"monto_alto": 1}, "La exposicion monetaria es alta y el riesgo tambien."),
    ("SIN_ACCION", "default", 10, {}, "No se detecta necesidad de gestion inmediata."),
]


def _parse_date(value) -> object | None:
    if pd.isna(value) or value == "":
        return None
    return pd.to_datetime(value).date()


def _bool01(value) -> bool:
    if pd.isna(value):
        return False
    return bool(int(value))


def reset_and_import_seed_data(db: Session) -> dict[str, int]:
    settings = get_settings()
    data_dir = settings.generated_data_dir

    clientes_df = pd.read_csv(data_dir / "clientes.csv")
    facturas_df = pd.read_csv(data_dir / "facturas.csv")
    gestiones_df = pd.read_csv(data_dir / "gestiones_cobranza.csv")
    promesas_df = pd.read_csv(data_dir / "promesas_pago.csv")
    segmentos_df = pd.read_csv(settings.frontend_segments_path)

    for model in [
        PrediccionFactura,
        SegmentoCliente,
        PromesaPago,
        GestionCobranza,
        Factura,
        Cliente,
        ReglaAccionSugerida,
        AccionSugerida,
    ]:
        db.execute(delete(model))
    db.flush()

    clientes = [
        Cliente(
            cliente_id=row.cliente_id,
            nombre=row.nombre,
            sector=row.sector,
            antiguedad_meses=int(row.antiguedad_meses),
            tiene_garantia=_bool01(row.tiene_garantia),
            perfil_pago_simulado=getattr(row, "perfil_pago", None),
        )
        for row in clientes_df.itertuples(index=False)
    ]
    db.add_all(clientes)
    db.flush()

    facturas = []
    for row in facturas_df.itertuples(index=False):
        fecha_pago = _parse_date(row.fecha_pago_real)
        estado = "pagada" if fecha_pago else "abierta"
        facturas.append(
            Factura(
                factura_id=row.factura_id,
                cliente_id=row.cliente_id,
                fecha_emision=_parse_date(row.fecha_emision),
                fecha_vencimiento=_parse_date(row.fecha_vencimiento),
                fecha_pago_real=fecha_pago,
                condicion_dias=int(row.condicion_dias),
                monto=float(row.monto),
                saldo_pendiente=0.0 if fecha_pago else float(row.monto),
                estado_factura=estado,
                target_mora_simulado=row.target_mora,
                dias_mora_real=int(row.dias_mora_real),
            )
        )
    db.add_all(facturas)
    db.flush()

    gestiones = [
        GestionCobranza(
            gestion_id=row.gestion_id,
            factura_id=row.factura_id,
            cliente_id=row.cliente_id,
            fecha_gestion=_parse_date(row.fecha_gestion),
            canal=row.canal,
            contacto_exitoso=_bool01(row.contacto_exitoso),
            resultado=row.resultado,
            motivo_no_pago=None if pd.isna(row.motivo_no_pago) else row.motivo_no_pago,
            dias_mora_en_gestion=int(row.dias_mora_en_gestion),
        )
        for row in gestiones_df.itertuples(index=False)
    ]
    db.add_all(gestiones)
    db.flush()

    promesas = []
    for row in promesas_df.itertuples(index=False):
        promesas.append(
            PromesaPago(
                promesa_id=row.promesa_id,
                gestion_id=row.gestion_id,
                factura_id=row.factura_id,
                cliente_id=row.cliente_id,
                fecha_promesa=_parse_date(row.fecha_promesa),
                fecha_compromiso=_parse_date(row.fecha_compromiso),
                se_cumplio=_bool01(row.se_cumplio),
                estado_promesa="cumplida" if _bool01(row.se_cumplio) else "incumplida",
            )
        )
    db.add_all(promesas)
    db.flush()

    segmentos = []
    for row in segmentos_df.itertuples(index=False):
        segmentos.append(
            SegmentoCliente(
                cliente_id=row.cliente_id,
                tipo_cliente=row.tipo_cliente,
                cluster=int(row.cluster),
                riesgo_0_100=float(row.riesgo_0_100),
                rating_estrellas=int(row.rating_estrellas),
                rating_label=row.rating_label,
                por_que_rating=row.por_que_rating,
                por_que_cluster=row.por_que_cluster,
                sector_dominante_modal=row.sector_dominante_modal,
                n_facturas_total=int(row.n_facturas_total),
                n_cortes_total=int(row.n_cortes_total),
                fecha_calculo=utc_now(),
            )
        )
    db.add_all(segmentos)
    db.flush()

    db.add_all(
        [
            AccionSugerida(
                codigo=codigo,
                nombre=nombre,
                descripcion=descripcion,
                canal_recomendado=canal,
                nivel_severidad=severidad,
            )
            for codigo, nombre, descripcion, canal, severidad in ACTION_CATALOG
        ]
    )
    db.flush()
    import json

    db.add_all(
        [
            ReglaAccionSugerida(
                accion_codigo=accion_codigo,
                nombre_regla=nombre_regla,
                prioridad_regla=prioridad,
                condiciones_json=json.dumps(condiciones),
                motivo_template=motivo,
            )
            for accion_codigo, nombre_regla, prioridad, condiciones, motivo in RULE_CATALOG
        ]
    )
    db.commit()

    return {
        "clientes": db.scalar(func.count(Cliente.cliente_id)) or 0,
        "facturas": db.scalar(func.count(Factura.factura_id)) or 0,
        "gestiones": db.scalar(func.count(GestionCobranza.gestion_id)) or 0,
        "promesas": db.scalar(func.count(PromesaPago.promesa_id)) or 0,
        "segmentos": db.scalar(func.count(SegmentoCliente.cliente_id)) or 0,
        "acciones": db.scalar(func.count(AccionSugerida.codigo)) or 0,
        "reglas": db.scalar(func.count(ReglaAccionSugerida.regla_id)) or 0,
    }

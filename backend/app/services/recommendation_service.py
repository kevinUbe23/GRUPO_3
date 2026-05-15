from __future__ import annotations

from dataclasses import dataclass

from app.db.models import SegmentoCliente


@dataclass
class Recommendation:
    codigo: str
    nombre: str
    canal_recomendado: str
    severidad: int
    motivo: str
    regla: str


def _monto_alto(features: dict) -> bool:
    return float(features.get("ratio_monto", 1.0)) >= 1.5 or float(features.get("monto", 0.0)) >= 20000


def recommend_action(
    *,
    features: dict,
    predicted_label_usuario: str,
    any_late_probability: float,
    high_risk_probability: float,
    priority_score_0_100: float,
    segmento: SegmentoCliente | None,
) -> Recommendation:
    rating = segmento.rating_estrellas if segmento else 3
    dias_mora = int(features.get("dias_mora_observable", 0))
    dias_hasta = int(features.get("dias_hasta_vence", 0))
    promesa_activa = int(features.get("tiene_promesa_activa", 0)) == 1
    disputa = int(features.get("tiene_disputa_activa", 0)) == 1
    no_contesta = int(features.get("num_no_contesta_cons", 0))
    gestiones = int(features.get("num_gestiones_factura", 0))
    tasa_contacto = float(features.get("tasa_contacto_cliente", 0.5))
    promesas_rotas = int(features.get("num_promesas_rotas", 0))
    tasa_cumpl_promesas = float(features.get("tasa_cumpl_promesas", 1.0))
    garantia = int(features.get("tiene_garantia", 0)) == 1
    cliente_nuevo = int(features.get("cliente_nuevo", 0)) == 1
    sin_gestion = int(features.get("sin_gestion_previa", 0)) == 1
    monto_alto = _monto_alto(features)
    estado_factura = str(features.get("estado_factura", "abierta"))

    def rec(codigo: str, nombre: str, canal: str, severidad: int, motivo: str, regla: str) -> Recommendation:
        return Recommendation(codigo, nombre, canal, severidad, motivo, regla)

    if estado_factura == "pagada":
        return rec(
            "SIN_ACCION",
            "Monitorear sin gestion inmediata",
            "Sistema",
            0,
            "La factura ya fue pagada y no requiere cobranza activa.",
            "factura_pagada",
        )
    if estado_factura in {"anulada", "castigada"}:
        return rec(
            "SIN_ACCION",
            "Monitorear sin gestion inmediata",
            "Sistema",
            0,
            "La factura no pertenece a la cola operativa normal.",
            "factura_cerrada",
        )
    if disputa:
        return rec(
            "REVISAR_DISPUTA",
            "Revisar disputa antes de cobrar",
            "Interno",
            5,
            "Existe una disputa activa; conviene resolverla antes de aumentar la presion de cobranza.",
            "disputa_activa",
        )
    if dias_mora >= 60 and high_risk_probability >= 0.70 and rating <= 2:
        return rec("ESCALAMIENTO_LEGAL", "Escalar a gestion legal", "Legal", 9, "Mora 60+ con riesgo alto y cliente de bajo rating.", "mora_60_alto_riesgo_rating_bajo")
    if dias_mora >= 60 and garantia:
        return rec("FORMALIZAR_GARANTIA", "Formalizar o revisar garantia", "Interno/legal", 8, "La mora supera 60 dias y existe garantia asociada.", "mora_60_con_garantia")
    if dias_mora >= 60:
        return rec("CARTA_FORMAL", "Enviar comunicacion formal", "Carta o email formal", 7, "La factura tiene mora mayor a 60 dias.", "mora_60")
    if dias_mora >= 31 and (high_risk_probability >= 0.60 or no_contesta >= 2):
        return rec("VISITA_CLIENTE", "Programar visita de cobranza", "Visita", 6, "La factura tiene mora alta y senales de riesgo operativo.", "mora_31_riesgo_o_no_contacto")
    if dias_mora >= 15 and (rating <= 2 or high_risk_probability >= 0.50):
        return rec("LLAMADA_URGENTE", "Realizar llamada urgente", "Llamada", 4, "La factura ya acumula mora relevante y requiere contacto urgente.", "mora_15_riesgo")
    if dias_mora >= 8:
        return rec("LLAMADA_SEGUIMIENTO", "Realizar llamada de seguimiento", "Llamada", 3, "La factura lleva mas de una semana vencida.", "mora_8")
    if dias_mora >= 1:
        return rec("RECORDATORIO_PREVENTIVO", "Enviar recordatorio preventivo", "WhatsApp", 2, "La factura ya vencio; se recomienda recordatorio inmediato.", "mora_inicial")
    if promesa_activa:
        return rec("SEGUIMIENTO_PROMESA", "Dar seguimiento a promesa activa", "WhatsApp o llamada", 3, "Existe una promesa activa; se recomienda seguimiento sin escalar todavia.", "promesa_activa")
    if dias_hasta <= 3 and any_late_probability >= 0.60:
        return rec("CONFIRMAR_FECHA_PAGO", "Confirmar fecha estimada de pago", "WhatsApp o llamada", 2, "La factura esta por vencer y el riesgo de atraso es alto.", "vence_3_riesgo")
    if dias_hasta <= 7 and any_late_probability >= 0.50 and rating <= 3:
        return rec("RECORDATORIO_PREVENTIVO", "Enviar recordatorio preventivo", "WhatsApp", 2, "Falta una semana o menos para vencer y el cliente no tiene rating alto.", "vence_7_riesgo_rating")
    if dias_hasta <= 7 and predicted_label_usuario != "Pago esperado dentro del plazo":
        return rec("RECORDATORIO_PREVENTIVO", "Enviar recordatorio preventivo", "WhatsApp", 2, "Es probable que no pague dentro del plazo y queda una semana o menos para el vencimiento.", "vence_7_pred_no_plazo")
    if dias_hasta <= 14 and high_risk_probability >= 0.50:
        return rec("LLAMADA_SEGUIMIENTO", "Realizar llamada de seguimiento", "Llamada", 3, "Aunque aun no vence, el riesgo de atraso alto o critico es relevante.", "vence_14_high_risk")
    if cliente_nuevo and any_late_probability >= 0.50:
        return rec("CONFIRMAR_FECHA_PAGO", "Confirmar fecha estimada de pago", "WhatsApp o llamada", 2, "Cliente con poco historial y riesgo moderado.", "cliente_nuevo_riesgo")
    if sin_gestion and priority_score_0_100 >= 60:
        return rec("RECORDATORIO_PREVENTIVO", "Enviar recordatorio preventivo", "WhatsApp", 2, "No hay gestion previa y el score de prioridad ya es alto.", "sin_gestion_score_alto")
    if no_contesta >= 3 or (tasa_contacto < 0.30 and gestiones >= 2):
        return rec("REVISAR_DATOS_CONTACTO", "Revisar datos de contacto", "Interno", 4, "Hay dificultad de contacto; conviene validar medios alternativos.", "contactabilidad_baja")
    if promesas_rotas >= 2 and any_late_probability >= 0.50:
        return rec("LLAMADA_URGENTE", "Realizar llamada urgente", "Llamada", 4, "El cliente acumula promesas incumplidas y la factura tiene riesgo de atraso.", "promesas_rotas_riesgo")
    if tasa_cumpl_promesas < 0.40 and dias_mora > 0:
        return rec("SOLICITAR_PROMESA", "Solicitar promesa formal de pago", "Llamada o WhatsApp", 4, "Bajo cumplimiento historico de promesas; cualquier compromiso debe quedar formalizado.", "bajo_cumpl_promesas")
    if monto_alto and priority_score_0_100 >= 70:
        return rec("REVISION_MANUAL_ALTO_MONTO", "Revision manual por exposicion alta", "Interno", 6, "La exposicion monetaria es alta y el riesgo tambien.", "monto_alto_score")
    if priority_score_0_100 >= 80:
        return rec("LLAMADA_URGENTE", "Realizar llamada urgente", "Llamada", 4, "El score de prioridad es critico.", "score_80")
    if priority_score_0_100 >= 60:
        return rec("LLAMADA_SEGUIMIENTO", "Realizar llamada de seguimiento", "Llamada", 3, "El score de prioridad es alto.", "score_60")
    if priority_score_0_100 >= 40:
        return rec("RECORDATORIO_PREVENTIVO", "Enviar recordatorio preventivo", "WhatsApp", 2, "El score de prioridad es medio.", "score_40")
    return rec("SIN_ACCION", "Monitorear sin gestion inmediata", "Sistema", 0, "No se detecta necesidad de gestion inmediata.", "default")

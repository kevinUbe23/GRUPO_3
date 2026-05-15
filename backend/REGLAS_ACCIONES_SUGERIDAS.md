# Reglas de acciones sugeridas

Este documento resume y justifica las reglas usadas por el backend para convertir una prediccion de riesgo en una accion operativa de cobranza. Su objetivo es dejar trazabilidad metodologica para el informe del proyecto y para la revision funcional del sistema.

## Proposito

El modelo estima probabilidades de pago dentro del plazo o atraso. Esa salida tecnica no es suficiente para operar cobranzas: el usuario necesita una accion concreta, proporcional al riesgo y al estado actual de la factura. Por eso el backend aplica un motor de reglas posterior al modelo.

La regla central es que la decision operativa se toma por factura. El cliente aporta contexto historico mediante rating, garantia, contactabilidad, promesas y comportamiento previo, pero la accion recomendada responde a la situacion observable de la factura en la fecha de corte.

## Variables usadas

Las reglas combinan variables de cuatro grupos:

- Estado operativo: `estado_factura`, `tiene_disputa_activa`, `tiene_promesa_activa`.
- Mora y temporalidad: `dias_mora_observable`, `dias_hasta_vence`.
- Riesgo estimado por IA: `any_late_probability`, `high_risk_probability`, `priority_score_0_100`, `predicted_label_usuario`.
- Contexto del cliente: `rating_estrellas`, `tiene_garantia`, `cliente_nuevo`, `tasa_contacto_cliente`, `num_no_contesta_cons`, `num_promesas_rotas`, `tasa_cumpl_promesas`.

El orden de evaluacion es deliberado: primero se descartan casos que no deben cobrarse, luego se atienden bloqueos o riesgos criticos, despues la mora observable, y finalmente senales preventivas antes del vencimiento.

## Catalogo de acciones

| Codigo | Accion | Canal | Severidad | Justificacion |
|---|---|---|---:|---|
| `SIN_ACCION` | Monitorear sin gestion inmediata | Sistema | 0 | Evita intervenciones innecesarias cuando la factura ya esta pagada, cerrada o sin riesgo operativo relevante. |
| `RECORDATORIO_SUAVE` | Enviar recordatorio suave | WhatsApp o email | 1 | Gestion preventiva de baja presion para casos con riesgo bajo y vencimiento lejano. |
| `RECORDATORIO_PREVENTIVO` | Enviar recordatorio preventivo | WhatsApp | 2 | Contacto temprano cuando hay vencimiento cercano, mora inicial o score medio. |
| `CONFIRMAR_FECHA_PAGO` | Confirmar fecha estimada de pago | WhatsApp o llamada | 2 | Busca reducir incertidumbre antes del vencimiento sin escalar la cobranza. |
| `LLAMADA_SEGUIMIENTO` | Realizar llamada de seguimiento | Llamada | 3 | Aplica cuando la factura ya tiene riesgo o atraso moderado y requiere contacto directo. |
| `LLAMADA_URGENTE` | Realizar llamada urgente | Llamada | 4 | Aplica cuando la mora, el riesgo o las promesas incumplidas justifican atencion prioritaria. |
| `SOLICITAR_PROMESA` | Solicitar promesa formal de pago | Llamada o WhatsApp | 4 | Formaliza un compromiso cuando existe atraso y riesgo de incumplimiento. |
| `SEGUIMIENTO_PROMESA` | Dar seguimiento a promesa activa | WhatsApp o llamada | 3 | Evita escalar mientras existe una promesa vigente, pero mantiene control operativo. |
| `REVISAR_DISPUTA` | Revisar disputa antes de cobrar | Interno | 5 | Una disputa debe resolverse antes de presionar el cobro para no deteriorar la relacion comercial. |
| `REVISAR_DATOS_CONTACTO` | Revisar datos de contacto | Interno | 4 | La baja contactabilidad puede impedir cualquier gestion efectiva. |
| `VISITA_CLIENTE` | Programar visita de cobranza | Visita | 6 | Se reserva para mora alta o contacto dificil, donde un canal remoto puede no ser suficiente. |
| `REVISION_MANUAL_ALTO_MONTO` | Revision manual por exposicion alta | Interno | 6 | Facturas de alto monto requieren validacion humana aunque el modelo ya priorice el caso. |
| `CARTA_FORMAL` | Enviar comunicacion formal | Carta o email formal | 7 | Formaliza la gestion cuando la mora supera un umbral severo. |
| `FORMALIZAR_GARANTIA` | Formalizar o revisar garantia | Interno/legal | 8 | Si existe garantia y mora severa, corresponde revisar respaldo y condiciones. |
| `ESCALAMIENTO_LEGAL` | Escalar a gestion legal | Legal | 9 | Ultimo nivel para mora severa, riesgo alto y bajo rating. |

## Reglas implementadas

| Orden | Condicion | Accion | Justificacion |
|---:|---|---|---|
| 1 | `estado_factura = pagada` | `SIN_ACCION` | Una factura pagada no debe recibir cobranza activa. |
| 2 | `estado_factura` en `anulada`, `castigada` | `SIN_ACCION` | Esas facturas no pertenecen a la cola operativa normal. |
| 3 | `tiene_disputa_activa = 1` | `REVISAR_DISPUTA` | La disputa bloquea la cobranza tradicional hasta ser aclarada. |
| 4 | `dias_mora_observable >= 60`, `high_risk_probability >= 0.70`, `rating_estrellas <= 2` | `ESCALAMIENTO_LEGAL` | Combina mora severa, prediccion critica y bajo rating; es el mayor riesgo operativo. |
| 5 | `dias_mora_observable >= 60`, `tiene_garantia = 1` | `FORMALIZAR_GARANTIA` | La garantia se vuelve relevante cuando el atraso ya es severo. |
| 6 | `dias_mora_observable >= 60` | `CARTA_FORMAL` | A partir de 60 dias conviene dejar evidencia formal de gestion. |
| 7 | `dias_mora_observable >= 31` y (`high_risk_probability >= 0.60` o `num_no_contesta_cons >= 2`) | `VISITA_CLIENTE` | La mora alta requiere mayor intensidad si ademas hay riesgo grave o dificultad de contacto. |
| 8 | `dias_mora_observable >= 15` y (`rating_estrellas <= 2` o `high_risk_probability >= 0.50`) | `LLAMADA_URGENTE` | Mora intermedia con cliente riesgoso amerita contacto inmediato. |
| 9 | `dias_mora_observable >= 8` | `LLAMADA_SEGUIMIENTO` | Despues de una semana vencida, el seguimiento telefonico es proporcional al atraso. |
| 10 | `dias_mora_observable >= 1` | `RECORDATORIO_PREVENTIVO` | En mora inicial basta una gestion temprana de baja severidad. |
| 11 | `tiene_promesa_activa = 1` | `SEGUIMIENTO_PROMESA` | Si hay promesa vigente, se prioriza seguimiento antes que escalamiento. |
| 12 | `dias_hasta_vence <= 3`, `any_late_probability >= 0.60` | `CONFIRMAR_FECHA_PAGO` | Cercania al vencimiento y riesgo alto justifican confirmar intencion de pago. |
| 13 | `dias_hasta_vence <= 7`, `any_late_probability >= 0.50`, `rating_estrellas <= 3` | `RECORDATORIO_PREVENTIVO` | Gestion preventiva para clientes no sobresalientes con riesgo moderado. |
| 14 | `dias_hasta_vence <= 7`, `predicted_label_usuario != Pago esperado dentro del plazo` | `RECORDATORIO_PREVENTIVO` | Si el modelo no espera pago puntual, conviene anticipar contacto. |
| 15 | `dias_hasta_vence <= 14`, `high_risk_probability >= 0.50` | `LLAMADA_SEGUIMIENTO` | Riesgo alto antes del vencimiento amerita seguimiento directo. |
| 16 | `cliente_nuevo = 1`, `any_late_probability >= 0.50` | `CONFIRMAR_FECHA_PAGO` | Menor historial aumenta incertidumbre; se busca confirmar comportamiento esperado. |
| 17 | `sin_gestion_previa = 1`, `priority_score_0_100 >= 60` | `RECORDATORIO_PREVENTIVO` | Facturas sin gestion y score alto no deben quedar sin contacto. |
| 18 | `num_no_contesta_cons >= 3` o (`tasa_contacto_cliente < 0.30`, `num_gestiones_factura >= 2`) | `REVISAR_DATOS_CONTACTO` | Varias fallas de contacto sugieren problema de datos antes que falta de voluntad de pago. |
| 19 | `num_promesas_rotas >= 2`, `any_late_probability >= 0.50` | `LLAMADA_URGENTE` | Promesas incumplidas repetidas elevan la urgencia de intervencion. |
| 20 | `monto_alto = 1`, `priority_score_0_100 >= 70` | `REVISION_MANUAL_ALTO_MONTO` | La exposicion monetaria alta requiere criterio humano adicional. |
| 21 | `priority_score_0_100 >= 80` | `LLAMADA_URGENTE` | Score critico por si solo justifica contacto urgente. |
| 22 | `priority_score_0_100 >= 60` | `LLAMADA_SEGUIMIENTO` | Score alto requiere gestion directa, aunque no haya mora severa. |
| 23 | `priority_score_0_100 >= 40` | `RECORDATORIO_PREVENTIVO` | Score medio se atiende con una accion preventiva. |
| 24 | Sin condiciones criticas | `SIN_ACCION` | Si no hay senales relevantes, se evita sobregestionar. |

Nota: `SOLICITAR_PROMESA` existe en el catalogo porque es una accion de negocio valida para futuras variantes del motor. En la version actual, las reglas de mora inicial y seguimiento tienen mayor precedencia, por lo que esa accion no queda como resultado principal del recomendador.

## Justificacion metodologica

El motor de acciones separa prediccion y decision. El modelo calcula probabilidades; las reglas traducen esas probabilidades a una respuesta operativa comprensible, trazable y defendible para un usuario de cobranzas.

Esta separacion reduce riesgos de automatizacion excesiva. El sistema no castiga automaticamente al cliente ni bloquea operaciones comerciales: recomienda una accion para revision humana. Por eso las acciones de mayor severidad se activan solo cuando coinciden mora observable, riesgo estimado y senales de deterioro historico.

La prioridad de las reglas protege la coherencia del flujo. Una factura pagada o anulada nunca debe recibir una llamada de cobro. Una disputa se revisa antes de cobrar. Una promesa activa se sigue, pero no se escala inmediatamente. Solo despues de esos casos se evalua la intensidad de cobranza segun mora, riesgo y contexto del cliente.

El score `priority_score_0_100` funciona como criterio transversal para ordenar la cola, pero no reemplaza las reglas de negocio. Dos facturas con score similar pueden recibir acciones distintas si una tiene disputa, garantia, promesa activa o problemas de contacto. Esto permite que la priorizacion sea sensible al contexto y no dependa exclusivamente de una probabilidad.

## Consideraciones para el informe

- Las reglas son deterministicas y auditables: la respuesta incluye el codigo de accion, el motivo y el nombre de la regla activada.
- La accion sugerida es una recomendacion, no una decision automatica irreversible.
- La severidad aumenta gradualmente desde monitoreo hasta escalamiento legal.
- El diseno evita fuga temporal porque las variables se calculan hasta la `fecha_corte`.
- La unidad de decision es la factura, alineada con el enfoque metodologico del proyecto.

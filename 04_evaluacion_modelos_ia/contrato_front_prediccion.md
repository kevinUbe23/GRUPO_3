# Contrato de entrada para prediccion desde el front

## Idea principal

El modelo de prediccion no recibe una factura cruda directamente. El modelo recibe una fila de features ya construidos, con las columnas exactas de `04_evaluacion_modelos_ia/outputs/model_feature_schema.csv`.

Para un sistema full stack se necesitan tres capas:

| Capa | Que recibe | Que produce |
|---|---|---|
| Front | Datos visibles/editables de factura, cliente y gestion | Payload de negocio para el backend |
| Backend feature builder | Payload del front + historial del cliente + gestiones + promesas | Features del modelo |
| Pipeline/modelo | Features del modelo | Clase predicha, probabilidades y score de prioridad |

El pipeline guardado dentro de `best_model_artifact.joblib` transforma features ya calculados: imputa, aplica `log1p`, escala variables numericas y hace one-hot de `ultimo_resultado_enc`. Ese pipeline no calcula por si solo variables como `mora_promedio_hist`, `dias_mora_observable`, `ratio_monto` o `friccion_contacto`. Esas variables deben construirse antes.

## Payload recomendado desde el front

Estos son los campos que conviene enviar o tener disponibles para construir una prediccion individual. Algunos pueden venir directamente del formulario y otros deberian venir del backend porque dependen del historial.

| Campo de negocio | Tipo | Origen esperado | Para que sirve |
|---|---|---|---|
| `factura_id` | texto | Front/backend | Trazabilidad. No entra como predictor. |
| `cliente_id` | texto | Front/backend | Trazabilidad y busqueda de historial. No entra como predictor. |
| `fecha_emision` | fecha | Factura | Permite calcular avance temporal. |
| `fecha_vencimiento` | fecha | Factura | Permite calcular vencimiento y mora observable. |
| `fecha_corte` | fecha | Backend/front | Fecha en la que se quiere hacer el scoring. Normalmente hoy. |
| `estado_factura` | categoria | Factura/cartera | Control operativo: abierta, pagada, anulada, castigada, en disputa, etc. No siempre entra al modelo, pero decide si se debe predecir. |
| `fecha_pago` | fecha/nulo | Pagos | Si existe pago final, la factura normalmente sale de la cola de prediccion activa. |
| `monto_pagado` | numero | Pagos | Sirve para control de estado y para actualizar historial. |
| `saldo_pendiente` | numero | Factura/pagos | Sirve para decidir si la factura sigue activa. El modelo actual usa `monto`, no saldo, pero el sistema operativo si debe controlar saldo. |
| `monto` | numero | Factura | Valor de la factura. |
| `condicion_dias` | numero | Factura/contrato | Plazo pactado de pago: 30, 45, 60 o 90 dias. |
| `sector` | categoria | Cliente | Sector del cliente: retail, manufactura, servicios, construccion, agro, tecnologia, salud o transporte. |
| `tiene_garantia` | 0/1 | Factura/cliente | Indica si existe garantia asociada. |
| `tiene_disputa_activa` | 0/1 | Gestion/cartera | Indica reclamo o disputa abierta. |
| `tiene_promesa_activa` | 0/1 | Promesas | Indica si existe promesa vigente. |
| `antiguedad_meses` | numero | Cliente | Tiempo del cliente en la cartera. |
| `num_facturas_prev` | numero | Historial | Facturas anteriores del cliente antes de esta factura. |
| `mora_promedio_hist` | numero | Historial | Promedio historico de dias de mora del cliente. |
| `mora_ultimo_tramo` | numero | Historial | Mora mas reciente observada en el historial. |
| `tasa_cumplimiento` | 0 a 1 | Historial | Proporcion de facturas pagadas a tiempo o dentro del criterio esperado. |
| `monto_promedio_hist` | numero | Historial | Monto promedio historico del cliente. |
| `moras_consecutivas` | numero | Historial | Racha reciente de moras. |
| `tasa_contacto_cliente` | 0 a 1 | Gestiones | Proporcion historica de contactos efectivos. |
| `num_gestiones_factura` | numero | Gestiones | Gestiones hechas sobre esta factura hasta el corte. |
| `fecha_ultima_gestion` | fecha/nulo | Gestiones | Permite calcular dias desde la ultima gestion. |
| `ultimo_resultado_enc` | categoria | Gestiones | Resultado codificado de la ultima gestion. Si no hay gestion: `cod_nan`. |
| `num_no_contesta_cons` | numero | Gestiones | No contestaciones consecutivas recientes. |
| `num_promesas_rotas` | numero | Promesas | Promesas incumplidas del cliente o de la factura segun la regla usada. |
| `promesas_total` | numero | Promesas | Total de promesas observadas. |
| `tasa_cumpl_promesas` | 0 a 1 | Promesas | Promesas cumplidas sobre promesas totales. |

## Features que consume el modelo

Estas son las columnas finales que deben existir antes de llamar al modelo.

| Feature | Significado didactico | Como se obtiene |
|---|---|---|
| `monto` | Valor de la factura. | Directo desde factura. |
| `monto_promedio_hist` | Monto promedio historico del cliente. | Agregado de facturas anteriores. |
| `ratio_monto` | Que tan grande es esta factura respecto al monto habitual del cliente. | `monto / monto_promedio_hist`; si no hay historial, usar una regla defensiva documentada. |
| `mora_promedio_hist` | Promedio historico de mora del cliente. | Agregado de pagos anteriores. |
| `mora_ultimo_tramo` | Mora mas reciente del cliente. | Ultimo tramo historico antes del corte. |
| `num_gestiones_factura` | Cuantas gestiones se hicieron sobre esta factura. | Conteo de gestiones hasta `fecha_corte`. |
| `dias_hasta_vence_pos` | Dias restantes si aun no vence; cero si ya vencio. | `max(fecha_vencimiento - fecha_corte, 0)`. |
| `dias_mora_observable` | Dias de mora que ya se pueden observar al corte. | `max(fecha_corte - fecha_vencimiento, 0)`. |
| `num_no_contesta_cons` | No contestaciones consecutivas. | Historial reciente de gestiones. |
| `num_promesas_rotas` | Promesas incumplidas. | Historial de promesas. |
| `promesas_total` | Total de promesas registradas. | Historial de promesas. |
| `dias_transcurridos_corte` | Cuantos dias han pasado desde la emision hasta el scoring. | `fecha_corte - fecha_emision`. |
| `condicion_dias` | Plazo pactado. | Directo o `fecha_vencimiento - fecha_emision`. |
| `antiguedad_meses` | Antiguedad del cliente. | Maestro de clientes. |
| `num_facturas_prev` | Volumen historico del cliente. | Conteo de facturas anteriores. |
| `tasa_cumplimiento` | Historial de cumplimiento de pago. | Pagos cumplidos / facturas historicas evaluables. |
| `moras_consecutivas` | Racha de moras. | Historial reciente del cliente. |
| `dias_desde_ultima_gestion` | Recencia de la ultima gestion. | `fecha_corte - fecha_ultima_gestion`; si no hay gestion, `-1`. |
| `dias_hasta_vence` | Dias hasta vencimiento; negativo si ya vencio. | `fecha_vencimiento - fecha_corte`. |
| `tasa_contacto_cliente` | Que tan contactable ha sido el cliente. | Contactos efectivos / gestiones. |
| `tasa_cumpl_promesas` | Que tanto cumple promesas. | Promesas cumplidas / promesas totales. |
| `intensidad_gestion` | Nivel de gestion relativo al tiempo transcurrido. | `num_gestiones_factura / (dias_transcurridos_corte + 1)`. |
| `friccion_contacto` | Dificultad de contacto. | Regla operacional basada en no contestaciones y gestiones; ejemplo: `num_no_contesta_cons / max(num_gestiones_factura, 1)`. |
| `ratio_promesas_rotas` | Proporcion de promesas incumplidas. | `num_promesas_rotas / max(promesas_total, 1)`. |
| `tiene_garantia` | Marca garantia. | Directo. |
| `sector_*` | Sector del cliente en formato one-hot. | Convertir `sector` a una columna activa y las demas en cero. |
| `tiene_disputa_activa` | Reclamo o disputa abierta. | Directo desde gestion/cartera. |
| `tiene_promesa_activa` | Promesa vigente. | Directo desde promesas. |
| `sin_gestion_previa` | Indica primer corte sin gestion. | `1` si `num_gestiones_factura == 0` o no hay ultima gestion. |
| `esta_vencida_al_corte` | Indica si ya vencio. | `1` si `fecha_corte > fecha_vencimiento`; si no, `0`. |
| `cliente_nuevo` | Cliente sin historial. | `1` si `num_facturas_prev == 0`; si no, `0`. |
| `ultimo_resultado_enc` | Resultado de la ultima gestion codificado. | Valor categorico compatible con entrenamiento; si no hay gestion, `cod_nan`. |

## Que devuelve el backend al front

El endpoint de prediccion deberia devolver algo parecido a:

| Campo | Uso visual |
|---|---|
| `factura_id` | Mostrar que factura se evaluo. |
| `cliente_id` | Mostrar cliente. |
| `predicted_class` | Clase mas probable: `on_time`, `+30`, `+60`, `+90`. |
| `prob_on_time` | Probabilidad de pago sin mora relevante. |
| `prob_plus_30` | Probabilidad de mora inicial. |
| `prob_plus_60` | Probabilidad de mora relevante. |
| `prob_plus_90` | Probabilidad de mora severa. |
| `confidence_probability` | Confianza de la clase ganadora. |
| `any_late_probability` | `prob_plus_30 + prob_plus_60 + prob_plus_90`; probabilidad de cualquier atraso. |
| `high_risk_probability` | `prob_plus_60 + prob_plus_90`; probabilidad de mora grave o severa. |
| `priority_score_0_100` | Score ponderado para ordenar facturas, considerando `+30`, `+60` y `+90`. |
| `feature_row` | Opcional para auditoria tecnica; no necesariamente visible al usuario final. |

## Score de priorizacion recomendado

Para la operacion de cobranza no conviene ignorar `+30`. Si una empresa ya dio 30, 45, 60 o 90 dias de credito, una prediccion `+30` significa que el pago podria llegar todavia mas tarde que el plazo pactado. Por eso se recomiendan tres lecturas:

| Metrica | Formula | Lectura |
|---|---|---|
| `any_late_probability` | `prob_plus_30 + prob_plus_60 + prob_plus_90` | Probabilidad de que no pague dentro del plazo esperado. |
| `high_risk_probability` | `prob_plus_60 + prob_plus_90` | Probabilidad de mora grave/severa. |
| `priority_score_0_100` | `100 * (0.40*prob_plus_30 + 0.70*prob_plus_60 + 1.00*prob_plus_90)` | Score para ordenar la cola: todo atraso suma, pero la mora severa pesa mas. |

Los pesos `0.40`, `0.70` y `1.00` son una regla operativa inicial. Si la empresa quiere cero tolerancia a atraso, puede subir el peso de `+30`, por ejemplo a `0.60` o `0.70`. Lo importante es que el score sea explicito y se mantenga estable para comparar facturas.

## Uso operativo de `fecha_corte`

`fecha_corte` es la fecha de scoring. Responde la pregunta: "con la informacion disponible hasta esta fecha, que riesgo tiene esta factura?".

Si hoy se crea una factura, `fecha_corte` normalmente es hoy. En ese primer scoring puede no existir ninguna gestion sobre esa factura, por lo que varios campos quedan en valores iniciales:

| Feature | Valor tipico al crear factura |
|---|---|
| `num_gestiones_factura` | 0 |
| `dias_desde_ultima_gestion` | -1 |
| `sin_gestion_previa` | 1 |
| `ultimo_resultado_enc` | `cod_nan` |
| `dias_mora_observable` | 0 si aun no vence |
| `dias_hasta_vence` | dias entre vencimiento y fecha de corte |

Aunque no haya gestiones de esa factura, el modelo si puede usar historial del cliente: monto promedio, mora historica, cumplimiento, contactabilidad, promesas rotas, facturas previas y sector.

Si al dia siguiente se vuelve a consultar la misma factura, `fecha_corte` cambia. Por eso cambian features temporales como:

| Feature | Por que cambia al avanzar el dia |
|---|---|
| `dias_transcurridos_corte` | Aumenta con el tiempo desde emision. |
| `dias_hasta_vence` | Disminuye cada dia antes del vencimiento y luego se vuelve negativo. |
| `dias_hasta_vence_pos` | Disminuye hasta llegar a cero. |
| `esta_vencida_al_corte` | Cambia a 1 cuando pasa el vencimiento. |
| `dias_mora_observable` | Empieza a crecer despues del vencimiento. |
| `intensidad_gestion` | Puede bajar si pasan dias sin nuevas gestiones, porque las gestiones se dividen para mas tiempo transcurrido. |
| `dias_desde_ultima_gestion` | Aumenta si existe una gestion previa y no se ha gestionado de nuevo. |

Entonces si el sistema tiene una fecha de simulacion global y se cambia de `2026-05-10` a `2026-05-15`, lo correcto es recalcular las predicciones de todas las facturas abiertas con esa nueva `fecha_corte`. No necesariamente se recalculan facturas ya pagadas, anuladas o cerradas, porque ya no son candidatas activas de priorizacion.

## Tiempo real vs lote diario

Hay dos formas razonables de usar el modelo:

| Modo | Cuando usarlo | Que recalcula |
|---|---|---|
| Tiempo real/on demand | Cuando el usuario abre una factura, crea una gestion, registra una promesa o simula una fecha. | Recalcula la factura consultada o las facturas afectadas. |
| Lote diario | Al inicio del dia operativo o al cambiar la fecha global de simulacion. | Recalcula todas las facturas abiertas para ordenar la cola del dia. |

Para un sistema academico o demo full stack, lo mas claro es combinar ambos:

1. Un boton o selector de `fecha_corte` global para simular el dia.
2. Una vista de cartera que recalcula todas las facturas abiertas con esa fecha.
3. Acciones como "agregar gestion", "agregar promesa", "registrar pago" o "abrir disputa" que recalculan inmediatamente la factura afectada.
4. Si la accion cambia estadisticas del cliente, recalcular tambien las demas facturas abiertas de ese mismo cliente.

## Eventos que deberian disparar recalculo

| Evento | Que pasa | Alcance recomendado |
|---|---|---|
| Cambia `fecha_corte` global | Cambian variables temporales y mora observable. | Recalcular todas las facturas abiertas. |
| Se crea una factura | Nace un nuevo caso de scoring. | Calcular prediccion inicial de esa factura. |
| Se edita monto, vencimiento, condicion o garantia | Cambian features directos y temporales. | Recalcular esa factura. |
| Se agrega una gestion | Cambian `num_gestiones_factura`, `dias_desde_ultima_gestion`, `ultimo_resultado_enc`, friccion y contacto. | Recalcular esa factura; si afecta tasas del cliente, recalcular facturas abiertas del mismo cliente. |
| Se agrega una promesa | Cambian promesas activas y totales. | Recalcular esa factura y posiblemente las del mismo cliente. |
| Se rompe o cumple una promesa | Cambian `num_promesas_rotas`, `tasa_cumpl_promesas`, `ratio_promesas_rotas`. | Recalcular factura y facturas abiertas del mismo cliente. |
| Se abre o cierra una disputa | Cambia `tiene_disputa_activa`. | Recalcular esa factura. |
| Se registra pago total | La factura sale de la cola activa. | No priorizar esa factura; actualizar historial del cliente y recalcular otras facturas abiertas del cliente. |
| Se registra pago parcial | Cambia el estado operativo y saldo. | El modelo actual no usa saldo, pero el sistema deberia recalcular estado; para modelo futuro convendria agregar saldo pendiente. |

## Estado de factura y prediccion

No toda factura debe pasar siempre por el modelo. Primero se debe decidir si esta activa para cobranza:

| Estado | Recomendacion |
|---|---|
| Abierta/no vencida | Predecir; puede servir para cobranza preventiva. |
| Abierta/vencida | Predecir; caso principal de priorizacion. |
| Con promesa activa | Predecir, pero mostrar contexto de promesa. |
| En disputa | Predecir con cautela; no usar como decision automatica. |
| Pagada total | No mostrar en cola activa; usar para actualizar historial. |
| Anulada/cancelada | No predecir como cobranza activa. |
| Castigada/cerrada | Depende del proceso; normalmente fuera de cola operativa normal. |

## Validaciones minimas antes de predecir

- Todas las columnas de `model_feature_schema.csv` deben existir.
- Las tasas deben estar entre 0 y 1.
- El sector debe pertenecer al catalogo entrenado.
- Las fechas deben producir una relacion coherente entre emision, vencimiento y corte.
- No se deben usar datos futuros al corte. Por ejemplo, no usar pagos o gestiones ocurridos despues de `fecha_corte`.
- Si el cliente es nuevo, se deben usar reglas defensivas y explicitas para historial faltante.

## Que falta para produccion

Para pasar de notebook a sistema full stack conviene crear un modulo reutilizable, por ejemplo `feature_builder_cobranzas.py`, que implemente las formulas anteriores y sea usado tanto por notebooks como por la API. Asi se evita que el backend, el notebook y el entrenamiento calculen features de forma distinta.

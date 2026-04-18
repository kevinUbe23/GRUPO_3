# Contexto actualizado del proyecto

Estoy desarrollando un **proyecto integrador de Inteligencia Artificial** académico cuyo problema central es: **Sistema inteligente de priorización de cobranzas para empresas que venden a crédito**.

---

## Problema que resuelve

Muchas empresas gestionan la cobranza con criterios tradicionales, como días de mora o experiencia del gestor, lo que limita la capacidad de anticipar incumplimientos y actuar oportunamente. Este sistema utiliza IA para predecir el riesgo de mora por factura, segmentar clientes según su comportamiento histórico y sugerir la acción de cobranza más adecuada en cada momento.

---

## Los tres componentes del sistema

### Componente 1 — Predicción de mora por factura (scoring dinámico)

Es el componente central del sistema. El modelo predice, para una factura dada, la probabilidad de que el pago termine en una de cuatro categorías: `on_time`, `+30`, `+60` o `+90`.

La predicción no ocurre una sola vez, sino en cada **corte de scoring**:

- **Corte 0:** al emitir la factura.
- **Corte N (N ≥ 1):** después de cada gestión registrada sobre esa factura.

Esto permite actualizar el riesgo conforme avanza el caso. Por ejemplo, si ya hubo varias gestiones y el cliente no contesta, esa información se incorpora como feature y modifica la predicción.

El dataset de entrenamiento refleja esta lógica temporal: por cada factura se generan múltiples filas, una por cada corte, todas compartiendo el mismo target final.

El output del modelo es un vector de probabilidades, por ejemplo:
`{on_time: 72%, +30: 18%, +60: 7%, +90: 3%}`

### Componente 2 — Segmentación de clientes por estrellas

Agrupa a los clientes según su comportamiento histórico de pago y les asigna un puntaje de **1 a 5 estrellas**:

- **5 estrellas:** cliente excelente
- **1 estrella:** cliente crítico

Este puntaje resume el perfil de riesgo histórico del cliente y sirve como input del componente 3.

### Componente 3 — Decisor de acciones de cobranza

Es una capa de reglas de negocio, no un modelo de machine learning. Combina la probabilidad de mora del componente 1 con la calificación por estrellas del componente 2 para sugerir una acción concreta de cobranza.

Las acciones pueden ser preventivas o reactivas, por ejemplo:

| Prob. mora | Estrellas | Situación | Acción sugerida |
|---|---|---|---|
| < 20% | 4–5 ★ | Dentro del plazo, bajo riesgo | Sin acción / monitoreo |
| 20–40% | 3–4 ★ | Dentro del plazo, riesgo leve | Mensaje recordatorio preventivo |
| 40–60% | 2–3 ★ | Dentro del plazo, riesgo medio | Llamada de seguimiento |
| 60–80% | 2–3 ★ | En mora o alto riesgo | Visita o llamada urgente |
| > 80% | 1–2 ★ | Mora grave | Escalamiento legal o formalización de garantía |

---

## Modelo de datos — 4 tablas

### `clientes`
Perfil base de cada empresa cliente.

Campos:
- `cliente_id`
- `nombre`
- `sector`
- `antiguedad_meses`
- `tiene_garantia` (0/1)
- `perfil_pago` (solo simulación, no entra al modelo)

> **Nota:** `limite_credito` fue eliminado del diseño por no aportar valor al modelo ni a la lógica actual del sistema.

### `facturas`
Una fila por factura emitida. Es la unidad principal de predicción del sistema.

Campos:
- `factura_id`
- `cliente_id`
- `fecha_emision`
- `fecha_vencimiento`
- `fecha_pago_real`
- `condicion_dias` (30/45/60/90)
- `monto`
- `target_mora` (`on_time`, `+30`, `+60`, `+90`)
- `dias_mora_real`

Reglas principales:
- `fecha_vencimiento = fecha_emision + condicion_dias`
- `dias_mora_real = max(0, fecha_pago_real - fecha_vencimiento)`
- `target_mora` es la variable objetivo del modelo, no una feature de entrada
- los `factura_id` deben respetar el orden cronológico global de `fecha_emision`

### `gestiones_cobranza`
Cada intento de contacto o seguimiento sobre una factura. Una factura puede tener múltiples gestiones a lo largo del tiempo.

Campos:
- `gestion_id`
- `factura_id`
- `cliente_id`
- `fecha_gestion`
- `canal` (`whatsapp`, `email`, `llamada`, `visita`, `carta_notarial`)
- `contacto_exitoso` (0/1)
- `resultado`
- `motivo_no_pago`
- `dias_mora_en_gestion`

Catálogo actualizado de resultados:
- **Resultados de no contacto:** `no_contesta`, `numero_invalido`, `cliente_ausente`
- **Resultados de contacto efectivo:** `pagado`, `promesa_de_pago`, `disputa_monto`, `rechazo_pago`, `en_proceso_interno`, `confirma_pago`

Reglas principales:
- puede haber gestiones **preventivas** antes del vencimiento y **reactivas** después del vencimiento
- `motivo_no_pago` solo se registra si hubo contacto exitoso, la factura ya estaba vencida y no hubo pago inmediato
- debe existir coherencia entre `canal`, `contacto_exitoso` y `resultado`

### `promesas_pago`
Compromisos de pago que surgen únicamente de una gestión con resultado `promesa_de_pago`.

Campos:
- `promesa_id`
- `gestion_id`
- `factura_id`
- `cliente_id`
- `fecha_promesa`
- `fecha_compromiso`
- `se_cumplio` (0/1)

Reglas principales:
- `fecha_promesa` coincide con la fecha de la gestión que originó la promesa
- la promesa siempre representa el pago total de la factura
- `se_cumplio = 1` si `fecha_pago_real <= fecha_compromiso`
- `se_cumplio = 0` si `fecha_pago_real > fecha_compromiso`

> **Campos eliminados del diseño:** `monto_comprometido`, `fecha_cumplimiento_real`, `dias_retraso_promesa`

---

## Dataset de entrenamiento — diseño de cortes temporales

El dataset de entrenamiento no tiene una sola fila por factura, sino **una fila por corte de scoring por factura**.

Los cortes ocurren en estos momentos:

- **Corte 0:** al emitir la factura. Solo usa historial previo del cliente.
- **Corte N:** después de cada gestión registrada. Usa historial del cliente y lo ocurrido con esa factura hasta ese momento.

Todas las filas de una misma factura comparten el mismo `target_mora` final.

Esto permite construir un esquema de **scoring dinámico**, donde el riesgo se actualiza conforme avanza el proceso de cobranza.

---

## Features del modelo por categoría

### Features de historial del cliente
Disponibles en todos los cortes y calculadas sobre facturas anteriores a la factura actual:

- `mora_promedio_hist`
- `mora_ultimo_tramo`
- `tasa_cumplimiento`
- `moras_consecutivas`
- `num_facturas_prev`
- `monto_promedio_hist`
- `ratio_monto`
- `antiguedad_meses`
- `tiene_garantia`
- variables one-hot de `sector`
- `condicion_dias`
- `monto`

### Features de gestiones sobre la misma factura
Disponibles desde el corte 1 en adelante y calculadas hasta la fecha del corte:

- `num_gestiones_factura`
- `dias_desde_ultima_gestion`
- `tasa_contacto_cliente`
- `ultimo_resultado_enc`
- `num_no_contesta_cons`
- `tiene_disputa_activa`

### Features de promesas
Calculadas sin fuga de información y respetando la fecha de corte:

- `num_promesas_rotas`
- `tasa_cumpl_promesas`
- `tiene_promesa_activa`
- `promesas_total`

### Features adicionales de tiempo
- `dias_desde_emision`
- `dias_hasta_vence`

### Target
- `target_mora` → `on_time / +30 / +60 / +90`

---

## Consideraciones metodológicas importantes

- Todas las features deben calcularse únicamente con información disponible hasta la `fecha_corte`.
- No se debe usar información futura ni variables derivadas del desenlace final como features.
- Si `features_ml.csv` contiene múltiples filas por factura, el split de entrenamiento y prueba debe hacerse por `factura_id`, no por fila.
- El EDA mostró que existe desbalance en el dataset de cortes, por lo que la métrica principal del componente 1 se mantiene como **F1-macro**.

---

## Plan de algoritmos — 5 en total

### Componente 1 — Clasificación multiclase
Se compararán tres algoritmos:
- XGBoost
- Random Forest
- Regresión Logística Multinomial

Métricas:
- F1-macro
- Accuracy
- AUC-ROC multiclase
- F1 por clase
- matriz de confusión
- tiempo de entrenamiento

### Componente 2 — Clustering
Se compararán dos algoritmos:
- K-Means (K=5)
- DBSCAN

Métricas:
- Silhouette Score
- Davies-Bouldin Index
- visualización PCA 2D

### Componente 3
No utiliza ML. Opera con una tabla de decisión basada en los outputs de los componentes 1 y 2.

---

## Stack tecnológico

- **Simulación y ML:** Python, pandas, numpy, faker, scikit-learn, xgboost, joblib, matplotlib, seaborn
- **Backend:** FastAPI local, con endpoints `/predict`, `/segment` y `/action`
- **Frontend:** Next.js + Tailwind CSS

---

## Lógica de simulación de datos

Los datos son completamente simulados. Los clientes tienen cuatro perfiles internos que controlan sus probabilidades de mora y cumplimiento de promesas:

| Perfil | % clientes | Prob. on_time | Prob. cumplir promesa |
|---|---|---|---|
| excelente | 25% | 85% | 85% |
| regular | 40% | 55% | 60% |
| riesgoso | 25% | 20% | 30% |
| crítico | 10% | 5% | 10% |

Las gestiones siguen una lógica de escalamiento por canal:
**WhatsApp → email → llamada → visita → carta notarial**

Además:
- las gestiones preventivas pueden aparecer antes del vencimiento
- las gestiones reactivas aparecen después del vencimiento
- el número de gestiones tiende a aumentar en clases de mora más severas
- el dataset de features se construye respetando el principio de no leakage temporal

---

## Estado actual del proyecto

A la fecha, el proyecto ya cuenta con:

- simulación corregida y consistente con las reglas de negocio
- dataset final para EDA
- notebook de EDA alineado a la rúbrica
- reporte académico preliminar de hallazgos

El siguiente paso es utilizar este contexto junto con el EDA para construir la fase experimental de comparación de algoritmos y entrenamiento de modelos.

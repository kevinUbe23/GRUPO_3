# Guia de predicciones por factura

## Proposito

Esta guia explica como leer y probar el componente supervisado de prediccion de mora de la fase 4. El modelo toma una fila de `features_ml_prepared.csv`, que representa una factura en un corte temporal, y estima la clase `target_mora`.

El modelo actualmente seleccionado es `Logistic Regression`, guardado en `04_evaluacion_modelos_ia/outputs/best_model_artifact.joblib`.

## Que predice

Las clases posibles son:

| Clase | Lectura operativa |
|---|---|
| `on_time` | Probable pago sin mora relevante. |
| `+30` | Riesgo de mora inicial. Ya es importante porque implica cobrar despues del credito pactado. |
| `+60` | Riesgo de mora grave para priorizacion. |
| `+90` | Riesgo severo; requiere mayor atencion operativa. |

El modelo no decide acciones automaticas. Su uso correcto es priorizar revision y gestion humana.

## Como leer los outputs nuevos

| Archivo | Uso |
|---|---|
| `outputs/prediction_test_rows.csv` | Predicciones para todos los cortes de test. |
| `outputs/prediction_invoice_summary.csv` | Una fila por factura de test, tomando el ultimo corte disponible. |
| `outputs/prediction_invoice_timeline_examples.csv` | Ejemplos concretos de facturas con evolucion del riesgo por corte. |
| `outputs/prediction_new_invoice_inputs.csv` | Facturas nuevas de ejemplo con todos los parametros requeridos. |
| `outputs/prediction_new_invoice_scenarios.csv` | Predicciones de esas facturas nuevas. |
| `outputs/prediction_feature_dictionary.csv` | Diccionario de parametros/features del modelo. |
| `outputs/prediction_model_parameters.csv` | Coeficientes completos del modelo logistico. |
| `outputs/prediction_top_model_parameters.csv` | Coeficientes mas influyentes por clase. |

## Columnas clave

| Columna | Interpretacion |
|---|---|
| `predicted_class` | Clase con mayor probabilidad. |
| `confidence_probability` | Probabilidad de la clase ganadora. |
| `any_late_probability` | Suma de probabilidades de `+30`, `+60` y `+90`; indica probabilidad de cualquier atraso. |
| `high_risk_probability` | Suma de probabilidades de `+60` y `+90`; indica mora grave o severa. |
| `priority_score_0_100` | Score ponderado para ordenar facturas: `+30` tambien cuenta, pero `+60` y `+90` pesan mas. |
| `actual_class` | Clase real conocida en test. |
| `is_correct` | Si la prediccion coincide con la clase real. |

## Como se prueba con test

La prueba usa `03_preparacion/outputs/test_facturas_ids.csv`. La separacion se respeta por `factura_id`, no por fila, porque cada factura puede tener varios cortes. Esto evita fuga de informacion entre train y test.

Hay dos lecturas complementarias:

- Por corte: evalua cada fila temporal de `features_ml_prepared.csv`.
- Por factura: usa el ultimo corte disponible para ver una decision consolidada por factura.

En la corrida actual, el script evaluo 3,936 cortes de test y 1,068 facturas de test en ultimo corte. La exactitud en ultimo corte fue 0.7406.

## Como probar facturas totalmente nuevas

Una factura nueva debe tener las mismas columnas de `outputs/model_feature_schema.csv`. El script `exploracion_predicciones_cobranzas.py` incluye cuatro escenarios:

| Escenario | Resultado actual |
|---|---|
| Factura preventiva con buen historial | `on_time`, score 19.97 |
| Factura nueva sin historial | `+30`, score 56.49 porque cualquier atraso preocupa |
| Factura vencida con contacto dificil | `+60`, score 82.05 |
| Factura critica con mora severa | `+90`, score 99.26 |

Para crear otros casos, se puede copiar una fila de `prediction_new_invoice_inputs.csv`, cambiar variables como `dias_mora_observable`, `tasa_contacto_cliente`, `num_promesas_rotas`, `tasa_cumplimiento` o `tiene_disputa_activa`, y volver a ejecutar:

```powershell
.\.venv\Scripts\python.exe 04_evaluacion_modelos_ia\exploracion_predicciones_cobranzas.py
```

## Como se conecta con un front

El front no deberia enviar directamente las 39 columnas tecnicas si el usuario solo esta creando o consultando una factura. Lo recomendable es que envie datos de negocio: factura, cliente, sector, fechas, monto, estado de disputa, promesas y gestion. Luego el backend debe construir los features con una capa de feature engineering antes de llamar al modelo.

La especificacion detallada queda en `04_evaluacion_modelos_ia/contrato_front_prediccion.md`.

Resumen del flujo:

```text
Front
  -> datos de factura, cliente, gestion y promesas
Backend feature builder
  -> calcula ratio_monto, dias_mora_observable, friccion_contacto, sector one-hot, etc.
Pipeline del modelo
  -> imputa, transforma, escala y codifica features ya calculados
Modelo
  -> predicted_class, probabilidades, any_late_probability, high_risk_probability y priority_score_0_100
```

Punto importante: el pipeline guardado en `best_model_artifact.joblib` no reemplaza al feature builder. El pipeline transforma columnas que ya existen; no consulta historiales ni calcula variables de negocio desde cero.

## Score de prioridad

La primera version del score usaba solo `+60` y `+90`, pero para una empresa que ya otorga credito a 30, 45, 60 o 90 dias, cualquier atraso adicional importa. Por eso la lectura recomendada separa:

| Medida | Formula conceptual | Uso |
|---|---|---|
| `any_late_probability` | `prob(+30) + prob(+60) + prob(+90)` | Saber si la factura probablemente se atrasara en cualquier nivel. |
| `high_risk_probability` | `prob(+60) + prob(+90)` | Medir mora grave/severa. |
| `priority_score_0_100` | `100 * (0.40*prob(+30) + 0.70*prob(+60) + 1.00*prob(+90))` | Ordenar la cola, castigando mas la mora severa pero sin ignorar `+30`. |

Asi, una factura con alta probabilidad de `+30` ya sube en prioridad, aunque todavia no tenga pinta de `+60` o `+90`.

## Como cambia con el tiempo

La `fecha_corte` es la fecha de scoring. Si hoy se sube una factura, se predice con lo que existe hasta hoy. Si manana se vuelve a consultar, la fecha de corte cambia y tambien cambian variables como `dias_hasta_vence`, `dias_transcurridos_corte`, `dias_mora_observable` y `dias_desde_ultima_gestion`.

En una simulacion, si cambias la fecha global, lo correcto es recalcular todas las facturas abiertas. En operacion real, esto puede hacerse con un lote diario para ordenar la cola del dia y tambien en tiempo real cuando ocurre un evento importante.

Eventos que deberian recalcular:

- Cambio de fecha de corte global: todas las facturas abiertas.
- Nueva factura: la factura creada.
- Nueva gestion: la factura gestionada y, si cambia metricas del cliente, sus otras facturas abiertas.
- Nueva promesa o promesa rota/cumplida: la factura y posiblemente el cliente.
- Disputa abierta/cerrada: la factura.
- Pago total: la factura sale de la cola activa y se actualiza historial del cliente.
- Pago parcial: se actualiza el estado/saldo; el modelo actual no usa saldo, pero el sistema debe reflejarlo.

## Figuras disponibles

| Figura | Lectura |
|---|---|
| `prediction_confusion_matrix_heatmap.png` | Donde acierta y donde confunde clases. |
| `prediction_test_risk_distribution.png` | Distribucion del score de prioridad en test. |
| `prediction_invoice_timeline_examples.png` | Evolucion de riesgo en facturas concretas. |
| `prediction_new_scenarios_probabilities.png` | Probabilidades por clase en facturas nuevas. |

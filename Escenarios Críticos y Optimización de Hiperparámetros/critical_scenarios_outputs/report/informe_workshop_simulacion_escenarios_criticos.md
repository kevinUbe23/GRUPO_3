# Workshop: Simulación de Escenarios Críticos
## Informe Técnico Académico
### Sistema Inteligente de Priorización de Cobranzas para PYMEs

**Repositorio:** https://github.com/kevinUbe23/GRUPO_3  
**Fecha de generación:** 24/04/2026  
**Versión del notebook:** Final corregida sin leakage v2  
**Directorio de outputs:** `critical_scenarios_outputs`

---

## 1. Introducción

El presente informe documenta la simulación de escenarios críticos aplicada al componente supervisado del sistema inteligente de priorización de cobranzas. El objetivo es evaluar la robustez del modelo multiclase de predicción de mora ante condiciones adversas que pueden ocurrir en producción: ruido en variables operativas, valores faltantes, outliers, drift temporal, drift por sector y pruebas de estrés de entrada.

Este análisis es necesario porque un modelo de cobranzas no solo debe obtener buen desempeño en un conjunto limpio, sino mantener estabilidad cuando cambian la calidad de captura, el comportamiento de pago y la composición de la cartera. Para el negocio, la clase `+90` es especialmente crítica, ya que un falso negativo en ese grupo puede retrasar acciones de cobro prioritarias.

---

## 2. Contexto del Modelo y Dataset

### 2.1 Fuente de datos

- **Archivo utilizado:** `features_ml_prepared.csv` (prepared).
- **Unidad de análisis:** corte de scoring por factura.
- **Registros originales:** 19,671 filas × 43 columnas.
- **Variables predictoras empleadas:** 39.
- **ID de control anti-leakage:** `factura_id`.
- **ID excluido como predictor:** `cliente_id`.

### 2.2 Control de leakage y selección de features

La partición principal se realizó a nivel de `factura_id`, no por fila. Esto evita que cortes distintos de la misma factura aparezcan simultáneamente en entrenamiento y prueba. La auditoría del split reportó:

| Elemento | Valor |
|---|---:|
| Facturas en train | 4270 |
| Facturas en test | 1068 |
| Intersección de `factura_id` train/test | 0 |

La selección de features excluyó únicamente la variable objetivo, identificadores, fechas reales y variables de leakage explícito. No se excluyeron variables por contener la palabra `corte`. Por tanto, variables derivadas válidas como `dias_transcurridos_corte` y `esta_vencida_al_corte` se conservaron cuando existían o pudieron construirse con información disponible al corte.

### 2.3 Preprocesamiento

El pipeline usa imputación por mediana para variables numéricas y `OneHotEncoder(handle_unknown="ignore")` para variables categóricas nominales. Esta corrección reemplaza el uso de `OrdinalEncoder` en variables nominales, evitando inducir un orden artificial en categorías como `ultimo_resultado_enc` o `sector_enc`.

---

## 3. Metodología de Simulación

### 3.1 Baseline limpio

El baseline corresponde a la evaluación del pipeline entrenado sobre el conjunto de prueba limpio. La degradación porcentual de cada escenario se calculó como:

\[
	ext{degradación (\%)} = rac{	ext{F1-macro baseline} - 	ext{F1-macro escenario}}{	ext{F1-macro baseline}} 	imes 100
\]

### 3.2 Ruido, missing y outliers

Se simularon perturbaciones de ruido gaussiano, missing aleatorio, missing dirigido por grupo de variables, outliers sobre monto y un perfil crítico combinado. El missing dirigido permite representar fallos operativos reales, por ejemplo, caída del CRM o ausencia sistemática de registros de gestión.

### 3.3 Drift temporal sin leakage

El drift temporal se corrigió para separar pasado y futuro a nivel de `factura_id`. La factura completa se asigna a train temporal o a test temporal, y el notebook valida explícitamente que la intersección de `factura_id` sea cero. Esto evita leakage entre cortes de una misma factura.

### 3.4 Drift por sector

Cuando el dataset contiene sectores one-hot encoded, el notebook evalúa todos los campos `sector_*` por separado. Para cada sector se exporta una tabla con sector, número de registros, F1-macro, recall de `+90` y degradación.

### 3.5 Pruebas de estrés

La sección de inputs fuera de rango evalúa estabilidad técnica del pipeline y necesidad de validadores antes de inferencia. No debe interpretarse como desempeño predictivo real, porque se modifican variables hacia valores imposibles en el dominio, como tasas mayores a 1, montos extremos o días negativos.

---

## 4. Resultados Baseline

| Métrica | Valor |
|---|---:|
| Accuracy | 0.5379 |
| Balanced Accuracy | 0.5572 |
| F1-macro | 0.5395 |
| F1-weighted | 0.5297 |
| Precision macro | 0.5310 |
| Recall macro | 0.5572 |
| Recall +90 | 0.5657 |
| F1 +90 | 0.5800 |
| Latencia ms/reg | 0.0480 |

---

## 5. Resultados por Escenario

| Escenario | F1-macro | Balanced Accuracy | Recall +90 | Degradación F1-macro |
|---|---:|---:|---:|---:|
| baseline_clean | 0.5395 | 0.5572 | 0.5657 | nan% |
| ruido_bajo_5pct | 0.5426 | 0.5600 | 0.5503 | -0.57% |
| ruido_medio_10pct | 0.5379 | 0.5534 | 0.5557 | 0.30% |
| ruido_alto_20pct | 0.5220 | 0.5366 | 0.5573 | 3.24% |
| missing_bajo_5pct | 0.5264 | 0.5355 | 0.5496 | 2.43% |
| missing_medio_15pct | 0.4980 | 0.4948 | 0.5088 | 7.69% |
| missing_alto_30pct | 0.4604 | 0.4511 | 0.4404 | 14.66% |
| missing_dirigido_gestion | 0.3223 | 0.3454 | 0.5926 | 40.26% |
| missing_dirigido_promesas | 0.5344 | 0.5511 | 0.5065 | 0.95% |
| missing_dirigido_historico | 0.5176 | 0.5130 | 0.4166 | 4.06% |
| outlier_monto_3x_5pct | 0.5406 | 0.5588 | 0.5642 | -0.20% |
| outlier_monto_5x_5pct | 0.5412 | 0.5593 | 0.5657 | -0.32% |
| outlier_monto_10x_5pct | 0.5428 | 0.5608 | 0.5657 | -0.61% |
| outlier_monto_5x_10pct | 0.5409 | 0.5589 | 0.5650 | -0.26% |
| outlier_critico_combinado | 0.5363 | 0.5489 | 0.5596 | 0.59% |
| drift_temporal_factura_pasado_vs_futuro | 0.5330 | 0.5850 | 0.3791 | 1.20% |
| drift_sector_retail | 0.5076 | 0.5369 | 0.4348 | 5.91% |
| drift_sector_manufactura | 0.5527 | 0.5746 | 0.6226 | -2.45% |
| drift_sector_servicios | 0.5494 | 0.5642 | 0.4474 | -1.84% |
| drift_sector_construccion | 0.5287 | 0.5441 | 0.5887 | 2.00% |
| drift_sector_agro | 0.4743 | 0.4977 | 0.5793 | 12.09% |
| drift_sector_tecnologia | 0.5174 | 0.5252 | 0.6102 | 4.10% |
| drift_sector_salud | 0.5495 | 0.5594 | 0.4453 | -1.85% |
| drift_sector_transporte | 0.5344 | 0.5684 | 0.5804 | 0.95% |
| drift_cartera_critica_q75 | 0.3684 | 0.3629 | 0.7526 | 31.71% |
| stress_inputs_fuera_de_rango | 0.4870 | 0.5744 | 0.5703 | 9.73% |
| stress_perfil_critico_combinado | 0.5363 | 0.5489 | 0.5596 | 0.59% |
| stress_volumen_10x | 0.5395 | 0.5572 | 0.5657 | 0.00% |
| stress_volumen_50x | 0.5395 | 0.5572 | 0.5657 | 0.00% |
| stress_volumen_100x | 0.5395 | 0.5572 | 0.5657 | 0.00% |
| stress_col_extra_irrelevante | 0.5395 | 0.5572 | 0.5657 | 0.00% |


### 5.1 Peor escenario identificado

El peor escenario de la ejecución fue **`missing_dirigido_gestion`**, con F1-macro **0.3223** y degradación **40.26%** frente al baseline. En particular, el escenario solicitado **`missing_dirigido_gestion`** obtuvo F1-macro **0.3223** y degradación cercana a **40.26%**. Este resultado confirma que la pérdida simultánea de variables de gestión afecta severamente la capacidad del modelo para separar las clases de mora.

La interpretación de negocio es directa: cuando se pierden variables como `dias_desde_ultima_gestion`, `ultimo_resultado_enc`, `tasa_contacto_cliente` y `num_gestiones_factura`, el modelo deja de observar señales recientes de interacción con el cliente. Por eso cae el F1-macro hasta **0.3223**. Aunque el recall de `+90` observado en este escenario fue **0.5926**, el desempeño global se deteriora porque el modelo pierde discriminación entre las demás clases, especialmente en fronteras intermedias como `+30` y `+60`.

---

## 6. Drift por Sector

| Sector | n_registros | F1-macro | Recall +90 | Degradación F1-macro |
|---|---:|---:|---:|---:|
| agro | 439 | 0.4743 | 0.5793 | 12.09% |
| retail | 368 | 0.5076 | 0.4348 | 5.91% |
| tecnologia | 316 | 0.5174 | 0.6102 | 4.10% |
| construccion | 686 | 0.5287 | 0.5887 | 2.00% |
| transporte | 589 | 0.5344 | 0.5804 | 0.95% |
| servicios | 276 | 0.5494 | 0.4474 | -1.84% |
| salud | 525 | 0.5495 | 0.4453 | -1.85% |
| manufactura | 737 | 0.5527 | 0.6226 | -2.45% |


El análisis por sector permite identificar si el modelo se comporta de forma desigual ante subconjuntos de la cartera. La salida `drift_sector_results.csv` debe revisarse junto con el tamaño de muestra, porque sectores con pocos registros pueden mostrar alta variabilidad.

---

## 7. Latencia y Pruebas de Volumen

| Registros | Latencia Total (s) | Latencia/Reg (ms) |
|---:|---:|---:|
| 1 | 0.0280 | 28.0184 |
| 10 | 0.0260 | 2.6036 |
| 100 | 0.0323 | 0.3228 |
| 1,000 | 0.0549 | 0.0549 |
| 3,936 | 0.1275 | 0.0324 |


Las pruebas de volumen evalúan escalabilidad operativa del pipeline. Si la latencia por registro se mantiene estable al aumentar el tamaño del lote, el modelo es viable para procesamiento batch de cartera. Para inferencia transaccional, conviene monitorear la latencia en tamaños de lote pequeños.

---

## 8. Matriz de Riesgo

| Escenario | Severidad | Métrica afectada | Acción de mitigación |
|---|---|---|---|
| Ruido Gaussiano Alto (20%) | Media | F1-macro, Recall +90 | Validación en fuente, robust scaling, winsorización controlada |
| Missing Dirigido — Gestión | Alta | Recall +90, F1-macro | Imputación semántica, alertas CRM, revisión humana obligatoria |
| Missing Dirigido — Historial | Alta | Balanced Accuracy, F1 +90 | Modelo de arranque en frío, imputación por segmento, retraining |
| Outlier Monto 10x | Media | F1-macro, Recall +90 | Winsorización, revisión manual de montos sobre percentil 99 |
| Outlier Crítico Combinado | Crítica | F1-macro, Recall +90, Precision | Revisión humana obligatoria para score > umbral crítico |
| Drift Temporal | Alta | F1-macro, Balanced Accuracy | Monitoreo mensual PSI; retraining si F1-macro cae >10% |
| Drift Cartera Crítica Q75 | Alta | Recall +90, F1-macro | Estratificación por segmento en retraining; alertas PSI > 0.2 |
| Inputs Fuera de Rango | Crítica | Estabilidad del pipeline | Validadores de rango antes de inferencia; contratos de datos |
| Volumen 100x | Media | Latencia de predicción | Inferencia batch asíncrona, escalado horizontal, caché de predicciones |
| Columna Predictora Faltante | Alta | F1-macro (o fallo total) | Schema validation en ingestión; contratos de API de datos |


---

## 9. Análisis Crítico

El modelo muestra robustez moderada ante ruido gaussiano y outliers aislados, lo cual es esperable en modelos de árboles y boosting porque las particiones por umbral suelen tolerar pequeñas perturbaciones. Sin embargo, la robustez disminuye ante missing dirigido, especialmente cuando se anula el bloque de gestión.

El drift temporal corregido por `factura_id` permite una lectura más confiable que un split por filas. Si se mezclaran cortes de una misma factura entre train y test, las métricas podrían inflarse artificialmente. La validación de intersección nula de `factura_id` es, por tanto, una condición metodológica indispensable.

El drift por sector aporta una visión de estabilidad por grupos de negocio. Si algún sector presenta una degradación superior al promedio o bajo recall en `+90`, la recomendación es revisar calibración por segmento, enriquecer datos de ese sector o establecer reglas de revisión humana.

La prueba de inputs fuera de rango no debe usarse para afirmar que el modelo predice bien o mal bajo esos valores. Su objetivo es evidenciar si el pipeline colapsa y justificar validadores de dominio antes de inferencia: tasas en [0,1], montos positivos, días no negativos y categorías conocidas o manejadas por `OneHotEncoder(handle_unknown="ignore")`.

---

## 10. Recomendaciones

1. Mantener particiones por `factura_id` en toda evaluación futura.
2. Implementar validadores de entrada antes de inferencia para evitar tasas imposibles, montos extremos no auditados o días negativos.
3. Monitorear drift mensual con PSI/KS y desempeño por sector.
4. Revisar manualmente facturas con alto riesgo y baja confianza del modelo.
5. Reentrenar el modelo si F1-macro cae más de 10% en una ventana reciente o si PSI supera umbrales operativos.
6. Mantener `OneHotEncoder(handle_unknown="ignore")` para variables nominales y evitar pseudo-ordinalidad.

---

## 11. Artefactos Exportados

Todos los artefactos quedan organizados en `critical_scenarios_outputs`:

- `tables/baseline_metrics.csv`
- `tables/classification_report_baseline.csv`
- `tables/classification_report_worst_scenario.csv`
- `tables/noise_missing_outlier_results.csv`
- `tables/drift_results.csv`
- `tables/drift_sector_results.csv`
- `tables/temporal_drift_split_summary.csv`
- `tables/stress_test_results.csv`
- `tables/all_scenarios_summary.csv`
- `figures/confusion_matrix_baseline.png`
- `figures/confusion_matrix_worst_scenario.png`
- `report/informe_workshop_simulacion_escenarios_criticos.md`
- `artifacts/model_baseline.joblib`
- `artifacts/preprocessing_pipeline.joblib`
- `artifacts/scenario_config.json`

---

## 12. Referencias APA

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys, 46*(4), 1–37. https://doi.org/10.1145/2523813

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering, 21*(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239

Moreno-Torres, J. G., Raeder, T., Alaiz-Rodríguez, R., Chawla, N. V., & Herrera, F. (2012). A unifying view on dataset shift in classification. *Pattern Recognition, 45*(1), 521–530. https://doi.org/10.1016/j.patcog.2011.06.019

Schafer, J. L., & Graham, J. W. (2002). Missing data: Our view of the state of the art. *Psychological Methods, 7*(2), 147–177. https://doi.org/10.1037/1082-989X.7.2.147

Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J. F., & Dennison, D. (2015). Hidden technical debt in machine learning systems. *Advances in Neural Information Processing Systems, 28*, 2503–2511.

---

*Informe generado automáticamente por el pipeline de evaluación de robustez. No contiene textos pendientes de completar.*

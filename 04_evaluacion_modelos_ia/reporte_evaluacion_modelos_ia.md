# Reporte de evaluacion de modelos IA

## 1. Proposito de la fase

Esta fase evalua el valor real del sistema de priorizacion de cobranzas despues de la preparacion de datos. Tiene dos componentes:

1. Prediccion supervisada de `target_mora` por factura y corte temporal.
2. Clustering no supervisado de clientes para segmentar perfiles de comportamiento.

La fase parte de los artefactos oficiales de `03_preparacion/outputs/` y mantiene la regla central del proyecto: el split de entrenamiento y prueba se reconstruye con `train_facturas_ids.csv` y `test_facturas_ids.csv`. No se regenera un split por fila.

## 2. Estructura de la fase

La fase quedo organizada en:

- `04_evaluacion_modelos_ia/notebook_evaluacion_modelos.ipynb`
- `04_evaluacion_modelos_ia/evaluacion_modelos_cobranzas.py`
- `04_evaluacion_modelos_ia/clustering_clientes_cobranzas.py`
- `04_evaluacion_modelos_ia/outputs/`
- `04_evaluacion_modelos_ia/reporte_evaluacion_modelos_ia.md`

El notebook principal es didactico y orquesta la ejecucion de los dos scripts. Los scripts producen artefactos tecnicos; no generan ni validan reportes narrativos `.md`.

## 3. Entradas utilizadas

La evaluacion supervisada usa:

| Archivo | Uso |
|---|---|
| `03_preparacion/outputs/features_ml_prepared.csv` | Dataset preparado con trazabilidad, target y variables listas para modelado. |
| `03_preparacion/outputs/train_facturas_ids.csv` | Facturas oficiales de entrenamiento. |
| `03_preparacion/outputs/test_facturas_ids.csv` | Facturas oficiales de prueba. |
| `03_preparacion/outputs/features_selected.csv` | Lista canonica de 39 predictores permitidos. |

El clustering usa `features_ml_prepared.csv` y agrega variables por `cliente_id`. El target se excluye del clustering porque ese componente es no supervisado.

## 4. Componente 1: benchmarking supervisado

El script `evaluacion_modelos_cobranzas.py` compara:

| Referencia | Rol en la comparacion |
|---|---|
| `Dummy Baseline` | Baseline minimo; predice la clase mas frecuente. |
| `Logistic Regression` | Modelo lineal interpretable y estable. |
| `Random Forest` | Modelo no lineal basado en arboles. |
| `XGBoost` | Referencia fuerte de boosting, cercana al estado del arte tabular cuando la libreria esta disponible. |

No se integro una API comercial externa porque Google, Amazon o Microsoft no ofrecen un modelo directo de mora B2B entrenado con este esquema academico y consumir una API generica no seria una comparacion metodologicamente equivalente. Para cubrir ese criterio, la comparacion comercial debe documentarse como referencia cualitativa: un proveedor comercial aportaria infraestructura, monitoreo y despliegue, pero requeriria los mismos datos historicos o un modelo entrenado para este dominio.

## 5. Resultados cuantitativos

Resultados principales en test:

| Modelo | Accuracy | Balanced accuracy | F1-macro | AUC weighted | Diagnostico |
|---|---:|---:|---:|---:|---|
| Dummy Baseline | 0.3305 | 0.2500 | 0.1242 | 0.5000 | Baseline sin valor predictivo real. |
| Logistic Regression | 0.5673 | 0.6006 | 0.5700 | 0.8131 | Mejor equilibrio entre desempeno y estabilidad. |
| Random Forest | 0.5711 | 0.5847 | 0.5727 | 0.8119 | F1-test alto, pero con sobreajuste fuerte. |
| XGBoost | 0.5257 | 0.5437 | 0.5346 | 0.7732 | Sobreajuste fuerte en esta configuracion. |

El modelo seleccionado fue `Logistic Regression`. Aunque `Random Forest` logra un F1-macro test ligeramente mayor, su F1 train es 1.0000 y su gap de F1 es 0.4273, lo que indica memorizacion del entrenamiento. La seleccion prioriza un modelo defendible y generalizable, no solo el mayor score puntual.

## 6. Diagnostico de overfitting y underfitting

| Modelo | F1 train | F1 test | Gap F1 | Lectura |
|---|---:|---:|---:|---|
| Dummy Baseline | 0.1273 | 0.1242 | 0.0031 | No sobreajusta, pero subajusta por falta de capacidad. |
| Logistic Regression | 0.5810 | 0.5700 | 0.0110 | Gap aceptable; generaliza de forma razonable. |
| Random Forest | 1.0000 | 0.5727 | 0.4273 | Sobreajuste claro. |
| XGBoost | 0.9987 | 0.5346 | 0.4642 | Sobreajuste claro. |

La curva de aprendizaje del mejor modelo queda en `learning_curve_best_model.csv` y `learning_curve_best_model.png`. En el punto final, el F1 train fue 0.5817 y el F1 de validacion interna fue 0.5632, con gap 0.0186. Esto respalda que la regresion logistica no esta sobreajustando de forma preocupante.

## 7. Fairness y sesgo algoritmico

Variables sensibles o proxies revisados:

| Variable | Justificacion |
|---|---|
| `sector_dominante` | Puede reflejar trato diferencial por industria. |
| `tiene_garantia` | Puede afectar condiciones de cobranza y representar diferencias estructurales de acceso a garantias. |
| `tiene_disputa_activa` | Puede afectar la severidad predicha y tiene implicacion operativa. |

El analisis exporta metricas por grupo en `fairness_by_group.csv` y gaps en `fairness_gap_summary.csv`. Los gaps mas relevantes fueron:

| Variable | Metrica | Gap |
|---|---|---:|
| `sector_dominante` | `worst_class_recall` | 0.2009 |
| `sector_dominante` | `predicted_high_risk_rate` | 0.2304 |
| `tiene_garantia` | `predicted_high_risk_rate` | 0.2248 |
| `tiene_disputa_activa` | `predicted_high_risk_rate` | 0.3830 |

Interpretacion: el modelo no debe usarse como decision automatica de castigo o bloqueo. Debe servir como priorizador operativo sujeto a revision humana, especialmente en grupos con gaps altos de tasa predicha de alto riesgo.

## 8. Componente 2: clustering de clientes

El clustering toma las filas por corte y construye una tabla agregada por `cliente_id`. Usa variables historicas y operativas, excluyendo `target_mora`, identificadores y categoricas nominales que podrian inducir ordinalidad artificial.

Resultados:

| Metodo | Clusters | Silhouette | Lectura |
|---|---:|---:|---|
| KMeans k=5 | 5 | 0.2638 | Segmentacion principal alineada al proyecto, interpretable pero moderada. |
| KMeans k=2 | 2 | 0.8180 | Mejor separacion matematica, probablemente divide clientes en dos macrogrupos. |
| DBSCAN | 0 | N/A | Con `eps=0.5`, todos los clientes quedan como ruido. |

Distribucion KMeans k=5:

| Cluster | Clientes | Porcentaje |
|---|---:|---:|
| 0 | 18 | 9.0% |
| 1 | 85 | 42.5% |
| 2 | 3 | 1.5% |
| 3 | 83 | 41.5% |
| 4 | 11 | 5.5% |

La conclusion critica es que `k=5` puede servir si se necesita una segmentacion mas granular para gestion, pero `k=2` tiene una separacion mucho mas fuerte. Para una defensa academica, conviene explicar que `k=5` responde a interpretabilidad/operacion, mientras `k=2` responde al criterio de silhouette.

## 9. Artefactos generados

| Output | Uso |
|---|---|
| `benchmark_metrics.csv` | Comparacion de modelos en train y test. |
| `train_test_gap.csv` | Diagnostico de overfitting/underfitting. |
| `classification_report_best_model.txt` | Reporte por clase del modelo seleccionado. |
| `confusion_matrix_best_model.csv` | Matriz de confusion del mejor modelo. |
| `class_metrics_best_model.csv` | Precision, recall y F1 por clase. |
| `fairness_by_group.csv` | Metricas por grupo. |
| `fairness_gap_summary.csv` | Brechas max-min por variable y metrica. |
| `learning_curve_best_model.csv` | Curva de aprendizaje tabular del mejor modelo. |
| `learning_curve_best_model.png` | Grafico de curva de aprendizaje. |
| `best_model_artifact.joblib` | Artefacto usable con preprocesador, modelo y metadata basica. |
| `model_feature_schema.csv` | Lista de features requeridas para prediccion. |
| `model_metadata.json` | Metadata del modelo seleccionado. |
| `clustering_metrics.csv` | Metricas de KMeans, DBSCAN y busqueda de k. |
| `client_cluster_assignments.csv` | Cluster asignado por cliente. |
| `cluster_profiles.csv` | Perfil promedio por cluster. |
| `summary.txt` | Resumen tecnico de la corrida supervisada. |

## 10. Implicaciones para la siguiente fase

La siguiente etapa de escenarios criticos e hiperparametros debe partir de estos hallazgos:

1. El modelo base defendible es `Logistic Regression`, no porque sea el score maximo, sino porque generaliza mejor.
2. Random Forest y XGBoost son candidatos para tuning, pero necesitan regularizacion, reduccion de profundidad, validacion mas estricta o busqueda controlada.
3. Los hiperparametros criticos a explorar son al menos: regularizacion de regresion logistica (`C`), profundidad/max leaf de arboles, numero de estimadores, learning rate en boosting y pesos de clase.
4. La metrica principal debe seguir siendo F1-macro, complementada con recall por clase y fairness gaps.
5. El artefacto `best_model_artifact.joblib` permite empezar a pensar en prediccion usable, siempre que los nuevos datos pasen por la misma construccion de features de preparacion.
6. Para clustering, se debe decidir si la defensa prioriza interpretabilidad operativa (`k=5`) o separacion matematica (`k=2`).

## 11. Respuesta a la duda sobre los dos componentes

Si, las etapas anteriores tambien dejaron base para clustering. La generacion produjo clientes, facturas, gestiones y promesas; el EDA reviso comportamiento y variables; la preparacion creo un dataset por cortes con variables historicas y operativas. Para clustering no se usa el target como etiqueta, pero si se reutilizan las variables preparadas y se agregan a nivel cliente.

La diferencia metodologica es esta:

- Prediccion: unidad tecnica `factura_id` + `fecha_corte`; target `target_mora`; split por factura.
- Clustering: unidad final `cliente_id`; no usa target; agrega comportamiento historico/operativo para perfilar clientes.

## 12. Conclusion

La fase queda ordenada y reproducible. Los outputs tecnicos viven dentro de `04_evaluacion_modelos_ia/outputs/`, el notebook principal es didactico sin cargar logica duplicada, y los scripts consumen las rutas canonicas de preparacion. El resultado principal es un modelo supervisado estable para priorizacion y una segmentacion de clientes que necesita decision metodologica antes de usarse como componente operativo.

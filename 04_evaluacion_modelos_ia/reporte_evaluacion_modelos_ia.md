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
| `03_preparacion/outputs/client_features_clustering_base.csv` | Base agregada por `cliente_id` para clustering. |
| `03_preparacion/outputs/client_clustering_features_selected.csv` | Lista canonica de 49 variables numericas para clustering. |

El clustering ya no reconstruye la agregacion por cliente dentro de la fase 4. Consume la base oficial exportada desde preparacion. El target se excluye del clustering porque ese componente es no supervisado.

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

El clustering toma `client_features_clustering_base.csv`, una tabla agregada por `cliente_id` generada en preparacion. Cada fila es un cliente y las 49 variables numericas resumen su comportamiento: montos, historial de mora, mora observable, friccion de contacto, promesas, gestiones, disputas, facturas previas, cortes observados y tasas de cumplimiento/contacto.

El flujo tecnico es:

1. Lee la base oficial por cliente y la lista canonica de variables de clustering.
2. Convierte las variables numericas a formato modelable.
3. Aplica `log1p` a variables sesgadas de montos, conteos y dias para reducir el peso de valores extremos.
4. Aplica `RobustScaler`, que centra por mediana y escala por rango intercuartil. Esto evita que una variable grande, como monto o numero de gestiones, domine solo por su escala.
5. Ejecuta KMeans. KMeans crea centroides, es decir, perfiles promedio en el espacio transformado.
6. Asigna cada cliente al cluster cuyo centroide esta mas cerca por distancia euclidiana.

En terminos practicos: el cliente no entra a un cluster porque tenga una etiqueta previa, sino porque su patron numerico de comportamiento se parece mas al centroide de ese grupo que al de los demas.

Resultados:

| Metodo | Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz | Lectura |
|---|---:|---:|---:|---:|---|
| KMeans k=2 | 2 | 0.8171 | 0.2592 | 138.15 | Mejor separacion matematica, pero deja 197 clientes en un grupo y 3 en otro; funciona mas como deteccion de atipicos que como segmentacion operativa. |
| KMeans k=3 | 3 | 0.4999 | 0.7005 | 152.09 | Opcion principal: separa clientes generales, atipicos de bajo volumen y alto riesgo operativo. |
| KMeans k=4 | 4 | 0.2819 | 0.9835 | 156.32 | Aumenta granularidad, pero baja bastante la separacion. |
| KMeans k=5 | 5 | 0.2760 | 1.0849 | 133.56 | Segmentacion mas detallada, pero menos clara; produce grupos pequenos y mas dificil de explicar. |
| DBSCAN | 0 | N/A | N/A | N/A | Con `eps=0.5`, todos los clientes quedan como ruido. |

Se agrego una prueba de sensibilidad de DBSCAN con distintos valores de `eps`:

| eps | Clusters encontrados | Clientes como ruido |
|---:|---:|---:|
| 0.50 | 0 | 100.0% |
| 0.75 | 0 | 100.0% |
| 1.00 | 0 | 100.0% |
| 1.25 | 0 | 100.0% |
| 1.50 | 0 | 100.0% |
| 2.00 | 0 | 100.0% |
| 2.50 | 1 | 85.5% |
| 3.00 | 2 | 52.5% |

Interpretacion: DBSCAN no fue adecuado para esta base. Con valores bajos de `eps` fue demasiado estricto y marco a todos los clientes como ruido. Al aumentar `eps`, empezo a formar grupos, pero dejo fuera a demasiados clientes: incluso con `eps=3.0`, mas de la mitad de la base quedo como ruido. Por eso DBSCAN se conserva solo como experimento comparativo descartado, no como modelo principal de segmentacion.

La decision final fue usar `KMeans k=3`. Aunque `k=2` tiene mejor silhouette, su salida no es suficientemente util para el sistema porque casi todos los clientes quedan en un solo grupo. `k=3` conserva una separacion aceptable y permite contar una historia operativa clara para el front: tipo general, tipo atipico y tipo de alto riesgo.

Distribucion KMeans k=3:

| Cluster | Clientes | Porcentaje |
|---|---:|---:|
| 0 | 175 | 87.5% |
| 1 | 3 | 1.5% |
| 2 | 22 | 11.0% |

Lectura operativa de los clusters:

| Cluster | Lectura resumida |
|---|---|
| 0 | Clientes recurrentes de comportamiento general. Es el grupo base: contiene la mayoria de clientes y mezcla distintos niveles de riesgo individual. |
| 1 | Clientes atipicos de bajo volumen. Solo tiene 3 clientes; sirve como alerta de baja evidencia y no como segmento comercial fuerte. |
| 2 | Clientes de alto riesgo operativo. Presenta mas mora reciente, moras consecutivas, friccion y menor cumplimiento. |

La conclusion critica es que el cluster responde "que tipo de cliente es", pero no debe confundirse con un rating. Un cliente dentro del cluster general puede ser 1 estrella si su riesgo individual es alto, y otro del mismo cluster puede ser 5 estrellas si su riesgo individual es bajo.

Para entender por que un cliente cae en un cluster, se exportaron tres niveles de lectura:

| Archivo | Como se interpreta |
|---|---|
| `client_cluster_assignments_detailed.csv` | Muestra la distancia del cliente a cada cluster. El cluster asignado es el de menor distancia. El margen contra el segundo mas cercano ayuda a detectar casos claros vs casos frontera. |
| `client_cluster_reasoning.csv` | Muestra, por cliente, las variables que mas caracterizan su cluster y compara valor del cliente, promedio del cluster y promedio global. |
| `client_cluster_examples.csv` | Lista clientes representativos de cada cluster y clientes frontera que estan cerca de cambiar de grupo. |

Ejemplo de lectura: si `CLI0001` aparece en el cluster 0, no significa que el modelo "sepa" su categoria real. Significa que, despues de transformar y escalar sus variables, queda mas cerca del centroide del cluster 0. En `client_cluster_reasoning.csv` se observa que para ese grupo pesan variables como dias de mora observable, moras consecutivas y ratio de monto; el archivo muestra el valor de `CLI0001` frente al promedio del cluster y al promedio global.

### 8.1 Conversion de clusters a rating de 1 a 5 estrellas

Los clusters no son un ranking por si mismos. El numero `cluster=2` no significa automaticamente 2 estrellas. Para convertir la segmentacion en un rating de clientes, se agrego una capa de riesgo individual encima del clustering:

1. Se calcula un `client_risk_score_0_100` por cliente.
2. El score sube con variables de riesgo: mora historica, mora reciente, dias de mora observable, moras consecutivas, promesas rotas, friccion de contacto, no contestacion, disputas y vencimiento.
3. El score baja cuando el cliente tiene mejor cumplimiento, mayor contactabilidad, mayor cumplimiento de promesas y mas margen preventivo antes del vencimiento.
4. Las estrellas se asignan por umbrales del riesgo individual:

| Riesgo individual | Rating |
|---:|---|
| 0 a 20 | 5 estrellas |
| >20 a 40 | 4 estrellas |
| >40 a 60 | 3 estrellas |
| >60 a 80 | 2 estrellas |
| >80 a 100 | 1 estrella |

Resultado agregado por cluster:

| Cluster | Tipo de cliente | Clientes | Riesgo promedio | Rating promedio del cluster |
|---|---|---:|---:|---|
| 2 | Clientes de alto riesgo operativo | 22 | 70.56 | 2 estrellas |
| 0 | Clientes recurrentes de comportamiento general | 175 | 47.78 | 3 estrellas |
| 1 | Clientes atipicos de bajo volumen | 3 | 35.50 | 4 estrellas |

Interpretacion: `rating_estrellas` es una escala ordinal por cliente; `cluster` sigue siendo la etiqueta tecnica del grupo. El front debe mostrar ambas cosas separadas.

### 8.2 Esquema para el front

El archivo principal para consumo visual es `frontend_customer_segments.csv`. La idea de pantalla por cliente es:

```text
Cliente: CLI0001
Tipo: Clientes recurrentes de comportamiento general
Riesgo: 61.99/100
Rating: 2 estrellas
Por que rating: promesas rotas | bajo cumplimiento de promesas | dias de mora observables | bajo cumplimiento historico
Por que cluster: cercania al centroide del grupo; variables distintivas de ese grupo
```

Campos recomendados para el front:

| Campo | Uso visual |
|---|---|
| `cliente_id` | Identificador del cliente. |
| `tipo_cliente` | Nombre entendible del cluster. |
| `cluster` | Etiqueta tecnica para auditoria. |
| `riesgo_0_100` | Score individual de riesgo. |
| `rating_estrellas` / `rating_label` | Traduccion simple del riesgo. |
| `por_que_rating` | Principales variables que empujan el riesgo del cliente. |
| `por_que_cluster` | Variables que explican por que pertenece al grupo. |
| `sector_dominante_modal`, `n_facturas_total`, `n_cortes_total` | Contexto para tarjetas, detalle o filtros. |

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
| `client_cluster_assignments_detailed.csv` | Cluster asignado, distancias a todos los centroides y margen contra el segundo cluster mas cercano. |
| `client_cluster_reasoning.csv` | Explicacion por cliente usando las variables que mas caracterizan su cluster. |
| `client_cluster_examples.csv` | Clientes representativos y clientes frontera por cluster. |
| `cluster_star_ratings.csv` | Resumen por cluster: tipo de cliente, riesgo promedio, rating promedio y distribucion de estrellas individuales. |
| `client_star_ratings.csv` | Rating final por cliente, con score individual, cluster, estrellas y variables de contexto. |
| `star_rating_feature_weights.csv` | Variables y pesos usados para construir el score de riesgo que ordena los clusters. |
| `client_star_rating_reasoning.csv` | Contribucion de cada variable al score de riesgo de cada cliente. |
| `frontend_customer_segments.csv` | Tabla compacta para el front: cliente, tipo, riesgo, rating y explicaciones. |
| `cluster_profiles.csv` | Perfil promedio por cluster. |
| `cluster_feature_drivers.csv` | Variables que mas diferencian cada cluster contra el promedio global. |
| `cluster_top_features.csv` | Version compacta con las variables principales de cada cluster. |
| `cluster_readable_summary.csv` | Resumen legible por cluster: tamano, sector modal y variables altas/bajas. |
| `cluster_pca_coordinates.csv` | Coordenadas 2D PCA para graficar clientes por cluster. |
| `dbscan_eps_search.csv` | Prueba de sensibilidad de DBSCAN con distintos valores de `eps`. |
| `clustering_model_features.csv` | Lista final de features usadas por el clustering despues de excluir varianza cero si aplica. |
| `clustering_k_search.png` | Grafico de silhouette e inertia para evaluar numero de clusters. |
| `cluster_pca_scatter.png` | Visual 2D de clientes por cluster usando PCA. |
| `cluster_sizes.png` | Distribucion visual de clientes por cluster. |
| `cluster_feature_drivers_heatmap.png` | Mapa de calor de variables que diferencian clusters. |
| `cluster_key_profile_heatmap.png` | Mapa de calor con variables clave de negocio por cluster. |
| `summary.txt` | Resumen tecnico de la corrida supervisada. |

## 10. Implicaciones para la siguiente fase

La siguiente etapa de escenarios criticos e hiperparametros debe partir de estos hallazgos:

1. El modelo base defendible es `Logistic Regression`, no porque sea el score maximo, sino porque generaliza mejor.
2. Random Forest y XGBoost son candidatos para tuning, pero necesitan regularizacion, reduccion de profundidad, validacion mas estricta o busqueda controlada.
3. Los hiperparametros criticos a explorar son al menos: regularizacion de regresion logistica (`C`), profundidad/max leaf de arboles, numero de estimadores, learning rate en boosting y pesos de clase.
4. La metrica principal debe seguir siendo F1-macro, complementada con recall por clase y fairness gaps.
5. El artefacto `best_model_artifact.joblib` permite empezar a pensar en prediccion usable, siempre que los nuevos datos pasen por la misma construccion de features de preparacion.
6. Para clustering, la defensa debe explicar que `k=3` se eligio por equilibrio entre metrica e interpretabilidad: `k=2` separa mejor matematicamente, pero casi todo queda en un solo grupo.

## 11. Respuesta a la duda sobre los dos componentes

Si, las etapas anteriores tambien dejaron base para clustering. La generacion produjo clientes, facturas, gestiones y promesas; el EDA reviso comportamiento y variables; y ahora la preparacion deja explicitamente `client_features_clustering_base.csv`, una base agregada por cliente. Para clustering no se usa el target como etiqueta, pero si se reutilizan variables preparadas de comportamiento historico y operativo.

La diferencia metodologica es esta:

- Prediccion: unidad tecnica `factura_id` + `fecha_corte`; target `target_mora`; split por factura.
- Clustering: unidad final `cliente_id`; no usa target; consume la base agregada por cliente generada en preparacion.

## 12. Conclusion

La fase queda ordenada y reproducible. Los outputs tecnicos viven dentro de `04_evaluacion_modelos_ia/outputs/`, el notebook principal es didactico sin cargar logica duplicada, y los scripts consumen las rutas canonicas de preparacion. El resultado principal es un modelo supervisado estable para priorizacion y una segmentacion de clientes construida sobre un dataset base preparado formalmente en la fase 3.

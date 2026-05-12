# Reporte de escenarios criticos e hiperparametros

## 1. Proposito de la fase

Esta fase evalua la robustez del modelo supervisado de cobranzas seleccionado en `04_evaluacion_modelos_ia`, revisa la sensibilidad de sus hiperparametros y agrega una revision de hiperparametros del componente de clustering. Para el componente supervisado se usa el dataset preparado y los splits oficiales por `factura_id`; no se reconstruyen particiones por fila.

Entradas principales:

| Entrada | Uso |
|---|---|
| `03_preparacion/outputs/features_ml_prepared.csv` | Dataset modelable por factura y corte temporal. |
| `03_preparacion/outputs/train_facturas_ids.csv` | Facturas oficiales de entrenamiento. |
| `03_preparacion/outputs/test_facturas_ids.csv` | Facturas oficiales de prueba externa. |
| `04_evaluacion_modelos_ia/outputs/best_model_artifact.joblib` | Modelo base defendible de fase 4: Logistic Regression. |
| `03_preparacion/outputs/client_features_clustering_base.csv` | Base por cliente para sensibilidad del componente de clustering. |
| `03_preparacion/outputs/client_clustering_features_selected.csv` | Variables oficiales del clustering. |

## 2. Estructura de la fase

La fase queda organizada como:

- `05_escenarios_criticos_hiperparametros/notebook_fase5_escenarios_criticos_hiperparametros.ipynb`
- `05_escenarios_criticos_hiperparametros/outputs/`
- `05_escenarios_criticos_hiperparametros/reporte_escenarios_criticos_hiperparametros.md`

El notebook principal concentra la ejecucion tecnica de la fase. El reporte queda separado como documento de continuidad e interpretacion.

## 3. Baseline usado

El baseline se recalculo con el artefacto oficial de fase 4 sobre el test oficial:

| Metrica | Valor |
|---|---:|
| Accuracy | 0.5673 |
| Balanced accuracy | 0.6006 |
| F1-macro | 0.5700 |
| F1-weighted | 0.5606 |
| Recall `+90` | 0.5796 |
| F1 `+90` | 0.6281 |
| AUC weighted | 0.8131 |

Este baseline corresponde al modelo seleccionado en fase 4 y se usa como referencia unica para medir degradacion.

Que significa cada metrica:

- `Baseline`: punto de comparacion. Es el resultado del modelo original antes de simular problemas o ajustar hiperparametros. Sirve para responder si una prueba mejora, empeora o mantiene el comportamiento del modelo.
- `Accuracy`: porcentaje total de predicciones correctas. En este proyecto no basta por si sola, porque las clases de mora pueden no tener el mismo peso operativo.
- `Balanced accuracy`: promedio del recall de cada clase. Es mas justa que accuracy cuando hay clases con diferente cantidad de casos.
- `F1-macro`: promedio simple del F1 de todas las clases. Cada clase pesa igual, por eso es la metrica principal para no esconder mal desempeno en clases minoritarias.
- `F1-weighted`: promedio del F1 ponderado por la cantidad de registros de cada clase. Refleja mejor el rendimiento global, pero puede favorecer clases grandes.
- `Recall +90`: de todas las facturas que realmente estaban en mora `+90`, que porcentaje detecto el modelo. Es clave porque `+90` representa riesgo alto.
- `F1 +90`: balance entre precision y recall especificamente para la clase `+90`.
- `AUC weighted`: mide que tan bien el modelo ordena probabilidades entre clases, ponderando por soporte de cada clase. Valores mas altos indican mejor separacion probabilistica.
- `Test oficial`: subconjunto reservado para evaluar el resultado final. No debe usarse para escoger hiperparametros, porque eso contaminaria la evaluacion.

## 4. Escenarios criticos

Se simularon escenarios con multiples semillas para no depender de una sola perturbacion aleatoria. Los niveles tienen interpretacion operativa:

| Tipo | Niveles | Justificacion |
|---|---|---|
| Ruido gaussiano | 5%, 10%, 20% de la desviacion estandar | Error de digitacion, medicion o sincronizacion. 5% representa ruido leve; 20% representa deterioro fuerte de calidad. |
| Missing aleatorio | 5%, 15%, 30% | Perdida parcial de campos por fallas de integracion. 30% representa un incidente severo. |
| Missing dirigido | Gestion, promesas, historial | Fallas sistematicas por modulo: CRM, promesas de pago o historico del cliente. |
| Outliers | Monto 3x, 5x, 10x; 5% y 10% de filas | Errores de carga, duplicacion de montos o facturas atipicas. |
| Escenario combinado | Outliers + missing gestion + ruido alto | Caso adverso compuesto para probar fragilidad operacional. |

Resultados principales:

| Escenario | F1-macro medio | Degradacion F1-macro |
|---|---:|---:|
| `outlier_critico_combinado` | 0.4180 | 26.67% |
| `missing_aleatorio_30pct` | 0.4430 | 22.28% |
| `ruido_gaussiano_20pct` | 0.4853 | 14.86% |
| `missing_dirigido_gestion` | 0.4902 | 14.00% |
| `missing_aleatorio_15pct` | 0.5075 | 10.96% |

La lectura critica es que el modelo tolera ruido pequeno, pero pierde capacidad cuando la perturbacion afecta senales operativas recientes o combina errores de varias fuentes. El escenario combinado es el mas riesgoso porque mezcla valores extremos y perdida de gestion; en produccion esto justificaria validadores de rango, monitoreo de completitud y revision humana para casos de alto riesgo.

Que significa cada termino:

- `Escenario critico`: simulacion de una falla o condicion adversa para ver si el modelo sigue siendo confiable.
- `Semilla`: numero que controla la aleatoriedad del experimento. Usar varias semillas evita concluir por una sola corrida afortunada o desafortunada.
- `Ruido gaussiano`: alteracion numerica aleatoria con forma de campana normal. Representa pequenos errores de digitacion, medicion o sincronizacion.
- `Desviacion estandar`: medida de dispersion de una variable. Usar 5%, 10% o 20% de esa dispersion permite agregar ruido proporcional al tamano real de cada variable.
- `Missing aleatorio`: valores faltantes distribuidos al azar en varias columnas.
- `Missing dirigido`: valores faltantes concentrados en un grupo de variables de negocio, por ejemplo gestion, promesas o historial. Es mas grave porque simula la caida de un modulo completo.
- `Outlier`: valor extremo que se aleja mucho del comportamiento normal, por ejemplo un monto multiplicado por 10.
- `Escenario combinado`: mezcla de varios problemas al mismo tiempo. Es util porque en operacion real las fallas rara vez aparecen aisladas.
- `Degradacion F1-macro`: porcentaje de perdida de F1 frente al baseline. Si el baseline era 0.5700 y un escenario baja a 0.4180, se mide cuanto se deterioro el modelo.

## 5. Drift de datos

Se agregaron metricas formales de drift:

- PSI para cuantificar cambio de distribucion por variable.
- KS statistic y p-value para contrastar train vs test.
- Evaluacion temporal separando facturas pasadas y futuras sin interseccion de `factura_id`.
- Evaluacion por sector y por cartera critica.

Variables con mayor PSI:

| Variable | PSI | KS statistic | Lectura |
|---|---:|---:|---|
| `ratio_monto` | 0.0320 | 0.0448 | Drift bajo, pero detectable. |
| `antiguedad_meses` | 0.0309 | 0.0285 | Cambio leve de composicion. |
| `tasa_cumpl_promesas` | 0.0231 | 0.0654 | Cambio leve en comportamiento de promesas. |
| `ratio_promesas_rotas` | 0.0231 | 0.0614 | Cambio leve en friccion de promesas. |

No hay PSI alto bajo umbrales comunes de referencia (`0.10` moderado, `0.25` alto), pero varios KS tienen p-values pequenos por el tamano de muestra. Esto significa que hay cambios estadisticamente detectables, aunque de magnitud operativa baja.

Escenarios de drift con mayor degradacion:

| Escenario | Tipo | F1-macro | Degradacion |
|---|---|---:|---:|
| `drift_cartera_critica_q75` | Covariate riesgo | 0.3785 | 33.60% |
| `drift_sector_retail` | Sector | 0.5353 | 6.10% |
| `drift_sector_tecnologia` | Sector | 0.5429 | 4.75% |
| `drift_temporal_futuro` | Temporal | 0.5647 | 0.94% |

La conclusion de produccion es que el mayor riesgo no viene del tiempo por si solo, sino de concentrar la inferencia en una cartera ya critica. Si el mix operativo cambia hacia facturas con historial de mora alto, el modelo debe monitorearse por segmento y no solo con metricas globales.

Que significa cada termino:

- `Drift`: cambio en la distribucion de los datos entre el entrenamiento y la evaluacion. Si los datos nuevos ya no se parecen a los datos usados para entrenar, el modelo puede perder confiabilidad.
- `PSI`: Population Stability Index. Resume cuanto cambio una variable entre dos poblaciones. Como regla practica, menos de 0.10 suele leerse como cambio bajo, 0.10 a 0.25 como moderado y mas de 0.25 como alto.
- `KS statistic`: distancia maxima entre dos distribuciones acumuladas. Mientras mas alto, mas diferentes son las distribuciones comparadas.
- `p-value`: medida estadistica para evaluar si una diferencia podria aparecer por azar. Con bases grandes puede salir pequeno incluso cuando el cambio operativo es pequeno.
- `Drift temporal`: comparacion entre registros mas antiguos y mas recientes. Sirve para ver si el paso del tiempo cambia el comportamiento de facturas o clientes.
- `Drift sectorial`: evaluacion del modelo en sectores especificos, como retail o tecnologia. Sirve para detectar si un sector se comporta diferente al promedio.
- `Covariate riesgo`: subconjunto definido por variables de entrada de alto riesgo, por ejemplo cartera con mora historica alta. No significa que el target cambio, sino que el perfil de entrada es mas dificil.
- `Cartera critica Q75`: grupo de facturas por encima del percentil 75 en una variable de riesgo. En palabras simples, es el 25% con mayor nivel de riesgo segun esa senal.
- `Metricas globales`: resultados calculados sobre todo el test. Pueden ocultar fallas en segmentos especificos.

## 6. Pruebas de estres

Las pruebas de estres cubren estabilidad de entrada, esquema, volumen y concurrencia:

| Prueba | Resultado |
|---|---|
| Inputs extremos | El pipeline predice, pero F1-macro cae a 0.1242; se requieren validadores de dominio. |
| Columna extra | No rompe inferencia; `ColumnTransformer` ignora columnas no esperadas. |
| Columna requerida faltante | Falla controlada con `KeyError`; se requiere validacion estricta de schema. |
| Volumen 1x, 10x, 50x, 100x | Prediccion completa sin degradacion de metrica porque se replica el mismo test; la medicion relevante es latencia. |
| Concurrencia | Se exporta `latency_concurrency_results.csv` con workers, p95 y throughput. |

Esta seccion separa dos conceptos: la metrica predictiva bajo datos alterados y el comportamiento del sistema bajo volumen o schema. Para produccion, la accion prioritaria es implementar contratos de entrada antes de inferencia.

Que significa cada termino:

- `Prueba de estres`: prueba que lleva el sistema a condiciones limite para ver si falla, se degrada o responde de forma controlada.
- `Pipeline`: cadena completa de procesamiento usada para predecir: seleccion de columnas, transformaciones, preprocesamiento y modelo.
- `Inputs extremos`: valores fuera del rango normal de negocio, por ejemplo tasas negativas o montos exageradamente altos.
- `Schema`: contrato de columnas esperadas por el modelo. Define que campos deben existir y con que estructura.
- `ColumnTransformer`: componente de scikit-learn que aplica transformaciones a columnas especificas. Puede ignorar columnas extra si no estan declaradas como requeridas.
- `KeyError`: error tecnico que aparece cuando el codigo intenta usar una columna que no existe.
- `Latencia`: tiempo que tarda el modelo en responder.
- `p95`: percentil 95 de latencia. Significa que 95% de las corridas tardaron ese tiempo o menos; es mas informativo que un promedio cuando hay picos.
- `Throughput`: cantidad de registros procesados por segundo.
- `Concurrencia`: varias predicciones o procesos ejecutandose al mismo tiempo.
- `Contrato de entrada`: validacion previa que revisa columnas, tipos, rangos y valores permitidos antes de llamar al modelo.

## 7. Sensibilidad de hiperparametros

El tuning se hizo sobre Logistic Regression porque fue el modelo seleccionado en fase 4 por estabilidad. Se evaluaron 40 combinaciones con:

- `C`: fuerza inversa de regularizacion.
- `class_weight`: `none` vs `balanced`.
- `max_iter`: condicion de convergencia.
- `tol`: tolerancia del solver.

El split interno de validacion se hizo por `factura_id` dentro del train oficial. El test externo se uso solo para reporte final.

Resultado general:

| Resultado | Valor |
|---|---:|
| Experimentos | 40 |
| Mejor F1-macro validacion | 0.5684 |
| Mejor F1-macro validacion convergido seleccionado | 0.5683 |
| F1-macro test externo final | 0.5697 |
| Configuracion final | `C=100`, `class_weight=balanced`, `max_iter=500`, `tol=0.0001` |

La configuracion ajustada queda practicamente empatada con el baseline externo (`0.5697` vs `0.5700`). El cambio metodologico importante es que el test externo no se usa para rankear configuraciones: los 40 experimentos se ordenan por validacion interna y el test se reserva para la comparacion final. Ademas, se selecciona la mejor configuracion convergida dentro de una tolerancia de 0.005 F1 frente al mejor score de validacion, evitando exportar como final una configuracion no convergida.

Que significa cada termino:

- `Hiperparametro`: configuracion que se elige antes de entrenar el modelo. No se aprende directamente de los datos como los coeficientes del modelo.
- `Tuning`: busqueda sistematica de combinaciones de hiperparametros para encontrar una configuracion mas estable o con mejor metrica.
- `Logistic Regression`: modelo supervisado lineal que estima probabilidades de clase. En este proyecto se escogio porque tuvo buen equilibrio entre desempeno, estabilidad e interpretabilidad.
- `Regularizacion`: penalizacion que evita que el modelo dependa demasiado de patrones especificos del entrenamiento.
- `C`: fuerza inversa de regularizacion en Logistic Regression. Un `C` pequeno regulariza mas; un `C` grande regulariza menos.
- `class_weight`: peso asignado a cada clase durante el entrenamiento. `balanced` aumenta la atencion sobre clases con menos registros.
- `max_iter`: maximo de iteraciones permitidas al optimizador para encontrar la solucion.
- `tol`: tolerancia de parada. Si la mejora entre iteraciones es menor que ese valor, el optimizador puede detenerse.
- `Solver`: algoritmo numerico que ajusta los parametros del modelo. Aqui se usa `lbfgs`.
- `Convergencia`: senal de que el optimizador alcanzo una solucion estable antes de agotar `max_iter`.
- `Validacion interna`: particion tomada solo desde el train oficial para comparar hiperparametros sin tocar el test.
- `Empatado con el baseline`: la diferencia de metrica es tan pequena que no justifica afirmar una mejora real.
- `Tolerancia de 0.005 F1`: margen aceptado para preferir una configuracion mas segura o convergida aunque su F1 sea levemente menor que el maximo observado.

## 8. Sensibilidad individual

| Hiperparametro | Mejor valor promedio | Rango F1 promedio | Lectura |
|---|---:|---:|---|
| `C` | 10.0 | 0.0248 | Es el parametro mas influyente. Valores muy bajos subregularizan; entre 1 y 100 aparece una zona estable. |
| `class_weight` | `balanced` | 0.0040 | Impacto menor que `C`; puede ayudar a recall macro, pero debe vigilarse el gap. |
| `tol` | 0.0001 | 0.0027 | Efecto bajo; afinar tolerancia no cambia sustancialmente el modelo. |
| `max_iter` | 200 | 0.0001 | Efecto practicamente nulo salvo convergencia. |

Hubo advertencias de convergencia para algunas configuraciones con `max_iter=200`. Esto no invalida el experimento: precisamente muestra que `max_iter` debe tratarse como condicion tecnica minima, no como fuente principal de mejora predictiva.

Que significa cada termino:

- `Sensibilidad individual`: analisis de un hiperparametro a la vez, promediando el efecto de los demas. Sirve para ver que variable de configuracion mueve mas el resultado.
- `Mejor valor promedio`: valor del hiperparametro que, en promedio, logro mejor metrica en la busqueda.
- `Rango F1 promedio`: diferencia entre el mejor y el peor promedio de F1 para ese hiperparametro. Si el rango es grande, el modelo es sensible a ese ajuste.
- `Zona estable`: rango de valores donde la metrica cambia poco. Es preferible porque da robustez.
- `Saturacion`: punto donde seguir aumentando o afinando un hiperparametro ya no produce mejora relevante.
- `Gap train-valid`: diferencia entre desempeno en entrenamiento y validacion. Si es alto, puede indicar sobreajuste.
- `Advertencia de convergencia`: aviso de que el optimizador llego al limite de iteraciones. No siempre invalida el resultado, pero obliga a interpretarlo con cuidado.

## 9. Ranking e interacciones

Ranking de importancia:

| Hiperparametro | Score importancia | Lectura |
|---|---:|---|
| `C` | 0.3734 | Principal palanca predictiva. |
| `class_weight` | 0.0322 | Ajuste secundario, util para balance de clases. |
| `tol` | 0.0122 | Bajo impacto. |
| `max_iter` | 0.0011 | Bajo impacto predictivo; relevante para convergencia. |

Interacciones principales:

| Interaccion | Rango F1 | Lectura |
|---|---:|---|
| `C` x `class_weight` | 0.0324 | Interaccion relevante: regularizacion y balance de clases deben elegirse juntos. |
| `C` x `tol` | 0.0264 | El efecto de tolerancia depende de la regularizacion. |
| `C` x `max_iter` | 0.0250 | Con regularizacion menos fuerte, la convergencia importa mas. |

Decision metodologica: mantener `C` como hiperparametro central de busqueda, revisar `class_weight` por efecto en clases minoritarias y fijar `max_iter` suficientemente alto para evitar conclusiones contaminadas por no convergencia.

Que significa cada termino:

- `Ranking de importancia`: orden de hiperparametros segun cuanto parecen influir en la metrica.
- `Score importancia`: puntaje combinado usado para resumir impacto. En este reporte mezcla el cambio promedio de F1 y una importancia estimada por modelo auxiliar.
- `Spearman`: correlacion de rangos. Indica si al subir un hiperparametro la metrica tiende a subir o bajar de forma monotona.
- `Surrogate RF`: Random Forest usado como modelo explicativo auxiliar para estimar que hiperparametros explican mejor las diferencias de F1. No reemplaza el modelo final.
- `Interaccion`: ocurre cuando el efecto de un hiperparametro depende del valor de otro. Por ejemplo, `C` puede comportarse distinto si `class_weight` esta en `balanced`.
- `Rango de interaccion`: diferencia entre el mejor y peor resultado observado para una combinacion de dos hiperparametros.
- `Clases minoritarias`: clases con menos ejemplos. En cobranzas pueden ser importantes aunque no sean la mayoria.
- `Sobreajuste`: cuando el modelo aprende demasiado el entrenamiento y pierde capacidad de generalizar.

## 10. Sensibilidad del componente de clustering

El segundo componente de IA no es predictivo: segmenta clientes sin target. Por eso no se evalua con F1, accuracy o recall, sino con metricas internas de clustering y criterios operativos:

- Silhouette, Davies-Bouldin y Calinski-Harabasz para cohesion y separacion.
- `noise_ratio` para DBSCAN.
- `min_cluster_share` y `max_cluster_share` para detectar segmentaciones poco utiles por grupos demasiado pequenos o dominantes.

Se evaluaron 141 configuraciones:

- KMeans: `n_clusters` 2 a 6, `init`, `n_init`, escalado (`robust`/`standard`) y uso de `log1p`.
- DBSCAN: `eps`, `min_samples`, escalado robusto y `log1p`.

Ranking de sensibilidad del clustering:

| Hiperparametro | Mejor valor promedio | Rango silhouette promedio | Lectura |
|---|---:|---:|---|
| `n_clusters` | 2 | 0.3620 | Es la decision mas influyente; tambien cambia la utilidad operativa de los segmentos. |
| `scaler` | `robust` | 0.2093 | El escalado afecta fuertemente la geometria del clustering. |
| `min_samples` | 5 | 0.1718 | Importante para DBSCAN, pero ese algoritmo mantiene demasiado ruido. |
| `algorithm` | `KMeans` | 0.1217 | KMeans domina a DBSCAN para esta base. |
| `eps` | 3.0 | 0.1074 | DBSCAN es muy sensible a `eps` y deja muchos clientes como ruido. |

La sensibilidad confirma algo que ya se habia decidido en fase 4: `k=2` maximiza silhouette, pero separa casi toda la base en un grupo y funciona mas como deteccion de atipicos que como segmentacion operativa. Por eso se mantiene `KMeans k=3` como componente principal: no es el optimo matematico puro, pero es mas defendible para lectura de negocio.

Interacciones principales del clustering:

| Interaccion | Rango silhouette | Lectura |
|---|---:|---|
| `n_clusters` x `scaler` | 0.6647 | La cantidad de clusters debe evaluarse junto al escalado. |
| `n_clusters` x `use_log1p` | 0.3782 | La transformacion de variables sesgadas cambia la separacion. |
| `n_clusters` x `n_init` | 0.3638 | La estabilidad de KMeans depende mas de `k` que de aumentar reinicios. |
| `n_clusters` x `init` | 0.3634 | La inicializacion importa, pero menos que elegir bien `k` y el escalado. |

Decision metodologica: para el componente 2 no corresponde prometer "mejor prediccion". Lo correcto es defender estabilidad, interpretabilidad y utilidad operativa. DBSCAN se descarta como modelo principal porque su `noise_ratio` promedio fue 0.82 en la busqueda de sensibilidad.

Que significa cada termino:

- `Clustering`: tecnica no supervisada que agrupa clientes por similitud. No usa una etiqueta correcta previa como `target_mora`.
- `No supervisado`: tipo de modelo donde no existe una respuesta correcta por fila para entrenar. Por eso se evalua con coherencia interna y utilidad de negocio.
- `KMeans`: algoritmo que divide los clientes en `k` grupos alrededor de centros llamados centroides.
- `Centroide`: punto promedio que representa el centro de un cluster.
- `DBSCAN`: algoritmo que forma grupos por densidad. Puede dejar puntos como ruido si no estan cerca de suficientes vecinos.
- `n_clusters` o `k`: numero de grupos que se le pide a KMeans.
- `init`: forma de inicializar los centroides antes de entrenar KMeans.
- `n_init`: cantidad de inicializaciones que prueba KMeans. Mas reinicios reducen el riesgo de quedarse con una mala solucion inicial.
- `scaler`: metodo para poner variables en escalas comparables. Sin escalado, variables grandes como monto pueden dominar la distancia.
- `RobustScaler`: escalado basado en mediana y rango intercuartil. Es util cuando hay outliers.
- `StandardScaler`: escalado basado en media y desviacion estandar. Puede ser mas sensible a outliers.
- `log1p`: transformacion `log(1+x)` que reduce el peso de valores muy grandes sin romper los ceros.
- `Silhouette`: metrica de separacion y cohesion. Cerca de 1 indica clusters bien separados; cerca de 0 indica fronteras ambiguas.
- `Davies-Bouldin`: metrica donde menor es mejor. Penaliza clusters poco compactos o muy parecidos entre si.
- `Calinski-Harabasz`: metrica donde mayor suele ser mejor. Compara separacion entre clusters contra dispersion interna.
- `noise_ratio`: porcentaje de clientes que DBSCAN dejo como ruido, es decir, sin asignar a un cluster util.
- `min_cluster_share`: porcentaje del total que tiene el cluster mas pequeno. Ayuda a detectar grupos demasiado pequenos para decisiones operativas.
- `max_cluster_share`: porcentaje del total que tiene el cluster mas grande. Si es demasiado alto, la segmentacion puede estar concentrando casi todo en un solo grupo.
- `eps`: radio de vecindad en DBSCAN. Si es bajo, muchos puntos quedan como ruido; si es alto, puede juntar grupos demasiado distintos.
- `min_samples`: minimo de vecinos que necesita DBSCAN para considerar una zona como densa.
- `Optimo matematico puro`: configuracion que maximiza una metrica tecnica, aunque no necesariamente sea la mas entendible o util para negocio.

## 11. Artefactos generados

Tablas clave:

- `outputs/tables/baseline_metrics.csv`
- `outputs/tables/scenario_run_metrics.csv`
- `outputs/tables/noise_missing_outlier_results.csv`
- `outputs/tables/drift_psi_ks.csv`
- `outputs/tables/drift_results.csv`
- `outputs/tables/drift_sector_results.csv`
- `outputs/tables/stress_test_results.csv`
- `outputs/tables/latency_concurrency_results.csv`
- `outputs/tables/experimentos_sensibilidad_hiperparametros.csv`
- `outputs/tables/sensibilidad_individual_hiperparametros.csv`
- `outputs/tables/ranking_importancia_hiperparametros.csv`
- `outputs/tables/interacciones_hiperparametros.csv`
- `outputs/tables/recomendaciones_tuning_hiperparametros.csv`
- `outputs/tables/clustering_hyperparameter_experiments.csv`
- `outputs/tables/clustering_sensibilidad_hiperparametros.csv`
- `outputs/tables/clustering_ranking_importancia_hiperparametros.csv`
- `outputs/tables/clustering_interacciones_hiperparametros.csv`
- `outputs/tables/clustering_recomendaciones_hiperparametros.csv`

Figuras clave:

- `outputs/figures/noise_missing_outlier_impact.png`
- `outputs/figures/drift_impact.png`
- `outputs/figures/drift_psi_top_variables.png`
- `outputs/figures/latency_by_batch_size.png`
- `outputs/figures/top_degradation_scenarios.png`
- `outputs/figures/ranking_importancia_hiperparametros.png`
- `outputs/figures/sensibilidad_C.png`
- `outputs/figures/interaccion_C_x_class_weight.png`
- `outputs/figures/clustering_sensibilidad_kmeans_k.png`
- `outputs/figures/clustering_ranking_hiperparametros.png`

Artefactos:

- `outputs/artifacts/scenario_config.json`
- `outputs/artifacts/best_tuned_logistic_regression.joblib`

## 12. Implicaciones para la siguiente fase

1. El modelo debe desplegarse con validacion de schema y rangos antes de inferencia.
2. El tuning supervisado no modifica la seleccion de fase 4, ya que la version ajustada queda practicamente empatada con el baseline externo y no justifica cambiar el modelo de referencia.
3. Para futuras iteraciones, conviene ampliar pruebas de estres con latencia p95 en un servicio real o API local, no solo inferencia batch en memoria.
4. Para clustering, la defensa debe separar claramente metrica matematica y utilidad de negocio: `k=2` tiene mejor silhouette, pero `k=3` mantiene una segmentacion mas interpretable.
5. La decision operacional debe priorizar robustez, explicabilidad y control humano para casos extremos antes que pequenas diferencias de F1.

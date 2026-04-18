# Reporte académico de hallazgos del análisis exploratorio de datos (EDA)

## 1. Introducción

El presente reporte documenta los hallazgos del análisis exploratorio de datos realizado sobre el proyecto **Sistema inteligente de priorización de cobranzas para empresas que venden a crédito**. El objetivo del EDA no es únicamente describir los datos, sino también identificar problemas de calidad, patrones relevantes, riesgos metodológicos y decisiones de preparación que impactarán directamente en el entrenamiento de los modelos posteriores. En concordancia con la guía de implementación, este análisis se orienta a **analizar, visualizar, interpretar y justificar decisiones sobre los datos antes del modelado**.

El problema de negocio consiste en mejorar la gestión de cobranzas mediante un sistema que anticipe el riesgo de mora por factura, segmente clientes por comportamiento histórico y sugiera acciones concretas de cobranza según el nivel de riesgo observado. En este contexto, el EDA cumple una función metodológica crítica, porque permite validar si la simulación construida representa un escenario coherente, si las variables contienen señal útil y si existen condiciones que deban tratarse antes del entrenamiento.

## 2. Objetivo del reporte

Este documento tiene cuatro finalidades:

1. Registrar de forma académica los resultados del EDA.
2. Dejar trazabilidad metodológica sobre las decisiones de limpieza e interpretación de datos.
3. Servir como insumo para la etapa posterior de modelado y comparación de algoritmos.
4. Proveer contexto suficiente para comprender cómo está construido el dataset y qué cuidados deberán observarse durante el entrenamiento.

## 3. Contexto del sistema analizado

El sistema se estructura en tres componentes:

- **Componente 1. Predicción de mora por factura:** problema de clasificación multiclase que estima si una factura terminará en `on_time`, `+30`, `+60` o `+90`.
- **Componente 2. Segmentación de clientes:** problema de clustering para agrupar clientes según comportamiento histórico y traducir el resultado a una escala de 1 a 5 estrellas.
- **Componente 3. Decisor de acción de cobranza:** capa de reglas de negocio que combina el riesgo de mora y el puntaje del cliente para recomendar una acción operativa.

La unidad principal de análisis del sistema es la **factura**. Sin embargo, en el dataset de entrenamiento del componente 1 no existe una sola fila por factura, sino múltiples filas por **corte temporal de scoring**, lo que vuelve imprescindible una lectura temporal correcta del dataset.

## 4. Descripción del dataset utilizado

El análisis se realizó sobre cinco tablas finales:

- `clientes.csv`: 200 registros y 6 variables.
- `facturas.csv`: 5,338 registros y 9 variables.
- `gestiones_cobranza.csv`: 14,333 registros y 9 variables.
- `promesas_pago.csv`: 1,741 registros y 7 variables.
- `features_ml.csv`: 19,671 registros y 36 variables.

Las cuatro primeras tablas representan el modelo de datos operativo del dominio de cobranzas. La quinta tabla, `features_ml.csv`, es el dataset analítico preparado para el entrenamiento del modelo supervisado del componente 1. Su estructura responde a una lógica de **scoring dinámico por cortes temporales**, donde cada factura genera una fila inicial en la fecha de emisión y filas adicionales después de cada gestión registrada.

Esta característica es central para la interpretación metodológica posterior, porque implica que el tamaño de `features_ml.csv` no refleja únicamente el número de facturas, sino también la intensidad y frecuencia de la gestión de cobranza aplicada sobre ellas.

## 5. Calidad de datos e integridad estructural

La revisión de calidad de datos arrojó resultados favorables en los aspectos estructurales principales.

### 5.1. Duplicados e integridad referencial

No se detectaron duplicados en ninguna de las cinco tablas. Asimismo, los identificadores primarios presentaron unicidad completa (`cliente_id`, `factura_id`, `gestion_id` y `promesa_id`). Tampoco se encontraron registros huérfanos en las relaciones entre tablas: todas las facturas remiten a clientes existentes, todas las gestiones remiten a facturas válidas y todas las promesas remiten a gestiones y facturas existentes.

### 5.2. Consistencia temporal

La estructura temporal del dataset también resultó consistente:

- no existen facturas con fecha de vencimiento anterior a la fecha de emisión;
- no existen gestiones anteriores a la fecha de emisión;
- no existen gestiones posteriores al pago final de la factura;
- la numeración de facturas quedó alineada con el orden cronológico global de emisión.

Estos resultados son importantes porque uno de los principales riesgos del proyecto era introducir inconsistencias temporales o fuga de información (data leakage) en la construcción del dataset.

### 5.3. Valores nulos

Los valores nulos observados no corresponden, en su mayoría, a errores de captura, sino a nulos estructurales explicables por la lógica del negocio.

En `gestiones_cobranza.csv`, la variable `motivo_no_pago` concentra una gran cantidad de nulos, lo cual es esperable, dado que solo aplica cuando hubo contacto exitoso, la factura ya estaba vencida y el cliente expresó una causa de no pago.

En `features_ml.csv`, las variables `dias_desde_ultima_gestion` y `ultimo_resultado_enc` presentan nulos en los cortes iniciales de cada factura. Esto ocurre porque en el **corte 0**, correspondiente a la fecha de emisión, aún no existen gestiones previas para esa factura. Por lo tanto, estos nulos no deben tratarse como error, sino como parte del diseño temporal del dataset.

## 6. Estadísticas descriptivas e interpretación inicial

El análisis descriptivo mostró que varias variables cuantitativas presentan alta dispersión y asimetría, especialmente las relacionadas con monto y comportamiento histórico de pago.

La variable `monto` presenta una distribución claramente sesgada a la derecha, con una media superior a la mediana, lo que sugiere la presencia de facturas de alto valor que elevan el promedio. Esta condición es coherente con escenarios reales de crédito empresarial, donde algunas cuentas pueden concentrar valores sustancialmente mayores al promedio.

Las variables históricas como `mora_promedio_hist`, `mora_ultimo_tramo`, `num_facturas_prev` y `tasa_cumplimiento` muestran suficiente variabilidad para considerarse potencialmente útiles en el modelado. De forma similar, variables derivadas del proceso de cobranza, como `num_gestiones_factura`, `num_promesas_rotas`, `promesas_total` o `num_no_contesta_cons`, exhiben dispersión compatible con un uso predictivo posterior.

Desde una perspectiva metodológica, estos resultados muestran que el dataset no es plano ni trivial: contiene heterogeneidad suficiente para justificar el uso de modelos supervisados y no supervisados.

## 7. Distribución de la variable objetivo

### 7.1. Distribución a nivel de factura

A nivel de `facturas.csv`, la distribución final de `target_mora` fue la siguiente:

- `on_time`: 41.53%
- `+30`: 23.34%
- `+60`: 20.38%
- `+90`: 14.74%

Esta distribución indica que, aunque la clase sin mora es la más frecuente, existe una proporción significativa de facturas con mora severa. Por tanto, el problema no debe interpretarse como un escenario trivial de clase mayoritaria dominante.

### 7.2. Distribución a nivel de cortes temporales

En `features_ml.csv`, la distribución cambia de forma importante:

- `on_time`: 16.86%
- `+30`: 18.88%
- `+60`: 30.32%
- `+90`: 33.94%

Este hallazgo es especialmente relevante. La razón no es que el comportamiento de los clientes cambie entre tablas, sino que las facturas más problemáticas generan más gestiones y, por lo tanto, más cortes temporales. En consecuencia, las clases severas quedan sobrerrepresentadas en el dataset analítico.

Desde el punto de vista del modelado, esto implica que el entrenamiento del componente 1 se realizará sobre un dataset con **desbalance moderado a fuerte**, lo que justifica el uso de **F1-macro** como métrica principal y obliga a considerar técnicas de ponderación o balanceo de clases.

## 8. Análisis de valores atípicos

El análisis de outliers mostró presencia de valores extremos en variables como:

- `monto`
- `ratio_monto`
- `num_promesas_rotas`
- `num_no_contesta_cons`
- `moras_consecutivas`
- `promesas_total`

Sin embargo, la interpretación de estos outliers no sugiere errores evidentes de simulación. Por el contrario, en varios casos representan precisamente los comportamientos de mayor interés para el problema de negocio: clientes con reiteradas promesas incumplidas, cadenas largas de no contacto o facturas excepcionalmente altas.

Por ello, la decisión metodológica recomendada no es eliminar estos valores de forma automática, sino conservarlos como parte del fenómeno a modelar. En todo caso, podrán evaluarse transformaciones o escalamiento robusto según las exigencias del algoritmo específico.

## 9. Análisis de correlaciones

El análisis de correlación permitió identificar tanto variables con fuerte asociación respecto a la severidad de mora como redundancias entre predictores.

Entre las variables con mayor relación con el target ordinal se encuentran:

- `num_corte`
- `num_gestiones_factura`
- `dias_hasta_vence`
- `dias_desde_emision`
- `tasa_cumplimiento`
- `mora_promedio_hist`
- `mora_ultimo_tramo`
- `tasa_cumpl_promesas`
- `num_promesas_rotas`
- `tiene_garantia`

La lectura sustantiva es consistente con el dominio:

- a mayor número de gestiones y mayor antigüedad del caso en el ciclo de cobranza, mayor probabilidad de mora severa;
- a mejor historial de cumplimiento, menor severidad esperada;
- la existencia de garantía se asocia con menor riesgo relativo.

De forma paralela, también se identificaron redundancias relevantes:

- `num_corte` y `num_gestiones_factura` presentan equivalencia práctica;
- `dias_desde_emision` y `dias_hasta_vence` tienen alta relación inversa;
- `mora_promedio_hist` y `mora_ultimo_tramo` capturan fenómenos cercanos;
- `num_promesas_rotas` y `promesas_total` también muestran fuerte dependencia.

Estas redundancias no invalidan el dataset, pero sí sugieren prudencia en modelos sensibles a colinealidad, como la regresión logística multinomial.

## 10. Variables categóricas y relaciones de negocio

El análisis de variables categóricas permitió verificar que la simulación conserva coherencia con el dominio del problema.

### 10.1. Perfiles de cliente

La distribución de perfiles muestra predominio de los clientes `regular`, seguidos por `excelente`, `riesgoso` y `critico`. Esta mezcla genera un escenario de riesgo heterogéneo adecuado para fines de entrenamiento.

### 10.2. Canales y resultados de gestión

Los canales más frecuentes fueron `whatsapp`, `llamada` y `email`, seguidos por `visita` y `carta_notarial`. Esto es consistente con una lógica de escalamiento operativo, donde los canales menos costosos aparecen con mayor frecuencia y los más severos se reservan para escenarios más complejos.

Asimismo, los resultados de gestión más frecuentes fueron `no_contesta`, `promesa_de_pago`, `rechazo_pago`, `en_proceso_interno` y `disputa_monto`. La coherencia entre `contacto_exitoso`, `canal` y `resultado` quedó corregida respecto de versiones anteriores del simulador, lo que mejora notablemente la validez del dataset.

### 10.3. Relación entre perfil y mora final

La relación entre perfil interno del cliente y `target_mora` es clara y útil para validación de la simulación. Los clientes excelentes concentran mayor proporción de pagos `on_time`, mientras que los perfiles críticos concentran una fracción mucho mayor de mora severa (`+90`).

Este hallazgo valida conceptualmente la lógica de generación sintética del dataset: el perfil del cliente está influyendo de manera coherente sobre su desenlace final.

## 11. Hallazgos clave para el modelado posterior

El EDA no solo describe la data; también identifica implicaciones concretas para la etapa de entrenamiento. Los principales hallazgos metodológicos son los siguientes.

### 11.1. El dataset de entrenamiento no está balanceado

El dataset `features_ml.csv` sobrerrepresenta facturas problemáticas porque estas generan más cortes temporales. Por ello, las métricas del componente 1 no deben evaluarse únicamente con accuracy. La métrica principal debe mantenerse como **F1-macro**, acompañada de F1 por clase, matriz de confusión y AUC-ROC multiclase.

### 11.2. El split debe hacerse por factura, no por fila

Dado que múltiples filas pertenecen a una misma factura, un particionado aleatorio por fila generaría fuga de información entre entrenamiento y prueba. La separación de datos debe realizarse por `factura_id`, de modo que todos los cortes de una factura queden en un solo conjunto.

### 11.3. Hay variables nulas estructurales que deben tratarse explícitamente

Los nulos de variables como `dias_desde_ultima_gestion` y `ultimo_resultado_enc` deben imputarse o codificarse con una lógica consistente con el corte inicial. No deben eliminarse filas por esta razón.

### 11.4. Existen pares de variables altamente redundantes

Antes de entrenar modelos lineales o comparar interpretabilidad, conviene revisar una reducción moderada de variables redundantes. Esto no significa descartar indiscriminadamente información, sino construir una versión más estable del dataset para experimentación.

### 11.5. Los outliers no deben eliminarse automáticamente

Muchos valores extremos contienen señal relevante del fenómeno de cobranza. En este contexto, eliminar outliers podría suprimir precisamente los casos de mayor interés operativo.

## 12. Decisiones recomendadas para la etapa siguiente

Con base en el EDA, se recomienda adoptar las siguientes decisiones antes del modelado:

1. Usar `features_ml.csv` como dataset principal del componente 1.
2. Mantener `facturas.csv` como referencia para validar distribución final de clases por factura.
3. Implementar el split por `factura_id`.
4. Tratar explícitamente los nulos estructurales de variables de gestión en el corte 0.
5. Evaluar una versión reducida del dataset para la regresión logística multinomial.
6. Mantener F1-macro como métrica principal del comparativo supervisado.
7. Documentar en el pipeline qué variables se mantienen, transforman o excluyen.
8. Justificar cualquier decisión de balanceo, escalado o selección de variables antes del entrenamiento formal.

## 13. Conclusiones

El análisis exploratorio realizado permite concluir que el dataset generado es metodológicamente útil para el proyecto y presenta coherencia general con el fenómeno que busca modelarse. La estructura temporal del sistema, la lógica de cortes de scoring, la relación entre historial del cliente y desenlace final, y la consistencia operativa de gestiones y promesas ofrecen una base sólida para avanzar hacia la etapa experimental.

No obstante, el EDA también muestra que el modelado no debe abordarse de forma ingenua. Existen riesgos concretos de desbalance, colinealidad parcial y fuga de información si se particionan mal los datos o si se ignora la estructura temporal del dataset. Por ello, este reporte debe asumirse como un documento de referencia metodológica para la fase de entrenamiento y comparación de algoritmos.

En síntesis, el dataset está listo para avanzar a la siguiente fase, pero con condiciones: respetar el carácter temporal de las observaciones, usar métricas acordes al desbalance, y documentar cuidadosamente las decisiones de preprocessing y selección de variables.

## 14. Uso de este reporte en la siguiente etapa

Este documento debe utilizarse como insumo para:

- diseñar el pipeline de entrenamiento del componente 1;
- justificar la selección de métricas del comparativo supervisado;
- explicar por qué el split debe ser por factura;
- sustentar decisiones de limpieza, imputación y selección de variables;
- contextualizar el comportamiento del dataset frente al componente 2 de clustering;
- redactar la sección metodológica del informe final del proyecto.

## 15. Referencias

Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. En *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)* (pp. 226-231).

Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861-874.

Han, J., Kamber, M., & Pei, J. (2011). *Data mining: Concepts and techniques* (3rd ed.). Morgan Kaufmann.

James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An introduction to statistical learning: With applications in Python*. Springer. https://doi.org/10.1007/978-3-031-38747-0

Kuhn, M., & Johnson, K. (2019). *Feature engineering and selection: A practical approach for predictive models*. CRC Press.

Material de clase. (2026). *Implementación completa EDA* [PDF].

Material de clase. (2026). *Rúbrica EDA* [PDF].


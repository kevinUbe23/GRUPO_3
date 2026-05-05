# Reporte académico del análisis exploratorio de datos (EDA)

## 1. Introducción

Este reporte documenta el análisis exploratorio de datos del proyecto **Sistema inteligente de priorización de cobranzas para empresas que venden a crédito**. El EDA se realizó sobre los archivos generados en `01_generacion/data/` y fue ejecutado con salidas documentadas en `02_eda/outputs/`.

La decisión predictiva concreta que habilita el dataset es: **priorizar facturas para gestión de cobranza según su riesgo esperado de mora en cada corte temporal**. Para cada combinación de `factura_id` y `fecha_corte`, el sistema debe estimar si la factura terminará en `on_time`, `+30`, `+60` o `+90`. Esta predicción sirve para ordenar casos, definir intensidad de contacto y alimentar una estrategia operativa de cobranza.

## 2. Comprensión del dataset

El análisis usa cinco tablas:

| Tabla | Filas | Columnas | Rol |
|---|---:|---:|---|
| `clientes.csv` | 200 | 6 | Maestro de clientes y atributos de negocio |
| `facturas.csv` | 5,338 | 9 | Unidad de negocio principal: obligación comercial |
| `gestiones_cobranza.csv` | 14,333 | 9 | Acciones de cobranza realizadas sobre facturas |
| `promesas_pago.csv` | 1,741 | 7 | Promesas derivadas de gestiones |
| `features_ml.csv` | 19,671 | 36 | Dataset analítico por cortes temporales |

La unidad de negocio es la **factura**, pero `features_ml.csv` contiene múltiples filas por factura porque representa cortes temporales de scoring. Esta estructura permite modelar riesgo dinámico conforme avanza el ciclo de cobranza, pero impone una regla metodológica estricta: el split train/test debe hacerse por `factura_id`, nunca por fila.

## 3. Calidad de datos

La revisión de calidad se formalizó en `02_eda/outputs/validation_checklist.csv`. Todas las reglas principales quedaron en estado **Cumple**.

| Regla validada | Resultado | Decisión metodológica |
|---|---|---|
| Archivos canónicos disponibles | Cumple | Continuar con entrada desde `01_generacion/data/` |
| Sin filas duplicadas exactas | Cumple | No eliminar filas por duplicidad |
| Llaves primarias únicas | Cumple | Usar llaves para trazabilidad |
| Integridad referencial | Cumple | No descartar registros por relaciones faltantes |
| Consistencia temporal | Cumple | Mantener `fecha_corte` como frontera temporal |
| Target consistente y único por factura | Cumple | Usar `target_mora` solo como etiqueta |
| Nulos estructurales identificados | Cumple | Imputar sin eliminar cortes iniciales |
| Split compatible con múltiples cortes | Cumple | Particionar por `factura_id` |

Los nulos relevantes aparecen en `dias_desde_ultima_gestion` y `ultimo_resultado_enc`, con 5,338 nulos cada una. Esta cifra coincide con los 5,338 cortes iniciales (`num_corte = 0`), por lo que no se interpreta como error de calidad sino como ausencia estructural de una gestión previa.

## 4. Estadísticas descriptivas

Se calcularon media, mediana, desviación estándar, percentiles adicionales e IQR. El archivo completo queda en `02_eda/outputs/descriptive_statistics.csv`.

| Variable | Media | Mediana | P25 | P75 | P95 | P99 | IQR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `monto` | 28,173.43 | 14,013.53 | 6,641.34 | 36,312.92 | 106,803.43 | 142,429.41 | 29,671.58 |
| `ratio_monto` | 1.09 | 1.00 | 0.56 | 1.49 | 2.09 | 3.26 | 0.93 |
| `mora_promedio_hist` | 23.92 | 18.89 | 3.75 | 42.14 | 58.59 | 67.75 | 38.39 |
| `num_gestiones_factura` | 2.31 | 2.00 | 0.00 | 4.00 | 7.00 | 9.00 | 4.00 |
| `dias_hasta_vence` | 1.06 | -6.00 | -30.00 | 33.00 | 90.00 | 90.00 | 63.00 |
| `num_promesas_rotas` | 2.72 | 1.00 | 0.00 | 4.00 | 10.00 | 16.00 | 4.00 |
| `promesas_total` | 5.00 | 3.00 | 1.00 | 8.00 | 16.00 | 22.00 | 7.00 |

La variable `monto` muestra asimetría positiva: la media duplica aproximadamente la mediana y los percentiles altos se alejan del rango intercuartílico. Esto justifica evaluar transformaciones como `log1p` o escalado robusto cuando se usen modelos lineales o basados en distancia.

## 5. Distribuciones e implicaciones de preprocessing

La distribución del target difiere entre el nivel factura y el nivel corte temporal:

| Clase | Facturas | % facturas | Cortes `features_ml` | % cortes |
|---|---:|---:|---:|---:|
| `on_time` | 2,217 | 41.53 | 3,317 | 16.86 |
| `+30` | 1,246 | 23.34 | 3,713 | 18.88 |
| `+60` | 1,088 | 20.38 | 5,965 | 30.32 |
| `+90` | 787 | 14.74 | 6,676 | 33.94 |

El cambio no es un error: las facturas con mayor mora generan más gestiones y, por lo tanto, más cortes. La implicación directa es que accuracy no debe ser la métrica principal. Se recomienda usar **F1-macro**, métricas por clase y matriz de confusión.

Las decisiones de preprocessing derivadas de las distribuciones son:

| Hallazgo | Riesgo | Decisión |
|---|---|---|
| Clases severas sobrerrepresentadas en `features_ml` | Métricas globales engañosas | Usar F1-macro y evaluar `class_weight` dentro de train |
| Múltiples cortes por factura | Fuga de información | Split agrupado por `factura_id` |
| `monto` y `ratio_monto` sesgados | Dominio por escala en modelos sensibles | Evaluar `log1p` o `RobustScaler` |
| `dias_hasta_vence` negativo | Interpretarlo erróneamente como inválido | Conservarlo; indica corte posterior al vencimiento |
| Nulos en corte 0 | Eliminar el inicio de todas las facturas | Imputar con sentinela e indicador de ausencia |

## 6. Valores atípicos

Los outliers se detectaron con el criterio IQR. El detalle completo está en `02_eda/outputs/outlier_summary.csv`.

| Variable | % outliers | Lectura |
|---|---:|---|
| `tasa_contacto_cliente` | 8.12 | Valores extremos posibles por diferencias de contactabilidad |
| `monto` | 7.88 | Facturas grandes plausibles en cartera empresarial |
| `num_promesas_rotas` | 4.83 | Señal relevante de incumplimiento |
| `num_no_contesta_cons` | 4.66 | Señal operativa de dificultad de contacto |
| `moras_consecutivas` | 3.41 | Historial de riesgo persistente |
| `promesas_total` | 2.75 | Intensidad de gestión y promesas acumuladas |

La decisión metodológica es **no eliminar outliers automáticamente**. En cobranzas, muchos extremos representan precisamente los casos críticos que el sistema debe priorizar. Para modelos sensibles a escala se recomienda transformación o escalado robusto, no eliminación ciega.

## 7. Correlaciones

Se codificó `target_mora` como variable ordinal (`on_time = 0`, `+30 = 1`, `+60 = 2`, `+90 = 3`) para explorar asociaciones lineales. Las correlaciones más relevantes fueron:

| Variable | Correlación con target ordinal |
|---|---:|
| `num_corte` | 0.571 |
| `num_gestiones_factura` | 0.571 |
| `dias_hasta_vence` | -0.567 |
| `dias_desde_emision` | 0.554 |
| `tasa_cumplimiento` | -0.405 |
| `mora_promedio_hist` | 0.386 |
| `mora_ultimo_tramo` | 0.355 |
| `tasa_cumpl_promesas` | -0.304 |
| `num_promesas_rotas` | 0.268 |
| `tiene_garantia` | -0.257 |

La lectura de negocio es coherente: más gestiones, más cortes y mayor antigüedad del caso se asocian con mayor severidad; mejores tasas de cumplimiento y garantía se asocian con menor riesgo.

También se detectan redundancias importantes. `num_corte` y `num_gestiones_factura` capturan información muy cercana; `dias_desde_emision` y `dias_hasta_vence` están fuertemente relacionados; `mora_promedio_hist` y `mora_ultimo_tramo` representan historial de mora; y `promesas_total` se relaciona con promesas rotas. Esto no invalida el dataset, pero debe considerarse en modelos lineales e interpretables.

## 8. Variables categóricas

Las variables categóricas revisadas para interpretación operativa fueron `sector`, `canal`, `resultado` y el target. `perfil_pago` existe en `clientes.csv`, pero se trata como **variable interna artificial de la simulación**; por ese motivo no se interpreta como predictor ni se usa para explicar el target.

La distribución por sector muestra una cartera relativamente diversificada. Los sectores con más clientes son `construccion` (35; 17.50%), `manufactura` (34; 17.00%) y `salud` (27; 13.50%). Los sectores menos frecuentes son `tecnologia` (18; 9.00%) y `servicios` (17; 8.50%). Esta composición permite conservar las dummies de sector como señales de segmentación, sin asumir que un único sector domina todo el comportamiento de cobranza.

En canales de gestión predominan `whatsapp` (4,221; 29.45%), `llamada` (3,854; 26.89%) y `email` (2,704; 18.87%). Las acciones más costosas o formales tienen menor frecuencia: `visita` (2,255; 15.73%) y `carta_notarial` (1,299; 9.06%). La lectura es coherente con una operación que inicia con canales remotos y escala hacia gestiones presenciales o formales cuando el riesgo lo amerita.

En resultados de gestión, `no_contesta` es el resultado más frecuente (5,010; 34.95%). Luego aparecen `promesa_de_pago` (1,741; 12.15%), `rechazo_pago` (1,592; 11.11%), `en_proceso_interno` (1,403; 9.79%) y `disputa_monto` (1,294; 9.03%). Esta distribución advierte que la contactabilidad y la respuesta del cliente son señales operativas relevantes: no deben leerse como etiquetas finales, sino como información disponible durante el ciclo de cobranza.

## 9. Selección final sugerida de variables

El EDA generó `02_eda/outputs/feature_selection_recommendation.csv` y `02_eda/outputs/selected_features_base.csv`.

Resumen de decisiones:

| Decisión | Cantidad de variables |
|---|---:|
| Mantener base | 26 |
| Mantener con imputación | 2 |
| Evaluar por redundancia | 4 |
| Excluir identificador | 3 |
| Excluir target | 1 |

Variables a revisar por redundancia antes del pipeline final:

- `num_corte`
- `ratio_monto`
- `mora_ultimo_tramo`
- `promesas_total`

Variables que deben excluirse directamente como predictores:

- `factura_id`
- `cliente_id`
- `fecha_corte`
- `target_mora`

La exclusión de `cliente_id` aplica solo al identificador crudo como predictor directo. El historial del cliente sí debe conservarse mediante variables agregadas disponibles al corte, como `num_facturas_prev`, `mora_promedio_hist`, `mora_ultimo_tramo`, `tasa_cumplimiento`, `monto_promedio_hist`, `moras_consecutivas`, `tasa_contacto_cliente`, `num_promesas_rotas`, `tasa_cumpl_promesas` y `promesas_total`. En un sistema web, `cliente_id` debe funcionar como llave operativa para recuperar y calcular ese historial, pero no como columna numérica directa dentro del modelo.

La selección base conserva variables numéricas disponibles en el corte, incluyendo monto, condición de pago, antigüedad, garantía, dummies de sector, historial de mora, historial de cumplimiento, señales de gestión, contacto, disputa y promesas.

## 10. Riesgos metodológicos para modelado

Los principales riesgos identificados son:

1. **Data leakage por split incorrecto:** si se separa por fila, cortes de una misma factura pueden quedar en train y test.
2. **Uso indebido del target:** `target_mora` y derivaciones futuras no deben entrar como features.
3. **Uso indebido de variables artificiales:** `perfil_pago` valida la simulación, pero no debe alimentar el modelo final.
4. **Eliminación agresiva de outliers:** puede borrar casos de mayor valor operativo.
5. **Imputación ingenua de nulos:** los nulos de corte 0 deben codificarse como ausencia estructural de gestión previa.
6. **Multicolinealidad:** algunas variables redundantes pueden afectar regresión logística e interpretación.

## 11. Artefactos generados

Las salidas tabulares del EDA quedaron en `02_eda/outputs/`:

- `validation_checklist.csv`
- `descriptive_statistics.csv`
- `target_distribution_facturas.csv`
- `target_distribution_features_ml.csv`
- `distribution_preprocessing_decisions.csv`
- `outlier_summary.csv`
- `correlation_with_target.csv`
- `modeling_risks_and_decisions.csv`
- `feature_selection_recommendation.csv`
- `selected_features_base.csv`

Este reporte es el documento de continuidad de la fase EDA y debe leerse antes de iniciar la preparación de datos.

## 12. Consideraciones para la siguiente fase

La preparación de datos debe usar `features_ml.csv` como base supervisada y mantener la frontera temporal de cada `fecha_corte`. La partición de entrenamiento y prueba debe hacerse por `factura_id`; separar filas individuales produciría fuga de información porque una misma factura puede aparecer en varios cortes.

El pipeline debe excluir `factura_id`, `cliente_id`, `fecha_corte` y `target_mora` del conjunto de predictores. También debe tratar `perfil_pago` como variable interna de simulación: sirve para validar coherencia del dataset, pero no debe alimentar el modelo final salvo que se justifique explícitamente como dato disponible en un escenario real.

Los nulos de `dias_desde_ultima_gestion` y `ultimo_resultado_enc` deben imputarse como ausencia estructural de gestión previa, idealmente con valor sentinela e indicador binario. Los outliers no deben eliminarse automáticamente; para modelos sensibles a escala conviene evaluar `log1p` o `RobustScaler`.

Antes del modelado conviene revisar redundancias en `num_corte`, `ratio_monto`, `mora_ultimo_tramo` y `promesas_total`. La evaluación posterior debe priorizar F1-macro, métricas por clase y matriz de confusión, porque las clases severas aparecen con más peso en `features_ml` que en el nivel factura.

## 13. Checklist de continuidad para Preparación y Preprocesamiento

La siguiente fase será evaluada por limpieza de datos, feature engineering, balanceamiento de clases, data augmentation, pipeline automatizado y entrega de un dataset listo para entrenamiento. Los hallazgos del EDA deben convertirse en decisiones reproducibles dentro del pipeline.

### 13.1 Limpieza de datos

La preparación debe validar la existencia de los archivos canónicos, columnas requeridas, tipos de datos, ausencia de duplicados exactos, integridad de llaves, consistencia temporal y dominio válido de variables discretas.

Validaciones mínimas recomendadas:

- `condicion_dias` debe pertenecer al catálogo `30`, `45`, `60`, `90`.
- `fecha_corte` debe respetar la lógica temporal de la factura.
- `target_mora` debe existir solo como etiqueta supervisada.
- `dias_desde_ultima_gestion` y `ultimo_resultado_enc` deben tratarse como nulos estructurales del corte inicial.
- Los outliers detectados por IQR no deben eliminarse automáticamente; deben conservarse salvo evidencia de error, porque representan casos relevantes de cobranza.

La salida de esta parte debe incluir una tabla o registro de validaciones de preparación y una justificación de cada corrección aplicada.

### 13.2 Feature engineering

El dataset final debe conservar variables de factura y variables históricas del cliente disponibles al corte. La construcción de features debe respetar la frontera temporal: ninguna variable puede usar información posterior a `fecha_corte`.

Features recomendadas para crear o conservar:

- Indicador de ausencia de gestión previa para los nulos estructurales.
- Imputación con sentinela para `dias_desde_ultima_gestion`.
- Imputación con categoría o código sentinela para `ultimo_resultado_enc`.
- Variables temporales derivadas como días transcurridos al corte, estado vencido al corte y mora observable al corte, siempre que se calculen solo con información disponible.
- Transformación `log1p` o escalado robusto para `monto` y variables sesgadas cuando el modelo lo requiera.
- Variables históricas agregadas del cliente, no el identificador crudo `cliente_id`.

Las variables `factura_id`, `cliente_id` y `fecha_corte` deben mantenerse para trazabilidad, auditoría y partición, pero excluirse de `X` al entrenar.

### 13.3 Balanceamiento de clases

La preparación debe documentar la distribución del target tanto a nivel factura como a nivel `features_ml`. Dado que `+60` y `+90` tienen más peso en cortes temporales, el dataset puede sesgar métricas globales si se evalúa solo con accuracy.

El balanceamiento debe aplicarse solo dentro del conjunto de entrenamiento, nunca antes del split y nunca sobre test. Opciones razonables para la fase siguiente:

- Usar `class_weight` en modelos que lo soporten.
- Evaluar pesos por clase o pesos por factura para reducir la sobrerrepresentación de facturas con muchos cortes.
- Comparar resultados con y sin balanceamiento usando F1-macro y métricas por clase.

### 13.4 Data augmentation

La data augmentation no debe aplicarse mecánicamente. En este problema tabular de cobranzas, generar filas sintéticas puede introducir patrones artificiales o fuga si se hace antes de separar train/test.

Si se evalúa augmentation, debe plantearse como experimento controlado:

- Aplicarlo solo sobre train.
- No generar datos sintéticos para test.
- Preferir técnicas compatibles con datos tabulares y clases minoritarias, por ejemplo SMOTE o SMOTENC si las variables finales lo permiten.
- Comparar contra alternativas más conservadoras como `class_weight`.
- Justificar si se descarta augmentation por riesgo metodológico.

### 13.5 Pipeline automatizado

La preparación debe implementarse como pipeline reproducible. El flujo mínimo esperado es:

1. Cargar `01_generacion/data/features_ml.csv`.
2. Validar entradas y dominios.
3. Separar train/test por `factura_id`.
4. Separar `X` e `y`, excluyendo identificadores y target.
5. Imputar nulos estructurales.
6. Aplicar transformaciones numéricas solo cuando correspondan.
7. Guardar dataset preparado y artefactos de trazabilidad.

El pipeline debe evitar que transformaciones aprendidas con train usen información de test. Imputadores, escaladores, codificadores y balanceadores deben ajustarse dentro de train.

### 13.6 Dataset listo para entrenamiento

La salida de preparación debe quedar documentada y lista para modelado. Como mínimo se esperan:

- `features_ml_prepared.csv` o `.parquet`.
- `train_facturas_ids.csv`.
- `test_facturas_ids.csv`.
- `preprocessing_metadata.json`.
- Resumen de columnas finales, tipos, nulos restantes y distribución del target.
- Confirmación de que no hay `factura_id` compartidas entre train y test.
- Confirmación de que `target_mora` queda fuera de las features y solo se usa como etiqueta.

## 14. Conclusiones

El dataset es coherente para avanzar hacia preparación y modelado, siempre que se respeten las reglas metodológicas detectadas en el EDA. La estructura por cortes temporales aporta valor porque permite un scoring dinámico, pero también exige controlar leakage con split por `factura_id`.

El EDA confirma que existen señales predictivas útiles: historial de mora, número de gestiones, antigüedad del caso, cumplimiento histórico, promesas rotas, tasa de contacto y garantía. También confirma que los outliers y nulos no deben tratarse mecánicamente, sino según su significado de negocio.

La siguiente fase debe construir un pipeline reproducible que valide entradas, impute nulos estructurales, excluya identificadores y target, revise redundancias, aplique transformaciones solo cuando el algoritmo lo requiera, evalúe balanceamiento y data augmentation solo dentro de train, y exporte artefactos listos para modelado.

## 15. Referencias

Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861-874.

Han, J., Kamber, M., & Pei, J. (2011). *Data mining: Concepts and techniques* (3rd ed.). Morgan Kaufmann.

James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An introduction to statistical learning: With applications in Python*. Springer. https://doi.org/10.1007/978-3-031-38747-0

Kuhn, M., & Johnson, K. (2019). *Feature engineering and selection: A practical approach for predictive models*. CRC Press.

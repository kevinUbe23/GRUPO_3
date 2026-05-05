# Reporte de preparacion y procesamiento de datos

## 1. Proposito de la fase

La fase de preparacion toma el dataset analitico `features_ml.csv` y lo transforma en una base lista para modelado. Esta etapa no entrena modelos; su responsabilidad es dejar los datos validados, ordenados, separados correctamente en train/test y documentados para que la siguiente fase pueda entrenar sin introducir fuga de informacion.

La regla metodologica central viene del EDA: la unidad de negocio es la `factura`, pero `features_ml.csv` tiene varias filas por factura porque cada fila representa un corte temporal de scoring. Por eso, la preparacion no puede separar train/test por fila. Si una misma factura aparece en train y test, el modelo podria aprender informacion del mismo caso que luego se evalua, generando una evaluacion artificialmente optimista. Para evitarlo, el split se hace por `factura_id`.

El notebook principal es:

`03_preparacion/notebook_preparacion.ipynb`

Las salidas quedan en:

`03_preparacion/outputs/`

## 2. Entrada utilizada

La entrada tecnica de la fase fue:

| Archivo | Para que se usa |
|---|---|
| `01_generacion/data/features_ml.csv` | Base supervisada por cortes temporales. Contiene identificadores, variables historicas, variables operativas al corte y la etiqueta `target_mora`. |

El dataset de entrada tenia:

| Metrica | Valor |
|---|---:|
| Filas | 19,671 |
| Columnas | 36 |
| Facturas unicas | 5,338 |
| Clientes unicos | 200 |
| Target | `target_mora` |

Cada fila representa una combinacion de `factura_id` y `fecha_corte`. Esto permite modelar el riesgo dinamicamente conforme avanza la cobranza, pero obliga a controlar muy bien la separacion entre train y test.

## 3. Que hizo el notebook, paso a paso

### 3.1 Configuracion de rutas y columnas

El notebook primero localiza la raiz del proyecto y define rutas relativas. Esto evita rutas fragiles como `/mnt/data` o rutas absolutas de una maquina especifica. La entrada se busca en `01_generacion/data/features_ml.csv` y los outputs se escriben en `03_preparacion/outputs/`.

Tambien define listas de columnas esperadas:

- Columnas identificadoras: `factura_id`, `cliente_id`, `fecha_corte`.
- Target: `target_mora`.
- Columnas de sector one-hot.
- Columnas numericas sesgadas que pueden necesitar `log1p`.
- Columnas binarias.
- Columnas numericas de conteo o tasa.
- Columna categorica `ultimo_resultado_enc`.

Esta separacion no es solo orden cosmetico. Sirve para que cada tipo de variable reciba un tratamiento compatible con su significado.

### 3.2 Carga y validacion estructural

Despues de cargar el CSV, el notebook valida que la base tenga sentido antes de transformarla. Las validaciones principales fueron:

| Validacion | Interpretacion |
|---|---|
| Columnas requeridas disponibles | Confirma que el dataset tiene las variables necesarias para preparar modelado. |
| Sin duplicados exactos | Evita entrenar con registros repetidos sin justificacion. |
| Sin duplicados `factura_id` + `num_corte` | Cada factura debe tener como maximo una fila por corte. |
| Target unico por factura | Una misma factura no debe tener dos etiquetas finales distintas. |
| Fechas de corte validas | `fecha_corte` debe poder interpretarse como fecha. |
| Montos positivos | Una factura con monto cero o negativo romperia la interpretacion de cobranza. |
| Condicion de pago valida | La condicion debe estar en el catalogo esperado: 30, 45, 60 o 90 dias. |
| Sector one-hot valido | Cada fila debe pertenecer a un solo sector. |
| Tasas en rango `[0, 1]` | Tasas fuera de ese rango indicarian error de calculo. |
| Sin `perfil_pago` como predictor | `perfil_pago` es una variable interna de simulacion y no debe alimentar el modelo. |

Resultado: `15/15` validaciones quedaron en estado `Cumple`, documentadas en `preprocessing_validation_checklist.csv`.

### 3.3 Tratamiento de nulos estructurales

El EDA detecto que `dias_desde_ultima_gestion` y `ultimo_resultado_enc` tienen nulos en los cortes iniciales. Esos nulos no significan que falte informacion por error; significan que todavia no habia una gestion previa.

Por eso no se eliminaron esas filas. Hacerlo habria eliminado el primer corte de todas las facturas y sesgaria el dataset hacia casos que ya tuvieron gestion. En su lugar:

| Variable | Tratamiento | Interpretacion |
|---|---|---|
| `dias_desde_ultima_gestion` | Se rellena con `-1` | Sentinela que significa "no existe gestion previa". |
| `ultimo_resultado_enc` | Se convierte a `sin_gestion_previa` | Categoria explicita para ausencia de resultado anterior. |
| `sin_gestion_previa` | Se crea como binaria | Permite que el modelo aprenda que el caso esta en su primer corte. |

Esta decision conserva informacion importante del ciclo de vida de la factura.

### 3.4 Revision de redundancias

El EDA pidio revisar variables redundantes antes del pipeline final. En preparacion se hizo una revision tecnica y se documento en `redundancy_review.csv`.

| Variable | Decision | Interpretacion |
|---|---|---|
| `num_corte` | Eliminar | En esta base equivale a `num_gestiones_factura`; mantener ambas duplicaria informacion. |
| `dias_desde_emision` | Eliminar | Se puede reconstruir como `condicion_dias - dias_hasta_vence`. |
| `ratio_monto` | Conservar | Aunque esta relacionado con `monto`, aporta lectura relativa contra el historial del cliente. |
| `mora_ultimo_tramo` | Conservar | Complementa `mora_promedio_hist` con una senal mas reciente. |
| `promesas_total` | Conservar | Aporta volumen historico de promesas y complementa `ratio_promesas_rotas`. |

La regla aplicada fue: eliminar redundancias deterministicas, pero no eliminar variables utiles solo por estar correlacionadas.

### 3.5 Feature engineering

Luego se crearon variables nuevas a partir de informacion disponible al corte. Estas variables intentan traducir mejor el estado operativo de la factura.

| Variable nueva | Que representa | Como se interpreta |
|---|---|---|
| `dias_transcurridos_corte` | Dias desde el inicio del credito hasta el corte | Mide avance temporal del caso. |
| `esta_vencida_al_corte` | Si la factura ya esta vencida en ese corte | Distingue casos preventivos vs vencidos. |
| `dias_mora_observable` | Dias de mora acumulables al corte | Solo crece cuando la factura ya vencio. |
| `dias_hasta_vence_pos` | Dias restantes si aun no vence | Separa tiempo preventivo de mora real. |
| `cliente_nuevo` | Cliente sin facturas previas | Indica baja historia disponible. |
| `intensidad_gestion` | Gestiones relativas al tiempo transcurrido | Mide presion operativa sobre el caso. |
| `friccion_contacto` | Proporcion de no respuesta sobre gestiones | Resume dificultad de contacto. |
| `ratio_promesas_rotas` | Promesas rotas sobre promesas totales | Resume incumplimiento en promesas. |

Estas variables no usan informacion futura; se calculan con datos disponibles en cada `fecha_corte`.

### 3.6 Split por factura

La separacion train/test se hizo con `StratifiedShuffleSplit`, pero no sobre las filas directamente. Primero se construyo una tabla unica por `factura_id` y su `target_mora`, y sobre esa tabla se separaron las facturas.

Resultado:

| Conjunto | Filas | Facturas |
|---|---:|---:|
| Train | 15,735 | 4,270 |
| Test | 3,936 | 1,068 |
| Facturas compartidas | 0 | 0 |

La distribucion por filas quedo:

| Split | on_time | +30 | +60 | +90 |
|---|---:|---:|---:|---:|
| Train | 16.82% | 18.82% | 30.20% | 34.16% |
| Test | 17.05% | 19.08% | 30.82% | 33.05% |

Interpretacion: train y test mantienen una distribucion parecida. Las clases severas `+60` y `+90` tienen mas peso por filas porque esas facturas generan mas cortes. Esto confirma que accuracy no debe ser la metrica principal en modelado.

### 3.7 Pipeline de preprocesamiento

El notebook construyo un `ColumnTransformer` de scikit-learn. El objetivo es que cada tipo de variable reciba el tratamiento adecuado:

| Grupo | Tratamiento | Por que se hace |
|---|---|---|
| Numericas sesgadas | Mediana, `log1p`, `RobustScaler` | Reduce influencia de colas largas y outliers. |
| Numericas de conteo/tasas | Mediana, `RobustScaler` | Escala variables sin asumir distribucion normal. |
| Binarias | Valor mas frecuente | Mantiene 0/1 sin escalarlas innecesariamente. |
| `ultimo_resultado_enc` | Categoria constante y `OneHotEncoder` | Convierte la categoria en columnas numericas para modelos. |

El pipeline se ajusta con `X_train` y luego transforma `X_test`. Esto es importante: imputadores, escaladores y codificadores aprenden parametros solo desde train. Si aprendieran desde todo el dataset, habria fuga de informacion desde test.

Una aclaracion importante: este pipeline no reemplaza al dataset preparado ni significa que todas las variables deban perder su escala original. `features_ml_prepared.csv` conserva variables como `num_promesas_rotas`, `promesas_total`, `monto` o `dias_mora_observable` en escala interpretable. El pipeline sirve para transformar esas variables cuando el algoritmo lo necesita.

La decision metodologica mas prudente para la siguiente fase es no asumir que existe un unico preprocesamiento ideal para todos los modelos. Hay dos familias de uso:

| Familia de modelo | Necesita escalado robusto? | Lectura |
|---|---|---|
| Modelos lineales, SVM, KNN, redes neuronales | Si, normalmente conviene | Estos modelos son sensibles a la magnitud de las variables; `RobustScaler` ayuda a comparar variables en escalas distintas. |
| Arboles, Random Forest, XGBoost, LightGBM | No necesariamente | Estos modelos trabajan con cortes por variable y suelen tolerar escalas originales; el escalado no suele ser indispensable. |

Por eso, el `preprocessing_pipeline.joblib` se interpreta como un pipeline reproducible de referencia, especialmente util para modelos sensibles a escala. No debe entenderse como obligacion para todos los modelos. En modelado se puede comparar:

- Una version con preprocesamiento robusto para modelos sensibles a escala.
- Una version mas directa para modelos basados en arboles, manteniendo variables numericas en escala original y codificando solo las categoricas cuando corresponda.

Lo que si es obligatorio es la consistencia: el preprocesamiento usado para entrenar el modelo elegido debe ser exactamente el mismo que se use luego para predecir nuevas facturas.

## 4. Como queda el dataset preparado

El archivo final `features_ml_prepared.csv` tiene:

| Metrica | Valor |
|---|---:|
| Filas | 19,671 |
| Columnas | 43 |
| Nulos restantes | 0 |
| Features sin procesar | 39 |
| Columnas de trazabilidad | 3 |
| Target | 1 |

Las columnas de trazabilidad son:

- `factura_id`
- `cliente_id`
- `fecha_corte`

Estas columnas se conservan para auditoria, seguimiento y split, pero no deben entrar al modelo como predictores.

El target es:

- `target_mora`

Las features son las 39 columnas listadas en `features_selected.csv`.

## 5. Para que sirve cada output

Esta es la parte mas importante para la siguiente fase. No todos los outputs se usan igual: algunos son datos para entrenar, otros son validaciones, otros son documentacion tecnica.

| Output | Que contiene | Para que sirve en modelado |
|---|---|---|
| `features_ml_prepared.csv` | Dataset preparado completo con trazabilidad, features y target. | Es la base principal para entrenar y evaluar modelos. |
| `train_facturas_ids.csv` | Lista de facturas asignadas a train y su target. | Permite reconstruir `df_train` sin regenerar el split. |
| `test_facturas_ids.csv` | Lista de facturas asignadas a test y su target. | Permite reconstruir `df_test` sin contaminar train/test. |
| `features_selected.csv` | Lista de las 39 columnas que pueden usarse como predictores antes de transformar. | Sirve para crear `X_train` y `X_test` sin incluir IDs, fecha ni target. |
| `processed_feature_names.csv` | Nombres de las 47 columnas despues del preprocesamiento. | Sirve para interpretar matrices ya transformadas por el pipeline. |
| `prepared_columns_summary.csv` | Rol, tipo, nulos y cantidad de valores unicos por columna. | Sirve para revisar rapidamente que es feature, trazabilidad o target. |
| `target_distribution_train_test.csv` | Distribucion del target por split, en filas y facturas. | Sirve para entender el desbalance y justificar metricas como F1-macro. |
| `preprocessing_validation_checklist.csv` | Validaciones estructurales de entrada. | Sirve para demostrar que la base paso controles minimos de calidad. |
| `split_integrity_check.csv` | Validaciones del split por factura. | Sirve para confirmar que no hay leakage por facturas compartidas. |
| `outlier_summary.csv` | Variables con outliers por IQR, limites y decision. | Sirve para justificar que los outliers se conservan y se tratan con transformaciones robustas. |
| `redundancy_review.csv` | Variables revisadas por redundancia y decision tomada. | Sirve para explicar por que algunas columnas se eliminaron y otras se conservaron. |
| `preprocessing_metadata.json` | Metadata de la corrida: shapes, rutas, validaciones, schema, class weights. | Sirve para trazabilidad y auditoria; no es obligatorio para entrenar. |
| `preprocessing_pipeline.joblib` | Objeto `ColumnTransformer` ajustado con train. | Sirve si se quiere reutilizar el preprocesamiento robusto de referencia, sobre todo para modelos sensibles a escala. No es obligatorio para arboles si modelado define otro pipeline. |

En terminos practicos, para entrenar modelos lo minimo es:

1. `features_ml_prepared.csv`
2. `train_facturas_ids.csv`
3. `test_facturas_ids.csv`
4. `features_selected.csv`

Los demas archivos sirven para trazabilidad, validacion, interpretacion o reutilizacion del pipeline.

## 6. Como se usaria en modelado

La siguiente fase puede reconstruir train y test asi:

```python
import pandas as pd

df = pd.read_csv("03_preparacion/outputs/features_ml_prepared.csv")
train_ids = pd.read_csv("03_preparacion/outputs/train_facturas_ids.csv")
test_ids = pd.read_csv("03_preparacion/outputs/test_facturas_ids.csv")
features = pd.read_csv("03_preparacion/outputs/features_selected.csv")["feature"].tolist()

df_train = df[df["factura_id"].isin(train_ids["factura_id"])]
df_test = df[df["factura_id"].isin(test_ids["factura_id"])]

X_train = df_train[features]
y_train = df_train["target_mora"]
X_test = df_test[features]
y_test = df_test["target_mora"]
```

Luego hay dos caminos validos:

**Camino A: usar el pipeline ya exportado**

```python
import joblib

preprocessor = joblib.load("03_preparacion/outputs/preprocessing_pipeline.joblib")
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)
```

Este camino es util si se quiere garantizar que modelado usa exactamente el mismo preprocesamiento definido en preparacion.

**Camino B: construir pipelines propios por modelo**

Este camino tambien es valido. Por ejemplo, un Random Forest puede necesitar menos escalado que una Regresion Logistica. En ese caso, la fase de modelado puede usar los CSV oficiales y definir pipelines propios, siempre respetando:

- El split oficial por `factura_id`.
- Las features de `features_selected.csv`.
- La exclusion de `factura_id`, `cliente_id`, `fecha_corte` y `target_mora` como predictores.
- El ajuste de transformaciones solo con train.

Este segundo camino es el recomendado para una comparacion seria de modelos: no forzar la misma transformacion a algoritmos con necesidades distintas. Por ejemplo, una Regresion Logistica deberia evaluarse con escalado robusto, mientras que un Random Forest o XGBoost puede evaluarse sin escalado numerico, conservando conteos como `num_promesas_rotas` en su escala original.

## 7. Interpretacion de resultados de preparacion

La preparacion confirma que el dataset esta listo para modelado porque:

- No quedan nulos en el dataset preparado.
- No hay facturas compartidas entre train y test.
- El target es consistente por factura.
- Las clases de target son validas.
- Las variables de trazabilidad se conservan, pero quedan fuera de predictores.
- Los nulos estructurales se transformaron en informacion util.
- Los outliers se conservaron porque pueden representar casos criticos reales.
- El pipeline de transformacion se ajusto solo con train.

Tambien confirma un punto importante del EDA: el dataset por cortes temporales no tiene la misma distribucion que el dataset por facturas. Las clases `+60` y `+90` tienen mas peso por filas, porque las facturas mas problematicas generan mas cortes. En modelado esto significa que la evaluacion debe mirar F1-macro, recall por clase y matriz de confusion, no solo accuracy.

## 8. Decisiones metodologicas tomadas

Quedan tomadas estas decisiones:

- La base oficial preparada es `features_ml_prepared.csv`.
- El split oficial esta definido por `train_facturas_ids.csv` y `test_facturas_ids.csv`.
- La particion train/test no debe regenerarse por fila.
- `factura_id`, `cliente_id` y `fecha_corte` se conservan solo para trazabilidad.
- `target_mora` se usa solo como etiqueta.
- `perfil_pago` no debe entrar al modelo porque es una variable interna de simulacion.
- Los nulos de gestion previa se tratan como ausencia estructural.
- Los outliers no se eliminan automaticamente.
- `class_weight` es la estrategia base de balanceamiento sugerida.
- La augmentation no se aplica al dataset oficial; queda solo como experimento controlado sobre train.
- El escalado robusto no es una obligacion universal: se usa cuando el modelo lo necesita, especialmente en modelos sensibles a escala.

## 9. Riesgos que debe cuidar la siguiente fase

| Riesgo | Por que importa | Como mitigarlo |
|---|---|---|
| Split por fila | Produce leakage entre cortes de la misma factura. | Usar siempre `train_facturas_ids.csv` y `test_facturas_ids.csv`. |
| IDs como predictores | El modelo podria memorizar clientes o facturas. | Usar `features_selected.csv` para armar `X`. |
| Target en features | Seria fuga directa de la respuesta. | Mantener `target_mora` solo en `y`. |
| Transformar usando todo el dataset | Test influiria en imputacion, escalado u OHE. | Ajustar transformaciones solo con train. |
| Accuracy enganosa | Las clases severas pesan mas por cortes. | Reportar F1-macro, recall por clase y matriz de confusion. |
| Overfitting a cortes repetidos | Varias filas de una factura comparten informacion. | Evaluar siempre por el split oficial de facturas. |
| Augmentation mal aplicada | Puede crear patrones artificiales o contaminar test. | Si se usa, aplicarla solo sobre train y compararla contra `class_weight`. |

## 10. Que debe considerar modelado

La siguiente fase debe partir de este orden:

1. Leer `features_ml_prepared.csv`.
2. Leer `train_facturas_ids.csv` y `test_facturas_ids.csv`.
3. Leer `features_selected.csv`.
4. Construir `X_train`, `y_train`, `X_test`, `y_test`.
5. Definir modelos y pipelines.
6. Ajustar transformaciones y modelos solo con train.
7. Evaluar en test con metricas por clase.

El archivo `preprocessing_pipeline.joblib` queda disponible si se quiere reutilizar el preprocesamiento ya definido. Si modelado construye pipelines propios, ese archivo queda como referencia o artefacto de reproducibilidad, no como obligacion.

La recomendacion final para modelado es:

1. Usar siempre el split oficial por `factura_id`.
2. Mantener `features_ml_prepared.csv` como dataset base interpretable.
3. Comparar al menos dos enfoques de pipeline:
   - Pipeline robusto con imputacion, `log1p`, `RobustScaler` y OHE para modelos sensibles a escala.
   - Pipeline orientado a arboles, sin escalado numerico obligatorio, pero con tratamiento de categoricas si el algoritmo lo requiere.
4. Elegir el pipeline junto con el modelo, no por separado.
5. Para uso diario, aplicar a cada nueva factura la misma construccion de features y el mismo preprocesamiento del modelo finalmente seleccionado.

## 11. Conclusion

La fase de preparacion deja el dataset en condiciones tecnicas para modelado. Su aporte principal no es solo "limpiar datos", sino fijar reglas metodologicas: separar por factura, preservar trazabilidad, excluir variables que producirian fuga, convertir nulos estructurales en informacion util, conservar outliers relevantes y documentar cada artefacto necesario para continuar.

Con esta fase, modelado ya no tiene que decidir desde cero que columnas usar ni como separar train/test. Debe consumir los outputs oficiales, entrenar modelos y concentrarse en comparar desempeno, estabilidad e interpretabilidad.

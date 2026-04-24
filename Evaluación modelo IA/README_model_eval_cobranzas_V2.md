# model_eval_cobranzas_V2.py

Script de evaluacion para el componente de priorizacion de cobranzas.

Esta version toma como fuente oficial el dataset y el split exportados desde la fase de preparacion, entrena un benchmark de modelos, compara su desempeno en train/test, selecciona el mejor modelo con un criterio de estabilidad y exporta un resumen de resultados.

## Que hace

1. Carga el dataset preparado desde la carpeta oficial de preparacion.
2. Carga los archivos oficiales de `train_facturas_ids.csv` y `test_facturas_ids.csv`.
3. Valida que el dataset tenga las columnas obligatorias.
4. Elimina filas con target nulo.
5. Verifica que cada `factura_id` tenga una sola etiqueta target.
6. Verifica que el split oficial no tenga:
   - interseccion entre train y test
   - IDs duplicados
   - facturas faltantes
   - facturas extra respecto al dataset
7. Construye el conjunto de features con control para evitar leakage accidental.
8. Preprocesa variables numericas, binarias y nominales.
9. Entrena estos modelos:
   - `Dummy Baseline`
   - `Logistic Regression`
   - `Random Forest`
   - `XGBoost` si la libreria esta instalada
10. Calcula metricas en train y test:
    - accuracy
    - balanced accuracy
    - precision macro
    - recall macro
    - f1 macro
    - AUC weighted
11. Calcula el gap train-test para diagnosticar estabilidad.
12. Selecciona el mejor modelo priorizando `f1_macro_test`, pero solo entre modelos con gap de F1 razonable.
13. Genera un `classification_report` del mejor modelo.
14. Ejecuta un analisis simple de fairness por subgrupos si las columnas existen.
    - accuracy
    - balanced accuracy
    - f1 macro
    - worst class recall
15. Exporta tablas y un resumen final en texto.

## Entradas esperadas

El script no recibe argumentos CLI. Usa rutas internas fijas.

Debe existir esta carpeta:

`Presentacion de la fase de preparacion y procesamiento de datos/data`

Y dentro de ella estos archivos:

- `features_ml_prepared.csv`
- `train_facturas_ids.csv`
- `test_facturas_ids.csv`

En esta implementacion, esas rutas se resuelven automaticamente desde la ubicacion del script.

## Salidas generadas

Las salidas se guardan en:

`Evaluacion modelo IA/data`

Archivos generados:

- `benchmark_metrics.csv`
- `train_test_gap.csv`
- `fairness_by_group.csv`
- `fairness_gap_summary.csv`
- `classification_report_best_model.txt`
- `summary.txt`

## Como ejecutarlo

Desde la raiz del proyecto:

```powershell
python ".\Evaluacion modelo IA\model_eval_cobranzas_V2.py"
```

Si usas un entorno virtual:

```powershell
.\.venv\Scripts\python.exe ".\Evaluacion modelo IA\model_eval_cobranzas_V2.py"
```

## Requisitos

Bibliotecas principales:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost` opcional

Si `xgboost` no esta instalado, el script sigue corriendo y simplemente omite ese modelo.

## Criterios importantes del script

- El split se toma de los archivos oficiales de preparacion, no se regenera al vuelo.
- Si una misma factura tiene mas de una clase target, el script falla para evitar una evaluacion metodologicamente invalida.
- El mejor modelo no se elige solo por score alto, sino tambien por estabilidad train-test.
- Las columnas numericas inesperadas quedan fuera del set final de features para reducir riesgo de leakage.

## Consideraciones

- El analisis de fairness usa `MIN_FAIRNESS_GROUP = 30`.
- La regla actual para estabilidad usa `MAX_SELECTION_GAP = 0.10` sobre `gap_f1_macro`.
- Las salidas anteriores dentro de `Evaluacion modelo IA/data` pueden ser sobreescritas en una nueva corrida.

## Recomendacion de uso

Antes de ejecutar:

1. Confirma que la fase de preparacion ya genero los 3 archivos oficiales.
2. Confirma que el dataset y los IDs pertenecen a la misma corrida de preparacion.
3. Ejecuta la V2.
4. Revisa primero `summary.txt`, `benchmark_metrics.csv` y `train_test_gap.csv`.

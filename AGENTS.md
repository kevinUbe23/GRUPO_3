# AGENTS.md

## Contexto Del Sistema
Este repositorio contiene un flujo academico de priorizacion de cobranzas. La logica principal vive en tres notebooks encadenados:
`01_generacion/simulacion_datos.ipynb` -> `02_eda/EDA_cobranzas.ipynb` -> `03_preparacion/notebook_preparacion.ipynb`.

## Regla Central
La unidad de negocio es la `factura`, pero el dataset de modelado usa `features_ml` por cortes temporales. Nunca se debe mezclar train y test por fila; el split correcto siempre es por `factura_id`.

## Rutas Canonicas
- Entrada generada: `01_generacion/data/`
- Salida del EDA: `artifacts/02_eda/`
- Salida de preparacion: `artifacts/03_preparacion/`

## Artefactos Clave
- `clientes.csv`, `facturas.csv`, `gestiones_cobranza.csv`, `promesas_pago.csv`, `features_ml.csv`
- `features_ml_prepared.csv` o `.parquet`
- `train_facturas_ids.csv`, `test_facturas_ids.csv`
- `outlier_summary.csv`
- `preprocessing_metadata.json`

## Orden Operativo
1. Generar datos sinteticos.
2. Ejecutar EDA solo de lectura.
3. Ejecutar preparacion y exportar artefactos listos para modelado.

## Criterios De Calidad
- Las rutas deben ser relativas al proyecto.
- Cada notebook debe validar sus entradas antes de continuar.
- Cada etapa debe imprimir un resumen final de sus outputs.
- Las salidas deben estar documentadas en un archivo de contexto compacto.

## Proxima Expansion
Despues de la fase de preparacion, este contexto debe servir para nuevas etapas de modelado, evaluacion y presentacion sin releer todos los notebooks.

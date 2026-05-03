# AGENTS.md

## Contexto Del Sistema
Este repositorio contiene un flujo academico de priorizacion de cobranzas. La logica principal vive en tres notebooks encadenados:
`01_generacion/simulacion_datos.ipynb` -> `02_eda/notebook_eda_cobranzas.ipynb` -> `03_preparacion/notebook_preparacion.ipynb`.

## Regla Central
La unidad de negocio es la `factura`, pero el dataset de modelado usa `features_ml` por cortes temporales. Nunca se debe mezclar train y test por fila; el split correcto siempre es por `factura_id`.

## Rutas Canonicas
- Entrada generada: `01_generacion/data/`
- Fase EDA: `02_eda/`
- Salida del EDA: `02_eda/outputs/`
- Fase de preparacion: `03_preparacion/`
- Salida de preparacion: `03_preparacion/outputs/`

## Patron De Carpetas Por Fase
Cada fase debe mantener su notebook, reporte de continuidad y outputs dentro de su propia carpeta numerada, salvo que exista una razon especifica para no seguir este patron.

Estructura esperada:

- `NN_nombre_fase/notebook_o_script_principal`
- `NN_nombre_fase/outputs/`
- `NN_nombre_fase/reporte_<nombre_fase>.md` o `reporte_academico_<nombre_fase>.md`
- `NN_nombre_fase/contexto_<nombre_fase>.md` solo si no existe un reporte narrativo suficiente

Los outputs intermedios o finales de una fase no deben escribirse en una carpeta global `artifacts/` si pueden vivir dentro de la carpeta de la fase. Esto facilita continuidad, revision y transferencia de contexto entre fases.

## Reporte Como Contexto De Continuidad
El reporte academico o reporte de fase es el contexto principal para humanos y agentes de IA. Debe ser autosuficiente: explicar que hizo el codigo, que entradas uso, que resultados obtuvo, como se interpretan esos resultados, que decisiones metodologicas se tomaron y que debe considerar la fase siguiente.

No se debe crear un `contexto_<fase>.md` separado si solo resume o duplica el reporte. El contexto compacto queda permitido unicamente cuando una fase no tenga reporte narrativo o cuando se necesite una version abreviada para transferencia rapida; en ese caso debe apuntar al reporte completo y no reemplazarlo.

## Separacion Entre Codigo Y Narrativa
Los notebooks y scripts de ejecucion no deben generar, leer ni validar reportes academicos, contextos narrativos ni archivos `.md`. Su responsabilidad es producir y validar artefactos tecnicos: datos, tablas, metricas, graficos y metadatos operativos.

Los reportes y contextos narrativos se escriben fuera del notebook, despues de revisar los outputs. Son documentos de interpretacion para personas y agentes de IA, no dependencias de ejecucion del codigo.

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

## Regla De Continuidad Entre Fases
Antes de iniciar una nueva fase, se debe leer el reporte de continuidad de la fase anterior. Ese reporte no reemplaza la revision de codigo ni la validacion de outputs, pero sirve para mantener en memoria las decisiones, supuestos, hallazgos, interpretaciones y riesgos metodologicos ya detectados.

Esta lectura es una practica de trabajo del agente o de la persona responsable, no un paso que deba implementarse dentro del notebook o script de la fase.

Ejemplos:
- Antes del EDA, leer `01_generacion/contexto_generacion_datos.md`.
- Antes de preparacion, leer `02_eda/reporte_academico_eda_cobranzas.md`.
- Antes de modelado, leer el reporte final de preparacion dentro de `03_preparacion/`.

## Criterios De Calidad
- Las rutas deben ser relativas al proyecto.
- Cada notebook debe validar sus entradas antes de continuar.
- Cada etapa debe imprimir un resumen final de sus outputs.
- Las salidas deben estar documentadas en el reporte de continuidad de la fase.

## Proxima Expansion
Despues de la fase de preparacion, este contexto debe servir para nuevas etapas de modelado, evaluacion y presentacion sin releer todos los notebooks.

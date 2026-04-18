# Salidas Del Sistema

## 1. Generacion De Datos

`clientes.csv`
: Catálogo base de clientes simulados. Sirve para conocer perfil de pago, sector, antiguedad y si el cliente tiene garantia.

`facturas.csv`
: Tabla central del negocio. Cada fila representa una factura con su fecha, vencimiento, mora real y clase objetivo.

`gestiones_cobranza.csv`
: Historial de contactos y acciones de cobranza. Se usa para estudiar canal, resultado y secuencia operativa.

`promesas_pago.csv`
: Registros de promesas derivadas de gestiones exitosas. Permite analizar cumplimiento y rotura de promesas.

`features_ml.csv`
: Dataset de modelado por cortes temporales. Una misma factura puede aparecer varias veces, una por cada momento de scoring.

## 2. EDA

`eda_resumen.md` o reporte equivalente
: Sintesis narrativa de hallazgos, calidad de datos y relaciones relevantes.

`eda_metricas.json` o `.csv`
: Indicadores estructurales y estadisticos como nulos, cardinalidad, balance de clases e integridad referencial.

Gráficos exportados
: Evidencia visual para la presentacion y el informe academico. Deben corresponder a distribuciones, outliers, correlaciones y relaciones de negocio.

## 3. Preparacion

`features_ml_prepared.csv` o `.parquet`
: Dataset final depurado y enriquecido, listo para entrenar modelos.

`train_facturas_ids.csv`
: Lista de facturas asignadas a entrenamiento. Es la referencia del split sin fuga.

`test_facturas_ids.csv`
: Lista de facturas asignadas a prueba. Debe ser disjunta de train.

`outlier_summary.csv`
: Resumen de valores extremos por variable con criterio IQR. No elimina datos, solo documenta su presencia.

`preprocessing_metadata.json`
: Bitacora tecnica del proceso. Resume validaciones, columnas derivadas, esquema del preprocesador y pesos de clase.

## Lectura Rapida

- Si necesitas trazabilidad de origen, revisa `facturas.csv` y `gestiones_cobranza.csv`.
- Si necesitas entendimiento del scoring por momento, revisa `features_ml.csv`.
- Si necesitas entrenamiento, usa `features_ml_prepared.csv` y los IDs de split.
- Si necesitas contexto tecnico del pipeline, usa `preprocessing_metadata.json` y `AGENTS.md`.


## Correccion aplicada

Se detecto una inconsistencia de nombres:

- `facturas.csv` usaba `target_mora`.
- `features_ml.csv` usaba `target`.

Se corrigio para que `features_ml.csv` tambien use `target_mora`. Esto evita ambiguedad en preparacion, modelado y evaluacion.

Tambien se alineo la estructura de carpetas:

- Notebook: `01_generacion/simulacion_datos.ipynb`
- Outputs: `01_generacion/data/`
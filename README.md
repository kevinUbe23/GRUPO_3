# Sistema inteligente de priorizacion de cobranzas

Proyecto academico para priorizar facturas por riesgo de mora en empresas que venden a credito. El flujo combina generacion de datos sinteticos, EDA, preparacion de datos, evaluacion de modelos, pruebas de robustez y una demo full stack con backend FastAPI y frontend Next.js.

La unidad de negocio principal es la `factura`. El dataset de modelado usa cortes temporales por `factura_id` y `fecha_corte`, por lo que el split de entrenamiento y prueba debe hacerse siempre por `factura_id`, nunca por fila.

## Estructura del proyecto

| Ruta | Contenido principal |
|---|---|
| `01_generacion/` | Notebook de simulacion y datos sinteticos base. |
| `02_eda/` | Analisis exploratorio, validaciones y reporte academico del EDA. |
| `03_preparacion/` | Preparacion de datos, split oficial y artefactos listos para modelado. |
| `04_evaluacion_modelos_ia/` | Benchmark supervisado, clustering de clientes y artefacto del modelo seleccionado. |
| `05_escenarios_criticos_hiperparametros/` | Robustez, drift, pruebas de estres y sensibilidad de hiperparametros. |
| `backend/` | API FastAPI para inicializacion, scoring, cola priorizada y operaciones de cobranza. |
| `frontend/` | Interfaz Next.js para explorar dashboard, clientes, facturas y predicciones. |

## Flujo academico

El orden recomendado de ejecucion es:

1. Ejecutar `01_generacion/simulacion_datos.ipynb`.
2. Ejecutar `02_eda/notebook_eda_cobranzas.ipynb`.
3. Ejecutar `03_preparacion/notebook_preparacion.ipynb`.
4. Ejecutar `04_evaluacion_modelos_ia/notebook_evaluacion_modelos.ipynb`.
5. Ejecutar `05_escenarios_criticos_hiperparametros/notebook_fase5_escenarios_criticos_hiperparametros.ipynb`.

Antes de iniciar una fase, revisar el reporte de continuidad de la fase anterior. Estos reportes documentan entradas, decisiones metodologicas, outputs y riesgos para la siguiente etapa.

## Artefactos principales

| Artefacto | Uso |
|---|---|
| `01_generacion/data/clientes.csv` | Clientes sinteticos. |
| `01_generacion/data/facturas.csv` | Facturas sinteticas. |
| `01_generacion/data/gestiones_cobranza.csv` | Historial de gestiones. |
| `01_generacion/data/promesas_pago.csv` | Promesas de pago simuladas. |
| `01_generacion/data/features_ml.csv` | Dataset analitico por factura y corte temporal. |
| `03_preparacion/outputs/features_ml_prepared.csv` | Dataset preparado para modelado. |
| `03_preparacion/outputs/train_facturas_ids.csv` | Facturas oficiales de entrenamiento. |
| `03_preparacion/outputs/test_facturas_ids.csv` | Facturas oficiales de prueba. |
| `04_evaluacion_modelos_ia/outputs/best_model_artifact.joblib` | Modelo seleccionado para prediccion. |
| `04_evaluacion_modelos_ia/outputs/model_feature_schema.csv` | Schema requerido por el modelo. |
| `04_evaluacion_modelos_ia/outputs/frontend_customer_segments.csv` | Segmentacion y rating de clientes para la demo. |

## Entorno Python y kernel de notebooks

Desde la raiz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name grupo3-cobranzas --display-name "Grupo 3 Cobranzas"
```

Despues de ejecutar esos comandos, seleccionar el kernel **Grupo 3 Cobranzas** en VS Code o Jupyter para correr los notebooks del proyecto.

Notas:

- `.venv/` no se sube a Git; cada computadora debe reconstruirlo con `requirements.txt`.
- El kernel queda registrado localmente en la maquina del usuario.
- Todos los notebooks del repositorio deben usar el mismo kernel para evitar diferencias entre fases.

## Ejecutar la demo local

La demo usa los artefactos generados por las fases previas. Antes de levantarla, verificar que existan los archivos de `01_generacion/data/`, `03_preparacion/outputs/` y `04_evaluacion_modelos_ia/outputs/`.

### Backend

Desde la raiz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API local:

```text
http://localhost:8000/api/v1
```

Documentacion interactiva:

```text
http://localhost:8000/docs
```

Inicializar la base operativa:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/admin/init-db
```

La base SQLite local se crea en `backend/data/cobranzas.db`.

### Frontend

En otra terminal:

```powershell
cd frontend
npm install
copy .env.example .env.local
npm run dev
```

Abrir:

```text
http://localhost:3000
```

Por defecto, el frontend consume:

```text
http://127.0.0.1:8000/api/v1
```

## Despliegue

La opcion documentada para demo es:

- Backend FastAPI en Render, usando `render.yaml`.
- Frontend Next.js en Vercel.

Las instrucciones completas estan en `DEPLOYMENT.md`.

## Reglas metodologicas clave

- La prediccion supervisada se evalua por `factura_id` y `fecha_corte`.
- `features_ml.csv` y `features_ml_prepared.csv` contienen multiples cortes por factura.
- El split correcto usa `train_facturas_ids.csv` y `test_facturas_ids.csv`.
- `factura_id`, `cliente_id` y `fecha_corte` son trazabilidad; no deben entrar como predictores.
- `target_mora` es etiqueta, no feature.
- La segmentacion de clientes es un componente separado: opera por `cliente_id` y no usa el target como variable de entrada.
- El score `priority_score_0_100` es un score operativo para ordenar facturas, no una probabilidad directa ni un castigo automatico.
- Las acciones sugeridas son reglas de negocio posteriores al modelo y deben quedar sujetas a revision humana.

## Reportes de continuidad

Los documentos narrativos principales son:

- `01_generacion/contexto_generacion_datos.md`
- `02_eda/reporte_academico_eda_cobranzas.md`
- `03_preparacion/reporte_preparacion.md`
- `04_evaluacion_modelos_ia/reporte_evaluacion_modelos_ia.md`
- `05_escenarios_criticos_hiperparametros/reporte_escenarios_criticos_hiperparametros.md`

Estos reportes explican que hizo cada fase, que entradas uso, que resultados obtuvo, como se interpretan y que debe cuidar la siguiente etapa.

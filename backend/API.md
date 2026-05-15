# API Backend - Sistema de priorizacion de cobranzas

Documentacion para integracion del frontend con el backend FastAPI.

## Base URL

Entorno local por defecto:

```text
http://localhost:8000/api/v1
```

FastAPI tambien expone documentacion interactiva en:

```text
http://localhost:8000/docs
http://localhost:8000/openapi.json
```

El backend permite CORS desde origenes locales de desarrollo:

```text
http://localhost:3000
http://127.0.0.1:3000
http://localhost:<puerto>
http://127.0.0.1:<puerto>
```

## Flujo de inicializacion

1. Verificar que existan los artefactos generados por las fases previas:
   - `01_generacion/data/clientes.csv`
   - `01_generacion/data/facturas.csv`
   - `01_generacion/data/gestiones_cobranza.csv`
   - `01_generacion/data/promesas_pago.csv`
   - `03_preparacion/outputs/features_ml_prepared.csv`
   - `04_evaluacion_modelos_ia/outputs/best_model_artifact.joblib`
   - `04_evaluacion_modelos_ia/outputs/model_feature_schema.csv`
   - `04_evaluacion_modelos_ia/outputs/frontend_customer_segments.csv`
2. Levantar el backend.
3. Inicializar o recargar la base con `POST /admin/init-db`.
4. Consultar catalogos y resumen.
5. Ejecutar scoring por factura con `POST /invoices/{factura_id}/score`.
6. Consumir `GET /invoices/prioritized` para ver facturas abiertas ordenadas por prioridad. Este endpoint solo lista facturas activas que ya tienen una prediccion persistida.

Importante: `POST /admin/init-db` resetea e importa la data semilla. Tambien elimina predicciones previamente guardadas.

## Comandos para correr

Desde la raiz del proyecto, en PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Inicializar la base desde HTTP:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/admin/init-db
```

Inicializar la base desde script:

```powershell
cd backend
..\.venv\Scripts\python.exe -m app.scripts.init_db
```

Base de datos por defecto:

```text
backend/data/cobranzas.db
```

Variable opcional para cambiar la conexion:

```powershell
$env:COBRANZAS_DATABASE_URL = "sqlite:///C:/ruta/cobranzas.db"
```

## Convenciones de integracion

- No hay autenticacion implementada actualmente.
- Todas las fechas viajan como strings ISO `YYYY-MM-DD`.
- Los montos y probabilidades viajan como numeros JSON.
- Los errores de negocio del scoring devuelven `400` con `{ "detail": "..." }`.
- Los recursos no encontrados devuelven `404` con `{ "detail": "..." }`.
- Errores de validacion de parametros o payload devuelven `422`.
- `limit` acepta `1..500`; `offset` acepta `0..n`.

## Endpoints disponibles

| Metodo | Endpoint | Uso |
| --- | --- | --- |
| `POST` | `/admin/init-db` | Reinicia e importa data semilla a la base operativa. |
| `GET` | `/dashboard/summary` | KPIs generales para dashboard. |
| `GET` | `/customers` | Lista clientes con paginacion y filtro opcional por sector. |
| `GET` | `/customers/{cliente_id}` | Detalle de cliente. |
| `GET` | `/customers/{cliente_id}/segment` | Segmentacion, rating y explicacion del cliente. |
| `GET` | `/invoices` | Lista facturas con filtros. |
| `POST` | `/invoices` | Crea una factura operativa. |
| `GET` | `/invoices/prioritized` | Lista facturas con prediccion persistida, ordenadas por prioridad. |
| `GET` | `/invoices/{factura_id}` | Detalle de factura. |
| `PATCH` | `/invoices/{factura_id}` | Actualiza una factura operativa. |
| `GET` | `/invoices/{factura_id}/interactions` | Gestiones historicas de cobranza de la factura. |
| `POST` | `/invoices/{factura_id}/score` | Calcula prediccion, score y accion sugerida para una factura. |
| `POST` | `/scoring/recalculate` | Recalcula scoring por lote para facturas activas a una fecha de corte. |
| `POST` | `/collections/interactions` | Registra una gestion de cobranza. |
| `POST` | `/payment-promises` | Registra una promesa nacida de una gestion. |
| `PATCH` | `/payment-promises/{promesa_id}` | Actualiza estado de una promesa. |
| `POST` | `/payments` | Registra pago total de una factura. |
| `GET` | `/actions` | Lista catalogo de acciones sugeridas. |
| `POST` | `/actions/recommend` | Calcula recomendacion sin persistir prediccion. |
| `GET` | `/model/status` | Estado basico del modelo cargado. |
| `GET` | `/model/metrics` | Metricas del modelo seleccionado. |

## Admin

### POST `/admin/init-db`

Carga clientes, facturas, gestiones, promesas y segmentos desde los CSV/artefactos canonicos.

Request: sin body.

Respuesta `200`:

```json
{
  "clientes": 200,
  "facturas": 5338,
  "gestiones": 14333,
  "promesas": 1741,
  "segmentos": 200
}
```

## Dashboard

### GET `/dashboard/summary`

Query params:

| Parametro | Tipo | Default | Notas |
| --- | --- | --- | --- |
| `fecha_corte` | date/null | null | Si se envia, calcula cartera activa historica a esa fecha. |

Respuesta `200`:

```json
{
  "total_facturas": 5338,
  "facturas_activas": 1320,
  "monto_pendiente": 1234567.89,
  "monto_vencido": 456789.12,
  "promesas_activas": 0,
  "facturas_en_disputa": 0
}
```

Campos:

- `total_facturas`: total de facturas importadas.
- `facturas_activas`: facturas con `estado_factura = "abierta"`.
- `monto_pendiente`: suma de `saldo_pendiente`.
- `monto_vencido`: suma pendiente de facturas abiertas con vencimiento anterior a hoy.
- `promesas_activas`: promesas con `estado_promesa = "activa"`.
- `facturas_en_disputa`: facturas con `estado_factura = "en_disputa"`.

## Clientes

### GET `/customers`

Query params:

| Parametro | Tipo | Default | Notas |
| --- | --- | --- | --- |
| `limit` | integer | `50` | Min `1`, max `500`. |
| `offset` | integer | `0` | Min `0`. |
| `sector` | string | null | Filtro exacto por sector. |

Ejemplo:

```text
GET /customers?limit=20&offset=0&sector=agro
```

Respuesta `200`:

```json
[
  {
    "cliente_id": "CLI0001",
    "nombre": "Alvarado-Laureano",
    "sector": "agro",
    "antiguedad_meses": 17,
    "tiene_garantia": true
  }
]
```

### GET `/customers/{cliente_id}`

Respuesta `200`:

```json
{
  "cliente_id": "CLI0001",
  "nombre": "Alvarado-Laureano",
  "sector": "agro",
  "antiguedad_meses": 17,
  "tiene_garantia": true
}
```

Errores:

```json
{ "detail": "Cliente no encontrado" }
```

### GET `/customers/{cliente_id}/segment`

Respuesta `200`:

```json
{
  "cliente_id": "CLI0001",
  "cluster": 1,
  "tipo_cliente": "Cliente estable de riesgo medio",
  "riesgo_0_100": 42.5,
  "rating_estrellas": 3,
  "rating_label": "Medio",
  "por_que_rating": "Explicacion del rating del cliente.",
  "por_que_cluster": "Explicacion del cluster asignado.",
  "sector_dominante_modal": "agro",
  "n_facturas_total": 24,
  "n_cortes_total": 120
}
```

Errores:

```json
{ "detail": "Segmento no encontrado" }
```

## Facturas

### GET `/invoices`

Query params:

| Parametro | Tipo | Default | Notas |
| --- | --- | --- | --- |
| `limit` | integer | `50` | Min `1`, max `500`. |
| `offset` | integer | `0` | Min `0`. |
| `active_only` | boolean | `false` | Si es `true`, devuelve solo facturas abiertas. |
| `cliente_id` | string | null | Filtro exacto por cliente. |
| `fecha_corte` | date/null | null | Si se combina con `active_only=true`, devuelve facturas activas a esa fecha historica. |

Ejemplo:

```text
GET /invoices?active_only=true&fecha_corte=2023-01-30&cliente_id=CLI0001&limit=20
```

Respuesta `200`:

```json
[
  {
    "factura_id": "FAC000001",
    "cliente_id": "CLI0051",
    "fecha_emision": "2023-03-02",
    "fecha_vencimiento": "2023-04-01",
    "fecha_pago_real": "2023-04-01",
    "condicion_dias": 30,
    "monto": 8050.57,
    "saldo_pendiente": 0.0,
    "estado_factura": "pagada",
    "target_mora_simulado": "on_time",
    "dias_mora_real": 0
  }
]
```

### GET `/invoices/{factura_id}`

Respuesta `200`: mismo objeto que `GET /invoices`.

Errores:

```json
{ "detail": "Factura no encontrada" }
```

### POST `/invoices`

Crea una factura en la base operativa. Si `factura_id` se omite, el backend genera un identificador con prefijo `FACAPP`.

Request:

```json
{
  "factura_id": "FACAPP000001",
  "cliente_id": "CLI0001",
  "fecha_emision": "2024-01-10",
  "fecha_vencimiento": "2024-02-09",
  "fecha_pago_real": null,
  "condicion_dias": 30,
  "monto": 1500.0,
  "saldo_pendiente": 1500.0,
  "estado_factura": "abierta",
  "target_mora_simulado": null,
  "dias_mora_real": null
}
```

Campos opcionales:

- `factura_id`: si no se envia, se autogenera.
- `condicion_dias`: si no se envia, se calcula como diferencia entre vencimiento y emision.
- `saldo_pendiente`: si no se envia, usa `monto` para facturas activas y `0` para facturas anuladas o castigadas.
- `fecha_pago_real`: si se envia, la factura queda como `pagada`, `saldo_pendiente = 0` y `dias_mora_real` se calcula automaticamente.

Validaciones:

- `cliente_id` debe existir.
- `fecha_vencimiento` debe ser mayor o igual a `fecha_emision`.
- `fecha_pago_real` no puede ser anterior a `fecha_emision`.
- `estado_factura` debe ser uno de `abierta`, `pagada`, `en_disputa`, `anulada`, `castigada`.
- `saldo_pendiente` no puede ser mayor que `monto`.

### PATCH `/invoices/{factura_id}`

Actualiza parcialmente una factura. Acepta campos operativos de factura, excepto `factura_id`. El `cliente_id` se conserva para no desalinear gestiones, promesas y predicciones historicas ya asociadas a la factura.

Ejemplo:

```json
{
  "fecha_pago_real": "2024-02-15"
}
```

Si se actualiza `fecha_pago_real`, el backend marca la factura como `pagada`, lleva `saldo_pendiente` a `0` y recalcula `dias_mora_real`.

Al editar una factura, el backend elimina predicciones persistidas de esa factura para evitar que la cola priorizada use scores calculados con datos anteriores. Si la factura debe volver a aparecer en `/invoices/prioritized`, se debe ejecutar nuevamente el scoring.

Campos no anulables en PATCH: `cliente_id`, `fecha_emision`, `fecha_vencimiento`, `condicion_dias`, `monto`, `saldo_pendiente` y `estado_factura`. Si se quieren dejar sin valor, solo pueden enviarse como `null` los campos que el modelo de datos permite nulos: `fecha_pago_real`, `target_mora_simulado` y `dias_mora_real`.

Errores:

```json
{ "detail": "No existe la factura FAC999999" }
```

```json
{ "detail": "estado_factura invalido: pendiente" }
```

```json
{ "detail": "cliente_id no puede cambiarse en una factura existente" }
```

### GET `/invoices/{factura_id}/interactions`

Respuesta `200`:

```json
[
  {
    "gestion_id": "GES000001",
    "fecha_gestion": "2023-04-05",
    "canal": "telefono",
    "contacto_exitoso": true,
    "resultado": "promesa_pago",
    "motivo_no_pago": "flujo_caja",
    "dias_mora_en_gestion": 4
  }
]
```

Nota: si la factura no tiene gestiones, devuelve `[]`.

### GET `/invoices/prioritized`

Devuelve facturas abiertas con prediccion persistida, ordenadas por `priority_score_0_100` descendente. Las facturas pagadas, anuladas o castigadas no aparecen en esta cola.

Query params:

| Parametro | Tipo | Default | Notas |
| --- | --- | --- | --- |
| `limit` | integer | `50` | Min `1`, max `500`. |
| `fecha_corte` | date | null | Si se envia, devuelve la ultima prediccion persistida para esa fecha de corte. |

Respuesta `200`:

```json
[
  {
    "factura_id": "FAC000123",
    "cliente_id": "CLI0001",
    "cliente_nombre": "Alvarado-Laureano",
    "sector": "agro",
    "monto": 25000.0,
    "fecha_vencimiento": "2024-07-30",
    "estado_factura": "abierta",
    "estado_factura_actual": "pagada",
    "fecha_corte": "2024-07-15",
    "predicted_label_usuario": "Atraso alto probable",
    "prob_pago_plazo": 0.18,
    "prob_atraso_leve": 0.19,
    "prob_atraso_alto": 0.43,
    "prob_atraso_critico": 0.20,
    "any_late_probability": 0.82,
    "high_risk_probability": 0.63,
    "priority_score_0_100": 68.4,
    "accion_sugerida": "Programar visita de cobranza",
    "rating_estrellas": 2
  }
]
```

## Scoring

### POST `/scoring/recalculate`

Recalcula predicciones para facturas activas a una `fecha_corte`. En datos simulados, una factura pagada en el futuro se considera activa si `fecha_pago_real > fecha_corte`.

Request:

```json
{
  "fecha_corte": "2023-01-30",
  "limit": 100,
  "persist": true
}
```

Respuesta `200`:

```json
{
  "fecha_corte": "2023-01-30",
  "total_evaluadas": 100,
  "total_con_error": 0,
  "errores": []
}
```

### POST `/invoices/{factura_id}/score`

Calcula prediccion de mora, probabilidades, score de prioridad y accion sugerida.

Request body:

```json
{
  "fecha_corte": "2024-08-15",
  "persist": true
}
```

Campos:

| Campo | Tipo | Default | Notas |
| --- | --- | --- | --- |
| `fecha_corte` | date/null | hoy | Fecha de scoring. Si coincide con un corte preparado, usa `features_ml_prepared.csv`; si no, construye features desde la base operativa. |
| `persist` | boolean | `true` | Si es `true`, guarda la prediccion en `predicciones_factura`. Necesario para que aparezca en `/invoices/prioritized`. |

Respuesta `200`:

```json
{
  "factura_id": "FAC000123",
  "cliente_id": "CLI0001",
  "fecha_corte": "2024-08-15",
  "modelo_version": "fase4_logistic_regression",
  "predicted_class_tecnica": "+60",
  "predicted_label_usuario": "Atraso alto probable",
  "prob_pago_plazo": 0.12,
  "prob_atraso_leve": 0.25,
  "prob_atraso_alto": 0.43,
  "prob_atraso_critico": 0.2,
  "any_late_probability": 0.88,
  "high_risk_probability": 0.63,
  "priority_score_0_100": 62.1,
  "accion_sugerida": {
    "codigo": "VISITA_CLIENTE",
    "nombre": "Programar visita de cobranza",
    "canal_recomendado": "Visita",
    "severidad": 6,
    "motivo": "La factura tiene mora alta y senales de riesgo operativo.",
    "regla": "mora_31_riesgo_o_no_contacto"
  },
  "feature_source": "prepared_features_snapshot"
}
```

Valores relevantes:

- `predicted_class_tecnica`: clase tecnica del modelo. Valores esperados: `on_time`, `+30`, `+60`, `+90`.
- `predicted_label_usuario`: etiqueta lista para UI.
- `any_late_probability`: suma de probabilidades de atraso `+30`, `+60`, `+90`.
- `high_risk_probability`: suma de probabilidades `+60` y `+90`.
- `priority_score_0_100`: score operativo para ordenar cobranza.
- `accion_sugerida.severidad`: escala operativa; mayor valor implica mayor urgencia.
- `feature_source`: `prepared_features_snapshot` si encontro el corte exacto en features preparadas; `operational_db` si lo construyo desde tablas operativas.

Errores:

```json
{ "detail": "No existe la factura FAC999999" }
```

```json
{ "detail": "fecha_corte no puede ser anterior a fecha_emision" }
```

```json
{ "detail": "Faltan features para predecir: [...]" }
```

Notas para frontend:

- Para priorizar un lote, el frontend puede listar facturas abiertas con `GET /invoices?active_only=true` y luego llamar `/score` por cada factura con `persist=true`.
- Despues del scoring persistido, refrescar `GET /invoices/prioritized`.
- Si se quiere previsualizar sin guardar ni alterar el ranking persistido, usar `persist=false`.
- El endpoint carga el modelo en memoria de forma lazy en la primera llamada; la primera respuesta puede tardar mas.

## Modelo

### GET `/model/status`

Respuesta `200`:

```json
{
  "modelo_version": "fase4_logistic_regression",
  "feature_count": 39,
  "class_names": ["+30", "+60", "+90", "on_time"]
}
```

Usar este endpoint para validar que el backend puede cargar el artefacto del modelo antes de habilitar acciones de scoring en UI.

### GET `/model/metrics`

Devuelve la fila de metricas del modelo `Logistic Regression` desde `benchmark_metrics.csv`.

## Gestiones, promesas y pagos

### POST `/collections/interactions`

Registra una gestion de cobranza.

```json
{
  "factura_id": "FAC000002",
  "fecha_gestion": "2023-03-03",
  "canal": "llamada",
  "contacto_exitoso": true,
  "resultado": "promesa_de_pago",
  "motivo_no_pago": "flujo_caja",
  "observacion": "Cliente confirma pago la proxima semana.",
  "recalculate": true
}
```

`resultado` debe ser coherente con `contacto_exitoso`. Si `recalculate=true`, el backend recalcula la factura usando `fecha_gestion` como `fecha_corte`.

### POST `/payment-promises`

Registra una promesa a partir de una gestion con `resultado = "promesa_de_pago"`.

```json
{
  "gestion_id": "GESAPP014334",
  "fecha_compromiso": "2023-03-10",
  "recalculate": true
}
```

### PATCH `/payment-promises/{promesa_id}`

Actualiza el estado de una promesa.

```json
{
  "estado_promesa": "cumplida",
  "se_cumplio": true
}
```

Estados validos: `activa`, `cumplida`, `incumplida`, `reemplazada`, `cancelada`.

### POST `/payments`

Registra pago total de una factura.

```json
{
  "factura_id": "FAC000002",
  "fecha_pago": "2023-03-06"
}
```

La factura pasa a `pagada`, `saldo_pendiente` queda en `0`, y las promesas activas de esa factura se marcan como cumplidas o incumplidas segun la fecha de compromiso.

## Acciones

### GET `/actions`

Devuelve el catalogo de acciones sugeridas disponible para el front.

### POST `/actions/recommend`

Calcula prediccion y accion sugerida sin persistir. Usa query params:

```text
POST /actions/recommend?factura_id=FAC000002&fecha_corte=2023-03-03
```

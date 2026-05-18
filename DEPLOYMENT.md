# Despliegue del sistema de cobranzas

Este proyecto tiene dos servicios:

- Backend FastAPI en `backend/`.
- Frontend Next.js en `frontend/`.

## Opcion recomendada para demo

Usar Render para el backend y Vercel para el frontend.

### 1. Subir el repositorio

Publicar este repositorio en GitHub, GitLab o Bitbucket. Los artefactos de datos y modelo que necesita el backend deben estar versionados:

- `01_generacion/data/`
- `03_preparacion/outputs/`
- `04_evaluacion_modelos_ia/outputs/`

La base SQLite local `backend/data/cobranzas.db` no se versiona; se inicializa desde la interfaz o desde el endpoint del backend.

### 2. Backend en Render

Crear un Blueprint en Render usando `render.yaml` desde la raiz del repositorio, o crear manualmente un Web Service con estos valores:

- Root directory: `backend`
- Runtime: `Python`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

Render entregara una URL parecida a:

```text
https://cobranzas-backend.onrender.com
```

El API quedara en:

```text
https://cobranzas-backend.onrender.com/api/v1
```

### 3. Frontend en Vercel

Crear un proyecto en Vercel apuntando al mismo repositorio:

- Root directory: `frontend`
- Framework preset: `Next.js`
- Build command: `npm run build`
- Install command: `npm install`

Agregar esta variable de entorno en Vercel antes de desplegar:

```text
NEXT_PUBLIC_API_BASE_URL=https://cobranzas-backend.onrender.com/api/v1
```

Reemplazar la URL por la URL real del backend en Render.

### 4. Inicializar datos

Abrir la URL publica del frontend y usar el boton `Inicializar`.

Tambien se puede llamar directamente:

```text
POST https://cobranzas-backend.onrender.com/api/v1/admin/init-db
```

## Nota sobre persistencia

Para una revision de tesis, SQLite en el filesystem del servicio es suficiente si se puede reinicializar la data. Si se necesita conservar cambios entre despliegues o reinicios, conviene agregar un disco persistente en Render o migrar `COBRANZAS_DATABASE_URL` a PostgreSQL.

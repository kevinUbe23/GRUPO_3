# Despliegue del sistema de cobranzas

Este proyecto tiene dos servicios:

- Backend FastAPI en `backend/`.
- Frontend Next.js en `frontend/`.

## Opcion recomendada para Portainer

Usar `docker-compose.yml` como Stack en Portainer. El stack levanta:

- `proxy`: Caddy, expone una sola URL publica.
- `frontend`: Next.js.
- `backend`: FastAPI.
- `cobranzas_backend_data`: volumen para la base SQLite.

Con esta opcion, la profesora abre una sola URL. Caddy enruta:

- `/api/*` hacia el backend.
- Todo lo demas hacia el frontend.

### 1. Subir el repositorio

Publicar este repositorio en GitHub, GitLab o Bitbucket. Los artefactos de datos y modelo que necesita el backend deben estar versionados:

- `01_generacion/data/`
- `03_preparacion/outputs/`
- `04_evaluacion_modelos_ia/outputs/`

La base SQLite local `backend/data/cobranzas.db` no se versiona; en Docker vive en el volumen `cobranzas_backend_data` y se inicializa desde la interfaz o desde el endpoint del backend.

### 2. Crear Stack en Portainer

En Portainer:

1. Ir a `Stacks`.
2. Crear un stack nuevo.
3. Usar la opcion de repositorio Git y apuntar al repo.
4. Compose path: `docker-compose.yml`.
5. Agregar variables de entorno. Como tu puerto `80` ya esta ocupado, usa por ejemplo `8080`:

```text
WEB_PORT=8080
PUBLIC_API_BASE_URL=/api/v1
```

No activar una opcion tipo `Pull latest image`, `Re-pull image` o `Pull and redeploy` para este stack. El frontend y el backend no son imagenes publicadas en Docker Hub; Portainer debe construirlas desde `backend/Dockerfile` y `frontend/Dockerfile`.

Si el puerto `8080` tambien esta ocupado, puedes cambiarlo por otro puerto libre:

```text
WEB_PORT=8081
PUBLIC_API_BASE_URL=/api/v1
```

Para probarlo fuera de Portainer, en un servidor con Docker instalado:

```bash
docker compose --env-file .env.docker.example up --build -d
docker compose logs -f
```

### 3. Abrir la URL publica

Con `WEB_PORT=8080`:

```text
http://IP_O_DOMINIO_DEL_SERVIDOR:8080
```

### 4. Inicializar datos

Abrir la URL publica del frontend y usar el boton `Inicializar`.

Tambien se puede llamar directamente:

```text
POST http://IP_O_DOMINIO_DEL_SERVIDOR:8080/api/v1/admin/init-db
```

## Variante Render + Vercel

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

En Portainer, SQLite queda en el volumen Docker `cobranzas_backend_data`. Ese volumen sobrevive reinicios del contenedor, pero se puede perder si se elimina el stack junto con sus volumenes.

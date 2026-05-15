# Frontend - Sistema de priorizacion de cobranzas

App Next.js para consumir el backend FastAPI del proyecto.

## Requisitos

- Node.js 20 o superior.
- Backend corriendo en `http://127.0.0.1:8000/api/v1`.

## Configuracion

Desde la raiz del proyecto:

```powershell
cd frontend
copy .env.example .env.local
npm install
npm run dev
```

Abrir:

```text
http://localhost:3000
```

Si Next.js usa otro puerto local porque `3000` esta ocupado, el backend tambien acepta origenes `localhost` y `127.0.0.1` en puertos alternos de desarrollo.

## Backend esperado

Por defecto usa:

```text
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api/v1
```

Endpoints principales usados:

- `GET /dashboard/summary?fecha_corte=YYYY-MM-DD`
- `POST /admin/init-db`
- `POST /scoring/recalculate`
- `GET /invoices/prioritized`
- `GET /invoices/{factura_id}`
- `GET /invoices/{factura_id}/interactions`
- `POST /invoices/{factura_id}/score`
- `GET /customers/{cliente_id}`
- `GET /customers/{cliente_id}/segment`

## Estructura

- `app/page.tsx`: entrada de la ruta principal.
- `components/dashboard/`: layout operativo, sidebar, toolbar, resumen, tabla paginada y panel de detalle.
- `components/ui/`: componentes base instalados con shadcn/ui.
- `lib/api.ts`: cliente HTTP del backend.
- `lib/types.ts`: tipos compartidos por la pantalla.
- `lib/formatters.ts`: formatos de dinero, fecha, porcentaje, prioridad y riesgo historico.

## UI

El frontend usa Next.js App Router, React 19, Tailwind CSS 4 y shadcn/ui. La tabla de la cola priorizada incluye busqueda, paginacion cliente y selector de filas por pagina. El menu principal vive en sidebar fijo para desktop y en sheet lateral para mobile.

## Flujo de uso

1. Levantar backend.
2. Abrir frontend.
3. Click en `Inicializar`.
4. Elegir fecha de corte.
5. Click en `Recalcular`.
6. Seleccionar una factura de la cola.
7. Usar `Actualizar prediccion` para recalcular el caso seleccionado.

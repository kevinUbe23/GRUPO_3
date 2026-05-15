# AGENTS.md - Frontend

Este archivo complementa el `AGENTS.md` raiz. Si hay conflicto, prevalece la regla raiz del proyecto.

## Tecnologias
- Next.js App Router con React 19 y TypeScript.
- Tailwind CSS 4 con variables CSS en `app/globals.css`.
- shadcn/ui estilo `radix-nova`, componentes en `components/ui/`.
- Iconos con `lucide-react`.
- Cliente HTTP propio en `lib/api.ts`, usando `NEXT_PUBLIC_API_BASE_URL`.

## Contexto A Mantener
- El frontend representa un sistema academico de priorizacion de cobranzas; la unidad de negocio sigue siendo la `factura`.
- No introducir logica que mezcle train/test ni reglas de modelado; el frontend consume resultados del backend.
- Mantener coherencia con el PRD, las reglas de negocio y los reportes de fases cuando afecten texto, estados o metricas mostradas.
- No revertir cambios de otros agentes o personas. Si hay cambios ajenos, trabajar alrededor de ellos o preguntar si bloquean la tarea.

## Estructura Principal
- `app/`: rutas, layout global y estilos.
- `components/dashboard/`: pantalla operativa, sidebar, toolbar, resumen, tabla, panel de detalle y visualizaciones.
- `components/ui/`: componentes base shadcn/ui; preferir componerlos antes que crear markup visual nuevo.
- `lib/`: cliente API, tipos, utilidades y formateadores.
- `.env.example`: contrato minimo de variables publicas del frontend.

## Flujo De Trabajo
- Validar entradas, estados de carga, error y vacio en cambios de UI.
- Usar rutas relativas y aliases definidos en `components.json`.
- Antes de entregar codigo escrito en `frontend/`, usar un agente de review para revisar calidad, regresiones, accesibilidad basica y consistencia visual.

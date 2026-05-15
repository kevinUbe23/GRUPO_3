# PRD - Sistema inteligente de priorizacion de cobranzas

## 1. Resumen del producto

El sistema es una aplicacion web para priorizar cobranzas de empresas que venden a credito. El producto permite consultar clientes, facturas, gestiones y promesas de pago, y usa inteligencia artificial para estimar el riesgo de atraso de cada factura abierta.

La unidad principal del sistema es la factura. El cliente sirve como contexto historico y como fuente de segmentacion, pero la priorizacion operativa se decide por factura.

El MVP se enfoca en un unico usuario administrador. No se implementaran roles, permisos por perfil ni flujos separados para gestor, supervisor o auditor.

## 2. Objetivo del sistema

Ayudar a decidir que facturas deben gestionarse primero, que tan urgente es cada caso y que accion de cobranza conviene ejecutar segun:

- Riesgo estimado de atraso de la factura.
- Dias restantes para vencimiento o dias de mora observable.
- Historial de pago del cliente.
- Rating de cliente de 1 a 5 estrellas.
- Contactabilidad del cliente.
- Promesas de pago activas o incumplidas.
- Existencia de disputa activa.
- Numero e intensidad de gestiones previas.
- Monto de la factura.

El sistema no debe funcionar como castigo automatico ni como bloqueo comercial automatico. Su proposito es priorizar y recomendar acciones para revision humana.

## 3. Alcance del MVP

### Incluido

- Base de datos operativa con clientes, facturas, gestiones y promesas.
- Carga inicial desde los CSV ya generados por el proyecto.
- Dashboard general de cartera.
- Cola priorizada de facturas activas.
- Detalle de factura con prediccion, score, accion sugerida e historial.
- Detalle de cliente con rating, segmento y explicacion.
- Registro de nuevas gestiones de cobranza.
- Registro de promesas de pago.
- Registro de pago o cierre de factura.
- Recalculo de riesgo por factura.
- Recalculo de cartera por fecha de corte.
- Tabla de acciones sugeridas basada en reglas de negocio.
- Validacion de datos antes de predecir.
- Persistencia de predicciones y versiones de modelo.
- Vista auditora de IA con distribucion del score de prioridad, evolucion temporal de predicciones por factura y comparacion contra el resultado real cuando exista.

### No incluido en MVP

- Roles y permisos.
- Alertas por clientes de 1 o 2 estrellas como alerta independiente.
- Pagos parciales complejos.
- Refinanciaciones.
- Notas de credito.
- Anulaciones complejas.
- Integracion real con WhatsApp, email o SMS.
- Automatizacion real de envio de mensajes.
- Reentrenamiento de modelos desde la interfaz.

## 4. Usuario objetivo del MVP

El sistema tendra un unico usuario administrador. Este usuario puede:

- Ver toda la cartera.
- Crear, editar y consultar facturas.
- Consultar clientes.
- Registrar gestiones.
- Registrar promesas.
- Registrar pagos.
- Recalcular predicciones.
- Consultar segmentos y ratings.
- Revisar acciones sugeridas.

## 5. Conceptos principales

### Factura

Es la unidad de negocio y la unidad principal de priorizacion. Una factura puede tener cero, una o varias gestiones, y puede tener promesas de pago asociadas.

### Fecha de corte

Es la fecha en la que se calcula el riesgo. Responde la pregunta:

> Con la informacion disponible hasta esta fecha, que tan riesgosa es esta factura?

Si cambia la fecha de corte, deben recalcularse variables temporales como dias hasta vencimiento, dias de mora observable y dias desde la ultima gestion.

### Prediccion de riesgo

El modelo tecnico usa cuatro clases internas:

- `on_time`
- `+30`
- `+60`
- `+90`

Pero el front no debe mostrar esos nombres tecnicos al usuario final. Se usaran etiquetas orientadas a negocio.

| Clase tecnica | Etiqueta para usuario | Lectura de negocio |
|---|---|---|
| `on_time` | Pago esperado dentro del plazo | La factura probablemente se pague sin atraso relevante. |
| `+30` | Atraso leve probable | Podria pagarse hasta 30 dias despues del vencimiento. |
| `+60` | Atraso alto probable | Podria pagarse entre 31 y 60 dias despues del vencimiento. |
| `+90` | Atraso critico probable | En el dataset actual representa mora mayor a 60 dias. Es decir, 60+ dias de atraso, aunque la clase tecnica se llame `+90`. |

Decision de producto: en la interfaz se debe usar "Atraso critico probable" o "60+ dias de atraso probable", no `+90`, para evitar confusion.

### Score de prioridad

El score recomendado sigue la logica ya documentada:

```text
priority_score_0_100 = 100 * (0.40 * prob_atraso_leve + 0.70 * prob_atraso_alto + 1.00 * prob_atraso_critico)
```

Este score permite ordenar la cola de facturas. Una factura con alta probabilidad de atraso leve tambien debe subir en prioridad, porque cualquier pago posterior al plazo pactado afecta caja.

Lectura operativa sugerida:

| Rango de score | Nivel de prioridad | Lectura para el usuario |
|---:|---|---|
| 0-39 | Baja | Monitorear. No requiere gestion inmediata salvo que exista otra alerta operativa. |
| 40-59 | Media | Hacer gestion preventiva o revisar si esta proxima a vencer. |
| 60-79 | Alta | Priorizar seguimiento; conviene contacto activo o confirmacion de pago. |
| 80-100 | Critica | Gestion urgente; revisar accion sugerida y escalar si hay mora, promesa vencida o monto alto. |

#### Lectura para la interfaz y el informe

La interfaz debe separar cuatro conceptos para evitar ambiguedad:

| Concepto | Que responde | Uso en pantalla |
|---|---|---|
| Clase predicha | Cual es el escenario mas probable segun el modelo. | Mostrar como etiqueta principal: pago dentro del plazo, atraso leve, atraso alto o atraso critico. |
| Probabilidades por clase | Que probabilidad tiene cada escenario. | Mostrar cuatro barras: pago dentro del plazo, atraso leve, atraso alto y atraso critico / 60+. |
| Probabilidad de mora grave | Que tan probable es una mora alta o critica. | Mostrar como lectura auxiliar: `prob_atraso_alto + prob_atraso_critico`. |
| Score de prioridad | Que factura debe gestionarse primero. | Mostrar como indice 0-100 para ordenar la cola de cobranza. No es una probabilidad directa. |

Los tooltips deben explicar los rangos de dias de cada clase:

- Pago dentro del plazo: pago sin atraso relevante frente al vencimiento pactado.
- Atraso leve: pago hasta 30 dias despues del vencimiento.
- Atraso alto: pago entre 31 y 60 dias despues del vencimiento.
- Atraso critico / 60+: mora mayor a 60 dias. Aunque la clase tecnica sea `+90`, la lectura de negocio debe ser 60+ dias.

En la cola principal, el antiguo texto "Riesgo" debe reemplazarse por "Score de prioridad", porque la metrica pondera gravedad y probabilidad para ordenar trabajo operativo. La probabilidad de atraso debe mostrarse aparte. La clase predicha debe aparecer como una columna o etiqueta independiente llamada "Prediccion", para que se identifiquen rapidamente casos de "Pago esperado dentro del plazo".

### Rating de cliente

El rating va de 1 a 5 estrellas:

| Rating | Lectura |
|---:|---|
| 5 | Cliente de muy bajo riesgo historico. |
| 4 | Cliente de buen comportamiento. |
| 3 | Cliente intermedio. |
| 2 | Cliente riesgoso. |
| 1 | Cliente critico. |

El rating no debe mostrarse como alerta independiente. Debe usarse como contexto dentro de la ficha del cliente, detalle de factura, filtros y reglas de accion sugerida.

El `riesgo_0_100` del cliente no es la misma metrica que el `priority_score_0_100` de una factura. El riesgo del cliente es historico y contextual: resume comportamiento pasado, cumplimiento, mora, promesas y contacto. El score de prioridad es operativo y se calcula por factura para ordenar la cola diaria.

## 6. Pantallas del frontend

### 6.1 Dashboard de cartera

Debe mostrar:

- Total de facturas activas.
- Monto total pendiente.
- Monto pendiente vencido.
- Facturas por nivel de riesgo.
- Facturas por estado operativo.
- Distribucion de facturas por etiqueta de prediccion.
- Promesas activas.
- Promesas vencidas.
- Facturas en disputa.
- Top facturas criticas por score.
- Sectores con mayor exposicion.

Alertas incluidas:

- Promesas vencidas.
- Facturas con atraso critico probable.
- Facturas vencidas sin gestion reciente.
- Facturas con disputa activa.
- Facturas de monto alto con riesgo alto o critico.

Alertas excluidas:

- Clientes 1 o 2 estrellas como alerta independiente.

### 6.2 Cola priorizada de cobranzas

Tabla principal del sistema. Debe ordenar por `priority_score_0_100` descendente.

Columnas recomendadas:

- Factura.
- Cliente.
- Sector.
- Monto.
- Fecha de vencimiento.
- Dias hasta vencer o dias de mora.
- Estado de factura.
- Riesgo estimado.
- Probabilidad de cualquier atraso.
- Probabilidad de atraso alto/critico.
- Score de prioridad.
- Rating del cliente.
- Promesa activa.
- Disputa activa.
- Ultima gestion.
- Accion sugerida.

Filtros:

- Riesgo estimado.
- Estado de factura.
- Sector.
- Rating de cliente.
- Con promesa activa.
- Con promesa vencida.
- Con disputa activa.
- Sin gestion previa.
- Vencidas.
- Por vencer en 7 dias.
- Monto alto.

### 6.3 Detalle de factura

Debe mostrar:

- Datos generales de la factura.
- Cliente asociado.
- Estado operativo.
- Monto y saldo pendiente.
- Fecha de emision.
- Fecha de vencimiento.
- Fecha de corte usada para scoring.
- Dias hasta vencer o dias de mora observable.
- Prediccion con nombres orientados al usuario.
- Probabilidades por nivel:
  - Pago dentro del plazo.
  - Atraso leve.
  - Atraso alto.
  - Atraso critico / 60+.
- Score de prioridad.
- Accion sugerida.
- Motivo de la accion sugerida.
- Historial de gestiones.
- Promesas de pago.
- Disputa activa si existe.

Acciones disponibles:

- Registrar gestion.
- Registrar promesa.
- Registrar pago.
- Abrir disputa.
- Cerrar disputa.
- Recalcular riesgo.

### 6.4 Detalle de cliente

Debe mostrar:

- Datos del cliente.
- Sector.
- Antiguedad.
- Garantia.
- Rating de estrellas.
- Riesgo de cliente 0-100.
- Tipo de cliente o cluster.
- Explicacion del rating.
- Explicacion del cluster.
- Facturas abiertas.
- Facturas historicas.
- Tasa de cumplimiento.
- Mora promedio historica.
- Moras consecutivas.
- Promesas rotas.
- Tasa de contacto.

### 6.5 Modulo de simulacion

Debe permitir:

- Cambiar la fecha de corte global.
- Recalcular toda la cartera activa.
- Ver como cambia la prioridad.
- Probar una factura nueva.

### 6.6 Observabilidad y evidencia de IA

Esta vista existe para que un auditor academico pueda entender si el sistema de prediccion aporta valor y como cambia su recomendacion al avanzar la fecha de corte.

Debe mostrar a nivel de cartera:

- Distribucion de facturas por nivel de score: bajo, medio, alto y critico.
- Probabilidad promedio de cualquier atraso.
- Probabilidad promedio de mora alta o critica.
- Concentracion monetaria de los casos mas prioritarios.
- Etiquetas de prediccion dominantes en la cola activa.
- Cantidad de casos que requieren revision por score alto o critico.

Debe mostrar a nivel de factura:

- Historial de predicciones persistidas por `fecha_corte`.
- Evolucion del `priority_score_0_100`.
- Evolucion de la etiqueta de prediccion orientada al usuario.
- Probabilidad de cualquier atraso y de mora alta/critica en cada corte.
- Accion sugerida vigente en cada corte.
- Resultado real de la factura cuando ya exista `fecha_pago_real`, `dias_mora_real` o `target_mora_simulado`.
- Comparacion entre la ultima prediccion previa al cierre y la realidad observada.

Metricas recomendadas para una version posterior:

- Error de direccion: casos en los que el sistema predijo bajo riesgo y la factura termino con mora alta o critica.
- Acierto preventivo: casos en los que el sistema marco riesgo alto antes del vencimiento y la factura efectivamente se atraso.
- Calibracion por rangos: de las facturas con 70%-80% de probabilidad de atraso, cuantas se atrasaron realmente.
- Curva de lift/top-k: que porcentaje de los atrasos reales aparece en el top 10%, 20% y 30% de prioridad.
- Estabilidad temporal: variacion del score entre cortes consecutivos para detectar predicciones demasiado volatiles.
- Matriz de confusion con etiquetas de negocio: pago dentro del plazo, atraso leve, atraso alto y atraso critico.

Requisito metodologico: estas metricas no deben usar informacion futura respecto a la `fecha_corte`. Para auditoria, la comparacion contra realidad se calcula despues del cierre de la factura y se presenta como evaluacion posterior, no como entrada del modelo.

## 7. Base de datos necesaria

El sistema necesita una base de datos relacional. Los CSV y notebooks sirven como fuente inicial y respaldo academico, pero no deben ser la forma operativa del sistema.

Recomendacion para MVP:

- Backend: FastAPI.
- ORM: SQLAlchemy.
- Migraciones: Alembic.
- Base de datos local inicial: SQLite.
- Base recomendada para version mas robusta: PostgreSQL.

### Tablas principales

#### `clientes`

- `cliente_id`
- `nombre`
- `sector`
- `antiguedad_meses`
- `tiene_garantia`
- `perfil_pago_simulado`
- `created_at`
- `updated_at`

#### `facturas`

- `factura_id`
- `cliente_id`
- `fecha_emision`
- `fecha_vencimiento`
- `fecha_pago_real`
- `condicion_dias`
- `monto`
- `saldo_pendiente`
- `estado_factura`
- `target_mora_simulado`
- `dias_mora_real`
- `created_at`
- `updated_at`

Estados sugeridos:

- `abierta`
- `pagada`
- `en_disputa`
- `anulada`
- `castigada`

#### `gestiones_cobranza`

- `gestion_id`
- `factura_id`
- `cliente_id`
- `fecha_gestion`
- `canal`
- `contacto_exitoso`
- `resultado`
- `motivo_no_pago`
- `dias_mora_en_gestion`
- `observacion`
- `created_at`

#### `promesas_pago`

- `promesa_id`
- `gestion_id`
- `factura_id`
- `cliente_id`
- `fecha_promesa`
- `fecha_compromiso`
- `se_cumplio`
- `estado_promesa`
- `created_at`
- `updated_at`

Estados sugeridos:

- `activa`
- `cumplida`
- `incumplida`
- `reemplazada`
- `cancelada`

#### `predicciones_factura`

- `prediccion_id`
- `factura_id`
- `cliente_id`
- `fecha_corte`
- `modelo_version`
- `predicted_class_tecnica`
- `predicted_label_usuario`
- `prob_pago_plazo`
- `prob_atraso_leve`
- `prob_atraso_alto`
- `prob_atraso_critico`
- `any_late_probability`
- `high_risk_probability`
- `priority_score_0_100`
- `accion_sugerida_id`
- `motivo_accion`
- `created_at`

#### `segmentos_cliente`

- `cliente_id`
- `cluster`
- `tipo_cliente`
- `riesgo_0_100`
- `rating_estrellas`
- `rating_label`
- `por_que_rating`
- `por_que_cluster`
- `fecha_calculo`
- `modelo_version`

#### `acciones_sugeridas`

- `accion_sugerida_id`
- `codigo`
- `nombre`
- `descripcion`
- `canal_recomendado`
- `nivel_severidad`
- `activa`

#### `reglas_accion_sugerida`

- `regla_id`
- `accion_sugerida_id`
- `nombre_regla`
- `prioridad_regla`
- `condiciones_json`
- `motivo_template`
- `activa`

#### `scoring_runs`

- `scoring_run_id`
- `fecha_corte`
- `tipo`
- `total_facturas_evaluadas`
- `modelo_version`
- `created_at`

#### `model_versions`

- `modelo_version`
- `tipo_modelo`
- `ruta_artifact`
- `ruta_schema`
- `f1_macro`
- `accuracy`
- `auc_weighted`
- `fecha_entrenamiento`
- `activo`

## 8. Backend requerido

### 8.1 API REST

Endpoints MVP:

- `GET /dashboard/summary`
- `GET /invoices`
- `GET /invoices/{factura_id}`
- `GET /invoices/{factura_id}/predictions`
- `POST /invoices`
- `PATCH /invoices/{factura_id}`
- `POST /invoices/{factura_id}/score`
- `POST /scoring/recalculate`
- `GET /customers`
- `GET /customers/{cliente_id}`
- `GET /customers/{cliente_id}/segment`
- `POST /collections/interactions`
- `POST /payment-promises`
- `PATCH /payment-promises/{promesa_id}`
- `POST /payments`
- `POST /actions/recommend`
- `GET /actions`
- `GET /model/status`
- `GET /model/metrics`

### 8.2 Feature builder

El modelo no debe recibir una factura cruda. El backend debe construir las 39 features requeridas por `model_feature_schema.csv`.

Responsabilidades:

- Leer factura.
- Leer cliente.
- Leer gestiones hasta `fecha_corte`.
- Leer promesas hasta `fecha_corte`.
- Calcular historial previo del cliente.
- Evitar datos futuros.
- Crear variables temporales.
- Crear variables de contacto.
- Crear variables de promesas.
- Crear one-hot de sector.
- Validar que existan todas las columnas requeridas.

### 8.3 Motor de prediccion

Debe:

- Cargar `best_model_artifact.joblib`.
- Validar schema.
- Predecir clase tecnica.
- Convertir clase tecnica a etiqueta de usuario.
- Calcular probabilidades.
- Calcular score de prioridad.
- Guardar resultado en `predicciones_factura`.

### 8.4 Motor de segmentacion

Para MVP puede iniciar cargando `frontend_customer_segments.csv` a la tabla `segmentos_cliente`.

En una version posterior puede recalcular los segmentos desde los artefactos de clustering.

### 8.5 Motor de acciones sugeridas

Debe evaluar reglas ordenadas por prioridad y devolver:

- Accion sugerida.
- Canal recomendado.
- Severidad.
- Motivo textual.
- Variables que activaron la regla.

## 9. Matriz de acciones sugeridas

Las acciones deben evaluarse de mayor prioridad a menor prioridad. Si una factura cumple varias reglas, se toma la regla mas prioritaria, salvo que el producto decida mostrar acciones secundarias.

### Variables usadas por las reglas

- `predicted_label_usuario`
- `any_late_probability`
- `high_risk_probability`
- `priority_score_0_100`
- `dias_hasta_vence`
- `dias_mora_observable`
- `esta_vencida_al_corte`
- `rating_estrellas`
- `tiene_promesa_activa`
- `promesa_vencida`
- `tiene_disputa_activa`
- `num_gestiones_factura`
- `dias_desde_ultima_gestion`
- `tasa_contacto_cliente`
- `num_no_contesta_cons`
- `num_promesas_rotas`
- `tasa_cumpl_promesas`
- `monto`
- `monto_alto`
- `cliente_nuevo`
- `sin_gestion_previa`

### Catalogo de acciones

| Codigo | Accion | Canal | Severidad |
|---|---|---|---:|
| `SIN_ACCION` | Monitorear sin gestion inmediata | Sistema | 0 |
| `RECORDATORIO_SUAVE` | Enviar recordatorio suave | WhatsApp o email | 1 |
| `RECORDATORIO_PREVENTIVO` | Enviar recordatorio preventivo | WhatsApp | 2 |
| `CONFIRMAR_FECHA_PAGO` | Confirmar fecha estimada de pago | WhatsApp o llamada | 2 |
| `LLAMADA_SEGUIMIENTO` | Realizar llamada de seguimiento | Llamada | 3 |
| `LLAMADA_URGENTE` | Realizar llamada urgente | Llamada | 4 |
| `SOLICITAR_PROMESA` | Solicitar promesa formal de pago | Llamada o WhatsApp | 4 |
| `SEGUIMIENTO_PROMESA` | Dar seguimiento a promesa activa | WhatsApp o llamada | 3 |
| `ESCALAR_PROMESA_VENCIDA` | Escalar promesa incumplida | Llamada | 5 |
| `REVISAR_DISPUTA` | Revisar disputa antes de cobrar | Interno | 5 |
| `VISITA_CLIENTE` | Programar visita de cobranza | Visita | 6 |
| `CARTA_FORMAL` | Enviar comunicacion formal | Carta o email formal | 7 |
| `FORMALIZAR_GARANTIA` | Formalizar o revisar garantia | Interno/legal | 8 |
| `ESCALAMIENTO_LEGAL` | Escalar a gestion legal | Legal | 9 |
| `REVISAR_DATOS_CONTACTO` | Revisar datos de contacto | Interno | 4 |
| `REVISION_MANUAL_ALTO_MONTO` | Revision manual por exposicion alta | Interno | 6 |

### Reglas sugeridas

| Prioridad | Condicion | Accion | Motivo para mostrar |
|---:|---|---|---|
| 100 | `estado_factura = pagada` | `SIN_ACCION` | La factura ya fue pagada y no requiere cobranza activa. |
| 99 | `estado_factura = anulada` o `castigada` | `SIN_ACCION` | La factura no pertenece a la cola operativa normal. |
| 95 | `tiene_disputa_activa = 1` | `REVISAR_DISPUTA` | Existe una disputa activa; se recomienda revisar el caso antes de aumentar la presion de cobranza. |
| 92 | `promesa_vencida = 1` y `rating_estrellas <= 2` | `ESCALAMIENTO_LEGAL` | La promesa vencio y el cliente tiene historial critico; se recomienda escalar el caso. |
| 90 | `promesa_vencida = 1` | `ESCALAR_PROMESA_VENCIDA` | La promesa de pago no fue cumplida; se recomienda contactar con urgencia y redefinir compromiso. |
| 88 | `dias_mora_observable >= 60` y `high_risk_probability >= 0.70` y `rating_estrellas <= 2` | `ESCALAMIENTO_LEGAL` | La factura ya tiene mora severa observable, alto riesgo estimado y cliente de bajo rating. |
| 86 | `dias_mora_observable >= 60` y `tiene_garantia = 1` | `FORMALIZAR_GARANTIA` | La mora supera 60 dias y existe garantia; se recomienda revisar o formalizar respaldo. |
| 84 | `dias_mora_observable >= 60` | `CARTA_FORMAL` | La factura tiene mora mayor a 60 dias; se recomienda comunicacion formal. |
| 82 | `dias_mora_observable >= 31` y `high_risk_probability >= 0.60` | `VISITA_CLIENTE` | La factura esta en mora alta y el modelo estima riesgo grave o critico. |
| 80 | `dias_mora_observable >= 31` y `num_no_contesta_cons >= 2` | `VISITA_CLIENTE` | Hay mora alta y dificultad de contacto; se recomienda una gestion presencial. |
| 78 | `dias_mora_observable >= 31` | `LLAMADA_URGENTE` | La factura tiene mas de 30 dias de mora; se requiere contacto urgente. |
| 76 | `dias_mora_observable >= 15` y `rating_estrellas <= 2` | `LLAMADA_URGENTE` | La factura ya esta vencida y el cliente tiene bajo rating. |
| 74 | `dias_mora_observable >= 15` y `high_risk_probability >= 0.50` | `SOLICITAR_PROMESA` | El atraso ya es relevante y el riesgo sigue alto; se recomienda pedir compromiso formal. |
| 72 | `dias_mora_observable >= 8` y `num_gestiones_factura = 0` | `LLAMADA_SEGUIMIENTO` | La factura ya vencio y aun no tiene gestiones registradas. |
| 70 | `dias_mora_observable >= 8` | `LLAMADA_SEGUIMIENTO` | La factura lleva mas de una semana vencida; se recomienda llamada de seguimiento. |
| 68 | `dias_mora_observable >= 1` y `any_late_probability >= 0.70` | `LLAMADA_SEGUIMIENTO` | La factura ya vencio y la probabilidad de atraso es alta. |
| 66 | `dias_mora_observable >= 1` | `RECORDATORIO_PREVENTIVO` | La factura ya vencio; se recomienda recordatorio inmediato. |
| 64 | `tiene_promesa_activa = 1` y `dias_hasta_compromiso <= 2` | `SEGUIMIENTO_PROMESA` | La promesa esta proxima a vencer; se recomienda confirmar cumplimiento. |
| 62 | `tiene_promesa_activa = 1` | `SEGUIMIENTO_PROMESA` | Existe una promesa activa; se recomienda seguimiento sin escalar todavia. |
| 60 | `dias_hasta_vence <= 3` y `any_late_probability >= 0.60` | `CONFIRMAR_FECHA_PAGO` | La factura esta por vencer y el riesgo de atraso es alto; se recomienda confirmar fecha de pago. |
| 58 | `dias_hasta_vence <= 7` y `any_late_probability >= 0.50` y `rating_estrellas <= 3` | `RECORDATORIO_PREVENTIVO` | Falta una semana o menos para vencer y el cliente no tiene rating alto. |
| 56 | `dias_hasta_vence <= 7` y `predicted_label_usuario != Pago esperado dentro del plazo` | `RECORDATORIO_PREVENTIVO` | Es probable que no pague dentro del plazo y queda una semana o menos para el vencimiento. |
| 54 | `dias_hasta_vence <= 7` y `monto_alto = 1` | `CONFIRMAR_FECHA_PAGO` | La factura esta proxima a vencer y el monto es alto; conviene confirmar pago preventivamente. |
| 52 | `dias_hasta_vence <= 14` y `high_risk_probability >= 0.50` | `LLAMADA_SEGUIMIENTO` | Aunque aun no vence, el riesgo de atraso alto o critico es relevante. |
| 50 | `dias_hasta_vence <= 14` y `any_late_probability >= 0.60` | `RECORDATORIO_PREVENTIVO` | La factura aun esta dentro de plazo, pero la probabilidad de atraso es elevada. |
| 48 | `cliente_nuevo = 1` y `any_late_probability >= 0.50` | `CONFIRMAR_FECHA_PAGO` | El cliente tiene poco historial y riesgo moderado; se recomienda confirmar intencion de pago. |
| 46 | `sin_gestion_previa = 1` y `priority_score_0_100 >= 60` | `RECORDATORIO_PREVENTIVO` | No hay gestion previa y el score de prioridad ya es alto. |
| 44 | `num_no_contesta_cons >= 3` | `REVISAR_DATOS_CONTACTO` | Hay varias no respuestas consecutivas; se recomienda validar datos de contacto. |
| 42 | `tasa_contacto_cliente < 0.30` y `num_gestiones_factura >= 2` | `REVISAR_DATOS_CONTACTO` | El cliente tiene baja contactabilidad; conviene revisar medios alternativos. |
| 40 | `num_promesas_rotas >= 2` y `any_late_probability >= 0.50` | `LLAMADA_URGENTE` | El cliente acumula promesas incumplidas y la factura tiene riesgo de atraso. |
| 38 | `tasa_cumpl_promesas < 0.40` y `tiene_promesa_activa = 0` y `dias_mora_observable > 0` | `SOLICITAR_PROMESA` | Hay bajo cumplimiento historico de promesas; si se negocia una nueva, debe quedar formalizada. |
| 36 | `monto_alto = 1` y `priority_score_0_100 >= 70` | `REVISION_MANUAL_ALTO_MONTO` | La exposicion monetaria es alta y el riesgo tambien; requiere revision manual prioritaria. |
| 34 | `priority_score_0_100 >= 80` | `LLAMADA_URGENTE` | El score de prioridad es critico; se recomienda contacto urgente. |
| 32 | `priority_score_0_100 >= 60` | `LLAMADA_SEGUIMIENTO` | El score de prioridad es alto; se recomienda llamada de seguimiento. |
| 30 | `priority_score_0_100 >= 40` | `RECORDATORIO_PREVENTIVO` | El score de prioridad es medio; se recomienda una gestion preventiva. |
| 20 | `dias_hasta_vence > 14` y `any_late_probability < 0.40` | `RECORDATORIO_SUAVE` | La factura aun no esta cerca de vencer y el riesgo es bajo; basta un recordatorio suave si se desea gestionar. |
| 10 | Sin condiciones criticas | `SIN_ACCION` | No se detecta necesidad de gestion inmediata. |

## 10. Validaciones de negocio

Antes de guardar o predecir:

- `cliente_id` debe existir para toda factura.
- `factura_id` debe existir para toda gestion o promesa.
- `monto` debe ser positivo.
- `saldo_pendiente` no debe ser negativo.
- `fecha_vencimiento` debe ser mayor o igual a `fecha_emision`.
- `fecha_corte` no debe ser anterior a `fecha_emision`.
- No se deben usar gestiones futuras a la fecha de corte.
- No se deben usar promesas futuras a la fecha de corte.
- Las tasas deben estar entre 0 y 1.
- El sector debe pertenecer al catalogo entrenado.
- El canal debe pertenecer al catalogo permitido.
- El resultado debe ser coherente con `contacto_exitoso`.
- Si no hay gestion previa, `dias_desde_ultima_gestion = -1` y `ultimo_resultado_enc = sin_gestion_previa` o equivalente esperado por el modelo.

## 11. Requisitos no funcionales

- Backend local reproducible.
- Rutas relativas al proyecto.
- Validacion estricta antes de inferencia.
- Persistencia de predicciones para auditoria.
- Separacion entre datos operativos y artefactos academicos.
- Latencia objetivo para prediccion individual: menor a 500 ms en ambiente local.
- Recalculo por lote suficiente para la base simulada.
- Manejo controlado de errores de schema.
- Logs basicos de recalculo y prediccion.

## 12. Estructura recomendada del proyecto

El desarrollo full stack puede vivir dentro del mismo repositorio:

```text
GRUPO_3/
  backend/
    app/
      api/
      core/
      db/
      models/
      schemas/
      services/
        feature_builder/
        prediction/
        segmentation/
        recommendation/
      scripts/
    tests/
    pyproject.toml
    README.md

  frontend/
    app/
    components/
    lib/
    public/
    package.json
    README.md
```

Backend recomendado:

- FastAPI.
- SQLAlchemy.
- Alembic.
- Pydantic.
- SQLite para MVP.
- PostgreSQL como siguiente paso.
- Joblib para cargar artefactos del modelo.
- Pandas solo en servicios de importacion, feature engineering o batch, no como capa principal de persistencia.

Frontend recomendado:

- Next.js.
- TypeScript.
- Tailwind CSS.
- Tabla de cartera con filtros.
- Componentes para score, probabilidad, rating y timeline.

## 13. Criterios de aceptacion del MVP

El MVP se considera completo cuando:

- La base inicial se carga desde CSV a la base de datos.
- La cola de cobranzas muestra facturas activas ordenadas por prioridad.
- El usuario puede abrir una factura y ver riesgo, probabilidades, score y accion sugerida.
- El usuario puede abrir un cliente y ver rating, segmento y explicacion.
- Una nueva gestion actualiza la factura y permite recalcular riesgo.
- Una nueva promesa se refleja en la factura y en las reglas de accion.
- Una factura pagada sale de la cola activa.
- El cambio de fecha de corte recalcula la cartera activa.
- El backend rechaza datos invalidos antes de llamar al modelo.
- Las acciones sugeridas se generan desde una tabla de reglas trazable.

## 14. Orden recomendado de implementacion

1. Crear backend FastAPI.
2. Crear modelos de base de datos.
3. Crear script de carga inicial desde CSV.
4. Implementar endpoints de lectura de clientes y facturas.
5. Implementar detalle de factura y cliente.
6. Implementar feature builder.
7. Integrar artefacto del modelo.
8. Persistir predicciones.
9. Implementar motor de acciones sugeridas.
10. Crear endpoints de gestiones, promesas y pagos.
11. Inicializar frontend Next.js.
12. Construir dashboard y cola priorizada.
13. Construir detalle de factura.
14. Construir detalle de cliente.

Decision operativa: conviene empezar por backend, porque el frontend depende de contratos claros de datos, prediccion, acciones sugeridas y estados de factura.

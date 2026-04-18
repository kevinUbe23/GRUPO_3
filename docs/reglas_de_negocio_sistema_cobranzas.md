# Reglas de negocio del sistema inteligente de priorización de cobranzas

## 1. Propósito del sistema

El sistema tiene como finalidad priorizar la gestión de cobranza de empresas que venden a crédito, mediante tres componentes integrados:

1. **Predicción de mora por factura**: estima el tramo final de mora esperado para cada factura.
2. **Segmentación de clientes**: agrupa clientes según su comportamiento histórico de pago y los traduce a una escala de 1 a 5 estrellas.
3. **Decisor de acción de cobranza**: sugiere una acción operativa a partir del riesgo de mora y la calidad histórica del cliente.

La **unidad principal de análisis y predicción** es la **factura**, no el cliente.

---

## 2. Principios generales de diseño

### 2.1. Principio temporal
Toda variable usada para entrenamiento o scoring debe construirse únicamente con información disponible **hasta la fecha de corte** correspondiente.

### 2.2. Prohibición de data leakage
No se permite usar como predictor ninguna variable que dependa de eventos ocurridos después de la fecha de corte o del desenlace final de la factura.

### 2.3. Trazabilidad
Toda gestión, promesa y feature debe poder vincularse de manera trazable con su factura y su cliente.

### 2.4. Coherencia operativa
Los catálogos y combinaciones de valores deben respetar lógica de negocio. No se permiten combinaciones imposibles o contradictorias.

### 2.5. Reproducibilidad
La simulación debe ser reproducible mediante semilla fija.

---

## 3. Reglas de negocio de clientes

### 3.1. Definición
Cada registro en `clientes.csv` representa una empresa cliente con comportamiento crediticio simulado.

### 3.2. Identificador
- `cliente_id` debe ser único.
- Un cliente puede tener múltiples facturas.

### 3.3. Perfil interno de simulación
- El cliente puede pertenecer a un perfil interno de simulación como `excelente`, `regular`, `riesgoso` o `critico`.
- Este perfil sirve para gobernar las probabilidades de mora, contacto, promesas y cumplimiento.
- El perfil puede conservarse solo para simulación y análisis interno, pero no necesariamente como feature del modelo productivo.

### 3.4. Campos del cliente
Se consideran válidos como atributos base:
- `cliente_id`
- `nombre`
- `sector`
- `antiguedad_meses`
- `tiene_garantia`
- `perfil_pago` o variable equivalente de simulación, si se mantiene solo para control interno

### 3.5. Límite de crédito
- El campo `limite_credito` se elimina del modelo de datos si no será utilizado en la lógica del negocio ni en el modelado.
- No deben mantenerse columnas sin aporte analítico o funcional.

### 3.6. Antigüedad
- `antiguedad_meses` debe ser un valor positivo.
- Debe representar la antigüedad del cliente al momento de la simulación.

### 3.7. Garantía
- `tiene_garantia` es binaria: `0` o `1`.
- Puede influir en la estrategia de cobranza, especialmente en decisiones más severas del componente 3.

### 3.8. Sector
- `sector` debe pertenecer a un catálogo predefinido.
- Para modelado, el sector debe codificarse sin inducir orden artificial; preferentemente mediante one-hot encoding.

---

## 4. Reglas de negocio de facturas

### 4.1. Definición
Cada registro en `facturas.csv` representa una factura emitida a un cliente. Esta es la unidad de predicción del componente 1.

### 4.2. Identificador
- `factura_id` debe ser único.
- Toda factura debe estar asociada a un `cliente_id` válido.

### 4.3. Orden cronológico global
- A medida que `factura_id` incrementa, la `fecha_emision` no debe retroceder.
- La numeración final de las facturas debe respetar el orden cronológico global de emisión.
- Regla práctica: se generan las facturas, luego se ordenan por `fecha_emision`, y finalmente se reasignan los IDs en ese orden.

### 4.4. Fecha de emisión
- `fecha_emision` es la fecha en que nace la obligación comercial.
- Debe ser la base para el primer corte temporal de la factura.

### 4.5. Condición de pago
- `condicion_dias` pertenece a un catálogo definido, por ejemplo: `30`, `45`, `60` o `90`.
- Representa el plazo otorgado al cliente antes del vencimiento.

### 4.6. Fecha de vencimiento
- `fecha_vencimiento = fecha_emision + condicion_dias`
- Debe calcularse directamente a partir de la fecha de emisión y la condición otorgada.

### 4.7. Fecha de pago real
- `fecha_pago_real` representa la fecha en que finalmente se realizó el pago.
- Debe cumplir consistencia temporal.

### 4.8. Consistencia temporal mínima de factura
Debe cumplirse:
- `fecha_emision <= fecha_vencimiento`
- `fecha_vencimiento <= fecha_pago_real` cuando el pago se realice al vencimiento o con atraso
- Si se permite pago anticipado, entonces `fecha_emision <= fecha_pago_real <= fecha_vencimiento`

### 4.9. Días de mora real
- `dias_mora_real = max(0, fecha_pago_real - fecha_vencimiento)`
- Es una variable final de resultado, no una feature de entrada.

### 4.10. Variable objetivo del modelo
La columna originalmente llamada `estado_pago` debe redefinirse semánticamente como una **clase objetivo**:
- nombre recomendado: `target_mora`

### 4.11. Regla de clasificación de `target_mora`
- `on_time`: `dias_mora_real = 0`
- `+30`: `1 <= dias_mora_real <= 30`
- `+60`: `31 <= dias_mora_real <= 60`
- `+90`: `dias_mora_real >= 61`

### 4.12. Naturaleza de `target_mora`
- `target_mora` sí aporta como etiqueta del problema de clasificación.
- No debe utilizarse como feature de entrada.
- Sirve además para análisis de distribución de clases y evaluación del modelo.

### 4.13. Monto de factura
- `monto` debe ser positivo.
- Puede influir en el riesgo de cobranza y en la priorización de acciones.

---

## 5. Reglas de negocio de gestiones de cobranza

### 5.1. Definición
Cada registro en `gestiones_cobranza.csv` representa un intento de contacto o acción de seguimiento asociado a una factura.

### 5.2. Cardinalidad
- Una factura puede tener cero, una o múltiples gestiones.
- Toda gestión debe pertenecer a una única factura y a un único cliente.

### 5.3. Fecha de gestión
- `fecha_gestion` debe ser una fecha válida dentro de la historia de la factura.
- Debe respetar la cronología de las gestiones de la misma factura.
- No debe ser anterior a `fecha_emision`.

### 5.4. Tipos de gestión según el momento
Se reconocen dos grandes tipos:

#### a) Gestiones preventivas
Ocurren **antes del vencimiento**.
Su objetivo es recordar, confirmar o monitorear el pago.

#### b) Gestiones reactivas
Ocurren **después del vencimiento**.
Su objetivo es recuperar cartera vencida y escalar la presión de cobranza.

### 5.5. Escalamiento de canal
La secuencia general esperada puede seguir una lógica de escalamiento como:
- `whatsapp`
- `email`
- `llamada`
- `visita`
- `carta_notarial`

Esta secuencia puede variar en casos particulares, pero el sistema debe tender a una progresión razonable de intensidad.

### 5.6. Catálogo de canal
El canal debe pertenecer a un catálogo cerrado, por ejemplo:
- `whatsapp`
- `email`
- `llamada`
- `visita`
- `carta_notarial`

### 5.7. Contacto exitoso
- `contacto_exitoso` es binario: `0` o `1`.
- Significa si se logró efectivamente interacción útil con el cliente o contraparte.

### 5.8. Regla de coherencia entre contacto y resultado
Si `contacto_exitoso = 0`, el `resultado` solo puede pertenecer al subconjunto de **no contacto**.

Si `contacto_exitoso = 1`, el `resultado` solo puede pertenecer al subconjunto de **contacto efectivo**.

### 5.9. Resultados de no contacto
Catálogo sugerido:
- `no_contesta`
- `numero_invalido`
- `cliente_ausente`

### 5.10. Resultados de contacto efectivo
Catálogo sugerido:
- `pagado`
- `promesa_de_pago`
- `disputa_monto`
- `rechazo_pago`
- `en_proceso_interno`
- `confirma_pago`

### 5.11. Coherencia entre canal y resultado
Deben evitarse combinaciones imposibles o poco lógicas. Ejemplos:
- `numero_invalido` es razonable en `whatsapp` o `llamada`
- `numero_invalido` no es coherente en `visita`
- `cliente_ausente` es particularmente coherente en `visita`
- `no_contesta` es coherente en `whatsapp`, `llamada` o `email`
- `carta_notarial` no representa una conversación bidireccional ordinaria, por lo que su interpretación de contacto debe tratarse de forma especial

### 5.12. Resultado en gestión preventiva
En gestiones preventivas puede existir un resultado como:
- `confirma_pago`

Este resultado indica una señal positiva previa al vencimiento.

### 5.13. Resultado en gestión reactiva
En gestiones posteriores al vencimiento pueden existir resultados como:
- `promesa_de_pago`
- `disputa_monto`
- `rechazo_pago`
- `en_proceso_interno`
- `pagado`

### 5.14. Motivo de no pago
`motivo_no_pago` solo debe registrarse cuando se cumplan simultáneamente estas condiciones:
- hubo `contacto_exitoso = 1`
- la factura ya estaba vencida en `fecha_gestion`
- el resultado no fue pago efectivo inmediato

### 5.15. Motivo de no pago nulo
`motivo_no_pago` debe ser `null` cuando:
- no hubo contacto exitoso
- la factura aún no estaba vencida
- no aplica el levantamiento del motivo

### 5.16. Días de mora en gestión
- `dias_mora_en_gestion` debe calcularse con respecto a `fecha_vencimiento` y `fecha_gestion`
- Regla sugerida: `max(0, fecha_gestion - fecha_vencimiento)`

### 5.17. Relación con el estado real de la factura
No debe existir una gestión fechada después del pago real si en la lógica del negocio ya no tendría sentido gestionar una factura totalmente cancelada, salvo que se desee modelar confirmaciones administrativas posteriores.

---

## 6. Reglas de negocio de promesas de pago

### 6.1. Definición
Cada registro en `promesas_pago.csv` representa un compromiso explícito de pago surgido a partir de una gestión de cobranza.

### 6.2. Origen obligatorio
- Solo puede existir una promesa si hubo una gestión cuyo `resultado = promesa_de_pago`.
- Toda promesa debe vincularse a:
  - `gestion_id`
  - `factura_id`
  - `cliente_id`

### 6.3. Fecha promesa
- `fecha_promesa` coincide con la `fecha_gestion` que originó la promesa.
- Puede mantenerse por claridad y autonomía de la tabla.

### 6.4. Fecha compromiso
- `fecha_compromiso` es la fecha prometida por el cliente para cumplir el pago.
- Puede ser anterior, igual o posterior al vencimiento contractual de la factura.

### 6.5. Monto comprometido
- Se elimina `monto_comprometido` si todas las promesas representan el pago total de la factura.
- No deben mantenerse campos redundantes sin variabilidad real.

### 6.6. Cumplimiento de promesa
La columna `se_cumplio` debe derivarse de la realidad de la factura, no generarse aleatoriamente.

Regla base:
- `se_cumplio = 1` si `fecha_pago_real <= fecha_compromiso`
- `se_cumplio = 0` si `fecha_pago_real > fecha_compromiso`

### 6.7. Reemplazo de promesas
Si una factura tiene múltiples promesas sucesivas:
- una promesa posterior puede considerarse reemplazo operativo de la anterior
- para simplificación del modelo actual, no se conservará `dias_retraso_promesa`
- la información relevante resumida podrá concentrarse en variables agregadas como número de promesas rotas o tasa histórica de cumplimiento

### 6.8. Campo eliminado
- `fecha_cumplimiento_real` se elimina
- la fuente de verdad del cumplimiento real es `fecha_pago_real` en la tabla de facturas

### 6.9. Múltiples promesas en una misma factura
- Se permiten múltiples promesas por factura
- Deben respetar orden cronológico
- No deben existir dos promesas activas simultáneamente sobre la misma factura en la misma fecha de corte, salvo que se modele explícitamente un caso excepcional

---

## 7. Reglas de negocio de fechas de corte y scoring dinámico

### 7.1. Definición de fecha de corte
La `fecha_corte` es el momento exacto en que el sistema toma una “fotografía” de la factura para construir features y estimar riesgo.

### 7.2. Fila por corte
El dataset de entrenamiento puede contener múltiples filas por una misma factura, una por cada corte relevante.

### 7.3. Cortes mínimos sugeridos
- **Corte 0**: `fecha_emision`
- **Cortes siguientes**: cada `fecha_gestion` registrada para la factura

### 7.4. Regla de visibilidad temporal
En cada fila del dataset, solo pueden usarse hechos ocurridos en fecha menor o igual a `fecha_corte`.

### 7.5. Prohibición de conocimiento futuro
No se permite que una fila de entrenamiento conozca:
- gestiones futuras
- promesas futuras
- resultados posteriores al corte
- datos del desenlace final, salvo la etiqueta objetivo, que se usa solo como target

### 7.6. Promesa activa
`tiene_promesa_activa = 1` solo si, a la fecha de corte:
- la promesa ya fue creada (`fecha_promesa <= fecha_corte`)
- la fecha compromiso aún no ha vencido (`fecha_corte < fecha_compromiso`)
- la promesa sigue vigente y no ha sido reemplazada o cerrada

### 7.7. Promesa no activa
`tiene_promesa_activa = 0` cuando:
- aún no existe una promesa al corte
- la promesa ya venció
- la promesa fue sustituida por otra
- la factura ya fue pagada

### 7.8. Prohibición de fecha global artificial
El estado temporal de una fila debe depender de su propia `fecha_corte`, no de una fecha global fija que distorsione el historial.

---

## 8. Reglas de negocio de features para modelado

### 8.1. Naturaleza de las features
Las features deben derivarse del historial acumulado conocido hasta cada corte.

### 8.2. Features permitidas por principio
Son válidas las variables históricas de comportamiento, frecuencia, severidad y respuesta a cobranza, siempre que respeten la fecha de corte.

### 8.3. Features no permitidas
No deben incluirse como predictores:
- `dias_mora_real`
- `target_mora`
- cualquier derivación directa del desenlace final

### 8.4. Variables de factura
Pueden incluirse, por ejemplo:
- `monto`
- `condicion_dias`
- `antiguedad_meses`
- `tiene_garantia`
- `sector` codificado correctamente
- contadores y promedios históricos previos

### 8.5. Variables de gestión
Pueden incluirse agregados históricos como:
- número de gestiones previas
- tasa de contacto
- secuencia de resultados previos
- disputas activas válidamente definidas al corte

### 8.6. Variables de promesas
Pueden incluirse agregados como:
- número de promesas previas
- número de promesas rotas
- tasa histórica de cumplimiento
- existencia de promesa activa al corte

### 8.7. Variables problemáticas
Variables como `dias_desde_ultima_gestion` deben revisarse para que tengan significado real y no queden constantes o nulas por construcción.

### 8.8. Codificación categórica
- Las variables categóricas no deben codificarse con enteros ordinales arbitrarios si ello induce orden inexistente.
- Para clasificación y clustering se prefiere codificación apropiada como one-hot, salvo justificación técnica distinta.

---

## 9. Reglas de negocio del componente 1: predicción de mora por factura

### 9.1. Objetivo
Predecir el tramo final de mora de una factura.

### 9.2. Tipo de problema
Clasificación multiclase.

### 9.3. Clases
- `on_time`
- `+30`
- `+60`
- `+90`

### 9.4. Métrica principal
La métrica principal es `F1-macro`, por ser apropiada para problemas con posible desbalance entre clases.

### 9.5. Otras métricas
- Accuracy
- AUC-ROC multiclase
- F1 por clase
- matriz de confusión
- tiempo de entrenamiento

### 9.6. Algoritmos a comparar
- XGBoost
- Random Forest
- Regresión Logística Multinomial

### 9.7. Elección final
El mejor modelo debe seleccionarse con base en desempeño, estabilidad, interpretabilidad y factibilidad del proyecto.

---

## 10. Reglas de negocio del componente 2: segmentación de clientes

### 10.1. Objetivo
Agrupar clientes según su comportamiento histórico de pago.

### 10.2. Tipo de problema
Aprendizaje no supervisado mediante clustering.

### 10.3. Algoritmos a comparar
- K-Means con `K=5`
- DBSCAN

### 10.4. Métricas de evaluación
- Silhouette Score
- Davies-Bouldin Index

### 10.5. Traducción a estrellas
Los grupos obtenidos deben mapearse a una escala de:
- 1 estrella = cliente crítico
- 5 estrellas = cliente excelente

### 10.6. Regla de interpretación
El mapeo a estrellas debe hacerse según el perfil de comportamiento observado en cada cluster, no de manera arbitraria.

---

## 11. Reglas de negocio del componente 3: decisor de acción

### 11.1. Naturaleza
El componente 3 no usa machine learning. Opera con reglas de negocio.

### 11.2. Entradas
- probabilidad o clase de mora estimada por el componente 1
- estrellas del cliente estimadas por el componente 2

### 11.3. Salida
Debe recomendar una acción concreta de cobranza, por ejemplo:
- monitoreo
- recordatorio
- llamada
- visita
- formalización de garantía
- escalamiento legal

### 11.4. Principio de severidad
La acción debe aumentar en intensidad cuando:
- aumenta la probabilidad de mora severa
- disminuye la calidad histórica del cliente
- existen antecedentes de incumplimiento o promesas rotas
- existe disputa activa o señales de riesgo operacional

---

## 12. Reglas de integridad y calidad de datos

### 12.1. Unicidad
Deben ser únicos:
- `cliente_id`
- `factura_id`
- `gestion_id`
- `promesa_id`

### 12.2. Integridad referencial
- Toda factura debe referenciar a un cliente existente.
- Toda gestión debe referenciar a una factura y cliente existentes.
- Toda promesa debe referenciar a una gestión, factura y cliente existentes.

### 12.3. Coherencia cronológica
- No debe existir una promesa anterior a la gestión que la originó.
- No debe existir una gestión anterior a la emisión de la factura.
- No debe existir una fecha de vencimiento anterior a la emisión.

### 12.4. Coherencia de catálogos
Los valores categóricos deben pertenecer a catálogos cerrados y documentados.

### 12.5. Nulos permitidos
Los nulos solo deben existir cuando el negocio lo justifique, por ejemplo:
- `motivo_no_pago` cuando no hubo contacto o la factura no estaba vencida
- campos de promesa cuando la factura nunca tuvo promesas

### 12.6. Variables eliminadas
Deben eliminarse del diseño final los campos que no aporten:
- `limite_credito`
- `monto_comprometido`
- `fecha_cumplimiento_real`
- `dias_retraso_promesa` en la versión simplificada acordada

---

## 13. Reglas metodológicas para entrenamiento y evaluación

### 13.1. Múltiples filas por factura
Si el dataset de features contiene múltiples cortes por factura, el particionado train/test debe realizarse **por `factura_id`**, no por fila individual.

### 13.2. Evitar fuga entre train y test
No deben coexistir cortes de una misma factura repartidos entre entrenamiento y prueba.

### 13.3. EDA previo
Antes del modelado se debe ejecutar un análisis exploratorio que valide:
- calidad de datos
- distribuciones
- balance de clases
- outliers
- correlaciones
- coherencia de variables categóricas

### 13.4. Comparativo teórico y experimental
Los algoritmos deben analizarse en dos planos:
- comparativo teórico
- comparativo experimental

---

## 14. Supuestos operativos adicionales recomendados

### 14.1. Pago total de factura
En esta versión del proyecto se asume que el pago registrado cancela completamente la factura, salvo que en una fase futura se decida modelar pagos parciales.

### 14.2. Una promesa por gestión
Se asume una sola promesa por gestión con resultado `promesa_de_pago`.

### 14.3. Sin notas de crédito ni anulaciones
En esta versión simplificada no se modelan anulaciones, castigos, notas de crédito ni reestructuraciones formales de deuda.

### 14.4. Sin facturas simultáneamente abiertas con lógica de aplicación parcial
Cada factura conserva su identidad propia y su desenlace final.

### 14.5. Gestión posterior al pago
Por defecto no se generan gestiones posteriores al pago final, salvo que se modelen explícitamente procesos administrativos adicionales.

---

## 15. Resumen ejecutivo de funcionamiento

1. Se simulan clientes con distintos perfiles de riesgo.
2. Se generan facturas con fechas consistentes y un desenlace final de pago.
3. Se registran gestiones preventivas y reactivas coherentes con el estado temporal de la factura.
4. Algunas gestiones generan promesas de pago con reglas consistentes de cumplimiento.
5. Para cada factura se crean cortes temporales, y en cada corte se construyen features solo con información disponible hasta ese momento.
6. Con esas features se entrena un clasificador multiclase para predecir el tramo final de mora.
7. En paralelo, se segmentan clientes para asignarles estrellas según su historial.
8. Finalmente, una capa de reglas combina riesgo por factura y calidad del cliente para recomendar la acción de cobranza más adecuada.

---

## 16. Pendientes opcionales para una versión futura

Estos puntos no son obligatorios para la versión actual, pero podrían considerarse en una fase posterior:

- modelar pagos anticipados explícitos
- modelar pagos parciales
- modelar refinanciaciones
- modelar múltiples tipos de garantía
- modelar severidad monetaria de la mora acumulada
- incorporar ventanas temporales más ricas para el scoring
- separar canales automáticos de canales asistidos por gestor


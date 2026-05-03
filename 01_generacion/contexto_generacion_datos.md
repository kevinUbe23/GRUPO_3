# Contexto de la fase 01 - Generacion de datos

## Proposito de la fase

Esta fase genera un dataset sintetico para un sistema inteligente de priorizacion de cobranzas. La unidad de negocio es la factura, pero el dataset de modelado se construye por cortes temporales, de modo que una misma factura puede aparecer varias veces en `features_ml.csv`.

La salida oficial de esta fase queda en:

`01_generacion/data/`

## Conceptos base para entender los datos

### Que es una factura en este proyecto

Una factura es la obligacion comercial que la empresa quiere cobrar. Por eso es la unidad principal del sistema.

Ejemplo:

- Cliente: Empresa ABC
- Factura: `FAC000123`
- Monto: 10,000
- Fecha de emision: 2024-03-01
- Fecha de vencimiento: 2024-04-01
- Fecha real de pago: 2024-05-10

El modelo no intenta responder solo "este cliente es bueno o malo". Intenta responder algo mas operativo: "esta factura especifica, en este momento especifico, que riesgo tiene de terminar en mora?".

### Que es `target_mora`

`target_mora` es la etiqueta final que queremos que el modelo aprenda a predecir. Resume como termino pagando la factura.

Se calcula con `dias_mora_real`, que es la diferencia entre la fecha real de pago y la fecha de vencimiento.

| Valor de `target_mora` | Significado |
|---|---|
| `on_time` | La factura se pago sin mora |
| `+30` | La factura se pago con 1 a 30 dias de mora |
| `+60` | La factura se pago con 31 a 60 dias de mora |
| `+90` | La factura se pago con 61 dias o mas de mora |

Importante: `target_mora` es la respuesta correcta que se usa para entrenar y evaluar el modelo. No debe usarse como predictor.

### Que son los cortes temporales

Un corte temporal es una fotografia del caso en un momento especifico. En vez de mirar la factura solo al final, el sistema la mira varias veces mientras avanza el ciclo de cobranza.

Ejemplo didactico:

| Momento | Fecha | Que sabe el sistema |
|---|---|---|
| Corte 0 | Fecha de emision | Monto, plazo, cliente, historial previo del cliente |
| Corte 1 | Primera gestion | Lo anterior + canal usado + resultado de la primera gestion |
| Corte 2 | Segunda gestion | Lo anterior + acumulado de gestiones hasta ese momento |
| Corte 3 | Promesa de pago | Lo anterior + si existe una promesa activa |

La misma factura puede aparecer varias veces en `features_ml.csv` porque cada fila representa un corte distinto.

Ejemplo:

| factura_id | num_corte | fecha_corte | num_gestiones_factura | target_mora |
|---|---:|---|---:|---|
| `FAC000123` | 0 | 2024-03-01 | 0 | `+60` |
| `FAC000123` | 1 | 2024-04-05 | 1 | `+60` |
| `FAC000123` | 2 | 2024-04-20 | 2 | `+60` |

Todas las filas comparten el mismo `target_mora` final, porque la factura termino en una sola clase de mora. Lo que cambia entre filas son las features disponibles en cada fecha de corte.

### Por que los cortes importan

Los cortes permiten simular un sistema real de scoring dinamico. El riesgo no se calcula una sola vez; se actualiza conforme llegan nuevas senales.

Una factura puede verse de bajo riesgo al emitirse, pero si luego el cliente no responde tres gestiones seguidas, el riesgo deberia subir. Esa es la razon de tener varias filas por factura.

### Que es data leakage

Data leakage significa que el modelo recibe informacion que no deberia conocer al momento de predecir. En otras palabras, el modelo "ve el futuro".

Ejemplo de leakage:

- Queremos predecir el riesgo el dia 2024-04-01.
- Pero usamos como feature una promesa creada el dia 2024-04-15.
- Eso esta mal, porque el 2024-04-01 esa promesa todavia no existia.

Otro ejemplo:

- Usar `dias_mora_real` como feature.
- Eso seria leakage porque `dias_mora_real` solo se conoce despues de saber la fecha real de pago.

### Como `fecha_corte` ayuda a evitar leakage

`fecha_corte` funciona como una frontera temporal. Para cada fila de `features_ml.csv`, solo se deben usar eventos que ocurrieron en esa fecha o antes.

Regla simple:

Si `fecha_evento <= fecha_corte`, se puede usar.

Si `fecha_evento > fecha_corte`, no se puede usar.

Esto aplica a gestiones, promesas, pagos, resultados y cualquier variable historica. Gracias a esta regla, cada fila del dataset representa lo que el sistema realmente habria sabido en ese momento.

## Artefactos generados

| Archivo | Filas | Columnas | Rol |
|---|---:|---:|---|
| `clientes.csv` | 200 | 6 | Tabla maestra de clientes simulados |
| `facturas.csv` | 5,338 | 9 | Tabla operacional principal; una fila por factura |
| `gestiones_cobranza.csv` | 14,333 | 9 | Historial de acciones de cobranza por factura |
| `promesas_pago.csv` | 1,741 | 7 | Promesas originadas por gestiones con `promesa_de_pago` |
| `features_ml.csv` | 19,671 | 36 | Dataset analitico por cortes temporales para modelado |

## Validaciones realizadas

Se revisaron 29 reglas de integridad y consistencia. Todas pasaron correctamente despues de corregir el nombre del target en `features_ml.csv`.

Validaciones principales:

- IDs unicos en clientes, facturas, gestiones y promesas.
- Integridad referencial entre clientes, facturas, gestiones y promesas.
- `factura_id` respeta el orden cronologico global de `fecha_emision`.
- `fecha_vencimiento = fecha_emision + condicion_dias`.
- `fecha_pago_real >= fecha_vencimiento`.
- `dias_mora_real` es consistente con las fechas.
- `target_mora` es consistente con `dias_mora_real`.
- Las gestiones no ocurren antes de la emision ni despues del pago real.
- `dias_mora_en_gestion` esta correctamente calculado.
- Existe coherencia entre `contacto_exitoso`, `resultado` y `canal`.
- Las promesas nacen solo de gestiones con `resultado = promesa_de_pago`.
- `se_cumplio` se deriva de `fecha_pago_real <= fecha_compromiso`.
- `features_ml.csv` cubre todas las facturas y tiene target unico por factura.
- `fecha_corte` siempre esta entre la emision y el pago real.

## Distribucion del target por factura

| Clase | Facturas | Porcentaje |
|---|---:|---:|
| `on_time` | 2,217 | 41.53% |
| `+30` | 1,246 | 23.34% |
| `+60` | 1,088 | 20.38% |
| `+90` | 787 | 14.74% |

A nivel de factura, la clase mayoritaria es `on_time`, lo cual tiene sentido para una cartera crediticia donde no todas las obligaciones terminan en mora severa.

## Distribucion del target en features_ml

| Clase | Filas | Porcentaje |
|---|---:|---:|
| `on_time` | 3,317 | 16.86% |
| `+30` | 3,713 | 18.88% |
| `+60` | 5,965 | 30.32% |
| `+90` | 6,676 | 33.94% |

Esta distribucion cambia porque `features_ml.csv` no tiene una fila por factura, sino una fila por corte. Las facturas mas problematicas generan mas gestiones y, por tanto, mas cortes temporales.

Este comportamiento no es un error. Es una consecuencia esperada del diseno de scoring dinamico.

## Cortes promedio por factura segun target

| Clase | Cortes promedio | Mediana | Maximo |
|---|---:|---:|---:|
| `on_time` | 1.50 | 1 | 2 |
| `+30` | 2.98 | 3 | 4 |
| `+60` | 5.48 | 6 | 7 |
| `+90` | 8.48 | 9 | 11 |

Lectura: a mayor severidad de mora, mayor numero de cortes. Esto refuerza la necesidad de hacer el split train/test por `factura_id`, no por fila.

## Relacion entre perfil simulado y mora final

| Perfil | Facturas | `on_time` | `+30` | `+60` | `+90` |
|---|---:|---:|---:|---:|---:|
| `excelente` | 1,487 | 85.0% | 9.1% | 4.6% | 1.2% |
| `regular` | 2,097 | 42.3% | 29.6% | 19.5% | 8.7% |
| `riesgoso` | 1,246 | 4.2% | 31.8% | 34.8% | 29.2% |
| `critico` | 508 | 3.0% | 18.5% | 34.6% | 43.9% |

La simulacion es coherente con la logica de negocio: los clientes excelentes pagan mayoritariamente a tiempo, mientras que los clientes criticos concentran mas casos `+60` y `+90`.

Nota metodologica: `perfil_pago` es una etiqueta interna de simulacion (`excelente`, `regular`, `riesgoso`, `critico`). Sirve para generar datos coherentes y para validar que la simulacion tenga sentido, pero no deberia usarse como predictor del modelo final. Debe revisarse en EDA y eliminarse como variable de modelado en la fase de Preparacion y Procesamiento de Datos, salvo que se justifique explicitamente como variable disponible en un escenario real.

## Gestiones y promesas

Promedio de gestiones por factura:

| Clase | Gestiones promedio | Mediana | Maximo |
|---|---:|---:|---:|
| `on_time` | 0.50 | 0 | 1 |
| `+30` | 1.98 | 2 | 3 |
| `+60` | 4.48 | 5 | 6 |
| `+90` | 7.48 | 8 | 10 |

Distribucion de canales:

| Canal | Registros |
|---|---:|
| `whatsapp` | 4,221 |
| `llamada` | 3,854 |
| `email` | 2,704 |
| `visita` | 2,255 |
| `carta_notarial` | 1,299 |

Promesas:

- Total de promesas: 1,741.
- Promesas cumplidas: 793.
- Promesas incumplidas: 948.
- Tasa de cumplimiento: 45.55%.

La tasa de cumplimiento no es alta, lo cual es razonable porque las promesas aparecen dentro de un contexto de cobranza y no sobre toda la cartera.

## Nulos estructurales

En `features_ml.csv` aparecen nulos en:

| Columna | Nulos | Interpretacion |
|---|---:|---|
| `dias_desde_ultima_gestion` | 5,338 | Corresponde al corte 0 de cada factura |
| `ultimo_resultado_enc` | 5,338 | Corresponde al corte 0 de cada factura |

Estos nulos no representan mala calidad de datos. Son nulos estructurales: en el primer corte todavia no existe una gestion previa para la factura.

## Implicaciones para fases siguientes

1. El EDA debe analizar dos niveles: factura y corte temporal.
2. La preparacion debe tratar los nulos de corte 0 como senales operativas, no como errores.
3. El split de modelado debe hacerse por `factura_id`.
4. La metrica principal debe mantenerse como F1-macro, porque `features_ml.csv` sobrerrepresenta las clases severas.
5. `target_mora` no debe entrar como feature; solo como etiqueta.
6. Las variables derivadas de gestiones y promesas deben respetar `fecha_corte` para evitar leakage temporal.
7. `perfil_pago` debe tratarse como variable interna de simulacion; puede analizarse en EDA, pero debe excluirse de las features finales en preparacion.

## Contexto suficiente para pasar a EDA

Con este documento si hay contexto suficiente para iniciar la siguiente fase, siempre que el EDA vuelva a revisar los archivos reales. Para no ir ciego a la siguiente etapa, hay que llevar estas ideas en memoria:

- La unidad de negocio es `factura_id`.
- `features_ml.csv` tiene una fila por corte temporal, no una fila por factura.
- `target_mora` es la etiqueta final y no puede usarse como predictor.
- Las clases severas aparecen mas en `features_ml.csv` porque generan mas gestiones y mas cortes.
- Los nulos de corte 0 son estructurales, no errores.
- `fecha_corte` es la frontera que evita usar informacion futura.
- `perfil_pago` es una variable artificial de simulacion y no debe llegar al modelo final.

La siguiente fase debe confirmar estas conclusiones con analisis exploratorio, revisar distribuciones, nulos, outliers, consistencia entre tablas y riesgos de leakage antes de preparar datos para modelado.

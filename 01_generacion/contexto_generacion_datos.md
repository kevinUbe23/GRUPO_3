# Contexto de la fase 01 - Generacion de datos

## Proposito de la fase

Esta fase genera un dataset sintetico para un sistema inteligente de priorizacion de cobranzas. La unidad de negocio es la factura, pero el dataset de modelado se construye por cortes temporales, de modo que una misma factura puede aparecer varias veces en `features_ml.csv`.

La salida oficial de esta fase queda en:

`01_generacion/data/`

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

## Correccion aplicada

Se detecto una inconsistencia de nombres:

- `facturas.csv` usaba `target_mora`.
- `features_ml.csv` usaba `target`.

Se corrigio para que `features_ml.csv` tambien use `target_mora`. Esto evita ambiguedad en preparacion, modelado y evaluacion.

Tambien se alineo la estructura de carpetas:

- Notebook: `01_generacion/simulacion_datos.ipynb`
- Outputs: `01_generacion/data/`

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

## Estado de la fase

La fase de generacion queda metodologicamente valida como insumo para EDA y preparacion. La logica sintetica es coherente con las reglas de negocio y los datos finales ya estan alineados con la nomenclatura canonica del proyecto.

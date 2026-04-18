# Resumen de la lógica de los notebooks del proyecto

Este documento explica la lógica y la continuidad entre los notebooks principales del proyecto:
- `0 Creación de datos/simulación_datos.ipynb`
- `EDA/EDA_cobranzas.ipynb`
- `Presentación de la fase de preparación y procesamiento de datos/notebook_preparación.ipynb`

También se anota cómo cada notebook se relaciona con el contexto y las reglas de negocio del sistema de cobranzas.

## 1. `0 Creación de datos/simulación_datos.ipynb`

### Propósito
Generar datos sintéticos consistentes con las reglas de negocio del proyecto. Produce las tablas base y el dataset de features con cortes temporales que serán la base del modelado.

### Lógica principal
1. Configura parámetros globales y catálogos: sectores, condiciones de pago, clases de mora y canales de cobranza.
2. Define perfiles internos de clientes (`excelente`, `regular`, `riesgoso`, `critico`) con probabilidades distintas de pago, contacto y cumplimiento de promesas.
3. Implementa funciones auxiliares para:
   - decidir perfil de cliente y clase de mora
   - generar montos según sector
   - distribuir fechas de emisión y gestiones
   - seleccionar canales de escalamiento
   - simular resultados de gestiones preventivas y reactivas
   - determinar promesas activas y fechas de compromiso
4. Genera clientes simulados con `cliente_id`, sector, antigüedad y garantía.
5. Genera facturas por cliente, respetando orden cronológico global y calculando:
   - `fecha_vencimiento` según `condicion_dias`
   - `dias_mora_real`
   - `target_mora` según la clase de mora simulada
   - `monto`
6. Simula gestiones de cobranza solo para facturas con mora, con:
   - gestiones preventivas y reactivas
   - escalamiento de canal
   - coherencia entre `contacto_exitoso` y `resultado`
   - registro de `motivo_no_pago` cuando aplica
7. Simula promesas de pago para gestiones cuyo resultado es `promesa_de_pago`, con cálculo de `fecha_compromiso` y `se_cumplio`.
8. Construye `features_ml` con cortes temporales:
   - corte 0 al emitir la factura
   - cortes posteriores tras cada gestión
   - mantiene solo información disponible hasta `fecha_corte`
   - conserva el mismo `target` final por todas las filas de una factura
9. Valida integridad de datos y coherencia temporal.

### Continuidad
Este notebook es la fuente de verdad de la simulación. Produce datos que luego son analizados en el EDA y preparados para entrenamiento en el notebook de preparación.

## 2. `EDA/EDA_cobranzas.ipynb`

### Propósito
Realizar el análisis exploratorio de datos (EDA) sobre las tablas simuladas y el dataset de features, validando calidad, estructuras y patrones relevantes para modelado.

### Lógica principal
1. Carga los datos transaccionales y de features desde archivos CSV.
2. Realiza diagnóstico inicial de la estructura: filas, columnas, tipos, nulos y cardinalidad.
3. Verifica integridad referencial y consistencia temporal entre tablas.
4. Analiza estadísticas descriptivas de variables numéricas clave.
5. Estudia distribuciones de `target` a nivel de factura y de cortes temporales.
6. Identifica outliers plausibles y discute su relevancia de negocio.
7. Calcula correlaciones y examina redundancias entre variables.
8. Explora variables categóricas como sector, perfil de pago, canal y resultado de gestión.
9. Revisa relaciones importantes, como `target` vs `perfil_pago`, `target` vs `condicion_dias` y `target` por número de corte.
10. Enumera problemas potenciales para el modelado y da recomendaciones metodológicas.

### Continuidad
El EDA valida que la simulación es coherente con las reglas de negocio, confirma que `features_ml` es el dataset correcto para entrenamiento y proporciona los criterios técnicos para la limpieza y el split de datos.

## 3. `Presentación de la fase de preparación y procesamiento de datos/notebook_preparación.ipynb`

### Propósito
Preparar el dataset final para modelado: limpiar, transformar, investigar outliers, dividir por factura y construir un pipeline reproducible.

### Lógica principal
1. Carga `features_ml.csv` desde `artifacts/01_generacion/` y renombra el target si es necesario.
2. Valida la estructura del dataset:
   - ausencia de duplicados exactos
   - ausencia de duplicados por `(factura_id, num_corte)`
   - fechas válidas y montos positivos
   - codificación sectorial correcta
3. Limpia nulos estructurales y crea indicadores como `sin_gestion_previa`.
4. Ajusta `ultimo_resultado_enc` para convertirlo en categoría segura y elimina columnas redundantes si son determinísticas.
5. Crea nuevas features operativas:
   - `dias_transcurridos_corte`
   - `esta_vencida_al_corte`
   - `dias_mora_observable`
   - `dias_hasta_vence_pos`
   - `cliente_nuevo`
   - `intensidad_gestion`
   - `friccion_contacto`
   - `ratio_promesas_rotas`
6. Detecta outliers usando IQR y documenta su presencia sin eliminarlos automáticamente.
7. Realiza el split por `factura_id` para evitar fuga de información temporal.
8. Construye un preprocesador con:
   - imputación
   - transformación logarítmica de variables sesgadas
   - escalado robusto
   - codificación one-hot de categorías
9. Calcula pesos de clase para balancear el entrenamiento.
10. Define un protocolo experimental de data augmentation como paso opcional, separado del flujo principal.
11. Exporta artefactos preparados: dataset, IDs de train/test y metadatos.

### Continuidad
Este notebook transforma `features_ml` en un conjunto listo para entrenamiento y respeta la regla clave: dividir por `factura_id`, no por fila. Es el eslabón final antes de la etapa de modelado.

## 4. Observaciones generales sobre continuidad y coherencia

- La lógica del proyecto es consistente: la simulación se alinea con las reglas de negocio, el EDA valida su calidad y la preparación crea un pipeline reproducible.
- Los tres notebooks siguen el flujo natural de un proyecto de IA académico:
  1. simulación de datos
  2. análisis exploratorio
  3. preparación para modelado
- Las reglas de negocio documentadas en `reglas_de_negocio_sistema_cobranzas.md` están implementadas de forma clara en la simulación y en las verificaciones.
- La arquitectura de datos responde al objetivo de un sistema de scoring dinámico por cortes.

## 5. Punto de atención

- El notebook de preparación ya usa una ruta canónica relativa al proyecto, por lo que no depende de `/mnt/data` ni de un entorno Linux específico.
- Los notebooks no están ejecutados actualmente en el repositorio (según el metadato de celdas), por lo que la validación de ejecución real debe hacerse en el entorno correcto.

## 6. Recomendación

Para completar la continuidad entre notebooks, se recomienda:
- verificar que `features_ml.csv` exista en `artifacts/01_generacion/` y contenga `target` o `target_mora` según corresponda
- revisar `AGENTS.md` si se quiere un contexto compacto del sistema antes de continuar con nuevas fases
- ejecutar el EDA y el notebook de preparación tras generar los datos con el notebook de simulación, asegurando que los artefactos intermedios se generan en el orden esperado.

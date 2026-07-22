# Modelo estocástico multivariado de afluencias

Módulo **aislado** del resto del proyecto (fase 1 de Sebastian): no importa ni
modifica `hydrology.balance` ni `environment.entorno`. Conectarlo al entorno
de simulación es un paso posterior, una vez validado — este módulo se
desarrolla y se prueba de forma completamente independiente.

## Propósito

Un solo modelo que recibe covariables exógenas (precipitación, RONI) y genera
4 salidas correlacionadas: caudal de Saucío y afluencia de Neusa, Sisga y
Tominé. Un modelo **conjunto**, no cuatro independientes, porque las cuatro
series pertenecen a la misma cuenca y están correlacionadas por el mismo
evento de lluvia — cuatro modelos separados podrían generar escenarios
físicamente inconsistentes (p. ej. Neusa en crecida mientras Sisga está en
sequía bajo la misma lluvia).

## Método

**VARX desestacionalizado con componente hurdle** (`modelo.py`) — ver el
docstring de ese módulo para la justificación completa frente a las otras dos
alternativas evaluadas (regresión + cópula; remuestreo por análogos) y las
simplificaciones documentadas.

**Remuestreo por análogos** (`analogos.py`) es una **línea base de
validación**, no el método final: sirve para confirmar que el VARX no se
aleja del comportamiento observado antes de confiarle la extrapolación fuera
del rango histórico (necesaria para estresar el shield con escenarios más
severos que el histórico, en el mismo espíritu que el escenario "ampliado" de
Tibitóc).

## Cómo entrenar

```bash
python -m pbcrl.stochastic.entrenamiento
```

Reutiliza los loaders ya existentes del proyecto (`dashboard.data_loader`,
`data_ingest.saucio`, `data_ingest.roni`) — no duplica ninguna lectura de
datos. Divide los datos de forma **temporal** (los últimos 2 años como
validación, no una partición aleatoria, porque son series de tiempo), ajusta
el VARX y el remuestreo por análogos sobre el período de entrenamiento, e
imprime un reporte comparando: correlación cruzada de las 4 series, % de días
en estado bajo por serie, y medias mensuales — histórico vs. VARX simulado
(promediado sobre varias semillas) vs. análogos.

El modelo VARX ajustado se guarda en `artefactos/` (ignorado por git: un
modelo entrenado con datos reales es un artefacto derivado de datos, no
código — ver `.gitignore`).

## Cómo usar el modelo ya entrenado

```python
from pbcrl.stochastic.modelo import ModeloEstocasticoAfluencias

modelo = ModeloEstocasticoAfluencias.cargar("src/pbcrl/stochastic/artefactos/modelo_afluencias_v1")
muestra = modelo.sample(covariables_futuras, semilla=42)
# muestra: DataFrame con columnas Saucio, Neusa, Sisga, Tomine [m³/s]
```

`covariables_futuras` es un DataFrame con índice de fechas y columnas
`precipitacion_mm` y `roni` (ver `ConfigModeloEstocastico.covariables`).

## Covariables: por qué solo precipitación y RONI

Decisión del usuario (2026-07-22): la temperatura no existe todavía en el
proyecto (no hay loader ni fuente integrada — no hay pipeline de GEE, pese a
que se mencionaba como "eventual" en `data_contracts/ventana.py`), y su efecto
es de segundo orden (evapotranspiración) frente a precipitación y ENSO.

El modelo queda preparado para una tercera covariable con el **mismo patrón
de configuración por escenario** ya usado para la extracción de Tibitóc
(`data_contracts.captaciones`): `ConfigModeloEstocastico.covariables` es una
tupla de nombres de columna, no un número fijo de argumentos. Agregar
temperatura el día que exista una fuente es agregar su nombre a esa tupla y su
columna al DataFrame de covariables — sin tocar la lógica de `modelo.py`.

## Validación — v1 ACEPTADO para el entregable (2026-07-22)

Se probó primero con un split simple (últimos 2 años), que mostró
sobreestimación sistemática en meses húmedos. Se descartó sesgo de
retransformación logarítmica como causa (`sample()` simula ruido real antes
de transformar — no necesita smearing — y el sesgo no aparecía igual en
entrenamiento que en validación, lo que sí sería necesario si fuera un
artefacto de la transformación). Se corrió después una validación cruzada
temporal de 5 bloques (ventana expansiva — ver "Deuda técnica" abajo).
Resultado, aceptado con tres salvedades declaradas:

1. **El error de validación está dominado por el evento operativo de
   vertimiento de Neusa** (jul-nov 2022, ver NOTAS.md 4f) cayendo en la
   ventana de validación de los bloques que lo contienen: sesgo de Neusa
   aislado hasta 4.87 m³/s en el bloque afectado, vs. ~1.08 m³/s del resto de
   series en el mismo bloque. Ese evento NO es predecible desde
   precipitación/RONI porque no lo generó el clima — es una anomalía
   operativa de un solo embalse, pendiente de verificación con la CAR, que
   podría implicar corrección de los propios datos de entrenamiento más
   adelante.
2. Excluido ese confusor, queda una señal **débil** de dependencia entre
   distancia climática (|ΔRONI| entre el bloque validado y su entrenamiento)
   y el error de validación (correlación 0.31, n=5 bloques — no concluyente).
   Se documenta como **hipótesis** a revisar si el modelo se refina después
   (candidatos: interacción RONI×estacionalidad, o coeficientes del VARX
   condicionados a la fase ENSO), **no** como defecto establecido.
3. El modelo **subestima la correlación cruzada de Neusa** con las demás
   series (0.15-0.20 simulado vs. 0.26-0.31 histórico). Lectura más probable:
   Neusa abastece acueductos además de responder al clima — una componente de
   demanda que un modelo puramente climático no puede capturar por diseño, no
   necesariamente un error del modelo.

Script de la validación de 5 bloques: `scratch_stochastic/validacion_bloques.py`
(fuera del paquete — es un diagnóstico, no código de producción).

## Deuda técnica (a corregir cuando se itere el modelo, no ahora)

`ModeloEstocasticoAfluencias.fit()` construye los rezagos autorregresivos del
VARX por **posición de fila, no por fecha**. Si se le pasan datos de
entrenamiento con un hueco temporal real (p. ej. para dejar un bloque de
validación en medio del período), el modelo trataría el último día antes del
hueco y el primero después como si fueran calendario-consecutivos — un
rezago espurio que contaminaría Phi. Por esta razón la validación de 5
bloques se hizo con **ventana expansiva** (cada pliegue entrena con un solo
tramo contiguo, antes o después del bloque, nunca ambos) en vez de "entrenar
con todo lo demás" como se planteó originalmente. Corregir esto (para poder
entrenar con huecos temporales reales) requiere que `fit()` reconozca la
fecha de cada fila y resetee la memoria autorregresiva en los saltos de
calendario, no solo en los días de estado "bajo" del hurdle (que ya maneja
correctamente).

## Otras simplificaciones a revisar después de validar más a fondo

Ver el docstring de `modelo.py`, sección "SIMPLIFICACIONES DOCUMENTADAS":
la ocurrencia del estado hurdle se sortea independientemente entre las 4
series (la correlación cruzada vive en el componente VARX continuo, no en el
hurdle); si una validación futura muestra que esto importa, el siguiente paso
es un mecanismo de cópula latente sobre la ocurrencia (p. ej. Wilks 1998).

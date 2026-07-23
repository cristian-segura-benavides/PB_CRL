# Shield de proyección cuadrática

**La contribución central de PB-CRL.** El módulo en sí (`restricciones.py`,
`proyeccion.py`) sigue aislado: no importa `environment.entorno` ni
`environment.hidraulica` ni `hydrology.balance` (ver más abajo el import
circular que esto evita). Lo que SÍ está conectado, opcionalmente, es
`environment.entorno`, que puede llamar a este módulo — ver "Conexión al
entorno" abajo. No se tocó `hidraulica.py` ni `balance.py`.

## Qué hace

Intercepta la acción propuesta por el agente de RL (â_t: los tres caudales de
suministro para Neusa, Sisga y Tominé) y la proyecta sobre el conjunto de
acciones factibles del paso actual:

```
a*_t = argmin_a ||a − â_t||²   sujeto a   g_k(s_t, a) ≤ 0  para todo k
```

Esto ocurre **antes** del recorte físico de `hidraulica.recortar_suministro`.
Son dos mecanismos distintos: el shield corrige la *intención* del agente
para no violar restricciones del sistema (incluida la conjunta entre los tres
embalses); el recorte físico verifica *después* si esa acción ya corregida es
alcanzable con el agua realmente disponible en cada embalse ese día.

## Las tres restricciones

1. **Cajas individuales** — capacidad de la torre de toma/compuertas de cada
   embalse (`data_contracts.embalses.EMBALSES[...].descarga_max_m3s`).
2. **Rata de descenso de Sisga** (solo Sisga, restricción dura del manual de
   operación CAR) — se traduce a un límite superior *dinámico* sobre Q_Sisga,
   linealizando el balance alrededor del volumen actual. Ver el docstring de
   `restricciones.py` para la derivación completa.
3. **Caudal ecológico conjunto** — la que justifica el Safe MARL: acopla las
   tres variables de decisión vía `Q_ElSol = Q_Saucío + Q_Neusa + Q_Sisga +
   Q_Tomine − Q_extracción_nominal ≥ Q_eco(mes)`. Reutiliza `data_contracts.
   captaciones` (extracción) y `data_contracts.caudal_ecologico` (umbral VMF)
   — no reimplementa ninguna de las dos.

## Método de solución

Bisección sobre el multiplicador de KKT de la única restricción de acople
(caja ∩ un semiespacio tiene una estructura muy simple: para cada λ≥0, el
minimizador es la caja proyectada de `â − λ·c`, y la función dual es
monótona). **No se usó una librería de QP** (el proyecto no tenía ninguna
dependencia de optimización, y con 3 variables y una sola restricción de
acople una librería externa sería desproporcionada frente a un método
analítico simple y exactamente verificable con pruebas unitarias). Si el
shield gana más restricciones de acople simultáneas en el futuro, este
método deja de alcanzar y ahí sí se justificaría una librería de QP general
— ver el docstring de `proyeccion.py` para el detalle.

## Uso

```python
from pbcrl.shield import EstadoShield, proyectar

estado = EstadoShield(
    volumen_mm3={"Neusa": 50.0, "Sisga": 50.0, "Tomine": 300.0},
    afluencia_m3s={"Neusa": 2.0, "Sisga": 1.5, "Tomine": 8.0},
    precipitacion_mm={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
    evaporacion_mm={"Neusa": 2.0, "Sisga": 1.0, "Tomine": 1.5},
    caudal_saucio_m3s=3.0,
    mes=7,
)
diagnostico = proyectar(estado, {"Neusa": 2.0, "Sisga": 2.0, "Tomine": 2.0})

diagnostico.accion_proyectada     # a*_t, dict por embalse
diagnostico.violaciones_previas   # qué violaba â_t antes de proyectar
diagnostico.restricciones_activas # qué queda en el borde en a*_t
diagnostico.factible              # False si el conjunto factible es VACÍO
```

## Verificación contra estados históricos

`scratch_shield/verificacion_historica.py` corre el shield contra los 4607 días
con los cuatro componentes completos (2012-01-02 → 2025-05-04), con la acción
propuesta = la descarga REAL observada ese día, sin ningún agente todavía —
solo para verificar factibilidad de los datos históricos.

| | Histórico (4.5 m³/s) | Ampliado (8.0 m³/s) |
|---|---|---|
| Ya factible sin corrección | 72.39% | 33.17% |
| El shield habría corregido | 27.61% (casi todo Nivel 2) | 66.83% (casi todo Nivel 2) |
| Infactibles (conjunto vacío) | 0 | 0 |
| Violación caja Sisga (descenso) | 4 días (0.09%) | 4 días (0.09%) |

Cero días verdaderamente infactibles en todo el rango histórico, en ambos
escenarios — el diseño de restricciones no colapsa el conjunto factible en
ningún estado observado.

**Validación cruzada no buscada:** el % de días donde el shield habría
corregido (27.61% / 66.83%) coincide, de forma independiente, con el % de
días en violación del VMF calculado en la sesión del umbral (27.13% / 66.75%,
ver `data_contracts/caudal_ecologico.py` y NOTAS.md 4f) — dos piezas del
proyecto construidas por separado llegan al mismo número.

## Conexión al entorno (`ConfigEntorno.con_shield`)

`environment.config.ConfigEntorno` gana `con_shield: bool = False` (por
defecto no cambia nada). Con `con_shield=True`,
`environment.entorno.EntornoEmbalses.step()` proyecta la acción propuesta
con `proyectar()` **antes** del recorte físico de
`hidraulica.recortar_suministro` — orden: acción propuesta → shield →
recorte físico → balance. `ResultadoPaso` gana el campo `diagnostico_shield`
(`None` si `con_shield=False`).

**Import circular evitado a propósito:** `restricciones.py` NO importa
`environment.config` (aunque comparte el mismo valor numérico que
`ConfigEntorno.sisga_descenso_umbral_cm`, documentado localmente con
referencia cruzada) — importar un submódulo de `environment` fuerza a cargar
`environment/__init__.py`, que importa `entorno.py`, que ahora importa
`shield`. Es el motivo concreto por el que el shield se mantiene sin ninguna
dependencia de `environment`.

**Dos bugs de precisión numérica encontrados y corregidos al conectar** (ver
NOTAS.md 4i para el detalle): la bisección de `proyeccion.py` dejaba una
holgura de `1e-9` en cada iteración (corregido: comparación estricta dentro
de la bisección, `tol` solo en los chequeos previos); y la detección de
violación en `entorno.py` (`q_sol < q_eco`) era estricta sin margen, lo que
en la primera corrida marcó como "violación" 366 de 400 casos que en
realidad eran ruido de punto flotante (~1e-15) del sistema quedando
exactamente en el borde de la restricción — corregido con una tolerancia de
`1e-9` (`_TOL_VIOLACION_ECOLOGICA_M3S`).

## Rollout histórico completo (dinámica + shield en cada paso)

`scratch_shield/simulacion_historica_con_shield.py` corre la simulación
CONTINUA (no snapshots independientes como la verificación de arriba):
volumen inicial = histórico del primer día, después evoluciona por la
dinámica propia del entorno; acción propuesta = descarga real observada
día a día; 4606 días, 2012-01-02 → 2025-05-04.

| | Histórico (4.5 m³/s) | Ampliado (8.0 m³/s) |
|---|---|---|
| Shield corrigió | 27.62% (1272 días) | 66.85% (3079 días) |
| Violación REAL de Q_eco (tras limpiar ruido numérico) | 34 días (0.74%) | 1262 días (27.4%) |
| Volumen fuera de rango | 0 días | 0 días |

**La garantía no se sostiene el 100% del tiempo — verificado por qué, no
dejado como caja negra:** el 100% de los déficits reales restantes coincide
exactamente con el recorte físico entregando menos de lo que el shield
pidió (agua genuinamente no disponible ese día) — el límite documentado
desde el diseño: el shield garantiza la acción *corregida*, no el resultado
físico final si el agua no existe. Bajo ampliado, el 22.4% de esos días
también tenía la cota física de extracción de Tibitóc activada.

**Lectura:** bajo extracción histórica el límite es marginal. Bajo
extracción ampliada, un shield puramente reactivo (sin ningún agente con
previsión) termina drenando los embalses lo suficiente en 13 años como para
no poder cumplir la garantía en más de una cuarta parte de los días
(déficit mediano 1.95 m³/s, máximo 6.53 m³/s) — evidencia concreta de que
hace falta el agente, o una versión del shield con la formulación de
"mínima violación"/recuperación, para el caso de estrés hídrico real.

Pruebas de integración: `tests/test_entorno.py::TestShieldEnElEntorno`
(regresión sin shield, garantía con agua disponible, barrido de los 12
meses, y el límite documentado sin agua disponible).

"""Captaciones sobre el sistema: extracción de la Planta de Tibitóc.

Topología confirmada (decisión del usuario 2026-07-21, ver NOTAS.md)
------------------------------------------------------------------------
Saucío -> confluencia Sisga -> confluencia Tominé -> confluencia Neusa ->
bocatoma Tibitóc (extracción) -> El Sol.

Las CUATRO entradas (Saucío + las descargas de Sisga, Tominé y Neusa) llegan a
la bocatoma ANTES de la extracción:

    Q_bocatoma(t) = Q_Saucío(t) + Q_desc_Sisga(t) + Q_desc_Tominé(t) + Q_desc_Neusa(t)

No hay término de afluencia lateral ni Q_otros: no hay datos que los sustenten y
la ecuación cierra sin ellos (ver NOTAS.md).

Por qué es un escenario y no un dato medido
-------------------------------------------------
No existe serie pública de captación de la Planta de Tibitóc: no está en datos
abiertos, y no hay ninguna estación de caudal aguas abajo de la bocatoma desde
la que se pueda estimar por diferencia (Puente Tocancipá, que se había
considerado, está AGUAS ARRIBA de la bocatoma). Por eso la extracción se modela
mediante ESCENARIOS documentados, seleccionables sin tocar el resto del código
(``environment.entorno.ForzantesExternos.caudal_tibitoc_m3s``) y reemplazables
por una serie real día a día si se obtiene — basta con pasar esa serie en vez
de una de las constantes de este módulo, sin cambiar la lógica del entorno.

Escenarios
----------
HISTÓRICO (``CAUDAL_TIBITOC_HISTORICO_M3S`` = 4.5 m³/s constante):
  Derivación promedio reportada por la CAR en su "informe de recorrido del río
  Bogotá". PENDIENTE DE TRAZABILIDAD: esta cita NO tiene radicado, fecha ni
  página — es más débil que el resto de fuentes del proyecto (que sí tienen
  radicado: 20261625095, 20261071858, ENL-002443-2026-S, Resolución 760/2011).
  Completar la cita exacta antes de publicar.
  VALIDACIÓN CRUZADA: con este escenario, el modelo produce una media de
  6.34 m³/s en El Sol sobre la ventana del proyecto, coherente con los
  6.57 m³/s que la CAR reporta aguas abajo de la desembocadura del Neusa — es
  decir, usando el valor de extracción que reporta la CAR se reproduce (sin
  ajustar nada) el caudal que la misma CAR mide aguas abajo. Corresponde a la
  operación durante la ventana de análisis del proyecto (2012-01-01 a
  2025-05-04). El valor de referencia (6.57 m³/s) también está PENDIENTE DE
  TRAZABILIDAD por la misma razón (sin radicado ni documento identificado).
  ADEMÁS, esta validación presupone que "aguas abajo de la desembocadura del
  Neusa" es el MISMO punto que el punto de control El Sol — un supuesto NO
  verificado. Dado que la topología de Neusa respecto a la bocatoma ya fue
  objeto de una confusión documental real (ver NOTAS.md 4e), esa equivalencia
  de puntos merece confirmarse explícitamente con el asesor antes de apoyarse
  en esta validación cruzada como evidencia fuerte.

AMPLIADO (``CAUDAL_TIBITOC_AMPLIADO_M3S`` = 8.0 m³/s constante):
  Caudal tratado reportado tras la optimización reciente de la planta (nuevos
  trenes de tratamiento). Representa la operación HACIA la que evoluciona el
  sistema, no la operación histórica de la ventana de análisis.

NOTA METODOLÓGICA — ambos escenarios tienen respaldo documental, pero NO
representan el mismo período: el histórico corresponde a la operación durante
la ventana de análisis; el ampliado, a una condición de mayor demanda urbana
posterior/futura. Se evalúan ambos para analizar la SENSIBILIDAD del sistema al
supuesto de extracción, no como alternativas igualmente representativas del
mismo momento — no promediarlos ni tratarlos como intercambiables.

Cota física sobre la extracción
--------------------------------
La planta no puede captar más agua de la que el río trae en la bocatoma en ese
instante. La extracción real (ver ``environment.hidraulica.calcular_extraccion_tibitoc``)
siempre respeta:

    Q_extraccion(t) = min(Q_Tibitoc_escenario, Q_bocatoma(t))
    Q_ElSol(t)       = Q_bocatoma(t) - Q_extraccion(t)      # siempre >= 0

Esta cota es la que elimina los caudales negativos que aparecían al restar un
valor nominal fijo sin considerar el caudal disponible en la bocatoma.
"""
from __future__ import annotations

ESCENARIO_HISTORICO = "historico"
ESCENARIO_AMPLIADO = "ampliado"

# Fuente: informe de recorrido del río Bogotá, CAR. PENDIENTE DE TRAZABILIDAD:
# falta radicado, fecha y página exactos (ver docstring del módulo). Validado
# cruzadamente contra el caudal medido por la CAR aguas abajo de la
# desembocadura del Neusa (6.57 m³/s; ese valor de referencia también está
# PENDIENTE DE TRAZABILIDAD, y la equivalencia de ese punto con El Sol no está
# verificada — ver docstring del módulo antes de citar esta validación).
CAUDAL_TIBITOC_HISTORICO_M3S: float = 4.5

# Fuente: caudal tratado reportado tras la optimización reciente de la planta
# (nuevos trenes de tratamiento). Condición de demanda futura, no histórica.
CAUDAL_TIBITOC_AMPLIADO_M3S: float = 8.0

CAUDAL_TIBITOC_ESCENARIOS: dict[str, float] = {
    ESCENARIO_HISTORICO: CAUDAL_TIBITOC_HISTORICO_M3S,
    ESCENARIO_AMPLIADO: CAUDAL_TIBITOC_AMPLIADO_M3S,
}


def caudal_tibitoc_nominal(escenario: str) -> float:
    """Caudal nominal de extracción de Tibitóc [m³/s] para el escenario dado.

    Parámetros
    ----------
    escenario : str
        ``ESCENARIO_HISTORICO`` o ``ESCENARIO_AMPLIADO``.

    Retorna
    -------
    float
        Caudal nominal [m³/s]. NO es la extracción efectiva: la extracción
        real se acota por el caudal disponible en la bocatoma (ver
        ``environment.hidraulica.calcular_extraccion_tibitoc``).

    Raises
    ------
    ValueError
        Si el escenario no está definido.
    """
    try:
        return CAUDAL_TIBITOC_ESCENARIOS[escenario]
    except KeyError as exc:
        raise ValueError(
            f"Escenario de captación de Tibitóc desconocido: {escenario!r}. "
            f"Escenarios válidos: {sorted(CAUDAL_TIBITOC_ESCENARIOS)}."
        ) from exc

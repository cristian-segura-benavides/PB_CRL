"""
Funciones hidráulicas puras para el entorno de simulación.

Implementan la dinámica de transición *hacia adelante* (forward) de cada embalse.
Son el complemento del balance hídrico *inverso* de hydrology/balance.py:
    - balance.py:    observaciones → afluencia (inferencia)
    - hidraulica.py: afluencia + acción → nuevo volumen (simulación)

CONVERSIONES DE UNIDADES (idénticas a las de balance.py)
---------------------------------------------------------
  Q [m³/s] × 0.0864                = V [Mm³/día]
  L [mm]   × A [km²] × 1e-3       = V [Mm³]
"""
from __future__ import annotations

from pbcrl.data_contracts.embalses import ParametrosEmbalse
from pbcrl.data_contracts.curvas import volumen_a_cota  # re-exportado desde curvas.py

# Constantes de conversión (duplicadas aquí para que este módulo sea autocontenido)
_S_POR_DIA: float = 86_400.0
_M3S_A_MM3_DIA: float = _S_POR_DIA / 1e6    # 0.0864  [m³/s → Mm³/día]
_MM_KM2_A_MM3: float = 1e-3                  # [mm × km² → Mm³]


def recortar_suministro(
    suministro_pedido_m3s: float,
    volumen_actual_mm3: float,
    params: ParametrosEmbalse,
) -> float:
    """Recorta el caudal suministrado a lo físicamente posible en un paso diario.

    Restricciones aplicadas en orden:
    1. No puede ser negativo (no se bombea agua al embalse por esta vía).
    2. No puede superar la capacidad máxima de las compuertas/torre de toma.
    3. No puede extraer agua por debajo del volumen muerto: el agua útil disponible
       es (volumen_actual - capacidad_min_mm3), expresada como caudal máximo equivalente.

    Parámetros
    ----------
    suministro_pedido_m3s : float
        Caudal que el agente desea suministrar [m³/s].
    volumen_actual_mm3 : float
        Volumen almacenado al inicio del paso [Mm³].
    params : ParametrosEmbalse
        Parámetros del embalse.

    Retorna
    -------
    float
        Caudal suministrado real [m³/s], nunca superior al disponible físicamente.
    """
    # Agua sobre el volumen muerto, expresada como caudal equivalente diario
    agua_util_mm3 = max(0.0, volumen_actual_mm3 - params.capacidad_min_mm3)
    suministro_max_fisico_m3s = agua_util_mm3 / _M3S_A_MM3_DIA

    suministro_real = max(0.0, suministro_pedido_m3s)
    suministro_real = min(suministro_real, params.descarga_max_m3s)
    suministro_real = min(suministro_real, suministro_max_fisico_m3s)
    return suministro_real


def paso_embalse(
    volumen_actual_mm3: float,
    afluencia_m3s: float,
    suministro_m3s: float,
    precipitacion_mm: float,
    evaporacion_mm: float,
    params: ParametrosEmbalse,
) -> tuple[float, float, float]:
    """Aplica la ecuación de balance hídrico hacia adelante en un paso diario.

    Ecuación (todas las cantidades en Mm³):

        V_bruto = V(t-1) + afluencia - suministro + precipitación - evaporación
        vertimiento = max(0, V_bruto - capacidad_max)
        V_sin_piso = V_bruto - vertimiento
        deficit = max(0, capacidad_min - V_sin_piso)
        V(t)    = V_sin_piso + deficit    # == clamp(V_sin_piso, capacidad_min, capacidad_max)

    El clamp inferior cubre el caso extremo de evaporación muy alta con embalse casi
    vacío; en condiciones normales, recortar_suministro ya garantiza V_bruto ≥
    capacidad_min. Este acotamiento es físicamente correcto (el volumen no puede bajar
    del volumen muerto), pero por sí solo "crea" masa silenciosamente cuando se activa:
    `deficit_mm3` cuantifica exactamente cuánta, para que quede diagnosticable y
    acumulable en vez de perderse sin dejar rastro (análogo a `deficit_extraccion_m3s`
    de la extracción de Tibitóc, ver `calcular_extraccion_tibitoc`). NO representa una
    entrada física real: es la magnitud del ajuste que el clamp tuvo que aplicar.

    Parámetros
    ----------
    volumen_actual_mm3 : float
        Volumen al inicio del paso [Mm³].
    afluencia_m3s : float
        Afluencia natural al embalse en este paso [m³/s].
    suministro_m3s : float
        Caudal suministrado (ya recortado por recortar_suministro) [m³/s].
    precipitacion_mm : float
        Precipitación sobre el espejo del embalse [mm/día].
    evaporacion_mm : float
        Evaporación sobre el espejo del embalse [mm/día].
    params : ParametrosEmbalse
        Parámetros físicos del embalse.

    Retorna
    -------
    tuple[float, float, float]
        (nuevo_volumen_mm3 [Mm³], vertimiento_mm3 [Mm³], deficit_volumen_mm3 [Mm³])
        deficit_volumen_mm3 es cero salvo cuando se activó el clamp inferior.
    """
    # Conversión de unidades
    afluencia_mm3 = afluencia_m3s * _M3S_A_MM3_DIA
    suministro_mm3 = suministro_m3s * _M3S_A_MM3_DIA
    prec_mm3 = precipitacion_mm * params.area_espejo_km2 * _MM_KM2_A_MM3
    evap_mm3 = evaporacion_mm * params.area_espejo_km2 * _MM_KM2_A_MM3

    v_bruto = volumen_actual_mm3 + afluencia_mm3 - suministro_mm3 + prec_mm3 - evap_mm3

    # Aliviadero: vierte el exceso sobre la capacidad máxima
    vertimiento_mm3 = max(0.0, v_bruto - params.capacidad_max_mm3)
    v_sin_piso = v_bruto - vertimiento_mm3

    # Cota inferior de seguridad (no bajar del volumen muerto). deficit_volumen_mm3
    # registra cuánta masa exigió el ajuste (ver docstring); cero si no se activó.
    deficit_volumen_mm3 = max(0.0, params.capacidad_min_mm3 - v_sin_piso)
    v_nuevo = v_sin_piso + deficit_volumen_mm3

    return v_nuevo, vertimiento_mm3, deficit_volumen_mm3


def calcular_extraccion_tibitoc(caudal_bocatoma_m3s: float, caudal_nominal_m3s: float) -> float:
    """Extracción real de la Planta de Tibitóc, acotada por el caudal disponible.

    La planta no puede captar más agua de la que el río trae en la bocatoma en
    ese instante (ver data_contracts.captaciones para la topología y los
    escenarios de caudal nominal):

        Q_extraccion = min(Q_nominal, Q_bocatoma)

    Esta cota es la que garantiza que el caudal en El Sol (Q_bocatoma -
    Q_extraccion) nunca sea negativo, a diferencia de restar un valor nominal
    fijo sin considerar el caudal disponible.

    Parámetros
    ----------
    caudal_bocatoma_m3s : float
        Caudal disponible en la bocatoma antes de la extracción [m³/s]
        (Saucío + descargas de Sisga, Tominé y Neusa).
    caudal_nominal_m3s : float
        Caudal nominal que la planta busca captar [m³/s] (escenario o serie real).

    Retorna
    -------
    float
        Extracción real [m³/s], nunca negativa ni superior al caudal disponible.
    """
    disponible = max(0.0, caudal_bocatoma_m3s)
    return min(max(0.0, caudal_nominal_m3s), disponible)

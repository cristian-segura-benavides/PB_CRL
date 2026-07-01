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

# Constantes de conversión (duplicadas aquí para que este módulo sea autocontenido)
_S_POR_DIA: float = 86_400.0
_M3S_A_MM3_DIA: float = _S_POR_DIA / 1e6    # 0.0864  [m³/s → Mm³/día]
_MM_KM2_A_MM3: float = 1e-3                  # [mm × km² → Mm³]


def volumen_a_cota(volumen_mm3: float, params: ParametrosEmbalse) -> float:
    """Convierte volumen almacenado a cota mediante interpolación lineal.

    PROVISIONAL: usa una relación lineal entre los puntos extremos operativos.
    Debe reemplazarse con la curva batimétrica real (tabla h-V) cuando esté disponible.

    Parámetros
    ----------
    volumen_mm3 : float
        Volumen almacenado [Mm³].
    params : ParametrosEmbalse
        Parámetros del embalse (define los extremos de la interpolación).

    Retorna
    -------
    float
        Cota estimada [m.s.n.m.]. Acotada entre cota_min_m y cota_max_m.
    """
    rango_vol = params.capacidad_max_mm3 - params.capacidad_min_mm3
    fraccion = (volumen_mm3 - params.capacidad_min_mm3) / rango_vol
    fraccion = max(0.0, min(1.0, fraccion))
    return params.cota_min_m + fraccion * (params.cota_max_m - params.cota_min_m)


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
) -> tuple[float, float]:
    """Aplica la ecuación de balance hídrico hacia adelante en un paso diario.

    Ecuación (todas las cantidades en Mm³):

        V_bruto = V(t-1) + afluencia - suministro + precipitación - evaporación
        vertimiento = max(0, V_bruto - capacidad_max)
        V(t)    = clamp(V_bruto - vertimiento, capacidad_min, capacidad_max)

    El clamp inferior cubre el caso extremo de evaporación muy alta con embalse casi vacío;
    en condiciones normales, recortar_suministro ya garantiza V_bruto ≥ capacidad_min.

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
    tuple[float, float]
        (nuevo_volumen_mm3 [Mm³], vertimiento_mm3 [Mm³])
    """
    # Conversión de unidades
    afluencia_mm3 = afluencia_m3s * _M3S_A_MM3_DIA
    suministro_mm3 = suministro_m3s * _M3S_A_MM3_DIA
    prec_mm3 = precipitacion_mm * params.area_espejo_km2 * _MM_KM2_A_MM3
    evap_mm3 = evaporacion_mm * params.area_espejo_km2 * _MM_KM2_A_MM3

    v_bruto = volumen_actual_mm3 + afluencia_mm3 - suministro_mm3 + prec_mm3 - evap_mm3

    # Aliviadero: vierte el exceso sobre la capacidad máxima
    vertimiento_mm3 = max(0.0, v_bruto - params.capacidad_max_mm3)
    v_nuevo = v_bruto - vertimiento_mm3

    # Cota inferior de seguridad (no bajar del volumen muerto)
    v_nuevo = max(params.capacidad_min_mm3, v_nuevo)

    return v_nuevo, vertimiento_mm3

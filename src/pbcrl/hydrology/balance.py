"""
Balance hídrico inverso para embalses.

Ecuación de conservación de masa (paso diario):

    afluencia(t) = ΔV(t) + descarga(t) + evaporación_vol(t)
                   - precipitación_vol(t) + vertimiento(t)

donde:
    ΔV(t) = V(t) - V(t-1)                    [Mm³]
    descarga(t)          : salida controlada   [Mm³]  ← convertida de m³/s
    evaporación_vol(t)   : lámina × área       [Mm³]  ← convertida de mm × km²
    precipitación_vol(t) : lámina × área       [Mm³]  ← convertida de mm × km²
    vertimiento(t)       : salida por aliviadero [Mm³] ← calculada internamente

CONVERSIONES DE UNIDADES (documentadas explícitamente para evitar errores)
---------------------------------------------------------------------------
1. Caudal → volumen diario:
       Q [m³/s] × 86 400 [s/día] = V [m³/día]
       V [m³/día] ÷ 1 000 000   = V [Mm³/día]
   Factor compuesto: Q [m³/s] × 86 400 / 1e6 = Q × 0.0864 [Mm³/día]

2. Lámina de agua → volumen:
       L [mm] × A [km²] = L [mm] × A × 1e6 [m²] ÷ 1000 [mm/m] = L × A × 1e3 [m³]
       L × A × 1e3 [m³] ÷ 1e6                                  = L × A × 1e-3 [Mm³]
   Factor compuesto: L [mm] × A [km²] × 1e-3 = vol [Mm³]

Todas las cantidades internas están en Mm³.  La afluencia se devuelve en m³/s
(unidad hidráulica estándar) para facilitar comparaciones y validaciones externas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pbcrl.data_contracts.embalses import ParametrosEmbalse
from pbcrl.data_contracts.schemas import validar_dataframe_embalse

# Constantes de conversión
_S_POR_DIA = 86_400          # segundos en un día
_M3S_A_MM3_DIA = _S_POR_DIA / 1e6   # m³/s → Mm³/día    (factor = 0.0864)
_MM_KM2_A_MM3 = 1e-3                 # mm × km² → Mm³    (factor = 0.001)


def _calcular_vertimiento(
    volumen_final_mm3: float,
    capacidad_max_mm3: float,
) -> float:
    """Calcula el volumen vertido por el aliviadero en un paso de tiempo.

    El aliviadero se activa cuando el volumen supera la capacidad máxima.
    El exceso se vierte instantáneamente (modelo de tasa de descarga libre).

    Parámetros
    ----------
    volumen_final_mm3 : float
        Volumen calculado antes de aplicar la restricción de capacidad [Mm³].
    capacidad_max_mm3 : float
        Capacidad máxima del embalse [Mm³].

    Retorna
    -------
    float
        Volumen vertido [Mm³]. Cero si el embalse no rebosa.
    """
    return max(0.0, volumen_final_mm3 - capacidad_max_mm3)


def calcular_afluencia(
    df: pd.DataFrame,
    params: ParametrosEmbalse,
    validar: bool = True,
) -> pd.Series:
    """Estima la afluencia diaria a un embalse mediante balance hídrico inverso.

    Función pura: no modifica el DataFrame de entrada ni tiene efectos secundarios.

    Ecuación aplicada en cada paso t (t ≥ 1):

        afluencia_mm3(t) = [V(t) - V(t-1)]          # cambio de almacenamiento
                         + descarga_mm3(t)           # salida controlada
                         + evaporacion_mm3(t)        # pérdida por evaporación
                         - precipitacion_mm3(t)      # ganancia por precipitación
                         + vertimiento_mm3(t)        # salida por aliviadero

    El primer paso (t=0) no tiene t-1, por lo que se devuelve NaN para ese día.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con DatetimeIndex diario y las columnas del esquema canónico.
        Unidades: ver módulo `data_contracts.schemas`.
    params : ParametrosEmbalse
        Parámetros físicos del embalse (capacidad máxima, área del espejo).
    validar : bool
        Si True, valida el DataFrame contra el contrato antes de procesar.

    Retorna
    -------
    pd.Series
        Serie de afluencia estimada en m³/s, con el mismo DatetimeIndex que `df`.
        El primer valor es NaN (se necesita V(t-1) para calcularlo).

    Raises
    ------
    ValueError
        Si `df` no cumple el contrato de datos (cuando validar=True).
    """
    if validar:
        validar_dataframe_embalse(df, nombre_embalse=params.nombre)

    n = len(df)
    afluencia_mm3 = np.full(n, np.nan)

    volumen = df["volumen_mm3"].to_numpy()
    descarga_m3s = df["descarga_m3s"].to_numpy()
    precipitacion_mm = df["precipitacion_mm"].to_numpy()
    evaporacion_mm = df["evaporacion_mm"].to_numpy()

    # Conversiones de unidades (ver encabezado del módulo)
    # Descarga: m³/s → Mm³/día
    descarga_mm3 = descarga_m3s * _M3S_A_MM3_DIA

    # Precipitación y evaporación: mm × km² → Mm³
    precipitacion_mm3 = precipitacion_mm * params.area_espejo_km2 * _MM_KM2_A_MM3
    evaporacion_mm3 = evaporacion_mm * params.area_espejo_km2 * _MM_KM2_A_MM3

    for t in range(1, n):
        delta_v = volumen[t] - volumen[t - 1]      # ΔV [Mm³]

        # Vertimiento: detecta si la diferencia de volumen implica rebose
        # El vertimiento es la fracción de afluencia que no pudo almacenarse.
        # Se estima como el exceso sobre la capacidad máxima en t.
        vertimiento_mm3 = _calcular_vertimiento(
            volumen_final_mm3=volumen[t],
            capacidad_max_mm3=params.capacidad_max_mm3,
        )

        afluencia_mm3[t] = (
            delta_v
            + descarga_mm3[t]
            + evaporacion_mm3[t]
            - precipitacion_mm3[t]
            + vertimiento_mm3
        )

    # Conversión de resultado: Mm³/día → m³/s
    afluencia_m3s = afluencia_mm3 / _M3S_A_MM3_DIA

    return pd.Series(
        afluencia_m3s,
        index=df.index,
        name="afluencia_m3s",
        dtype="float64",
    )


def reconstruir_volumen(
    afluencia_m3s: pd.Series,
    descarga_m3s: pd.Series,
    precipitacion_mm: pd.Series,
    evaporacion_mm: pd.Series,
    params: ParametrosEmbalse,
    volumen_inicial_mm3: float,
) -> pd.Series:
    """Reconstruye la serie de volumen a partir de la afluencia estimada.

    Función inversa de `calcular_afluencia`: si la afluencia estimada es correcta,
    el volumen reconstruido debe coincidir con el volumen original del DataFrame.

    Se usa en las pruebas de conservación de masa.

    Parámetros
    ----------
    afluencia_m3s : pd.Series
        Afluencia estimada en m³/s (salida de `calcular_afluencia`).
    descarga_m3s, precipitacion_mm, evaporacion_mm : pd.Series
        Columnas del DataFrame original de entrada.
    params : ParametrosEmbalse
        Parámetros físicos del embalse.
    volumen_inicial_mm3 : float
        Volumen en el primer paso de tiempo [Mm³].

    Retorna
    -------
    pd.Series
        Serie de volumen reconstruido [Mm³], con el mismo DatetimeIndex.
    """
    n = len(afluencia_m3s)
    volumen = np.full(n, np.nan)
    volumen[0] = volumen_inicial_mm3

    afl_mm3 = afluencia_m3s.to_numpy() * _M3S_A_MM3_DIA
    desc_mm3 = descarga_m3s.to_numpy() * _M3S_A_MM3_DIA
    prec_mm3 = precipitacion_mm.to_numpy() * params.area_espejo_km2 * _MM_KM2_A_MM3
    evap_mm3 = evaporacion_mm.to_numpy() * params.area_espejo_km2 * _MM_KM2_A_MM3

    for t in range(1, n):
        v_bruto = (
            volumen[t - 1]
            + afl_mm3[t]
            - desc_mm3[t]
            - evap_mm3[t]
            + prec_mm3[t]
        )
        vertimiento = _calcular_vertimiento(v_bruto, params.capacidad_max_mm3)
        volumen[t] = v_bruto - vertimiento

    return pd.Series(
        volumen,
        index=afluencia_m3s.index,
        name="volumen_reconstruido_mm3",
        dtype="float64",
    )

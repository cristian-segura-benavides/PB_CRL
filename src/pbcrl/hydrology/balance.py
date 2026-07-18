"""
Balance hídrico inverso para embalses.

Ecuación de conservación de masa (paso diario):

    afluencia(t) = ΔV(t) + descarga(t) + evaporación_vol(t)
                   - precipitación_vol(t) - bombeo(t) + vertimiento(t)

donde:
    ΔV(t) = V(t) - V(t-1)                    [Mm³]
    descarga(t)          : salida controlada   [Mm³]  ← convertida de m³/s
    evaporación_vol(t)   : lámina × área       [Mm³]  ← convertida de mm × km²
    precipitación_vol(t) : lámina × área       [Mm³]  ← convertida de mm × km²
    bombeo(t)            : entrada artificial   [Mm³]  ← ya en Mm³/día (sin conversión)
    vertimiento(t)       : salida por aliviadero [Mm³] ← calculada internamente

TÉRMINO DE BOMBEO (opcional; solo Tominé)
-----------------------------------------
Tominé es el único de los tres embalses que bombea agua desde el río HACIA el vaso
(entrada artificial). Parte del aumento de volumen proviene de ese bombeo y no de la
afluencia natural, por lo que el bombeo se RESTA para aislar la afluencia natural.
Es un parámetro OPCIONAL con valor por defecto cero: Neusa y Sisga (sin bombeo) se
calculan exactamente igual que antes, sin cambio alguno en sus resultados.
El bombeo ya viene en Mm³/día desde el loader de Tominé (convertido allí desde m³
absolutos: m³ ÷ 1e6 = Mm³), así que entra DIRECTAMENTE al balance, sin factor de
conversión adicional.

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


def _bombeo_a_array(
    bombeo_mm3: "pd.Series | np.ndarray | None",
    index: pd.DatetimeIndex,
    n: int,
) -> np.ndarray:
    """Normaliza el bombeo a un array de longitud n en Mm³/día.

    None -> serie de ceros (embalse sin bombeo, p.ej. Neusa/Sisga). Un pd.Series se
    alinea por índice a `index`. Los NaN se tratan como cero (día sin bombeo). El
    bombeo NO se convierte: se asume ya en Mm³/día (ver encabezado del módulo).
    """
    if bombeo_mm3 is None:
        return np.zeros(n)
    if isinstance(bombeo_mm3, pd.Series):
        arr = bombeo_mm3.reindex(index).to_numpy(dtype=float)
    else:
        arr = np.asarray(bombeo_mm3, dtype=float)
    if arr.shape[0] != n:
        raise ValueError(
            f"bombeo_mm3 debe tener longitud {n} (una por paso de tiempo); "
            f"se recibió {arr.shape[0]}."
        )
    return np.nan_to_num(arr, nan=0.0)


def calcular_afluencia(
    df: pd.DataFrame,
    params: ParametrosEmbalse,
    validar: bool = True,
    bombeo_mm3: "pd.Series | np.ndarray | None" = None,
) -> pd.Series:
    """Estima la afluencia diaria a un embalse mediante balance hídrico inverso.

    Función pura: no modifica el DataFrame de entrada ni tiene efectos secundarios.

    Ecuación aplicada en cada paso t (t ≥ 1):

        afluencia_mm3(t) = [V(t) - V(t-1)]          # cambio de almacenamiento
                         + descarga_mm3(t)           # salida controlada
                         + evaporacion_mm3(t)        # pérdida por evaporación
                         - precipitacion_mm3(t)      # ganancia por precipitación
                         - bombeo_mm3(t)             # entrada artificial (solo Tominé)
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
    bombeo_mm3 : pd.Series | np.ndarray | None
        Entrada artificial por bombeo, YA en Mm³/día (sin conversión). Opcional; por
        defecto None = sin bombeo (serie de ceros), de modo que Neusa y Sisga se
        calculan idénticamente que antes. Solo Tominé pasa una serie real. Un pd.Series
        se alinea por índice; los NaN se tratan como cero.

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

    # Bombeo: ya en Mm³/día (sin conversión). None -> ceros (Neusa/Sisga inalterados).
    bombeo_mm3_arr = _bombeo_a_array(bombeo_mm3, df.index, n)

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
            - bombeo_mm3_arr[t]
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
    bombeo_mm3: "pd.Series | np.ndarray | None" = None,
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
    bombeo_mm3 : pd.Series | np.ndarray | None
        Entrada artificial por bombeo, YA en Mm³/día. Opcional (None = ceros). Debe
        ser la MISMA serie que se pasó a `calcular_afluencia` para cerrar el balance.
        Se suma como entrada al reconstruir el volumen.

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
    # Bombeo: ya en Mm³/día (sin conversión). None -> ceros.
    bomb_mm3 = _bombeo_a_array(bombeo_mm3, afluencia_m3s.index, n)

    for t in range(1, n):
        v_bruto = (
            volumen[t - 1]
            + afl_mm3[t]
            - desc_mm3[t]
            - evap_mm3[t]
            + prec_mm3[t]
            + bomb_mm3[t]
        )
        vertimiento = _calcular_vertimiento(v_bruto, params.capacidad_max_mm3)
        volumen[t] = v_bruto - vertimiento

    return pd.Series(
        volumen,
        index=afluencia_m3s.index,
        name="volumen_reconstruido_mm3",
        dtype="float64",
    )

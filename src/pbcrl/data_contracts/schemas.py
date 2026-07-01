"""
Contrato de datos para las series diarias de cada embalse.

INTERFAZ CANÓNICA
-----------------
Cualquier DataFrame que entre al pipeline de balance hídrico debe cumplir este esquema.
La función `validar_dataframe_embalse` lo verifica y lanza excepciones descriptivas
para facilitar la depuración cuando se conecten datos reales.

Columnas requeridas y sus unidades
-----------------------------------
| Columna              | Unidad  | Descripción                                         |
|----------------------|---------|-----------------------------------------------------|
| fecha                | –       | Índice DatetimeIndex diario (frecuencia 'D')        |
| cota_m               | m       | Nivel/cota del embalse [m.s.n.m.]                   |
| volumen_mm3          | Mm³     | Volumen almacenado [millones de m³]                 |
| descarga_m3s         | m³/s    | Descarga controlada por compuertas                  |
| precipitacion_mm     | mm/día  | Precipitación sobre el espejo del embalse           |
| evaporacion_mm       | mm/día  | Evaporación sobre el espejo del embalse             |

NOTA: La afluencia NO es columna de entrada; es la SALIDA que calcula el balance hídrico.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Esquema: nombre de columna → tipo esperado de pandas
# ---------------------------------------------------------------------------

ESQUEMA_EMBALSE: dict[str, str] = {
    "cota_m": "float64",
    "volumen_mm3": "float64",
    "descarga_m3s": "float64",
    "precipitacion_mm": "float64",
    "evaporacion_mm": "float64",
}

# Rangos físicos de referencia (usados en validación de sanidad, no para síntesis).
# El límite inferior de cota_m es 2550 m para cubrir el Tominé (cota mínima batimétrica ~2559 m).
_RANGOS_SANIDAD: dict[str, tuple[float, float]] = {
    "cota_m": (2550.0, 3100.0),
    "volumen_mm3": (0.0, 1000.0),
    "descarga_m3s": (0.0, 100.0),
    "precipitacion_mm": (0.0, 200.0),
    "evaporacion_mm": (0.0, 20.0),
}


def validar_dataframe_embalse(
    df: pd.DataFrame,
    nombre_embalse: str = "",
    verificar_rango_fechas: bool = True,
    verificar_sanidad: bool = True,
) -> None:
    """Verifica que un DataFrame cumple el contrato de datos del embalse.

    Lanza ValueError con mensaje descriptivo ante cualquier incumplimiento.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a validar. Debe tener DatetimeIndex de frecuencia diaria.
    nombre_embalse : str
        Nombre del embalse (solo para mensajes de error).
    verificar_rango_fechas : bool
        Si True, comprueba que el índice sea diario continuo sin huecos.
    verificar_sanidad : bool
        Si True, comprueba que los valores estén dentro de rangos físicos plausibles.
    """
    prefijo = f"[{nombre_embalse}] " if nombre_embalse else ""

    # 1. Índice debe ser DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"{prefijo}El índice debe ser DatetimeIndex, "
            f"se recibió {type(df.index).__name__}."
        )

    # 2. Columnas requeridas
    faltantes = set(ESQUEMA_EMBALSE) - set(df.columns)
    if faltantes:
        raise ValueError(
            f"{prefijo}Columnas faltantes: {sorted(faltantes)}. "
            f"Se requieren: {sorted(ESQUEMA_EMBALSE)}."
        )

    # 3. Tipos numéricos
    for col, tipo_esperado in ESQUEMA_EMBALSE.items():
        tipo_real = str(df[col].dtype)
        if not pd.api.types.is_float_dtype(df[col]):
            raise ValueError(
                f"{prefijo}Columna '{col}' debe ser float64, "
                f"se recibió '{tipo_real}'."
            )

    # 4. Sin NaN en ninguna columna del esquema
    for col in ESQUEMA_EMBALSE:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            raise ValueError(
                f"{prefijo}Columna '{col}' tiene {n_nan} valores NaN."
            )

    # 5. Serie diaria continua (sin huecos)
    if verificar_rango_fechas and len(df) > 1:
        freq_inferida = pd.infer_freq(df.index)
        if freq_inferida not in ("D", "B", None):
            pass  # infer_freq puede devolver None con pocos puntos; se verifica abajo
        delta_dias = (df.index[1:] - df.index[:-1]).days
        huecos = (delta_dias != 1).sum()
        if huecos > 0:
            raise ValueError(
                f"{prefijo}El índice tiene {huecos} huecos (la frecuencia no es diaria continua)."
            )

    # 6. Sanidad de rangos
    if verificar_sanidad:
        for col, (lo, hi) in _RANGOS_SANIDAD.items():
            fuera = ((df[col] < lo) | (df[col] > hi)).sum()
            if fuera > 0:
                raise ValueError(
                    f"{prefijo}Columna '{col}': {fuera} valores fuera del rango "
                    f"de sanidad [{lo}, {hi}]."
                )

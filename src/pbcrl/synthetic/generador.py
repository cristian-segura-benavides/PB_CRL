"""
Generador de datos sintéticos "tontos".

Produce DataFrames que cumplen el contrato de datos definido en `data_contracts.schemas`,
con valores aleatorios dentro de rangos físicamente plausibles.  El propósito es
exclusivamente probar que la tubería de procesamiento funciona; NO hay realismo
estadístico (sin autocorrelación, sin estacionalidad, sin correlaciones entre variables).

Cuando lleguen los datos reales de la CAR/EEB, basta sustituir la llamada a esta función
por la función de carga de datos reales, que debe devolver un DataFrame con el mismo esquema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pbcrl.data_contracts.embalses import ParametrosEmbalse


def generar_serie_sintetica(
    params: ParametrosEmbalse,
    fecha_inicio: str = "2015-01-01",
    fecha_fin: str = "2020-12-31",
    semilla: int | None = 42,
    descarga_max_m3s: float = 10.0,
    precipitacion_max_mm: float = 70.0,
    evaporacion_max_mm: float = 6.0,
) -> pd.DataFrame:
    """Genera una serie diaria sintética que cumple el contrato de datos.

    Parámetros
    ----------
    params : ParametrosEmbalse
        Parámetros físicos del embalse (define rangos de cota y volumen).
    fecha_inicio, fecha_fin : str
        Rango de fechas en formato 'YYYY-MM-DD'.
    semilla : int | None
        Semilla del generador aleatorio para reproducibilidad.
    descarga_max_m3s : float
        Límite superior de descarga controlada [m³/s].
    precipitacion_max_mm : float
        Límite superior de precipitación diaria [mm/día].
    evaporacion_max_mm : float
        Límite superior de evaporación diaria [mm/día].

    Retorna
    -------
    pd.DataFrame
        DataFrame con DatetimeIndex diario y columnas del esquema canónico.
        La columna 'afluencia_m3s' NO se incluye (es la salida del balance hídrico).
    """
    rng = np.random.default_rng(semilla)

    indice = pd.date_range(start=fecha_inicio, end=fecha_fin, freq="D")
    n = len(indice)

    # Cota: uniforme entre mínimo y máximo operativo
    cota = rng.uniform(params.cota_min_m, params.cota_max_m, size=n)

    # Volumen: uniforme entre volumen mínimo y máximo
    volumen = rng.uniform(params.capacidad_min_mm3, params.capacidad_max_mm3, size=n)

    # Descarga: uniforme entre 0 y descarga_max_m3s
    descarga = rng.uniform(0.0, descarga_max_m3s, size=n)

    # Precipitación: uniforme entre 0 y precipitacion_max_mm
    precipitacion = rng.uniform(0.0, precipitacion_max_mm, size=n)

    # Evaporación: uniforme entre 0 y evaporacion_max_mm
    evaporacion = rng.uniform(0.0, evaporacion_max_mm, size=n)

    df = pd.DataFrame(
        {
            "cota_m": cota.astype("float64"),
            "volumen_mm3": volumen.astype("float64"),
            "descarga_m3s": descarga.astype("float64"),
            "precipitacion_mm": precipitacion.astype("float64"),
            "evaporacion_mm": evaporacion.astype("float64"),
        },
        index=indice,
    )
    df.index.name = "fecha"
    return df

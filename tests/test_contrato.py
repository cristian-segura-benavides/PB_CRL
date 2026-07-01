"""
Pruebas del contrato de datos (schemas.py).

Verifica que la función de validación detecta correctamente DataFrames
válidos e inválidos.
"""
import numpy as np
import pandas as pd
import pytest

from pbcrl.data_contracts.schemas import ESQUEMA_EMBALSE, validar_dataframe_embalse


def _df_base(n: int = 30, inicio: str = "2020-01-01") -> pd.DataFrame:
    """Crea un DataFrame mínimo válido para pruebas."""
    idx = pd.date_range(inicio, periods=n, freq="D")
    return pd.DataFrame(
        {
            "cota_m": np.full(n, 2990.0),
            "volumen_mm3": np.full(n, 50.0),
            "descarga_m3s": np.full(n, 2.0),
            "precipitacion_mm": np.full(n, 5.0),
            "evaporacion_mm": np.full(n, 2.0),
        },
        index=idx,
    )


def test_valido_no_lanza():
    """Un DataFrame bien formado no debe lanzar ninguna excepción."""
    validar_dataframe_embalse(_df_base(), nombre_embalse="TestEmbalse")


def test_falta_columna():
    """Detecta columna faltante."""
    df = _df_base().drop(columns=["descarga_m3s"])
    with pytest.raises(ValueError, match="Columnas faltantes"):
        validar_dataframe_embalse(df)


def test_columna_con_nan():
    """Detecta NaN en columna de datos."""
    df = _df_base()
    df.loc[df.index[5], "volumen_mm3"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validar_dataframe_embalse(df)


def test_indice_no_datetime():
    """Detecta índice que no es DatetimeIndex."""
    df = _df_base().reset_index(drop=True)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        validar_dataframe_embalse(df)


def test_hueco_en_serie():
    """Detecta huecos (días faltantes) en el índice."""
    idx = pd.date_range("2020-01-01", periods=10, freq="D").delete(5)
    df = pd.DataFrame(
        {col: np.ones(len(idx)) for col in ESQUEMA_EMBALSE},
        index=idx,
    )
    # Ajustar valores a rangos de sanidad
    df["cota_m"] = 2990.0
    df["volumen_mm3"] = 50.0
    df["descarga_m3s"] = 2.0
    df["precipitacion_mm"] = 5.0
    df["evaporacion_mm"] = 2.0
    with pytest.raises(ValueError, match="huecos"):
        validar_dataframe_embalse(df)


def test_valor_fuera_de_rango():
    """Detecta valores fuera del rango de sanidad."""
    df = _df_base()
    df.loc[df.index[0], "precipitacion_mm"] = 500.0  # físicamente imposible
    with pytest.raises(ValueError, match="sanidad"):
        validar_dataframe_embalse(df)

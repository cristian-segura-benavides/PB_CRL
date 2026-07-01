"""
Pruebas del generador de datos sintéticos.

Verifica que los DataFrames generados cumplen el contrato de datos.
"""
import pytest

from pbcrl.data_contracts.embalses import EMBALSES
from pbcrl.data_contracts.schemas import validar_dataframe_embalse
from pbcrl.synthetic.generador import generar_serie_sintetica


@pytest.mark.parametrize("nombre", list(EMBALSES))
def test_sintetico_cumple_contrato(nombre: str):
    """El generador produce DataFrames válidos para cada embalse."""
    params = EMBALSES[nombre]
    df = generar_serie_sintetica(params, fecha_inicio="2020-01-01", fecha_fin="2020-12-31")
    validar_dataframe_embalse(df, nombre_embalse=nombre)


def test_rango_de_fechas():
    """El índice del DataFrame cubre exactamente el rango solicitado."""
    params = EMBALSES["Neusa"]
    df = generar_serie_sintetica(params, fecha_inicio="2018-01-01", fecha_fin="2018-01-31")
    assert df.index[0].strftime("%Y-%m-%d") == "2018-01-01"
    assert df.index[-1].strftime("%Y-%m-%d") == "2018-01-31"
    assert len(df) == 31


def test_reproducibilidad():
    """La misma semilla produce el mismo DataFrame."""
    params = EMBALSES["Sisga"]
    df1 = generar_serie_sintetica(params, semilla=7)
    df2 = generar_serie_sintetica(params, semilla=7)
    assert df1.equals(df2)


def test_semillas_distintas():
    """Semillas distintas producen DataFrames distintos."""
    params = EMBALSES["Tomine"]
    df1 = generar_serie_sintetica(params, semilla=1)
    df2 = generar_serie_sintetica(params, semilla=2)
    assert not df1.equals(df2)

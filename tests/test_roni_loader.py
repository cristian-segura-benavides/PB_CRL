"""Pruebas del cargador del RONI (interpolación mensual -> diaria).

Dependen del CSV real (info_CAR/), que no se versiona; por eso se saltan
automáticamente si el archivo no está presente en el entorno.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pbcrl.data_contracts.ventana import VENTANA_FIN, VENTANA_INICIO
from pbcrl.data_ingest.roni import DEFAULT_RONI_CSV, cargar_roni

pytestmark = pytest.mark.skipif(
    not DEFAULT_RONI_CSV.exists(),
    reason=f"CSV del RONI no disponible en {DEFAULT_RONI_CSV}",
)


@pytest.fixture(scope="module")
def roni():
    return cargar_roni()


def test_ventana_sin_huecos(roni):
    """(a) La serie diaria cubre la ventana completa, sin huecos."""
    df, diag = roni
    assert df.index.min() == pd.Timestamp(VENTANA_INICIO)
    assert df.index.max() == pd.Timestamp(VENTANA_FIN)
    esperado = pd.date_range(VENTANA_INICIO, VENTANA_FIN, freq="D")
    pd.testing.assert_index_equal(df.index, esperado, check_names=False)
    assert diag.dias_totales == len(esperado)
    assert not df["roni"].isna().any()
    assert not df["fase"].isna().any()


def test_interpolacion_continua_sin_saltos(roni):
    """(b) La interpolacion es continua: sin escalones bruscos dentro de un mes.

    El cambio maximo dia-a-dia debe ser pequeño y uniforme (fraccion del cambio
    mensual total, no un salto de mes completo como haria un broadcast).
    """
    df, _ = roni
    delta_diario = df["roni"].diff().dropna().abs()
    # Un broadcast produciria saltos de hasta ~3 puntos RONI el dia 1 de mes;
    # la interpolacion lineal mantiene el paso diario muy por debajo de eso.
    assert delta_diario.max() < 0.15


def test_valores_mensuales_preservados_en_anclas(roni):
    """(c) Los valores mensuales originales se preservan en sus fechas ancla (dia 1)."""
    df, _ = roni
    anclas = df.index[df.index.day == 1]
    assert len(anclas) > 0
    # Los valores en dia 1 deben coincidir con el CSV original (sin distorsion).
    crudo = pd.read_csv(DEFAULT_RONI_CSV)
    crudo["fecha"] = pd.to_datetime(crudo["fecha"])
    crudo_indexado = crudo.set_index("fecha")["roni"]
    for ancla in anclas:
        assert abs(df.loc[ancla, "roni"] - crudo_indexado.loc[ancla]) < 1e-9


def test_evento_nino_2023_2024(roni):
    """(d) El evento El Niño 2023-24 aparece correctamente.

    - El ancla de julio 2023 (JJA) ya supera 0.5 (primer mes oficialmente Niño).
    - Bajo interpolacion lineal estricta entre el ancla de junio (0.4) y julio
      (0.6), el cruce diario de 0.5 cae a mitad de camino: 2023-06-17 (no en
      julio) -- es el comportamiento ESPERADO de interpolar entre anclas
      distintas, no un error.
    - El pico de 1.5 se sostiene de 2023-11-01 a 2023-12-01 (anclas OND y NDJ,
      ambas 1.5 -> la interpolacion entre dos anclas iguales es plana).
    """
    df, _ = roni
    assert df.loc["2023-07-01", "roni"] == pytest.approx(0.6)
    assert df.loc["2023-07-01", "fase"] == "Nino"

    cruce = df.loc["2023-01-01":"2023-12-31", "roni"]
    primer_dia_sobre_umbral = cruce[cruce > 0.5].index.min()
    assert primer_dia_sobre_umbral == pd.Timestamp("2023-06-17")

    pico = df.loc["2023-11-01":"2023-12-01", "roni"]
    assert (pico.sub(1.5).abs() < 1e-9).all()


def test_fase_categorias_validas(roni):
    df, _ = roni
    assert set(df["fase"].unique()) <= {"Nino", "Nina", "Neutral"}

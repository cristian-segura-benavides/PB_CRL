"""Pruebas del cargador/empalme del caudal de Saucío (CAR + Enlaza).

Dependen de los archivos reales (info_CAR/), que no se versionan; por eso se saltan
automáticamente si no están presentes en el entorno.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pbcrl.data_contracts.ventana import VENTANA_FIN, VENTANA_INICIO
from pbcrl.data_ingest.saucio import (
    CORTE_EMPALME,
    DEFAULT_CAR_ESTACIONES_CSV,
    DEFAULT_TOMINE_XLSX,
    DIA_SOSPECHOSO,
    cargar_saucio,
)

pytestmark = pytest.mark.skipif(
    not (DEFAULT_CAR_ESTACIONES_CSV.exists() and DEFAULT_TOMINE_XLSX.exists()),
    reason="Fuentes de Saucío (CAR/Enlaza) no disponibles en info_CAR/",
)


@pytest.fixture(scope="module")
def saucio():
    return cargar_saucio()


def test_ventana_y_continuidad(saucio):
    df, diag = saucio
    assert df.index.min() == pd.Timestamp(VENTANA_INICIO)
    assert df.index.max() == pd.Timestamp(VENTANA_FIN)
    esperado = pd.date_range(VENTANA_INICIO, VENTANA_FIN, freq="D")
    pd.testing.assert_index_equal(df.index, esperado, check_names=False)
    assert diag.dias_totales == len(esperado)


def test_corte_limpio_sin_solape(saucio):
    """Antes del corte todo es CAR; desde el corte en adelante, todo es Enlaza."""
    df, diag = saucio
    corte = pd.Timestamp(CORTE_EMPALME)
    antes = df.loc[: corte - pd.Timedelta(days=1), "fuente_caudal"].dropna()
    despues = df.loc[corte:, "fuente_caudal"].dropna()
    assert (antes == "CAR").all()
    assert (despues == "Enlaza").all()
    # dias_car/dias_enlaza cuentan solo dias con dato real; sumados a los huecos
    # (dias sin dato en ninguna fuente para ese tramo) deben cubrir la ventana completa.
    assert diag.dias_car + diag.dias_enlaza + diag.huecos_totales == diag.dias_totales


def test_dia_sospechoso_no_eliminado(saucio):
    """El día sospechoso (2021-06-21) sigue presente en la serie, sin corregir."""
    df, diag = saucio
    assert diag.dia_sospechoso == DIA_SOSPECHOSO
    valor = df.loc[DIA_SOSPECHOSO, "caudal_m3s"]
    assert pd.notna(valor)
    # Cae en el tramo CAR (antes del corte de empalme).
    assert diag.dia_sospechoso_fuente_usada == "CAR"
    assert diag.dia_sospechoso_car_m3s is not None
    assert diag.dia_sospechoso_enlaza_m3s is not None
    assert abs(valor - diag.dia_sospechoso_car_m3s) < 1e-6


def test_huecos_reportados_no_rellenados(saucio):
    """Los huecos conocidos (CAR: 3 días; Enlaza: 90 días) quedan como NaN."""
    df, diag = saucio
    assert diag.huecos_tramo_car == 3
    assert diag.huecos_tramo_enlaza == 90
    assert diag.huecos_totales == diag.huecos_tramo_car + diag.huecos_tramo_enlaza
    assert int(df["caudal_m3s"].isna().sum()) == diag.huecos_totales


def test_unidades_m3s_sin_negativos(saucio):
    df, _ = saucio
    serie = df["caudal_m3s"].dropna()
    assert (serie >= 0).all()

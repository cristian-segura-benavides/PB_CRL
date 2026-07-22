"""Pruebas del cargador de la serie operativa de Tominé.

Dependen del Excel real de Enlaza (info_CAR/), que no se versiona; por eso se saltan
automáticamente si el archivo no está presente en el entorno.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pbcrl.data_contracts import ESQUEMA_EMBALSE, validar_dataframe_embalse
from pbcrl.data_contracts.ventana import VENTANA_FIN, VENTANA_INICIO
from pbcrl.data_ingest import DEFAULT_TOMINE_XLSX, cargar_tomine

pytestmark = pytest.mark.skipif(
    not DEFAULT_TOMINE_XLSX.exists(),
    reason=f"Excel de Tominé no disponible en {DEFAULT_TOMINE_XLSX}",
)


@pytest.fixture(scope="module")
def tomine():
    return cargar_tomine()


def test_cumple_contrato(tomine):
    df, _ = tomine
    # No debe lanzar.
    validar_dataframe_embalse(df, nombre_embalse="Tomine")


def test_columnas_esquema_mas_bombeo(tomine):
    df, _ = tomine
    assert set(ESQUEMA_EMBALSE).issubset(df.columns)
    assert "bombeo_mm3" in df.columns
    assert all(str(t) == "float64" for t in df.dtypes)


def test_ventana_y_continuidad(tomine):
    df, diag = tomine
    assert df.index.min() == pd.Timestamp(VENTANA_INICIO)
    assert df.index.max() == pd.Timestamp(VENTANA_FIN)
    # Índice diario continuo sin huecos.
    esperado = pd.date_range(VENTANA_INICIO, VENTANA_FIN, freq="D")
    pd.testing.assert_index_equal(df.index, esperado, check_names=False)
    assert diag.dias_resultado == len(esperado)


def test_salto_2012_fuera_de_ventana(tomine):
    """El salto real de volumen del 2012-01-01 (~138 Mm3) ocurria en la transicion
    2011-12-31 -> 2012-01-01. Con la ventana arrancando exactamente en 2012-01-01,
    esa transicion queda excluida y no hay nada que corregir: la serie es
    internamente consistente desde su primer dia (valor crudo, sin interpolar)."""
    df, diag = tomine
    assert diag.saltos_volumen_corregidos == 0
    primer_valor = df.loc[VENTANA_INICIO, "volumen_mm3"]
    assert abs(primer_valor - 512.704290) < 1e-6  # es el valor crudo, no interpolado


def test_duplicados_eliminados(tomine):
    df, diag = tomine
    assert not df.index.duplicated().any()
    assert diag.fechas_duplicadas_eliminadas == 2


def test_sin_nan(tomine):
    df, _ = tomine
    assert not df.isna().any().any()


def test_ceros_de_descarga_preservados(tomine):
    df, diag = tomine
    # Los días sin descarga son reales; no deben interpolarse a valores positivos.
    assert diag.dias_descarga_cero > 0
    assert (df["descarga_m3s"] == 0).sum() == diag.dias_descarga_cero

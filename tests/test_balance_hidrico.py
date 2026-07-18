"""
Pruebas del balance hídrico inverso.

La prueba más importante es la de conservación de masa:
si reconstruyo el volumen a partir de la afluencia estimada,
debo recuperar el volumen original.
"""
import numpy as np
import pandas as pd
import pytest

from pbcrl.data_contracts.embalses import EMBALSES, ParametrosEmbalse
from pbcrl.hydrology.balance import (
    _M3S_A_MM3_DIA,
    calcular_afluencia,
    reconstruir_volumen,
)
from pbcrl.synthetic.generador import generar_serie_sintetica


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def params_neusa() -> ParametrosEmbalse:
    return EMBALSES["Neusa"]


@pytest.fixture
def df_sintetico(params_neusa: ParametrosEmbalse) -> pd.DataFrame:
    return generar_serie_sintetica(
        params_neusa, fecha_inicio="2020-01-01", fecha_fin="2020-12-31", semilla=0
    )


# ---------------------------------------------------------------------------
# Pruebas de forma
# ---------------------------------------------------------------------------

def test_afluencia_misma_longitud(params_neusa, df_sintetico):
    """La afluencia estimada tiene el mismo número de filas que la entrada."""
    afluencia = calcular_afluencia(df_sintetico, params_neusa)
    assert len(afluencia) == len(df_sintetico)


def test_afluencia_mismo_indice(params_neusa, df_sintetico):
    """La afluencia comparte el DatetimeIndex de la entrada."""
    afluencia = calcular_afluencia(df_sintetico, params_neusa)
    assert afluencia.index.equals(df_sintetico.index)


def test_primer_valor_nan(params_neusa, df_sintetico):
    """El primer valor de afluencia es NaN (no hay V(t-1) disponible)."""
    afluencia = calcular_afluencia(df_sintetico, params_neusa)
    assert np.isnan(afluencia.iloc[0])


def test_resto_sin_nan(params_neusa, df_sintetico):
    """Todos los valores excepto el primero son valores numéricos finitos."""
    afluencia = calcular_afluencia(df_sintetico, params_neusa)
    assert afluencia.iloc[1:].notna().all()
    assert np.isfinite(afluencia.iloc[1:].to_numpy()).all()


def test_nombre_serie(params_neusa, df_sintetico):
    """La serie devuelta tiene el nombre correcto."""
    afluencia = calcular_afluencia(df_sintetico, params_neusa)
    assert afluencia.name == "afluencia_m3s"


# ---------------------------------------------------------------------------
# Prueba de conservación de masa (la más importante)
# ---------------------------------------------------------------------------

def test_conservacion_de_masa(params_neusa, df_sintetico):
    """
    Prueba de conservación de masa (round-trip).

    Pasos:
    1. Calcular la afluencia estimada a partir del DataFrame sintético.
    2. Reconstruir el volumen usando esa afluencia.
    3. Verificar que el volumen reconstruido coincide con el original.

    Si esta prueba pasa, el balance hídrico está correctamente implementado.
    Tolerancia: 1e-9 Mm³ (errores de punto flotante únicamente).
    """
    afluencia = calcular_afluencia(df_sintetico, params_neusa)

    volumen_reconstruido = reconstruir_volumen(
        afluencia_m3s=afluencia,
        descarga_m3s=df_sintetico["descarga_m3s"],
        precipitacion_mm=df_sintetico["precipitacion_mm"],
        evaporacion_mm=df_sintetico["evaporacion_mm"],
        params=params_neusa,
        volumen_inicial_mm3=df_sintetico["volumen_mm3"].iloc[0],
    )

    volumen_original = df_sintetico["volumen_mm3"]

    # El primer valor es idéntico por construcción (volumen_inicial)
    np.testing.assert_allclose(
        volumen_reconstruido.iloc[0],
        volumen_original.iloc[0],
        atol=1e-9,
        err_msg="Volumen inicial no coincide.",
    )

    # Desde el segundo paso, debe coincidir con el original
    np.testing.assert_allclose(
        volumen_reconstruido.iloc[1:].to_numpy(),
        volumen_original.iloc[1:].to_numpy(),
        atol=1e-9,
        err_msg="Conservación de masa violada: el volumen reconstruido difiere del original.",
    )


# ---------------------------------------------------------------------------
# Bombeo (solo Tominé): término opcional del balance
# ---------------------------------------------------------------------------

def _serie_bombeo_mm3(index: pd.DatetimeIndex) -> pd.Series:
    """Serie de bombeo determinista en Mm³/día: muchos ceros y algunos pulsos."""
    valores = np.zeros(len(index))
    valores[2::7] = 0.30   # pulsos periódicos de bombeo
    valores[5::11] = 0.75
    return pd.Series(valores, index=index, name="bombeo_mm3")


def test_bombeo_none_identico_a_ceros(params_neusa, df_sintetico):
    """bombeo=None produce resultados BIT-idénticos a bombeo=ceros.

    Garantiza que Neusa/Sisga (sin bombeo) no cambian en absoluto.
    """
    afl_none = calcular_afluencia(df_sintetico, params_neusa)
    afl_ceros = calcular_afluencia(
        df_sintetico, params_neusa, bombeo_mm3=np.zeros(len(df_sintetico))
    )
    np.testing.assert_array_equal(afl_none.to_numpy(), afl_ceros.to_numpy())


def test_bombeo_resta_afluencia_natural(params_neusa, df_sintetico):
    """El bombeo desplaza la afluencia natural exactamente en bombeo/0.0864 [m³/s]."""
    bombeo = _serie_bombeo_mm3(df_sintetico.index)
    afl_sin = calcular_afluencia(df_sintetico, params_neusa)
    afl_con = calcular_afluencia(df_sintetico, params_neusa, bombeo_mm3=bombeo)

    esperado = afl_sin - bombeo / _M3S_A_MM3_DIA
    np.testing.assert_allclose(
        afl_con.iloc[1:].to_numpy(),
        esperado.iloc[1:].to_numpy(),
        atol=1e-9,
        err_msg="La afluencia con bombeo debe ser la natural menos el bombeo (en m³/s).",
    )


def test_conservacion_de_masa_con_bombeo():
    """Conservación de masa CON bombeo (caso Tominé).

    Si reconstruyo el volumen incluyendo el bombeo como entrada, recupero el volumen
    original con tolerancia de punto flotante.
    """
    params = EMBALSES["Tomine"]
    df = generar_serie_sintetica(params, semilla=7)
    bombeo = _serie_bombeo_mm3(df.index)

    afluencia = calcular_afluencia(df, params, bombeo_mm3=bombeo)
    volumen_rec = reconstruir_volumen(
        afluencia_m3s=afluencia,
        descarga_m3s=df["descarga_m3s"],
        precipitacion_mm=df["precipitacion_mm"],
        evaporacion_mm=df["evaporacion_mm"],
        params=params,
        volumen_inicial_mm3=df["volumen_mm3"].iloc[0],
        bombeo_mm3=bombeo,
    )

    np.testing.assert_allclose(
        volumen_rec.iloc[1:].to_numpy(),
        df["volumen_mm3"].iloc[1:].to_numpy(),
        atol=1e-9,
        err_msg="Conservación de masa violada con bombeo incluido.",
    )


# ---------------------------------------------------------------------------
# Prueba de sanidad: afluencia cero cuando no hay cambios
# ---------------------------------------------------------------------------

def test_afluencia_en_equilibrio():
    """
    Cuando el volumen es constante y no hay precipitación ni evaporación,
    la afluencia debe ser igual a la descarga.

    Ecuación simplificada: afluencia = ΔV + descarga = 0 + descarga = descarga.
    """
    params = EMBALSES["Sisga"]
    n = 10
    idx = pd.date_range("2021-01-01", periods=n, freq="D")

    descarga_m3s = 3.0  # m³/s constante

    df = pd.DataFrame(
        {
            "cota_m": np.full(n, 2660.0),           # dentro del rango operativo del Sisga
            "volumen_mm3": np.full(n, 45.0),        # volumen constante → ΔV = 0
            "descarga_m3s": np.full(n, descarga_m3s),
            "precipitacion_mm": np.zeros(n),         # sin precipitación
            "evaporacion_mm": np.zeros(n),           # sin evaporación
        },
        index=idx,
    )

    afluencia = calcular_afluencia(df, params)

    # Desde t=1 en adelante: afluencia_m3s debe ser igual a descarga_m3s
    np.testing.assert_allclose(
        afluencia.iloc[1:].to_numpy(),
        np.full(n - 1, descarga_m3s),
        atol=1e-9,
        err_msg="En equilibrio (ΔV=0, sin lluvia ni evaporación), "
                "la afluencia debe igualar la descarga.",
    )


# ---------------------------------------------------------------------------
# Prueba paramétrica: los tres embalses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nombre", list(EMBALSES))
def test_balance_todos_los_embalses(nombre: str):
    """El balance hídrico y la conservación de masa se cumplen para cada embalse."""
    params = EMBALSES[nombre]
    df = generar_serie_sintetica(params, semilla=99)

    afluencia = calcular_afluencia(df, params)

    volumen_rec = reconstruir_volumen(
        afluencia_m3s=afluencia,
        descarga_m3s=df["descarga_m3s"],
        precipitacion_mm=df["precipitacion_mm"],
        evaporacion_mm=df["evaporacion_mm"],
        params=params,
        volumen_inicial_mm3=df["volumen_mm3"].iloc[0],
    )

    np.testing.assert_allclose(
        volumen_rec.iloc[1:].to_numpy(),
        df["volumen_mm3"].iloc[1:].to_numpy(),
        atol=1e-9,
        err_msg=f"[{nombre}] Conservación de masa violada.",
    )

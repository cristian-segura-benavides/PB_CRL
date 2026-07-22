"""Pruebas del modelo estocástico de afluencias (src/pbcrl/stochastic/).

Usa ÚNICAMENTE datos sintéticos generados en este archivo (con una estructura
de correlación y estacionalidad conocidas) — no depende de info_CAR/ ni de
los loaders reales, para que el módulo se pruebe de forma aislada.

Cobertura:
  (a) fit/sample corren sin error con datos sintéticos pequeños.
  (b) sample() preserva aproximadamente la correlación cruzada observada en
      los datos de entrenamiento.
  (c) Reproducibilidad: misma semilla -> misma muestra.
  (d) guardar/cargar reproduce las mismas muestras que el modelo original.
  (e) RemuestreoAnalogos (línea base): fit/sample corren sin error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pbcrl.stochastic.analogos import ConfigAnalogos, RemuestreoAnalogos
from pbcrl.stochastic.modelo import ConfigModeloEstocastico, ModeloEstocasticoAfluencias

SERIES = ("Saucio", "Neusa", "Sisga", "Tomine")


def _generar_datos_sinteticos(n_dias: int = 1500, semilla: int = 123) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera (covariables, series_objetivo) sintéticas con correlación y
    estacionalidad conocidas, y una fracción de días en cero (simula el
    artefacto de acotamiento que el componente hurdle debe aislar)."""
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2015-01-01", periods=n_dias, freq="D")
    doy = fechas.dayofyear.to_numpy()
    meses = fechas.month.to_numpy()

    precip = np.clip(5.0 + 4.0 * np.sin(2 * np.pi * (doy - 60) / 365.0) + rng.normal(0, 1.5, n_dias), 0, None)
    roni = np.cumsum(rng.normal(0, 0.05, n_dias))
    roni = roni - roni.mean()

    covariables = pd.DataFrame({"precipitacion_mm": precip, "roni": roni}, index=fechas)

    k = len(SERIES)
    corr_objetivo = np.full((k, k), 0.65)
    np.fill_diagonal(corr_objetivo, 1.0)
    chol = np.linalg.cholesky(corr_objetivo)

    b_precip = np.array([0.05, 0.06, 0.04, 0.03])
    b_roni = np.array([0.30, 0.20, 0.25, 0.15])
    fases = np.array([0.0, 0.3, 0.6, 0.9])
    m_arr = np.arange(12)
    media_estacional = 1.0 + 0.4 * np.sin(2 * np.pi * m_arr[None, :] / 12 + fases[:, None])  # (4, 12)

    valores = np.zeros((n_dias, k))
    for t in range(n_dias):
        ruido = chol @ rng.standard_normal(k)
        m = meses[t] - 1
        val_log = media_estacional[:, m] + b_precip * precip[t] + b_roni * roni[t] + ruido
        valores[t] = np.expm1(val_log)
    valores = np.clip(valores, 0.0, None)  # no-negatividad: genera ceros cuando val_log << 0

    objetivo = pd.DataFrame(valores, index=fechas, columns=list(SERIES))
    return covariables, objetivo


@pytest.fixture(scope="module")
def datos_sinteticos():
    return _generar_datos_sinteticos()


# ---------------------------------------------------------------------------
# (a) fit/sample sin error
# ---------------------------------------------------------------------------

class TestFitSampleSinError:
    def test_fit_sample_corren_sin_error(self, datos_sinteticos):
        covariables, objetivo = datos_sinteticos
        modelo = ModeloEstocasticoAfluencias().fit(covariables, objetivo)
        muestra = modelo.sample(covariables.iloc[-200:], semilla=1)

        assert list(muestra.columns) == list(SERIES)
        assert len(muestra) == 200
        assert muestra.index.equals(covariables.iloc[-200:].index)
        assert (muestra.to_numpy() >= 0.0).all(), "las afluencias/caudal nunca deben ser negativos"

    def test_fit_lanza_error_con_muy_pocos_datos(self):
        covariables, objetivo = _generar_datos_sinteticos(n_dias=10)
        with pytest.raises(ValueError):
            ModeloEstocasticoAfluencias().fit(covariables, objetivo)

    def test_sample_sin_ajustar_lanza_error(self, datos_sinteticos):
        covariables, _ = datos_sinteticos
        modelo = ModeloEstocasticoAfluencias()
        with pytest.raises(RuntimeError):
            modelo.sample(covariables.iloc[-10:], semilla=1)


# ---------------------------------------------------------------------------
# (b) Preservación aproximada de la correlación cruzada
# ---------------------------------------------------------------------------

class TestCorrelacionPreservada:
    def test_correlacion_simulada_cercana_a_la_observada(self, datos_sinteticos):
        """La correlación cruzada de una muestra grande generada por el modelo
        debe parecerse a la observada en el histórico — no exacta (es un
        modelo estocástico), pero del mismo orden y signo."""
        covariables, objetivo = datos_sinteticos
        modelo = ModeloEstocasticoAfluencias().fit(covariables, objetivo)

        # Varias semillas, concatenadas, para una estimación más estable.
        muestras = [modelo.sample(covariables, semilla=s) for s in range(5)]
        muestra_grande = pd.concat(muestras, ignore_index=True)

        corr_observada = objetivo.corr()
        corr_simulada = muestra_grande.corr()

        diferencias = (corr_observada - corr_simulada).abs()
        # Tolerancia generosa: es una comparación estadística, no exacta.
        assert diferencias.to_numpy().max() < 0.30, (
            f"Diferencia máxima de correlación: {diferencias.to_numpy().max():.3f}\n"
            f"Observada:\n{corr_observada.round(3)}\nSimulada:\n{corr_simulada.round(3)}"
        )
        # Todas las correlaciones cruzadas simuladas deben seguir siendo
        # claramente positivas (la estructura de correlación conocida es 0.65).
        off_diag = corr_simulada.to_numpy()[~np.eye(4, dtype=bool)]
        assert (off_diag > 0.2).all(), "la correlación cruzada simulada debe seguir siendo claramente positiva"


# ---------------------------------------------------------------------------
# (c) Reproducibilidad
# ---------------------------------------------------------------------------

class TestReproducibilidad:
    def test_misma_semilla_misma_muestra(self, datos_sinteticos):
        covariables, objetivo = datos_sinteticos
        modelo = ModeloEstocasticoAfluencias().fit(covariables, objetivo)

        muestra_1 = modelo.sample(covariables.iloc[-100:], semilla=42)
        muestra_2 = modelo.sample(covariables.iloc[-100:], semilla=42)

        pd.testing.assert_frame_equal(muestra_1, muestra_2)

    def test_semillas_distintas_muestras_distintas(self, datos_sinteticos):
        covariables, objetivo = datos_sinteticos
        modelo = ModeloEstocasticoAfluencias().fit(covariables, objetivo)

        muestra_1 = modelo.sample(covariables.iloc[-100:], semilla=1)
        muestra_2 = modelo.sample(covariables.iloc[-100:], semilla=2)

        assert not muestra_1.equals(muestra_2)


# ---------------------------------------------------------------------------
# (d) Serialización (guardar/cargar)
# ---------------------------------------------------------------------------

class TestSerializacion:
    def test_guardar_cargar_reproduce_las_mismas_muestras(self, datos_sinteticos, tmp_path):
        covariables, objetivo = datos_sinteticos
        modelo = ModeloEstocasticoAfluencias().fit(covariables, objetivo)

        ruta = tmp_path / "modelo_prueba"
        modelo.guardar(ruta)
        modelo_cargado = ModeloEstocasticoAfluencias.cargar(ruta)

        muestra_original = modelo.sample(covariables.iloc[-50:], semilla=7)
        muestra_cargada = modelo_cargado.sample(covariables.iloc[-50:], semilla=7)

        pd.testing.assert_frame_equal(muestra_original, muestra_cargada)

    def test_guardar_sin_ajustar_lanza_error(self, tmp_path):
        modelo = ModeloEstocasticoAfluencias()
        with pytest.raises(RuntimeError):
            modelo.guardar(tmp_path / "no_deberia_crearse")


# ---------------------------------------------------------------------------
# (e) Línea base de remuestreo por análogos
# ---------------------------------------------------------------------------

class TestRemuestreoAnalogos:
    def test_fit_sample_corren_sin_error(self, datos_sinteticos):
        covariables, objetivo = datos_sinteticos
        modelo = RemuestreoAnalogos().fit(covariables, objetivo)
        muestra = modelo.sample(covariables.iloc[-100:], semilla=1)

        assert list(muestra.columns) == list(SERIES)
        assert len(muestra) == 100

    def test_valores_remuestreados_provienen_del_historico(self, datos_sinteticos):
        """Cada fila remuestreada debe ser EXACTAMENTE un vector observado en
        el entrenamiento (no un valor sintetizado) — es la propiedad central
        del método de análogos."""
        covariables, objetivo = datos_sinteticos
        modelo = RemuestreoAnalogos().fit(covariables, objetivo)
        muestra = modelo.sample(covariables.iloc[-30:], semilla=3)

        valores_train = set(map(tuple, objetivo.to_numpy().round(6)))
        for _, fila in muestra.round(6).iterrows():
            assert tuple(fila.to_numpy()) in valores_train

    def test_misma_semilla_misma_muestra(self, datos_sinteticos):
        covariables, objetivo = datos_sinteticos
        modelo = RemuestreoAnalogos().fit(covariables, objetivo)

        muestra_1 = modelo.sample(covariables.iloc[-50:], semilla=9)
        muestra_2 = modelo.sample(covariables.iloc[-50:], semilla=9)
        pd.testing.assert_frame_equal(muestra_1, muestra_2)

"""Pruebas de la fuente configurable de forzantes (environment/fuente_forzantes.py).

Usa datos SINTÉTICOS pequeños (no depende de info_CAR/) para que el módulo
se pruebe de forma aislada y rápida.

Cobertura:
  (a) ConfigFuenteForzantes valida la fuente y exige semilla en modo estocástico.
  (b) Modo "historico": ForzantesExternos coincide exactamente con los datos
      de entrada — regresión, cero cambio de comportamiento.
  (c) Modo "estocastico": corre sin error y es reproducible (misma semilla).
  (d) generar_episodio despacha correctamente según config.fuente.
  (e) Integración aislada: un episodio corto con fuente "estocastico" corre
      en EntornoEmbalses sin error y con volúmenes dentro de rango.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pbcrl.data_contracts.captaciones import CAUDAL_TIBITOC_HISTORICO_M3S
from pbcrl.data_contracts.embalses import EMBALSES
from pbcrl.environment.entorno import EntornoEmbalses
from pbcrl.environment.fuente_forzantes import (
    ConfigFuenteForzantes,
    generar_episodio,
    generar_episodio_estocastico,
    generar_episodio_historico,
)
from pbcrl.stochastic.modelo import ModeloEstocasticoAfluencias

NOMBRES = ("Neusa", "Sisga", "Tomine")


def _datos_historicos_sinteticos(n_dias: int = 10, semilla: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2020-03-01", periods=n_dias, freq="D")
    datos = {"caudal_saucio_m3s": rng.uniform(1, 5, n_dias)}
    for n in NOMBRES:
        datos[f"afluencia_{n}"] = rng.uniform(1, 10, n_dias)
        datos[f"precip_{n}"] = rng.uniform(0, 15, n_dias)
        datos[f"evap_{n}"] = rng.uniform(1, 4, n_dias)
    return pd.DataFrame(datos, index=fechas)


def _modelo_ajustado_sintetico(n_dias: int = 1200, semilla: int = 7) -> ModeloEstocasticoAfluencias:
    """Modelo ajustado sobre datos sintéticos grandes (solo para que fit()
    tenga suficientes días válidos); no representa ningún dato real."""
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2010-01-01", periods=n_dias, freq="D")
    precip = np.clip(5.0 + 3.0 * np.sin(2 * np.pi * fechas.dayofyear.to_numpy() / 365.0) + rng.normal(0, 1, n_dias), 0, None)
    roni = rng.normal(0, 0.5, n_dias)
    covariables = pd.DataFrame({"precipitacion_mm": precip, "roni": roni}, index=fechas)

    valores = np.clip(
        2.0 + 0.1 * precip[:, None] + rng.normal(0, 1.0, (n_dias, 4)),
        0.0,
        None,
    )
    objetivo = pd.DataFrame(valores, index=fechas, columns=["Saucio", "Neusa", "Sisga", "Tomine"])
    return ModeloEstocasticoAfluencias().fit(covariables, objetivo)


# ---------------------------------------------------------------------------
# (a) ConfigFuenteForzantes
# ---------------------------------------------------------------------------

class TestConfigFuenteForzantes:
    def test_fuente_por_defecto_es_historico(self):
        assert ConfigFuenteForzantes().fuente == "historico"

    def test_fuente_invalida_lanza_error(self):
        with pytest.raises(ValueError):
            ConfigFuenteForzantes(fuente="no_existe")

    def test_estocastico_sin_semilla_lanza_error(self):
        with pytest.raises(ValueError):
            ConfigFuenteForzantes(fuente="estocastico")

    def test_estocastico_con_semilla_ok(self):
        config = ConfigFuenteForzantes(fuente="estocastico", semilla=42)
        assert config.semilla == 42


# ---------------------------------------------------------------------------
# (b) Modo histórico: regresión (cero cambio de comportamiento)
# ---------------------------------------------------------------------------

class TestModoHistorico:
    def test_forzantes_coinciden_exactamente_con_los_datos(self):
        datos = _datos_historicos_sinteticos(n_dias=5)
        episodio = generar_episodio_historico(datos, tibitoc_nominal_m3s=CAUDAL_TIBITOC_HISTORICO_M3S)

        assert len(episodio) == 5
        for i, (fecha, fila) in enumerate(datos.iterrows()):
            forzantes = episodio[i]
            for n in NOMBRES:
                assert forzantes.afluencia_m3s[n] == pytest.approx(fila[f"afluencia_{n}"])
                assert forzantes.precipitacion_mm[n] == pytest.approx(fila[f"precip_{n}"])
                assert forzantes.evaporacion_mm[n] == pytest.approx(fila[f"evap_{n}"])
            assert forzantes.caudal_natural_m3s == pytest.approx(fila["caudal_saucio_m3s"])
            assert forzantes.mes == fecha.month
            assert forzantes.caudal_tibitoc_m3s == CAUDAL_TIBITOC_HISTORICO_M3S


# ---------------------------------------------------------------------------
# (c) Modo estocástico: corre sin error y es reproducible
# ---------------------------------------------------------------------------

class TestModoEstocastico:
    def test_corre_sin_error(self):
        modelo = _modelo_ajustado_sintetico()
        datos_fisicos = _datos_historicos_sinteticos(n_dias=10)
        covariables = pd.DataFrame(
            {"precipitacion_mm": np.full(10, 5.0), "roni": np.zeros(10)},
            index=datos_fisicos.index,
        )
        episodio = generar_episodio_estocastico(
            modelo, covariables, datos_fisicos, CAUDAL_TIBITOC_HISTORICO_M3S, semilla=1
        )
        assert len(episodio) == 10
        for forzantes in episodio:
            assert all(v >= 0.0 for v in forzantes.afluencia_m3s.values())
            assert forzantes.caudal_natural_m3s >= 0.0

    def test_misma_semilla_mismos_forzantes(self):
        modelo = _modelo_ajustado_sintetico()
        datos_fisicos = _datos_historicos_sinteticos(n_dias=10)
        covariables = pd.DataFrame(
            {"precipitacion_mm": np.full(10, 5.0), "roni": np.zeros(10)},
            index=datos_fisicos.index,
        )
        episodio_1 = generar_episodio_estocastico(
            modelo, covariables, datos_fisicos, CAUDAL_TIBITOC_HISTORICO_M3S, semilla=99
        )
        episodio_2 = generar_episodio_estocastico(
            modelo, covariables, datos_fisicos, CAUDAL_TIBITOC_HISTORICO_M3S, semilla=99
        )
        for f1, f2 in zip(episodio_1, episodio_2):
            assert f1.afluencia_m3s == pytest.approx(f2.afluencia_m3s)
            assert f1.caudal_natural_m3s == pytest.approx(f2.caudal_natural_m3s)

    def test_semillas_distintas_dan_forzantes_distintos(self):
        modelo = _modelo_ajustado_sintetico()
        datos_fisicos = _datos_historicos_sinteticos(n_dias=10)
        covariables = pd.DataFrame(
            {"precipitacion_mm": np.full(10, 5.0), "roni": np.zeros(10)},
            index=datos_fisicos.index,
        )
        episodio_1 = generar_episodio_estocastico(
            modelo, covariables, datos_fisicos, CAUDAL_TIBITOC_HISTORICO_M3S, semilla=1
        )
        episodio_2 = generar_episodio_estocastico(
            modelo, covariables, datos_fisicos, CAUDAL_TIBITOC_HISTORICO_M3S, semilla=2
        )
        distinto = any(
            f1.afluencia_m3s != pytest.approx(f2.afluencia_m3s)
            for f1, f2 in zip(episodio_1, episodio_2)
        )
        assert distinto


# ---------------------------------------------------------------------------
# (d) Despacho de generar_episodio
# ---------------------------------------------------------------------------

class TestGenerarEpisodioDespacho:
    def test_dispatch_historico(self):
        datos = _datos_historicos_sinteticos(n_dias=5)
        config = ConfigFuenteForzantes(fuente="historico")
        episodio = generar_episodio(config, datos, CAUDAL_TIBITOC_HISTORICO_M3S)
        esperado = generar_episodio_historico(datos, CAUDAL_TIBITOC_HISTORICO_M3S)
        for f1, f2 in zip(episodio, esperado):
            assert f1 == f2

    def test_dispatch_estocastico_requiere_modelo_y_covariables(self):
        datos = _datos_historicos_sinteticos(n_dias=5)
        config = ConfigFuenteForzantes(fuente="estocastico", semilla=1)
        with pytest.raises(ValueError):
            generar_episodio(config, datos, CAUDAL_TIBITOC_HISTORICO_M3S)

    def test_dispatch_estocastico_coincide_con_funcion_directa(self):
        modelo = _modelo_ajustado_sintetico()
        datos_fisicos = _datos_historicos_sinteticos(n_dias=5)
        covariables = pd.DataFrame(
            {"precipitacion_mm": np.full(5, 5.0), "roni": np.zeros(5)},
            index=datos_fisicos.index,
        )
        config = ConfigFuenteForzantes(fuente="estocastico", semilla=3)
        episodio = generar_episodio(
            config, datos_fisicos, CAUDAL_TIBITOC_HISTORICO_M3S, covariables_episodio=covariables, modelo=modelo
        )
        esperado = generar_episodio_estocastico(
            modelo, covariables, datos_fisicos, CAUDAL_TIBITOC_HISTORICO_M3S, semilla=3
        )
        for f1, f2 in zip(episodio, esperado):
            assert f1.afluencia_m3s == pytest.approx(f2.afluencia_m3s)


# ---------------------------------------------------------------------------
# (e) Integración aislada: episodio estocástico corriendo en EntornoEmbalses
# ---------------------------------------------------------------------------

class TestIntegracionEntorno:
    def test_episodio_estocastico_corre_sin_error_y_en_rango(self):
        modelo = _modelo_ajustado_sintetico()
        datos_fisicos = _datos_historicos_sinteticos(n_dias=30, semilla=5)
        covariables = pd.DataFrame(
            {
                "precipitacion_mm": np.full(30, 5.0),
                "roni": np.zeros(30),
            },
            index=datos_fisicos.index,
        )
        config = ConfigFuenteForzantes(fuente="estocastico", semilla=11)
        episodio = generar_episodio(
            config, datos_fisicos, CAUDAL_TIBITOC_HISTORICO_M3S,
            covariables_episodio=covariables, modelo=modelo,
        )

        env = EntornoEmbalses()  # con_shield=False (default), sin agente: acción = 0
        env.reset()
        for forzantes in episodio:
            resultado = env.step({"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0}, forzantes)
            for n in NOMBRES:
                v = resultado.estado.volumen_mm3[n]
                assert EMBALSES[n].capacidad_min_mm3 - 1e-6 <= v <= EMBALSES[n].capacidad_max_mm3 + 1e-6, (
                    f"{n} fuera de rango: {v}"
                )

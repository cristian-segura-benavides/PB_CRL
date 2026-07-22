"""Remuestreo condicional por análogos climáticos — LÍNEA BASE de validación.

NO es el método final del modelo estocástico (ver `modelo.py` para el VARX
desestacionalizado, que sí lo es). Este módulo existe únicamente para
verificar, en `entrenamiento.py`, que el VARX no se aleja del comportamiento
observado antes de confiarle la extrapolación fuera del rango histórico —
algo que el remuestreo por análogos, por diseño, no puede hacer: solo
reproduce combinaciones YA vistas en el histórico.

MÉTODO: para cada día a simular, se buscan los k días históricos de
entrenamiento con las covariables (precipitación, RONI) más parecidas
(distancia euclidiana sobre covariables estandarizadas), y se remuestrea
—con reemplazo, uniformemente entre los k vecinos— el vector conjunto
observado de las 4 series objetivo en uno de esos días análogos. Al ser un
valor REAL observado, la correlación cruzada y la estacionalidad quedan
preservadas exactamente, sin ningún supuesto distribucional.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConfigAnalogos:
    """Configuración del remuestreo por análogos.

    Atributos
    ---------
    covariables : tuple[str, ...]
        Nombres de columna de las covariables usadas para medir similitud.
    series_objetivo : tuple[str, ...]
        Nombres de las 4 series objetivo a remuestrear.
    k_vecinos : int
        Número de días análogos más cercanos entre los que se sortea uno.
    """

    covariables: tuple[str, ...] = ("precipitacion_mm", "roni")
    series_objetivo: tuple[str, ...] = ("Saucio", "Neusa", "Sisga", "Tomine")
    k_vecinos: int = 15


class RemuestreoAnalogos:
    """Línea base de validación por k-NN sobre covariables. Ver docstring del módulo."""

    def __init__(self, config: ConfigAnalogos | None = None) -> None:
        self.config = config or ConfigAnalogos()
        self._ajustado = False
        self._cov_train_estandarizada: np.ndarray | None = None
        self._media_cov: np.ndarray | None = None
        self._desv_cov: np.ndarray | None = None
        self._objetivo_train: np.ndarray | None = None

    def fit(
        self,
        covariables_historicas: pd.DataFrame,
        series_objetivo_historicas: pd.DataFrame,
    ) -> "RemuestreoAnalogos":
        """Guarda el conjunto de entrenamiento (estandarizado) para buscar análogos."""
        cov_cols = list(self.config.covariables)
        obj_cols = list(self.config.series_objetivo)

        datos = covariables_historicas[cov_cols].join(
            series_objetivo_historicas[obj_cols], how="inner"
        ).dropna()
        if len(datos) < self.config.k_vecinos:
            raise ValueError(
                f"Muy pocos días válidos ({len(datos)}) para buscar "
                f"{self.config.k_vecinos} vecinos."
            )

        cov_arr = datos[cov_cols].to_numpy(dtype=float)
        self._media_cov = cov_arr.mean(axis=0)
        self._desv_cov = cov_arr.std(axis=0, ddof=1)
        self._desv_cov[self._desv_cov == 0] = 1.0  # evita división por cero (covariable constante)
        self._cov_train_estandarizada = (cov_arr - self._media_cov) / self._desv_cov
        self._objetivo_train = datos[obj_cols].to_numpy(dtype=float)

        self._ajustado = True
        return self

    def sample(self, covariables: pd.DataFrame, semilla: int) -> pd.DataFrame:
        """Remuestrea, para cada día, el vector conjunto de un análogo histórico."""
        if not self._ajustado:
            raise RuntimeError("El modelo no ha sido ajustado. Llama fit() primero.")

        cov_cols = list(self.config.covariables)
        obj_cols = list(self.config.series_objetivo)
        k_vecinos = self.config.k_vecinos

        cov_arr = covariables[cov_cols].to_numpy(dtype=float)
        cov_estandarizada = (cov_arr - self._media_cov) / self._desv_cov

        rng = np.random.default_rng(semilla)
        n = len(covariables)
        resultado = np.zeros((n, len(obj_cols)))

        for t in range(n):
            distancias = np.linalg.norm(self._cov_train_estandarizada - cov_estandarizada[t], axis=1)
            indices_vecinos = np.argpartition(distancias, min(k_vecinos, len(distancias) - 1))[:k_vecinos]
            elegido = rng.choice(indices_vecinos)
            resultado[t] = self._objetivo_train[elegido]

        return pd.DataFrame(resultado, index=covariables.index, columns=obj_cols)

"""Modelo estocástico multivariado de afluencias — VARX desestacionalizado con
componente hurdle.

Módulo AISLADO del resto del proyecto (fase 1 de Sebastian): no importa ni
modifica `hydrology.balance` ni `environment.entorno`. Se entrena y se prueba
de forma independiente; conectarlo al entorno de simulación es un paso
posterior, una vez validado.

POR QUÉ UN MODELO CONJUNTO Y NO CUATRO INDEPENDIENTES
----------------------------------------------------------------------------
Las cuatro salidas (caudal de Saucío, afluencia de Neusa, afluencia de Sisga,
afluencia de Tominé) pertenecen a la misma cuenca y están correlacionadas por
el mismo evento de lluvia. Cuatro modelos ajustados por separado podrían
generar escenarios físicamente inconsistentes — por ejemplo, Neusa en crecida
mientras Sisga está en sequía bajo la misma lluvia, algo que no ocurre en la
cuenca real. Un modelo conjunto, con una matriz de covarianza compartida entre
las cuatro salidas, preserva esa correlación por construcción.

MÉTODO ELEGIDO (decisión del usuario, 2026-07-22): VARX desestacionalizado
----------------------------------------------------------------------------
Se evaluaron tres alternativas (regresión + cópula, VARX, remuestreo por
análogos). Se eligió VARX porque es el único que modela en un solo marco
tanto la correlación CRUZADA entre las 4 series (vía la matriz de covarianza
Σ de las innovaciones) como la persistencia TEMPORAL (rachas de varios días
húmedos/secos, vía la estructura autorregresiva) — y porque, a diferencia del
remuestreo por análogos, puede EXTRAPOLAR fuera del rango histórico. Esto
importa porque el modelo va a alimentar al entorno de RL, donde interesa poder
generar escenarios más severos que el histórico para estresar el shield de
recuperación (mismo espíritu que el escenario "ampliado" de Tibitóc en
`data_contracts.captaciones`). El remuestreo por análogos (`analogos.py`) se
construyó primero, como línea base de validación: sirve para verificar que el
VARX no se aleja del comportamiento observado antes de confiarle la
extrapolación, pero NO es el método final.

TRATAMIENTO DE CEROS: HURDLE, NO LOGARITMO DESPLAZADO
----------------------------------------------------------------------------
Entre 5% y 10% de los días de cada serie objetivo son exactamente cero — pero
esto es mayormente un ARTEFACTO de la limpieza del balance inverso (afluencias
negativas acotadas a cero en la "capa 2", ver NOTAS.md 4b), no sequía real. Un
logaritmo desplazado trataría ese artefacto como si fuera información
hidrológica genuina. En su lugar, cada serie se modela con un componente
"hurdle": una regresión logística que predice la probabilidad de estado BAJO
(valor ≤ `umbral_bajo_m3s`, por defecto 0.0 — el punto exacto del acotamiento)
en función de las covariables y el mes; el valor CONTINUO condicional a estado
ALTO se modela por separado, vía el VARX, sobre `log1p(valor - umbral)`
desestacionalizado (media mensual restada, calculada solo con días en estado
alto, para no contaminar la estacionalidad con el artefacto).

SIMPLIFICACIONES DOCUMENTADAS (para iterar después de la validación, no para
ocultar):
  - La ocurrencia de estado bajo se sortea de forma INDEPENDIENTE entre las 4
    series y entre días consecutivos (solo la estacionalidad mensual, vía las
    variables dummy del mes en la regresión logística, introduce estructura).
    La correlación cruzada del modelo vive en el componente VARX continuo, no
    en el hurdle. Si la validación muestra que la correlación observada entre
    las 4 series depende de forma importante de ocurrencias conjuntas de
    estado bajo (no solo de la magnitud continua), esto necesitaría un
    mecanismo de cópula latente (p. ej. Wilks 1998) como refinamiento.
  - Cuando un día se sortea en estado bajo, la memoria autorregresiva (el
    término A_{t-1} del VARX) se resetea a cero para esa serie ese día, en vez
    de propagar una anomalía indefinida.
  - Las covariables entran de forma CONTEMPORÁNEA (X_t, sin rezagos propios)
    al componente VARX; solo las salidas tienen memoria autorregresiva.

EXTENSIBILIDAD A UNA TERCERA COVARIABLE (temperatura)
----------------------------------------------------------------------------
Por decisión del usuario (2026-07-22), el modelo se entrena SOLO con
precipitación y RONI — la temperatura no existe todavía en el proyecto (no hay
loader ni fuente integrada) y su efecto es de segundo orden (evapotranspiración)
frente a precipitación y ENSO. El modelo queda preparado para una tercera
covariable con el MISMO patrón de configuración por escenario ya usado para la
extracción de Tibitóc (`data_contracts.captaciones`): `ConfigModeloEstocastico
.covariables` es una tupla de nombres de columna, no un número fijo de
argumentos — agregar temperatura el día que exista una fuente es agregar su
nombre a esa tupla y su columna al DataFrame de covariables pasado a `fit`/
`sample`, sin tocar la lógica de este módulo.

INTERFAZ
----------------------------------------------------------------------------
    modelo = ModeloEstocasticoAfluencias(config)
    modelo.fit(covariables_historicas, series_objetivo_historicas)
    muestra = modelo.sample(covariables_futuras, semilla=42)
    modelo.guardar("ruta/al/modelo")
    modelo2 = ModeloEstocasticoAfluencias.cargar("ruta/al/modelo")

Serialización en `.npz` (arrays numéricos) + `.json` (configuración) — NO se
usa pickle, para no depender de un formato que ejecute código arbitrario al
cargar un modelo entrenado con datos reales.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

_MESES_DUMMY = tuple(range(2, 13))  # enero es la referencia (capturada por el intercepto)


@dataclass
class ConfigModeloEstocastico:
    """Configuración del modelo estocástico VARX + hurdle.

    Atributos
    ---------
    covariables : tuple[str, ...]
        Nombres de columna de las covariables exógenas, en el DataFrame que se
        pasa a `fit`/`sample`. Por defecto, precipitación y RONI (ver
        docstring del módulo — decisión de excluir temperatura por ahora).
        Agregar una covariable nueva (p. ej. temperatura) es agregar su nombre
        aquí y su columna en los DataFrames de entrada, sin tocar la lógica.
    series_objetivo : tuple[str, ...]
        Nombres de las 4 series objetivo, en el orden usado internamente por
        la matriz de covarianza Σ.
    orden_varx : int
        Orden autorregresivo del componente VARX (rezagos de la propia serie).
    umbral_bajo_m3s : float
        Umbral que define el estado "bajo" del componente hurdle (valor ≤
        umbral). Por defecto 0.0: el punto exacto del acotamiento de
        afluencias negativas en el balance inverso (ver NOTAS.md 4b) — no es
        un umbral de sequía hidrológica, es el artefacto de limpieza que el
        hurdle aísla del componente continuo.
    """

    covariables: tuple[str, ...] = ("precipitacion_mm", "roni")
    series_objetivo: tuple[str, ...] = ("Saucio", "Neusa", "Sisga", "Tomine")
    orden_varx: int = 1
    umbral_bajo_m3s: float = 0.0


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    eta_acotado = np.clip(eta, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-eta_acotado))


def _ajustar_logistica(X: np.ndarray, y: np.ndarray, max_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
    """Ajusta una regresión logística por Newton-Raphson (IRLS), en numpy puro.

    `X` debe incluir la columna de intercepto (unos). Devuelve el vector de
    coeficientes beta.
    """
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        p = _sigmoid(X @ beta)
        w = np.clip(p * (1.0 - p), 1e-6, None)
        XtW = X.T * w
        hessiano = XtW @ X + 1e-8 * np.eye(k)
        gradiente = X.T @ (y - p)
        delta = np.linalg.solve(hessiano, gradiente)
        beta = beta + delta
        if np.max(np.abs(delta)) < tol:
            break
    return beta


def _dummies_mes(meses: np.ndarray) -> np.ndarray:
    """11 columnas dummy (una por mes, febrero a diciembre; enero es la referencia)."""
    return np.column_stack([(meses == m).astype(float) for m in _MESES_DUMMY])


def _diseno_hurdle(covariables_arr: np.ndarray, meses: np.ndarray) -> np.ndarray:
    """Matriz de diseño del componente hurdle: [intercepto, covariables, dummies de mes]."""
    n = covariables_arr.shape[0]
    intercepto = np.ones((n, 1))
    return np.hstack([intercepto, covariables_arr, _dummies_mes(meses)])


class ModeloEstocasticoAfluencias:
    """VARX desestacionalizado con componente hurdle para las 4 series objetivo.

    Ver el docstring del módulo para el método, las decisiones y las
    simplificaciones documentadas.
    """

    def __init__(self, config: ConfigModeloEstocastico | None = None) -> None:
        self.config = config or ConfigModeloEstocastico()
        self._ajustado = False

        # Parámetros aprendidos por fit() — nombrados con un solo guion bajo
        # (no un guion doble) porque se serializan explícitamente en guardar().
        self._hurdle_beta: dict[str, np.ndarray] = {}
        self._media_g: dict[str, np.ndarray] = {}  # 12 valores (uno por mes) de log1p(valor-umbral) medio
        self._intercepto_varx: np.ndarray | None = None  # (4,)
        self._phi: np.ndarray | None = None               # (4,4): coeficientes autorregresivos
        self._b_exog: np.ndarray | None = None            # (4, n_cov): coeficientes de covariables
        self._sigma: np.ndarray | None = None             # (4,4): covarianza residual
        self._chol_sigma: np.ndarray | None = None        # (4,4): factor de Cholesky de _sigma

    # ------------------------------------------------------------------
    # Ajuste
    # ------------------------------------------------------------------

    def fit(
        self,
        covariables_historicas: pd.DataFrame,
        series_objetivo_historicas: pd.DataFrame,
    ) -> "ModeloEstocasticoAfluencias":
        """Ajusta el modelo con datos históricos.

        Parámetros
        ----------
        covariables_historicas : pd.DataFrame
            Índice de fechas (DatetimeIndex), columnas = `config.covariables`.
        series_objetivo_historicas : pd.DataFrame
            Mismo índice, columnas = `config.series_objetivo` [m³/s].
            Se asume ya limpia (sin duplicados de índice); las filas con NaN
            en cualquier covariable o serie objetivo se excluyen del ajuste.

        Retorna
        -------
        ModeloEstocasticoAfluencias
            self, para poder encadenar `ModeloEstocasticoAfluencias().fit(...)`.
        """
        cov_cols = list(self.config.covariables)
        obj_cols = list(self.config.series_objetivo)

        datos = covariables_historicas[cov_cols].join(
            series_objetivo_historicas[obj_cols], how="inner"
        ).dropna()
        if len(datos) < 30:
            raise ValueError(
                f"Muy pocos días válidos para ajustar el modelo ({len(datos)}); "
                "se requieren al menos 30."
            )

        meses = datos.index.month.to_numpy()
        cov_arr = datos[cov_cols].to_numpy(dtype=float)
        umbral = self.config.umbral_bajo_m3s

        estado_bajo = {}  # serie -> array bool (True = bajo)
        g_alto = {}       # serie -> array float con NaN en días bajos (log1p(valor-umbral))

        for serie in obj_cols:
            valores = datos[serie].to_numpy(dtype=float)
            bajo = valores <= umbral
            estado_bajo[serie] = bajo

            # --- Componente hurdle: regresión logística ---
            X_hurdle = _diseno_hurdle(cov_arr, meses)
            self._hurdle_beta[serie] = _ajustar_logistica(X_hurdle, bajo.astype(float))

            # --- Media estacional del componente continuo (solo días "alto") ---
            g = np.full(len(datos), np.nan)
            g[~bajo] = np.log1p(valores[~bajo] - umbral)
            g_alto[serie] = g

            media_g_mes = np.zeros(12)
            for m in range(1, 13):
                mascara_mes = (meses == m) & ~bajo
                if mascara_mes.sum() == 0:
                    raise ValueError(
                        f"Serie '{serie}': no hay ningún día en estado alto en el mes {m} "
                        "en los datos de entrenamiento; no se puede estimar su media estacional."
                    )
                media_g_mes[m - 1] = g[mascara_mes].mean()
            self._media_g[serie] = media_g_mes

        # --- Anomalías desestacionalizadas (NaN en días "bajo") ---
        anomalia = np.full((len(datos), len(obj_cols)), np.nan)
        for j, serie in enumerate(obj_cols):
            media_por_dia = self._media_g[serie][meses - 1]
            anomalia[:, j] = g_alto[serie] - media_por_dia

        self._ajustar_varx(anomalia, cov_arr)
        self._ajustado = True
        return self

    def _ajustar_varx(self, anomalia: np.ndarray, cov_arr: np.ndarray) -> None:
        """Ajusta el componente VARX(p) sobre las anomalías (numpy puro, OLS)."""
        p = self.config.orden_varx
        n, k = anomalia.shape
        n_cov = cov_arr.shape[1]

        # Fila t valida como TARGET solo si las 4 series estan en estado alto en t.
        fila_completa = ~np.isnan(anomalia).any(axis=1)

        filas_y: list[np.ndarray] = []
        filas_z: list[np.ndarray] = []
        for t in range(p, n):
            if not fila_completa[t]:
                continue
            # Orden de los rezagos: del más antiguo (t-p) al más reciente (t-1),
            # para que coincida EXACTAMENTE con el orden de
            # `historial_anomalia[-p:]` (lista cronológica) usado en sample().
            # Con orden_varx=1 (el valor por defecto) no hay ambigüedad de orden.
            rezagos = []
            for lag in range(p, 0, -1):
                fila_lag = anomalia[t - lag]
                if np.isnan(fila_lag).any():
                    # Estado bajo en el rezago: memoria autorregresiva reseteada a cero
                    # para esa(s) serie(s) (ver docstring del módulo).
                    fila_lag = np.nan_to_num(fila_lag, nan=0.0)
                rezagos.append(fila_lag)
            z = np.concatenate([[1.0], *rezagos, cov_arr[t]])
            filas_z.append(z)
            filas_y.append(anomalia[t])

        if len(filas_y) < 20:
            raise ValueError(
                f"Muy pocas filas completas para ajustar el VARX ({len(filas_y)}); "
                "se requieren al menos 20 días con las 4 series en estado alto."
            )

        Z = np.vstack(filas_z)
        Y = np.vstack(filas_y)

        coef, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
        # coef: (1 + p*k + n_cov, k). Filas: intercepto, rezagos (p bloques de k,
        # del más antiguo al más reciente), covariables.
        self._intercepto_varx = coef[0, :]
        # _phi queda en forma (k, p*k) — la misma forma en la que se usa
        # directamente en sample() contra `rezagos` (también (p*k,)), sin
        # reshapes intermedios.
        self._phi = coef[1 : 1 + p * k, :].T if p > 0 else np.zeros((k, 0))
        self._b_exog = coef[1 + p * k :, :].T

        residuos = Y - Z @ coef
        self._sigma = np.cov(residuos, rowvar=False, ddof=1) + 1e-9 * np.eye(k)
        self._chol_sigma = np.linalg.cholesky(self._sigma)

    # ------------------------------------------------------------------
    # Muestreo
    # ------------------------------------------------------------------

    def sample(
        self,
        covariables: pd.DataFrame,
        semilla: int,
        anomalia_inicial: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Genera UNA realización de las 4 series, preservando su correlación conjunta.

        Parámetros
        ----------
        covariables : pd.DataFrame
            Índice de fechas, columnas = `config.covariables`, para el período
            a simular.
        semilla : int
            Semilla del generador de números aleatorios (reproducible: la
            misma semilla y las mismas covariables producen la misma muestra).
        anomalia_inicial : dict[str, float] | None
            Anomalía (en espacio log1p desestacionalizado) de cada serie en el
            día anterior al primero de `covariables`, para la memoria
            autorregresiva. Por defecto None: se asume 0.0 para las 4 series
            (arranca "en la media estacional").

        Retorna
        -------
        pd.DataFrame
            Mismo índice que `covariables`, columnas = `config.series_objetivo`
            [m³/s].

        Raises
        ------
        RuntimeError
            Si el modelo no ha sido ajustado (fit()) ni cargado (cargar()).
        """
        if not self._ajustado:
            raise RuntimeError("El modelo no ha sido ajustado. Llama fit() o cargar() primero.")

        cov_cols = list(self.config.covariables)
        obj_cols = list(self.config.series_objetivo)
        k = len(obj_cols)
        p = self.config.orden_varx
        umbral = self.config.umbral_bajo_m3s

        cov_arr = covariables[cov_cols].to_numpy(dtype=float)
        meses = covariables.index.month.to_numpy()
        n = len(covariables)

        rng = np.random.default_rng(semilla)

        # --- Sorteo del componente hurdle: independiente entre series y entre días
        # (ver simplificación documentada en el docstring del módulo) ---
        estado_bajo = np.zeros((n, k), dtype=bool)
        for j, serie in enumerate(obj_cols):
            X_hurdle = _diseno_hurdle(cov_arr, meses)
            p_bajo = _sigmoid(X_hurdle @ self._hurdle_beta[serie])
            estado_bajo[:, j] = rng.random(n) < p_bajo

        # --- Estado inicial de la memoria autorregresiva ---
        historial_anomalia = [
            np.array([(anomalia_inicial or {}).get(s, 0.0) for s in obj_cols])
            for _ in range(p)
        ]

        valores = np.zeros((n, k))
        for t in range(n):
            rezagos = np.concatenate(historial_anomalia[-p:]) if p > 0 else np.array([])

            media_condicional = self._intercepto_varx + (
                self._phi @ rezagos if p > 0 else 0.0
            ) + self._b_exog @ cov_arr[t]
            ruido = self._chol_sigma @ rng.standard_normal(k)
            a_t = media_condicional + ruido

            anomalia_efectiva = np.zeros(k)
            for j, serie in enumerate(obj_cols):
                if estado_bajo[t, j]:
                    valores[t, j] = umbral
                    anomalia_efectiva[j] = 0.0  # memoria reseteada (ver docstring)
                else:
                    media_est = self._media_g[serie][meses[t] - 1]
                    valores[t, j] = max(np.expm1(media_est + a_t[j]) + umbral, umbral)
                    anomalia_efectiva[j] = a_t[j]

            historial_anomalia.append(anomalia_efectiva)

        return pd.DataFrame(valores, index=covariables.index, columns=obj_cols)

    # ------------------------------------------------------------------
    # Serialización (npz + json — nunca pickle)
    # ------------------------------------------------------------------

    def guardar(self, ruta: str | Path) -> None:
        """Serializa el modelo ajustado a disco (config.json + parametros.npz).

        `ruta` se crea como directorio si no existe. NO usa pickle: los
        parámetros numéricos van en `.npz` (numpy) y la configuración en
        `.json`, para no depender de un formato que ejecute código arbitrario
        al cargar un modelo entrenado con datos reales.
        """
        if not self._ajustado:
            raise RuntimeError("No se puede guardar un modelo sin ajustar.")

        directorio = Path(ruta)
        directorio.mkdir(parents=True, exist_ok=True)

        with open(directorio / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)

        arrays: dict[str, np.ndarray] = {
            "intercepto_varx": self._intercepto_varx,
            "phi": self._phi,
            "b_exog": self._b_exog,
            "sigma": self._sigma,
            "chol_sigma": self._chol_sigma,
        }
        for serie in self.config.series_objetivo:
            arrays[f"hurdle_beta__{serie}"] = self._hurdle_beta[serie]
            arrays[f"media_g__{serie}"] = self._media_g[serie]
        np.savez(directorio / "parametros.npz", **arrays)

    @classmethod
    def cargar(cls, ruta: str | Path) -> "ModeloEstocasticoAfluencias":
        """Carga un modelo previamente guardado con `guardar()`."""
        directorio = Path(ruta)
        with open(directorio / "config.json", "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config_dict["covariables"] = tuple(config_dict["covariables"])
        config_dict["series_objetivo"] = tuple(config_dict["series_objetivo"])
        config = ConfigModeloEstocastico(**config_dict)

        modelo = cls(config=config)
        with np.load(directorio / "parametros.npz") as datos:
            modelo._intercepto_varx = datos["intercepto_varx"]
            modelo._phi = datos["phi"]
            modelo._b_exog = datos["b_exog"]
            modelo._sigma = datos["sigma"]
            modelo._chol_sigma = datos["chol_sigma"]
            for serie in config.series_objetivo:
                modelo._hurdle_beta[serie] = datos[f"hurdle_beta__{serie}"]
                modelo._media_g[serie] = datos[f"media_g__{serie}"]

        modelo._ajustado = True
        return modelo

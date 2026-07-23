"""Fuente configurable de forzantes para un episodio: histórica (datos
observados, comportamiento actual) o estocástica (modelo entrenado de
afluencias, src/pbcrl/stochastic/).

NO MODIFICA `environment.entorno` NI `environment.config` — decisión de
diseño, no descuido: `EntornoEmbalses.step()` ya es agnóstico a de dónde
vienen los `ForzantesExternos` (los recibe como argumento, sin preguntar su
origen). La fuente de forzantes es responsabilidad de quien CONDUCE el
episodio completo (el loop `reset()` + `step()` repetido), un nivel por
encima de `step()` — por eso no se agregó como campo de `ConfigEntorno`
(a diferencia de `con_shield`, que sí vive ahí porque gobierna el
comportamiento INTERNO de `step()`). Mezclarlo en `ConfigEntorno` mezclaría
responsabilidades de dos capas distintas.

DOS FUENTES:
  - "historico": construye ForzantesExternos directamente de datos
    observados (Saucío + afluencias por balance inverso + precipitación/
    evaporación por embalse) — EXACTAMENTE lo que ya hacían los scripts de
    scratch_shield/, ahora formalizado en una función reutilizable.
  - "estocastico": usa `stochastic.modelo.ModeloEstocasticoAfluencias.sample()`
    para generar las 4 series objetivo (Saucío, afluencia Neusa/Sisga/Tominé)
    a partir de una secuencia de covariables (precipitación, RONI) — ver
    `generar_episodio_estocastico`. La precipitación/evaporación POR EMBALSE
    que necesita el balance físico de cada embalse NO la genera el modelo
    (que solo produce las 4 series objetivo): sigue viniendo de una fuente
    externa (histórica, en la verificación inicial) incluso en modo
    estocástico — ver el parámetro `datos_fisicos` de `generar_episodio`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from pbcrl.environment.entorno import ForzantesExternos
from pbcrl.stochastic.modelo import ModeloEstocasticoAfluencias

NOMBRES_EMBALSES: tuple[str, ...] = ("Neusa", "Sisga", "Tomine")
FUENTES_VALIDAS: tuple[str, ...] = ("historico", "estocastico")


@dataclass
class ConfigFuenteForzantes:
    """Configuración explícita de la fuente de forzantes de un episodio.

    Atributos
    ---------
    fuente : Literal["historico", "estocastico"]
        "historico" (por defecto): datos observados, comportamiento actual
        sin cambios. "estocastico": genera las 4 series objetivo con el
        modelo entrenado.
    semilla : int | None
        Semilla del generador estocástico (reproducibilidad). Requerida si
        `fuente == "estocastico"`; ignorada si `fuente == "historico"`.
    """

    fuente: Literal["historico", "estocastico"] = "historico"
    semilla: int | None = None

    def __post_init__(self) -> None:
        if self.fuente not in FUENTES_VALIDAS:
            raise ValueError(f"fuente inválida: {self.fuente!r}. Válidas: {FUENTES_VALIDAS}")
        if self.fuente == "estocastico" and self.semilla is None:
            raise ValueError(
                "fuente='estocastico' requiere una semilla explícita (reproducibilidad)."
            )


def generar_episodio_historico(
    datos_historicos: pd.DataFrame,
    tibitoc_nominal_m3s: float,
) -> list[ForzantesExternos]:
    """Construye la secuencia de ForzantesExternos a partir de datos observados.

    Comportamiento IDÉNTICO al que ya usaban los scripts de scratch_shield/ —
    esta función solo lo formaliza para reutilizarlo, no cambia nada.

    Parámetros
    ----------
    datos_historicos : pd.DataFrame
        Índice de fechas; columnas `caudal_saucio_m3s`, y para cada embalse
        en NOMBRES_EMBALSES: `afluencia_{n}`, `precip_{n}`, `evap_{n}` (mismo
        formato que `scratch_shield.simulacion_historica_con_shield.
        construir_serie_diaria`).
    tibitoc_nominal_m3s : float
        Caudal nominal del escenario de extracción de Tibitóc (constante
        para todo el episodio).
    """
    episodio: list[ForzantesExternos] = []
    for fecha, fila in datos_historicos.iterrows():
        episodio.append(
            ForzantesExternos(
                afluencia_m3s={n: float(fila[f"afluencia_{n}"]) for n in NOMBRES_EMBALSES},
                precipitacion_mm={n: float(fila[f"precip_{n}"]) for n in NOMBRES_EMBALSES},
                evaporacion_mm={n: float(fila[f"evap_{n}"]) for n in NOMBRES_EMBALSES},
                caudal_natural_m3s=float(fila["caudal_saucio_m3s"]),
                mes=fecha.month,
                caudal_tibitoc_m3s=tibitoc_nominal_m3s,
            )
        )
    return episodio


def generar_episodio_estocastico(
    modelo: ModeloEstocasticoAfluencias,
    covariables_episodio: pd.DataFrame,
    datos_fisicos: pd.DataFrame,
    tibitoc_nominal_m3s: float,
    semilla: int,
) -> list[ForzantesExternos]:
    """Construye la secuencia de ForzantesExternos generando las 4 series con el modelo.

    Parámetros
    ----------
    modelo : ModeloEstocasticoAfluencias
        Modelo ya ajustado (fit() o cargar()).
    covariables_episodio : pd.DataFrame
        Índice de fechas, columnas = `modelo.config.covariables` (por
        defecto, `precipitacion_mm` agregada de cuenca + `roni`) — ver
        `stochastic.entrenamiento.construir_datos`. Inicialmente, una
        secuencia tomada del histórico (para comparar directamente contra
        resultados ya calculados); más adelante puede ser sintética.
    datos_fisicos : pd.DataFrame
        MISMO índice que `covariables_episodio`. Columnas `precip_{n}` y
        `evap_{n}` por embalse — el modelo NO genera precipitación ni
        evaporación por embalse (solo las 4 series objetivo), así que estos
        valores siguen viniendo de una fuente externa incluso en modo
        estocástico.
    tibitoc_nominal_m3s : float
        Caudal nominal del escenario de extracción de Tibitóc.
    semilla : int
        Semilla del generador (reproducibilidad: misma semilla, mismos
        forzantes).
    """
    muestra = modelo.sample(covariables_episodio, semilla=semilla)

    episodio: list[ForzantesExternos] = []
    for fecha in covariables_episodio.index:
        fila_fisica = datos_fisicos.loc[fecha]
        episodio.append(
            ForzantesExternos(
                afluencia_m3s={n: float(muestra.loc[fecha, n]) for n in NOMBRES_EMBALSES},
                precipitacion_mm={n: float(fila_fisica[f"precip_{n}"]) for n in NOMBRES_EMBALSES},
                evaporacion_mm={n: float(fila_fisica[f"evap_{n}"]) for n in NOMBRES_EMBALSES},
                caudal_natural_m3s=float(muestra.loc[fecha, "Saucio"]),
                mes=fecha.month,
                caudal_tibitoc_m3s=tibitoc_nominal_m3s,
            )
        )
    return episodio


def generar_episodio(
    config: ConfigFuenteForzantes,
    datos_historicos: pd.DataFrame,
    tibitoc_nominal_m3s: float,
    covariables_episodio: pd.DataFrame | None = None,
    modelo: ModeloEstocasticoAfluencias | None = None,
) -> list[ForzantesExternos]:
    """Punto de entrada único: despacha según `config.fuente`.

    En modo "historico", `datos_historicos` provee TODO (afluencias,
    caudal de Saucío, precipitación/evaporación). En modo "estocastico",
    `datos_historicos` se usa SOLO para precipitación/evaporación por
    embalse (ver `generar_episodio_estocastico`); `covariables_episodio` y
    `modelo` son obligatorios en ese caso.
    """
    if config.fuente == "historico":
        return generar_episodio_historico(datos_historicos, tibitoc_nominal_m3s)

    if modelo is None or covariables_episodio is None:
        raise ValueError(
            "fuente='estocastico' requiere pasar `modelo` y `covariables_episodio`."
        )
    return generar_episodio_estocastico(
        modelo=modelo,
        covariables_episodio=covariables_episodio,
        datos_fisicos=datos_historicos,
        tibitoc_nominal_m3s=tibitoc_nominal_m3s,
        semilla=config.semilla,
    )

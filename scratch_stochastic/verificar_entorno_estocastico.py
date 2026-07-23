"""Verifica, de forma AISLADA (sin agente, sin shield), que un episodio
completo corrido con fuente "estocastico" produce una dinámica de embalses
físicamente sensata: volúmenes dentro de rango, sin errores, y magnitudes
del mismo orden que el histórico real para el MISMO período de covariables.

Compara lado a lado, para el mismo período y las MISMAS acciones (descargas
históricas observadas — no hay agente todavía), dos episodios:
  - "historico": forzantes de datos observados (comportamiento actual).
  - "estocastico": forzantes generados por el modelo entrenado, a partir de
    la MISMA secuencia de covariables (precipitación, RONI) de ese período.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from pbcrl.data_contracts.captaciones import CAUDAL_TIBITOC_HISTORICO_M3S  # noqa: E402
from pbcrl.data_contracts.embalses import EMBALSES  # noqa: E402
from pbcrl.environment.config import ConfigEntorno  # noqa: E402
from pbcrl.environment.entorno import EntornoEmbalses  # noqa: E402
from pbcrl.environment.fuente_forzantes import (  # noqa: E402
    ConfigFuenteForzantes,
    generar_episodio,
)
from pbcrl.stochastic.entrenamiento import construir_datos  # noqa: E402
from pbcrl.stochastic.modelo import ModeloEstocasticoAfluencias  # noqa: E402

from scratch_shield.simulacion_historica_con_shield import (  # noqa: E402
    NOMBRES,
    construir_serie_diaria,
)

MODELO_RUTA = ROOT_DIR / "src" / "pbcrl" / "stochastic" / "artefactos" / "modelo_afluencias_v1"
PERIODO_INICIO = "2015-01-01"
PERIODO_FIN = "2017-12-31"  # 3 años, dentro del rango de entrenamiento del modelo guardado
SEMILLA = 2026


def correr_episodio(episodio, acciones_por_fecha: pd.DataFrame) -> pd.DataFrame:
    env = EntornoEmbalses(config=ConfigEntorno(con_shield=False))
    env.reset()
    registros = []
    for forzantes, (fecha, accion) in zip(episodio, acciones_por_fecha.iterrows()):
        resultado = env.step(
            {n: float(accion[f"descarga_{n}"]) for n in NOMBRES},
            forzantes,
        )
        fila = {"fecha": fecha}
        for n in NOMBRES:
            fila[f"volumen_{n}"] = resultado.estado.volumen_mm3[n]
        fila["caudal_saucio_m3s"] = forzantes.caudal_natural_m3s
        for n in NOMBRES:
            fila[f"afluencia_{n}"] = forzantes.afluencia_m3s[n]
        registros.append(fila)
    return pd.DataFrame(registros).set_index("fecha")


def main() -> None:
    print(f"Cargando modelo entrenado desde: {MODELO_RUTA}")
    modelo = ModeloEstocasticoAfluencias.cargar(MODELO_RUTA)

    print("Cargando datos históricos reales (covariables + físicos)...")
    covariables_completas, _ = construir_datos()
    datos_fisicos_completos = construir_serie_diaria()

    covariables_periodo = covariables_completas.loc[PERIODO_INICIO:PERIODO_FIN]
    datos_fisicos_periodo = datos_fisicos_completos.loc[PERIODO_INICIO:PERIODO_FIN]

    # Alinear al mismo índice exacto (por si acaso hay huecos distintos entre fuentes).
    indice_comun = covariables_periodo.index.intersection(datos_fisicos_periodo.index)
    covariables_periodo = covariables_periodo.loc[indice_comun]
    datos_fisicos_periodo = datos_fisicos_periodo.loc[indice_comun]
    print(f"Período: {indice_comun.min().date()} -> {indice_comun.max().date()} ({len(indice_comun)} días)\n")

    # --- Episodio histórico (comportamiento actual) ---
    config_historico = ConfigFuenteForzantes(fuente="historico")
    episodio_historico = generar_episodio(
        config_historico, datos_fisicos_periodo, CAUDAL_TIBITOC_HISTORICO_M3S
    )

    # --- Episodio estocástico (mismo período de covariables) ---
    config_estocastico = ConfigFuenteForzantes(fuente="estocastico", semilla=SEMILLA)
    episodio_estocastico = generar_episodio(
        config_estocastico,
        datos_fisicos_periodo,
        CAUDAL_TIBITOC_HISTORICO_M3S,
        covariables_episodio=covariables_periodo,
        modelo=modelo,
    )

    # Mismas acciones (descargas históricas observadas) en ambos episodios.
    resultado_historico = correr_episodio(episodio_historico, datos_fisicos_periodo)
    resultado_estocastico = correr_episodio(episodio_estocastico, datos_fisicos_periodo)

    print("=" * 78)
    print("(3) VERIFICACIÓN DE RANGO DE VOLUMEN — episodio estocástico")
    print("=" * 78)
    fuera_de_rango = 0
    for n in NOMBRES:
        params = EMBALSES[n]
        serie = resultado_estocastico[f"volumen_{n}"]
        malos = ((serie < params.capacidad_min_mm3 - 1e-6) | (serie > params.capacidad_max_mm3 + 1e-6)).sum()
        fuera_de_rango += malos
        print(f"  {n}: min={serie.min():.2f}, max={serie.max():.2f} "
              f"(rango válido [{params.capacidad_min_mm3:.1f}, {params.capacidad_max_mm3:.1f}]), "
              f"días fuera de rango: {malos}")
    print(f"  Total días fuera de rango (los 3 embalses): {fuera_de_rango}  "
          f"{'-> SIN ERRORES' if fuera_de_rango == 0 else '-> REVISAR'}")

    print("\n" + "=" * 78)
    print("(3) COMPARACIÓN DE MAGNITUDES — histórico real vs. estocástico (mismo período)")
    print("=" * 78)
    print("\nCaudal de Saucío y afluencias (m³/s) — media y desviación estándar:")
    columnas_series = ["caudal_saucio_m3s"] + [f"afluencia_{n}" for n in NOMBRES]
    comparacion = pd.DataFrame({
        "historico_media": resultado_historico[columnas_series].mean(),
        "estocastico_media": resultado_estocastico[columnas_series].mean(),
        "historico_std": resultado_historico[columnas_series].std(),
        "estocastico_std": resultado_estocastico[columnas_series].std(),
    })
    print(comparacion.round(3).to_string())

    print("\nVolumen por embalse (Mm³) — media y rango:")
    for n in NOMBRES:
        h = resultado_historico[f"volumen_{n}"]
        e = resultado_estocastico[f"volumen_{n}"]
        print(f"  {n}: histórico media={h.mean():.2f} [{h.min():.2f},{h.max():.2f}]  |  "
              f"estocástico media={e.mean():.2f} [{e.min():.2f},{e.max():.2f}]")

    print("\n" + "=" * 78)
    print("(4) REPRODUCIBILIDAD — misma semilla, mismo resultado")
    print("=" * 78)
    episodio_repetido = generar_episodio(
        config_estocastico,
        datos_fisicos_periodo,
        CAUDAL_TIBITOC_HISTORICO_M3S,
        covariables_episodio=covariables_periodo,
        modelo=modelo,
    )
    resultado_repetido = correr_episodio(episodio_repetido, datos_fisicos_periodo)
    idénticos = np.allclose(
        resultado_estocastico.to_numpy(), resultado_repetido.to_numpy(), equal_nan=True
    )
    print(f"  Misma semilla ({SEMILLA}) reproduce el episodio exactamente: {idénticos}")


if __name__ == "__main__":
    main()

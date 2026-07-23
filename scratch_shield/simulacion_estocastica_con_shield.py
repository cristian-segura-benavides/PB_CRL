"""Verificación AISLADA (sin agente) del shield y el modelo estocástico
CONECTADOS entre sí, en el mismo episodio — hasta ahora cada uno se había
verificado por separado (el shield contra forzantes históricos; el modelo
estocástico sin el shield). Este script corre el mismo período (2015-2017,
mismas acciones históricas observadas y misma semilla) usado en
`verificar_entorno_estocastico.py`, ahora con `ConfigEntorno(con_shield=True)`,
y compara CON shield vs. SIN shield sobre las MISMAS forzantes generadas por
el modelo.

No conecta ningún agente: la "acción propuesta" sigue siendo la descarga
históricamente observada, como en todas las verificaciones aisladas previas.
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
PERIODO_FIN = "2017-12-31"
SEMILLA = 2026


def correr_episodio(episodio, acciones_por_fecha: pd.DataFrame, con_shield: bool) -> pd.DataFrame:
    env = EntornoEmbalses(config=ConfigEntorno(con_shield=con_shield))
    env.reset()
    registros = []
    for forzantes, (fecha, accion) in zip(episodio, acciones_por_fecha.iterrows()):
        resultado = env.step(
            {n: float(accion[f"descarga_{n}"]) for n in NOMBRES},
            forzantes,
        )
        fila = {
            "fecha": fecha,
            "q_sol": resultado.estado.caudal_sol_m3s,
            "q_eco": resultado.q_eco_aplicado_m3s,
            "violacion": resultado.violacion_ecologica,
        }
        for n in NOMBRES:
            fila[f"volumen_{n}"] = resultado.estado.volumen_mm3[n]
        if resultado.diagnostico_shield is not None:
            fila["shield_actuo"] = any(resultado.diagnostico_shield.violaciones_previas.values())
        else:
            fila["shield_actuo"] = False
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
    indice_comun = covariables_periodo.index.intersection(datos_fisicos_periodo.index)
    covariables_periodo = covariables_periodo.loc[indice_comun]
    datos_fisicos_periodo = datos_fisicos_periodo.loc[indice_comun]
    print(f"Período: {indice_comun.min().date()} -> {indice_comun.max().date()} ({len(indice_comun)} días)\n")

    config_estocastico = ConfigFuenteForzantes(fuente="estocastico", semilla=SEMILLA)
    episodio_estocastico = generar_episodio(
        config_estocastico,
        datos_fisicos_periodo,
        CAUDAL_TIBITOC_HISTORICO_M3S,
        covariables_episodio=covariables_periodo,
        modelo=modelo,
    )

    print("Corriendo episodio CON shield (con_shield=True)...")
    resultado_con_shield = correr_episodio(episodio_estocastico, datos_fisicos_periodo, con_shield=True)
    print("Corriendo el MISMO episodio SIN shield (con_shield=False)...\n")
    resultado_sin_shield = correr_episodio(episodio_estocastico, datos_fisicos_periodo, con_shield=False)

    print("=" * 78)
    print("COMPARACIÓN: CON shield vs. SIN shield (mismas forzantes estocásticas, mismas acciones)")
    print("=" * 78)

    dias_shield_actuo = int(resultado_con_shield["shield_actuo"].sum())
    pct_shield_actuo = 100 * dias_shield_actuo / len(resultado_con_shield)
    print(f"\nDías donde el shield habría corregido la acción: {dias_shield_actuo} "
          f"({pct_shield_actuo:.2f}%)")

    viol_con = int(resultado_con_shield["violacion"].sum())
    viol_sin = int(resultado_sin_shield["violacion"].sum())
    print(f"\nDías con violación real de Q_eco:")
    print(f"  CON shield: {viol_con} ({100*viol_con/len(resultado_con_shield):.2f}%)")
    print(f"  SIN shield: {viol_sin} ({100*viol_sin/len(resultado_sin_shield):.2f}%)")

    if viol_con > 0:
        deficit_con = (resultado_con_shield["q_eco"] - resultado_con_shield["q_sol"]).clip(lower=0)
        deficit_con = deficit_con[resultado_con_shield["violacion"]]
        print(f"  Déficit medio en los días con violación CON shield: {deficit_con.mean():.3f} m³/s")
    deficit_sin = (resultado_sin_shield["q_eco"] - resultado_sin_shield["q_sol"]).clip(lower=0)
    deficit_sin_viol = deficit_sin[resultado_sin_shield["violacion"]]
    if not deficit_sin_viol.empty:
        print(f"  Déficit medio en los días con violación SIN shield: {deficit_sin_viol.mean():.3f} m³/s")

    print("\n" + "=" * 78)
    print("RANGO DE VOLUMEN (los 3 embalses, ambos modos)")
    print("=" * 78)
    for modo, resultado in [("CON shield", resultado_con_shield), ("SIN shield", resultado_sin_shield)]:
        fuera = 0
        for n in NOMBRES:
            params = EMBALSES[n]
            serie = resultado[f"volumen_{n}"]
            malos = ((serie < params.capacidad_min_mm3 - 1e-6) | (serie > params.capacidad_max_mm3 + 1e-6)).sum()
            fuera += malos
        print(f"  {modo}: días fuera de rango (suma los 3 embalses) = {fuera}")

    print("\n" + "=" * 78)
    print("REPRODUCIBILIDAD")
    print("=" * 78)
    episodio_repetido = generar_episodio(
        config_estocastico, datos_fisicos_periodo, CAUDAL_TIBITOC_HISTORICO_M3S,
        covariables_episodio=covariables_periodo, modelo=modelo,
    )
    resultado_repetido = correr_episodio(episodio_repetido, datos_fisicos_periodo, con_shield=True)
    identico = np.allclose(
        resultado_con_shield[["q_sol", "q_eco"]].to_numpy(),
        resultado_repetido[["q_sol", "q_eco"]].to_numpy(),
    )
    print(f"  Misma semilla ({SEMILLA}) reproduce el episodio CON shield exactamente: {identico}")


if __name__ == "__main__":
    main()

"""Simulación COMPLETA (dinámica + shield en cada paso) sobre la ventana
histórica, alimentada con las descargas REALMENTE observadas como "acción
propuesta" día a día. Sin agente de RL todavía — esto verifica que el
entorno con el shield conectado corre correctamente de punta a punta, no que
un agente aprenda nada.

A diferencia de scratch_shield/verificacion_historica.py (que evalúa el
shield de forma AISLADA, re-inyectando el volumen histórico cada día), este
script hace un ROLLOUT continuo: el volumen inicial es el histórico del
primer día, y de ahí en adelante el estado evoluciona por la dinámica propia
del entorno (afectada por las correcciones del shield), no se re-inyecta el
histórico. Es exactamente el "verificar que la simulación completa corre
correctamente" pedido — la trayectoria de volumen puede (y se espera que)
diverja algo de la histórica en los días donde el shield corrige.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from pbcrl.data_contracts.captaciones import (  # noqa: E402
    CAUDAL_TIBITOC_AMPLIADO_M3S,
    CAUDAL_TIBITOC_HISTORICO_M3S,
)
from pbcrl.data_contracts.embalses import EMBALSES  # noqa: E402
from pbcrl.data_ingest.saucio import cargar_saucio  # noqa: E402
from dashboard.data_loader import load_dashboard_context  # noqa: E402

from pbcrl.environment.config import ConfigEntorno  # noqa: E402
from pbcrl.environment.entorno import EntornoEmbalses, ForzantesExternos  # noqa: E402

NOMBRES = ("Neusa", "Sisga", "Tomine")


def construir_serie_diaria() -> pd.DataFrame:
    contexts, _, _ = load_dashboard_context()
    saucio_df, _ = cargar_saucio()

    columnas = {"caudal_saucio_m3s": saucio_df["caudal_m3s"]}
    for n in NOMBRES:
        frame = contexts[n].frame
        columnas[f"volumen_{n}"] = frame["volumen_mm3"]
        columnas[f"afluencia_{n}"] = frame["afluencia_m3s"]
        columnas[f"precip_{n}"] = frame["precipitacion_mm"]
        columnas[f"evap_{n}"] = frame["evaporacion_mm"]
        columnas[f"descarga_{n}"] = frame["descarga_m3s"]

    datos = pd.DataFrame(columnas)
    columnas_requeridas = [c for c in datos.columns]
    return datos.dropna(subset=columnas_requeridas)


def correr_simulacion(datos: pd.DataFrame, tibitoc_nominal: float) -> dict:
    volumenes_iniciales = {n: float(datos.iloc[0][f"volumen_{n}"]) for n in NOMBRES}

    config = ConfigEntorno(con_shield=True)
    env = EntornoEmbalses(config=config)
    env.reset(
        volumenes_iniciales_mm3=volumenes_iniciales,
        caudal_natural_inicial_m3s=float(datos.iloc[0]["caudal_saucio_m3s"]),
    )

    dias_totales = 0
    dias_shield_actua = 0
    dias_violacion = 0
    dias_fuera_de_rango_volumen = 0
    correcciones_abs = {n: [] for n in NOMBRES}  # magnitud |a*-â| en días de correccion
    fechas_violacion: list[str] = []
    fechas_fuera_de_rango: list[str] = []

    # Se salta la primera fila: ya se usó como estado inicial de reset().
    for fecha, fila in datos.iloc[1:].iterrows():
        forzantes = ForzantesExternos(
            afluencia_m3s={n: float(fila[f"afluencia_{n}"]) for n in NOMBRES},
            precipitacion_mm={n: float(fila[f"precip_{n}"]) for n in NOMBRES},
            evaporacion_mm={n: float(fila[f"evap_{n}"]) for n in NOMBRES},
            caudal_natural_m3s=float(fila["caudal_saucio_m3s"]),
            mes=fecha.month,
            caudal_tibitoc_m3s=tibitoc_nominal,
        )
        accion_propuesta = {n: float(fila[f"descarga_{n}"]) for n in NOMBRES}

        resultado = env.step(accion_propuesta, forzantes)
        dias_totales += 1

        diag = resultado.diagnostico_shield
        assert diag is not None
        actuo = any(diag.violaciones_previas.values())
        if actuo:
            dias_shield_actua += 1
            for n in NOMBRES:
                correcciones_abs[n].append(abs(diag.accion_proyectada[n] - diag.accion_propuesta[n]))

        if resultado.violacion_ecologica:
            dias_violacion += 1
            fechas_violacion.append(str(fecha.date()))

        for n in NOMBRES:
            v = resultado.estado.volumen_mm3[n]
            if not (EMBALSES[n].capacidad_min_mm3 - 1e-6 <= v <= EMBALSES[n].capacidad_max_mm3 + 1e-6):
                dias_fuera_de_rango_volumen += 1
                fechas_fuera_de_rango.append(f"{fecha.date()} ({n}={v:.3f})")
                break

    return {
        "dias_totales": dias_totales,
        "dias_shield_actua": dias_shield_actua,
        "dias_violacion": dias_violacion,
        "fechas_violacion": fechas_violacion,
        "dias_fuera_de_rango_volumen": dias_fuera_de_rango_volumen,
        "fechas_fuera_de_rango": fechas_fuera_de_rango,
        "correcciones_abs": correcciones_abs,
    }


def main() -> None:
    datos = construir_serie_diaria()
    print(f"Días con datos completos: {len(datos)} "
          f"({datos.index.min().date()} -> {datos.index.max().date()})\n")

    for nombre_escenario, nominal in [
        ("HISTÓRICO (4.5 m3/s)", CAUDAL_TIBITOC_HISTORICO_M3S),
        ("AMPLIADO (8.0 m3/s)", CAUDAL_TIBITOC_AMPLIADO_M3S),
    ]:
        print("=" * 78)
        print(f"ESCENARIO TIBITÓC: {nombre_escenario}")
        print("=" * 78)
        r = correr_simulacion(datos, nominal)

        pct_actua = 100 * r["dias_shield_actua"] / r["dias_totales"]
        print(f"Días simulados (rollout continuo): {r['dias_totales']}")
        print(f"Días donde el shield corrigió: {r['dias_shield_actua']} ({pct_actua:.2f}%)")

        print(f"\n(a) Garantía Q_ElSol >= Q_eco(mes): días con violación = {r['dias_violacion']}")
        if r["dias_violacion"] > 0:
            print(f"    NO se sostiene siempre. Primeras fechas: {r['fechas_violacion'][:10]}")
        else:
            print("    Se sostiene en el 100% de los días simulados.")

        print(f"\n(d) Volumen fuera de [min,max] en algún embalse: {r['dias_fuera_de_rango_volumen']} días")
        if r["dias_fuera_de_rango_volumen"] > 0:
            print(f"    Primeras fechas: {r['fechas_fuera_de_rango'][:10]}")
        else:
            print("    Los tres embalses se mantuvieron dentro de rango en todo el rollout.")

        print("\n(c) Magnitud típica de la corrección (m3/s), solo días donde el shield actuó:")
        for n in NOMBRES:
            valores = r["correcciones_abs"][n]
            if valores:
                arr = np.array(valores)
                print(f"    {n}: media={arr.mean():.3f}, mediana={np.median(arr):.3f}, "
                      f"p90={np.quantile(arr,0.90):.3f}, max={arr.max():.3f} "
                      f"(n={len(arr)} correcciones no nulas)")
            else:
                print(f"    {n}: sin correcciones no nulas")
        print()


if __name__ == "__main__":
    main()

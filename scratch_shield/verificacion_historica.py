"""Corre el shield de proyección contra los estados históricos reales, SIN
ningún agente todavía: la "acción propuesta" en cada día es la descarga
REALMENTE observada ese día para Neusa, Sisga y Tominé. Esto no evalúa si el
shield "funciona" con un agente — solo si el conjunto factible que define
está bien planteado, y qué tan seguido la operación histórica ya lo hubiera
satisfecho sin corrección.

Estado usado en el día t: volumen al INICIO del día t (= volumen reportado el
día t-1, ya que la columna `volumen_mm3` del proyecto representa el volumen
al final de cada día — ver hydrology/balance.py), afluencia/precipitación/
evaporación observadas el día t, caudal de Saucío el día t, mes del día t.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from pbcrl.data_contracts.captaciones import (  # noqa: E402
    CAUDAL_TIBITOC_AMPLIADO_M3S,
    CAUDAL_TIBITOC_HISTORICO_M3S,
)
from pbcrl.data_ingest.saucio import cargar_saucio  # noqa: E402
from dashboard.data_loader import load_dashboard_context  # noqa: E402

from pbcrl.shield.restricciones import EstadoShield  # noqa: E402
from pbcrl.shield.proyeccion import proyectar  # noqa: E402

NOMBRES = ("Neusa", "Sisga", "Tomine")


def construir_serie_estados() -> pd.DataFrame:
    """Un DataFrame indexado por fecha con todo lo necesario para EstadoShield
    y la acción histórica observada, alineado (dropna) sobre los cuatro
    componentes del sistema."""
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
    # Volumen AL INICIO del día t = volumen reportado el día t-1.
    for n in NOMBRES:
        datos[f"volumen_ini_{n}"] = datos[f"volumen_{n}"].shift(1)

    columnas_requeridas = (
        ["caudal_saucio_m3s"]
        + [f"volumen_ini_{n}" for n in NOMBRES]
        + [f"afluencia_{n}" for n in NOMBRES]
        + [f"precip_{n}" for n in NOMBRES]
        + [f"evap_{n}" for n in NOMBRES]
        + [f"descarga_{n}" for n in NOMBRES]
    )
    return datos.dropna(subset=columnas_requeridas)


def evaluar(datos: pd.DataFrame, tibitoc_nominal: float) -> dict:
    total = len(datos)
    dias_ya_factibles = 0
    dias_shield_actua = 0
    dias_infactibles = 0
    conteo_violaciones = {
        "caja_Neusa": 0, "caja_Sisga": 0, "caja_Tomine": 0,
        "caudal_ecologico_conjunto": 0,
    }
    fechas_infactibles: list[str] = []

    for fecha, fila in datos.iterrows():
        estado = EstadoShield(
            volumen_mm3={n: fila[f"volumen_ini_{n}"] for n in NOMBRES},
            afluencia_m3s={n: fila[f"afluencia_{n}"] for n in NOMBRES},
            precipitacion_mm={n: fila[f"precip_{n}"] for n in NOMBRES},
            evaporacion_mm={n: fila[f"evap_{n}"] for n in NOMBRES},
            caudal_saucio_m3s=fila["caudal_saucio_m3s"],
            mes=fecha.month,
            caudal_tibitoc_nominal_m3s=tibitoc_nominal,
        )
        accion_observada = {n: fila[f"descarga_{n}"] for n in NOMBRES}

        diag = proyectar(estado, accion_observada)

        if not any(diag.violaciones_previas.values()):
            dias_ya_factibles += 1
        else:
            dias_shield_actua += 1
            for k, violada in diag.violaciones_previas.items():
                if violada:
                    conteo_violaciones[k] = conteo_violaciones.get(k, 0) + 1

        if not diag.factible:
            dias_infactibles += 1
            fechas_infactibles.append(str(fecha.date()))

    return {
        "total_dias": total,
        "dias_ya_factibles": dias_ya_factibles,
        "dias_shield_actua": dias_shield_actua,
        "dias_infactibles": dias_infactibles,
        "conteo_violaciones": conteo_violaciones,
        "fechas_infactibles": fechas_infactibles,
    }


def main() -> None:
    datos = construir_serie_estados()
    print(f"Días con los cuatro componentes completos (volumen, afluencia, "
          f"precip, evap, descarga, Saucío): {len(datos)}")
    print(f"Rango: {datos.index.min().date()} -> {datos.index.max().date()}\n")

    for nombre_escenario, nominal in [
        ("HISTÓRICO (4.5 m3/s)", CAUDAL_TIBITOC_HISTORICO_M3S),
        ("AMPLIADO (8.0 m3/s)", CAUDAL_TIBITOC_AMPLIADO_M3S),
    ]:
        print("=" * 78)
        print(f"ESCENARIO TIBITÓC: {nombre_escenario}")
        print("=" * 78)
        r = evaluar(datos, nominal)
        pct_factible = 100 * r["dias_ya_factibles"] / r["total_dias"]
        pct_shield = 100 * r["dias_shield_actua"] / r["total_dias"]
        print(f"Total de días evaluados: {r['total_dias']}")
        print(f"  Ya factibles sin corrección: {r['dias_ya_factibles']} ({pct_factible:.2f}%)")
        print(f"  El shield habría corregido:  {r['dias_shield_actua']} ({pct_shield:.2f}%)")
        print(f"  Días REALMENTE infactibles (conjunto vacío): {r['dias_infactibles']}")
        print("\n  Desglose de violaciones (un día puede violar más de una):")
        for k, v in r["conteo_violaciones"].items():
            print(f"    {k}: {v} días ({100*v/r['total_dias']:.2f}%)")
        if r["fechas_infactibles"]:
            print(f"\n  Fechas infactibles (primeras 20 de {len(r['fechas_infactibles'])}):")
            print("   ", r["fechas_infactibles"][:20])
        print()


if __name__ == "__main__":
    main()

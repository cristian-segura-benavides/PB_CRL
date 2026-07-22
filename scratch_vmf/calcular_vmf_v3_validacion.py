"""Tres piezas para cerrar el umbral VMF (sobre el cálculo v2, régimen natural):

Pieza 1: validación del patrón estacional de Q_natural contra Saucío (caudal medido,
         aguas arriba de todos los embalses, referencia independiente).
Pieza 2: análisis de sensibilidad del umbral EFR (x0.8, x1.0, x1.2) para ambos
         escenarios de extracción.
Pieza 3 (verificación del caveat 2): ¿cambia la clasificación mensual VMF si se excluye
         el evento de vertimiento de Neusa (jul-nov 2022) del cálculo de MAF/MMF?

No toca el core del modelo (src/pbcrl); solo lee datos ya cargados por los loaders
existentes y reutiliza la lógica de calcular_vmf_v2.py.
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
from pbcrl.data_ingest.saucio import cargar_saucio  # noqa: E402
from dashboard.data_loader import load_dashboard_context  # noqa: E402

from calcular_vmf_v2 import (  # noqa: E402
    construir_q_elsol,
    construir_series_base,
    diagnostico_regimen,
)


def clasificar(mmf_m: float, maf: float) -> tuple[float, str]:
    if mmf_m <= 0.4 * maf:
        return 0.60, "bajo"
    if mmf_m <= 0.8 * maf:
        return 0.45, "intermedio"
    return 0.30, "alto"


def tabla_mmf_clasificacion(serie: pd.Series) -> pd.DataFrame:
    s = serie.dropna()
    maf = float(s.mean())
    mmf = s.groupby(s.index.month).mean()
    filas = []
    for mes, v in mmf.items():
        frac, reg = clasificar(v, maf)
        filas.append({"mes": mes, "MMF": v, "regimen": reg, "EFR": frac * v})
    df = pd.DataFrame(filas).set_index("mes")
    df.attrs["MAF"] = maf
    df.attrs["n"] = int(s.notna().sum())
    return df


def pieza1(df_natural: pd.DataFrame) -> None:
    print("=" * 78)
    print("PIEZA 1 — VALIDACIÓN DEL PATRÓN ESTACIONAL CONTRA SAUCÍO")
    print("=" * 78)

    saucio_df, _ = cargar_saucio()
    saucio = saucio_df["caudal_m3s"]

    tabla_saucio = tabla_mmf_clasificacion(saucio)
    tabla_natural = tabla_mmf_clasificacion(df_natural["q_natural_m3s"])

    print(f"\nMAF Saucío (solo, medido) = {tabla_saucio.attrs['MAF']:.3f} m3/s "
          f"({tabla_saucio.attrs['n']} días válidos)")
    print(f"MAF Q_natural (Saucío + 3 afluencias) = {tabla_natural.attrs['MAF']:.3f} m3/s "
          f"({tabla_natural.attrs['n']} días válidos)\n")

    comparacion = pd.DataFrame({
        "MMF_saucio": tabla_saucio["MMF"],
        "MMF_saucio_norm": tabla_saucio["MMF"] / tabla_saucio.attrs["MAF"],
        "regimen_saucio": tabla_saucio["regimen"],
        "MMF_natural": tabla_natural["MMF"],
        "MMF_natural_norm": tabla_natural["MMF"] / tabla_natural.attrs["MAF"],
        "regimen_natural": tabla_natural["regimen"],
    })
    comparacion["coincide"] = comparacion["regimen_saucio"] == comparacion["regimen_natural"]
    print(comparacion.round(3).to_string())

    # Spearman = correlacion de Pearson sobre los rangos (scipy no disponible en el entorno)
    rho = tabla_saucio["MMF"].rank().corr(tabla_natural["MMF"].rank())
    print(f"\nCorrelación de Spearman (patrón mensual Saucío vs. Q_natural): rho={rho:.4f} "
          f"(n=12 meses; calculada como Pearson sobre rangos)")
    n_coincide = int(comparacion["coincide"].sum())
    print(f"Meses con clasificación VMF coincidente: {n_coincide}/12")
    if n_coincide < 12:
        difieren = comparacion.index[~comparacion["coincide"]].tolist()
        print(f"Meses que difieren: {difieren}")
        print(comparacion.loc[difieren].round(3).to_string())
    print()


def pieza2(df_natural: pd.DataFrame, df_descargas: pd.DataFrame) -> pd.Series:
    print("=" * 78)
    print("PIEZA 2 — ANÁLISIS DE SENSIBILIDAD DEL UMBRAL (EFR x0.8 / x1.0 / x1.2)")
    print("=" * 78)

    tabla_natural = tabla_mmf_clasificacion(df_natural["q_natural_m3s"])
    efr_base = tabla_natural["EFR"]

    escenarios = {
        "historico (4.5 m3/s)": CAUDAL_TIBITOC_HISTORICO_M3S,
        "ampliado (8.0 m3/s)": CAUDAL_TIBITOC_AMPLIADO_M3S,
    }
    multiplicadores = [0.8, 1.0, 1.2]

    filas = []
    for nombre_esc, q_nominal in escenarios.items():
        q_elsol = construir_q_elsol(df_descargas, q_nominal)
        for mult in multiplicadores:
            umbral = efr_base * mult
            diag = diagnostico_regimen(q_elsol, umbral)
            filas.append({
                "escenario": nombre_esc,
                "multiplicador_EFR": mult,
                "pct_dias_violacion": diag["pct_dias_en_violacion"],
                "num_episodios": diag["num_episodios"],
                "racha_maxima": diag["racha_maxima_dias"],
                "deficit_medio_m3s": diag["deficit_medio_m3s"],
            })

    tabla = pd.DataFrame(filas).set_index(["escenario", "multiplicador_EFR"])
    print(tabla.to_string())
    print()
    return efr_base


def pieza3_verificacion(df_natural: pd.DataFrame) -> None:
    print("=" * 78)
    print("PIEZA 3 (verificación caveat 2) — ¿cambia la clasificación mensual si se")
    print("excluye jul-nov 2022 (evento de vertimiento de Neusa) del cálculo de MAF/MMF?")
    print("=" * 78)

    q_natural_completo = df_natural["q_natural_m3s"]
    tabla_completa = tabla_mmf_clasificacion(q_natural_completo)

    excluir = (q_natural_completo.index.year == 2022) & (q_natural_completo.index.month.isin([7, 8, 9, 10, 11]))
    q_natural_sin_evento = q_natural_completo[~excluir]
    dias_excluidos = int(excluir.sum())
    print(f"Días excluidos (jul-nov 2022): {dias_excluidos}")

    tabla_sin_evento = tabla_mmf_clasificacion(q_natural_sin_evento)

    comparacion = pd.DataFrame({
        "MMF_con_evento": tabla_completa["MMF"],
        "regimen_con_evento": tabla_completa["regimen"],
        "MMF_sin_evento": tabla_sin_evento["MMF"],
        "regimen_sin_evento": tabla_sin_evento["regimen"],
    })
    comparacion["cambia_regimen"] = comparacion["regimen_con_evento"] != comparacion["regimen_sin_evento"]
    comparacion["cambio_pct_MMF"] = 100 * (comparacion["MMF_sin_evento"] - comparacion["MMF_con_evento"]) / comparacion["MMF_con_evento"]

    print(f"\nMAF con evento: {tabla_completa.attrs['MAF']:.3f} m3/s")
    print(f"MAF sin evento (jul-nov 2022 excluido): {tabla_sin_evento.attrs['MAF']:.3f} m3/s")
    print(f"Diferencia MAF: {100*(tabla_sin_evento.attrs['MAF']-tabla_completa.attrs['MAF'])/tabla_completa.attrs['MAF']:.2f}%\n")
    print(comparacion.round(3).to_string())

    n_cambia = int(comparacion["cambia_regimen"].sum())
    print(f"\nMeses cuya clasificación (bajo/intermedio/alto) CAMBIA al excluir el evento: {n_cambia}/12")
    if n_cambia > 0:
        print("Meses afectados:", comparacion.index[comparacion["cambia_regimen"]].tolist())
    print()


def main() -> None:
    df_natural, df_descargas = construir_series_base()

    pieza1(df_natural)
    pieza2(df_natural, df_descargas)
    pieza3_verificacion(df_natural)


if __name__ == "__main__":
    main()

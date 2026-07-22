"""Cálculo limpio del VMF (Variable Monthly Flow, Gerten et al. 2013) para el punto
de control El Sol, y diagnóstico de régimen de transgresión histórica.

Reconstruye Q_ElSol(t) a partir de las descargas HISTÓRICAS observadas de los tres
embalses (Neusa, Sisga, Tominé) más el caudal natural de Saucío, aplicando la
topología confirmada y la cota física de extracción de Tibitóc:

    Q_bocatoma(t)   = Q_Saucío(t) + Q_desc_Sisga(t) + Q_desc_Tominé(t) + Q_desc_Neusa(t)
    Q_extraccion(t) = min(Q_Tibitoc_escenario, Q_bocatoma(t))
    Q_ElSol(t)      = Q_bocatoma(t) - Q_extraccion(t)

NOTA: este script NO reutiliza scratch_vmf/data.pkl (intento anterior, contaminado por
un término de afluencia lateral erróneo ya eliminado del código, ver NOTAS.md /
memoria). Se recalcula desde cero con la topología y las series actuales.

Método VMF (Pastor et al. 2014 / Gerten et al. 2013):
    MAF = caudal medio anual (media de todo el período)
    MMF_m = caudal medio de largo plazo del mes calendario m (media entre años)
    Si MMF_m <= 0.4*MAF  -> mes de caudal bajo   -> EFR_m = 0.60 * MMF_m
    Si 0.4*MAF < MMF_m <= 0.8*MAF -> intermedio  -> EFR_m = 0.45 * MMF_m
    Si MMF_m > 0.8*MAF   -> caudal alto          -> EFR_m = 0.30 * MMF_m
Un umbral por mes calendario (12 valores), aplicado a todos los años.
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


def construir_q_elsol(caudal_nominal_tibitoc_m3s: float) -> pd.DataFrame:
    """Reconstruye Q_bocatoma y Q_ElSol históricos para un escenario de extracción."""
    contexts, date_min, date_max = load_dashboard_context()
    saucio_df, diag_saucio = cargar_saucio()

    desc_neusa = contexts["Neusa"].frame["descarga_m3s"]
    desc_sisga = contexts["Sisga"].frame["descarga_m3s"]
    desc_tomine = contexts["Tomine"].frame["descarga_m3s"]
    caudal_saucio = saucio_df["caudal_m3s"]

    idx = desc_neusa.index
    assert idx.equals(desc_sisga.index) and idx.equals(desc_tomine.index), (
        "Los índices de los tres embalses deben coincidir (ventana única del proyecto)."
    )
    caudal_saucio = caudal_saucio.reindex(idx)

    q_bocatoma = caudal_saucio + desc_sisga + desc_tomine + desc_neusa
    q_extraccion = np.minimum(caudal_nominal_tibitoc_m3s, q_bocatoma.clip(lower=0.0))
    q_elsol = q_bocatoma - q_extraccion

    df = pd.DataFrame(
        {
            "caudal_saucio_m3s": caudal_saucio,
            "descarga_sisga_m3s": desc_sisga,
            "descarga_tomine_m3s": desc_tomine,
            "descarga_neusa_m3s": desc_neusa,
            "q_bocatoma_m3s": q_bocatoma,
            "q_extraccion_m3s": q_extraccion,
            "q_elsol_m3s": q_elsol,
        },
        index=idx,
    )
    return df


def calcular_umbral_vmf(q_elsol: pd.Series) -> pd.Series:
    """Calcula el umbral VMF mensual (12 valores, uno por mes calendario).

    Devuelve una Serie indexada 1..12 con el caudal ambiental EFR_m [m3/s].
    """
    serie = q_elsol.dropna()
    maf = serie.mean()

    mmf_por_mes = serie.groupby(serie.index.month).mean()

    def _clasificar(mmf_m: float) -> tuple[float, str]:
        if mmf_m <= 0.4 * maf:
            return 0.60, "bajo"
        if mmf_m <= 0.8 * maf:
            return 0.45, "intermedio"
        return 0.30, "alto"

    fracciones = {}
    regimen = {}
    for mes, mmf_m in mmf_por_mes.items():
        frac, reg = _clasificar(mmf_m)
        fracciones[mes] = frac
        regimen[mes] = reg

    efr = pd.Series({mes: fracciones[mes] * mmf_por_mes[mes] for mes in mmf_por_mes.index})
    efr.name = "vmf_m3s"

    resumen = pd.DataFrame(
        {
            "MMF_m3s": mmf_por_mes,
            "fraccion_preservada": pd.Series(fracciones),
            "regimen": pd.Series(regimen),
            "VMF_m3s": efr,
        }
    )
    resumen.index.name = "mes"
    resumen.attrs["MAF_m3s"] = maf
    return resumen


def diagnostico_regimen(q_elsol: pd.Series, umbral_mensual: pd.Series) -> dict:
    """Frecuencia, magnitud y duración de violaciones diarias de Q_ElSol vs VMF mensual."""
    serie = q_elsol.copy()
    umbral_diario = serie.index.month.map(umbral_mensual)
    umbral_diario = pd.Series(umbral_diario, index=serie.index, dtype=float)

    valido = serie.notna() & umbral_diario.notna()
    serie_v = serie[valido]
    umbral_v = umbral_diario[valido]

    deficit = (umbral_v - serie_v).clip(lower=0.0)
    en_violacion = deficit > 0.0

    # Duración de rachas consecutivas en violación
    grupo = (en_violacion != en_violacion.shift()).cumsum()
    rachas = en_violacion.groupby(grupo).sum()
    rachas = rachas[rachas > 0]

    return {
        "dias_totales": int(valido.sum()),
        "dias_en_violacion": int(en_violacion.sum()),
        "pct_dias_en_violacion": round(100.0 * en_violacion.sum() / valido.sum(), 2),
        "deficit_medio_m3s": round(float(deficit[en_violacion].mean()), 3) if en_violacion.any() else 0.0,
        "deficit_maximo_m3s": round(float(deficit.max()), 3),
        "deficit_acumulado_m3s_dia": round(float(deficit.sum()), 1),
        "num_episodios": int(len(rachas)),
        "racha_maxima_dias": int(rachas.max()) if not rachas.empty else 0,
        "racha_media_dias": round(float(rachas.mean()), 1) if not rachas.empty else 0.0,
    }


def main() -> None:
    escenarios = {
        "historico (4.5 m3/s)": CAUDAL_TIBITOC_HISTORICO_M3S,
        "ampliado (8.0 m3/s)": CAUDAL_TIBITOC_AMPLIADO_M3S,
    }

    for nombre_escenario, q_nominal in escenarios.items():
        print("=" * 70)
        print(f"ESCENARIO TIBITÓC: {nombre_escenario}")
        print("=" * 70)

        df = construir_q_elsol(q_nominal)
        q_elsol = df["q_elsol_m3s"]

        print(f"\nCobertura: {q_elsol.index.min().date()} -> {q_elsol.index.max().date()}")
        print(f"Días totales: {len(q_elsol)}  |  Días con dato válido: {int(q_elsol.notna().sum())}")
        print(f"\nQ_ElSol: media={q_elsol.mean():.2f} m3/s, mediana={q_elsol.median():.2f}, "
              f"min={q_elsol.min():.2f}, p10={q_elsol.quantile(0.10):.2f}")

        resumen_vmf = calcular_umbral_vmf(q_elsol)
        maf = resumen_vmf.attrs["MAF_m3s"]
        print(f"\nMAF (caudal medio anual) = {maf:.3f} m3/s")
        print("\nUmbral VMF por mes calendario:")
        print(resumen_vmf.round(3).to_string())

        diag = diagnostico_regimen(q_elsol, resumen_vmf["VMF_m3s"])
        print("\nDiagnóstico de régimen (Q_ElSol diario vs. VMF mensual):")
        for k, v in diag.items():
            print(f"  {k}: {v}")
        print()

    # Guardar resultado detallado del escenario histórico para inspección
    df_hist = construir_q_elsol(CAUDAL_TIBITOC_HISTORICO_M3S)
    out_path = Path(__file__).parent / "q_elsol_historico.csv"
    df_hist.to_csv(out_path)
    print(f"Serie detallada (escenario histórico) guardada en: {out_path}")


if __name__ == "__main__":
    main()

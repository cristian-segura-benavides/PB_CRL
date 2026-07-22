"""Cálculo CORREGIDO del VMF (Variable Monthly Flow, Pastor et al. 2014 / Gerten et al.
2013) para el punto de control El Sol.

CORRECCIÓN METODOLÓGICA respecto a calcular_vmf.py (v1, con error): el MAF y los MMF
deben derivarse del RÉGIMEN NATURAL de la cuenca, no del caudal ya regulado y disminuido
por la extracción de Tibitóc. En v1, MAF salía distinto por escenario (6.34 vs 3.02
m3/s) porque se calculaba sobre Q_ElSol (ya alterado) — eso invierte la lógica del
método: mientras más se extrae, más bajo queda el umbral.

CÁLCULO CORREGIDO:
    Paso 1: Q_natural(t) = Q_Saucío(t) + afluencia_Neusa(t) + afluencia_Sisga(t)
                           + afluencia_Tominé(t)
            Usa AFLUENCIAS (balance inverso, lo que la cuenca produce), no descargas
            (que ya están reguladas por la operación de los embalses).
    Paso 2: MAF = media de Q_natural sobre toda la ventana.
    Paso 3: MMF = media de Q_natural por mes calendario (12 valores).
    Paso 4: Clasificar cada mes y asignar EFR (60/45/30% de MMF) -> UN SOLO conjunto
            de 12 umbrales, derivado del régimen natural.
    Paso 5: Comparar Q_ElSol de CADA escenario de extracción contra ESE MISMO umbral.

Q_ElSol(t) sigue construyéndose igual que en v1 (a partir de las DESCARGAS observadas
más la extracción de Tibitóc) — lo que cambia es contra qué umbral se evalúa, no cómo
se calcula el caudal regulado.
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


def construir_series_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye Q_natural (régimen natural) y las descargas observadas por separado.

    Devuelve (df_natural, df_descargas), ambos indexados por fecha sobre la ventana
    completa del proyecto.
    """
    contexts, _, _ = load_dashboard_context()
    saucio_df, _ = cargar_saucio()

    afl_neusa = contexts["Neusa"].frame["afluencia_m3s"]
    afl_sisga = contexts["Sisga"].frame["afluencia_m3s"]
    afl_tomine = contexts["Tomine"].frame["afluencia_m3s"]
    caudal_saucio = saucio_df["caudal_m3s"]

    idx = afl_neusa.index
    caudal_saucio = caudal_saucio.reindex(idx)

    q_natural = caudal_saucio + afl_neusa + afl_sisga + afl_tomine

    df_natural = pd.DataFrame(
        {
            "caudal_saucio_m3s": caudal_saucio,
            "afluencia_neusa_m3s": afl_neusa,
            "afluencia_sisga_m3s": afl_sisga,
            "afluencia_tomine_m3s": afl_tomine,
            "q_natural_m3s": q_natural,
        },
        index=idx,
    )

    desc_neusa = contexts["Neusa"].frame["descarga_m3s"]
    desc_sisga = contexts["Sisga"].frame["descarga_m3s"]
    desc_tomine = contexts["Tomine"].frame["descarga_m3s"]
    q_bocatoma = caudal_saucio + desc_sisga + desc_tomine + desc_neusa

    df_descargas = pd.DataFrame(
        {
            "caudal_saucio_m3s": caudal_saucio,
            "descarga_sisga_m3s": desc_sisga,
            "descarga_tomine_m3s": desc_tomine,
            "descarga_neusa_m3s": desc_neusa,
            "q_bocatoma_m3s": q_bocatoma,
        },
        index=idx,
    )

    return df_natural, df_descargas


def reportar_cobertura(df_natural: pd.DataFrame) -> None:
    n = len(df_natural)
    print("Cobertura de términos de Q_natural (días sin dato válido, sobre "
          f"{n} días de ventana):")
    for col, etiqueta in [
        ("caudal_saucio_m3s", "Saucío"),
        ("afluencia_neusa_m3s", "Afluencia Neusa"),
        ("afluencia_sisga_m3s", "Afluencia Sisga"),
        ("afluencia_tomine_m3s", "Afluencia Tominé"),
    ]:
        faltantes = int(df_natural[col].isna().sum())
        print(f"  {etiqueta}: {faltantes} días ({100*faltantes/n:.2f}%)")
    faltantes_q = int(df_natural["q_natural_m3s"].isna().sum())
    print(f"  Q_natural (unión, cualquier término faltante invalida el día): "
          f"{faltantes_q} días ({100*faltantes_q/n:.2f}%)")
    print()


def calcular_umbral_vmf_natural(q_natural: pd.Series) -> pd.DataFrame:
    """Umbral VMF (12 valores mensuales) derivado del caudal NATURAL."""
    serie = q_natural.dropna()
    maf = float(serie.mean())
    mmf_por_mes = serie.groupby(serie.index.month).mean()

    def _clasificar(mmf_m: float) -> tuple[float, str]:
        if mmf_m <= 0.4 * maf:
            return 0.60, "bajo"
        if mmf_m <= 0.8 * maf:
            return 0.45, "intermedio"
        return 0.30, "alto"

    fracciones, regimen = {}, {}
    for mes, mmf_m in mmf_por_mes.items():
        frac, reg = _clasificar(mmf_m)
        fracciones[mes] = frac
        regimen[mes] = reg

    efr = pd.Series({mes: fracciones[mes] * mmf_por_mes[mes] for mes in mmf_por_mes.index})

    resumen = pd.DataFrame(
        {
            "MMF_m3s": mmf_por_mes,
            "fraccion_preservada": pd.Series(fracciones),
            "regimen": pd.Series(regimen),
            "EFR_m3s": efr,
        }
    )
    resumen.index.name = "mes"
    resumen.attrs["MAF_m3s"] = maf
    resumen.attrs["dias_validos"] = int(serie.notna().sum())
    return resumen


def construir_q_elsol(df_descargas: pd.DataFrame, caudal_nominal_tibitoc_m3s: float) -> pd.Series:
    q_bocatoma = df_descargas["q_bocatoma_m3s"]
    q_extraccion = np.minimum(caudal_nominal_tibitoc_m3s, q_bocatoma.clip(lower=0.0))
    return q_bocatoma - q_extraccion


def diagnostico_regimen(q_elsol: pd.Series, umbral_mensual: pd.Series) -> dict:
    serie = q_elsol.copy()
    umbral_diario = pd.Series(serie.index.month.map(umbral_mensual), index=serie.index, dtype=float)

    valido = serie.notna() & umbral_diario.notna()
    serie_v = serie[valido]
    umbral_v = umbral_diario[valido]

    deficit = (umbral_v - serie_v).clip(lower=0.0)
    en_violacion = deficit > 0.0

    grupo = (en_violacion != en_violacion.shift()).cumsum()
    rachas = en_violacion.groupby(grupo).sum()
    rachas = rachas[rachas > 0]

    cero = int((serie_v <= 0.001).sum())

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
        "dias_cero": cero,
        "pct_dias_cero": round(100.0 * cero / valido.sum(), 2),
        "_deficit": deficit,
        "_en_violacion": en_violacion,
    }


def distribucion_temporal(en_violacion: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Conteo de días en violación por mes calendario y por año."""
    idx = en_violacion[en_violacion].index
    por_mes = idx.month.value_counts().sort_index()
    por_anio = idx.year.value_counts().sort_index()
    return por_mes, por_anio


def main() -> None:
    df_natural, df_descargas = construir_series_base()
    reportar_cobertura(df_natural)

    resumen_vmf = calcular_umbral_vmf_natural(df_natural["q_natural_m3s"])
    maf = resumen_vmf.attrs["MAF_m3s"]
    dias_validos = resumen_vmf.attrs["dias_validos"]

    print("=" * 78)
    print("UMBRAL VMF ÚNICO, DERIVADO DEL RÉGIMEN NATURAL (Q_Saucío + afluencias)")
    print("=" * 78)
    print(f"Días válidos usados para MAF/MMF: {dias_validos} / {len(df_natural)}")
    print(f"MAF (caudal medio anual natural) = {maf:.3f} m3/s\n")
    print(resumen_vmf.round(3).to_string())
    print()

    umbral_mensual = resumen_vmf["EFR_m3s"]

    escenarios = {
        "historico (4.5 m3/s)": CAUDAL_TIBITOC_HISTORICO_M3S,
        "ampliado (8.0 m3/s)": CAUDAL_TIBITOC_AMPLIADO_M3S,
    }

    resultados = {}
    for nombre, q_nominal in escenarios.items():
        q_elsol = construir_q_elsol(df_descargas, q_nominal)
        diag = diagnostico_regimen(q_elsol, umbral_mensual)
        resultados[nombre] = (q_elsol, diag)

        print("=" * 78)
        print(f"ESCENARIO: {nombre}  (evaluado contra el umbral único de arriba)")
        print("=" * 78)
        print(f"Q_ElSol: media={q_elsol.mean():.2f} m3/s, mediana={q_elsol.median():.2f}, "
              f"min={q_elsol.min():.2f}")
        for k, v in diag.items():
            if k.startswith("_"):
                continue
            print(f"  {k}: {v}")

        por_mes, por_anio = distribucion_temporal(diag["_en_violacion"])
        print("\n  Días en violación por mes calendario:")
        print("  " + por_mes.reindex(range(1, 13), fill_value=0).to_string().replace("\n", "\n  "))
        print("\n  Días en violación por año:")
        print("  " + por_anio.to_string().replace("\n", "\n  "))
        print()

    print("=" * 78)
    print("COMPARACIÓN LADO A LADO (umbral común)")
    print("=" * 78)
    filas = []
    for nombre, (q_elsol, diag) in resultados.items():
        filas.append({
            "escenario": nombre,
            "pct_dias_violacion": diag["pct_dias_en_violacion"],
            "num_episodios": diag["num_episodios"],
            "racha_maxima": diag["racha_maxima_dias"],
            "deficit_medio_m3s": diag["deficit_medio_m3s"],
            "pct_dias_cero": diag["pct_dias_cero"],
        })
    print(pd.DataFrame(filas).set_index("escenario").to_string())
    print()

    print("VERIFICACIÓN DE CONSISTENCIA: el MAF es UNA sola cifra "
          f"({maf:.3f} m3/s), no depende del escenario de extracción — correcto por "
          "construcción, ya que se deriva únicamente de Q_natural.")

    # Guardar series detalladas
    df_natural.to_csv(Path(__file__).parent / "q_natural.csv")
    for nombre, (q_elsol, _diag) in resultados.items():
        slug = nombre.split(" ")[0]
        q_elsol.to_csv(Path(__file__).parent / f"q_elsol_{slug}_v2.csv")
    resumen_vmf.to_csv(Path(__file__).parent / "umbral_vmf_natural.csv")
    print("\nSeries y umbral guardados en scratch_vmf/*.csv")


if __name__ == "__main__":
    main()

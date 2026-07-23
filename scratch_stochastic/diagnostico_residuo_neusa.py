"""Diagnóstico: ¿el residuo de Neusa (lo que el VARX climático NO explica)
tiene un patrón estacional propio, consistente año a año (evidencia de
demanda de acueducto), o es más bien errático/puntual (ruido, eventos como
el vertimiento de 2022)?

Ajusta un modelo FRESCO sobre la ventana COMPLETA 2012-2025 (sin split
train/validación) — a propósito: el objetivo aquí es diagnosticar la
ESTRUCTURA DEL RESIDUO a lo largo de los 13 años, no evaluar generalización;
usar el modelo ya guardado (entrenado solo hasta 2023-05-04) mezclaría el
efecto de "no generaliza bien" (ya documentado en la validación de 5 bloques)
con la pregunta real de si hay un patrón de demanda recurrente.

Reconstruye el residuo accediendo a los parámetros ya ajustados del modelo
(_media_g, _intercepto_varx, _phi, _b_exog) — NO se modifica modelo.py; es
solo una re-derivación externa de lo mismo que fit() ya calcula internamente
(residuos = Y - Z@coef), aplicada por separado a cada serie en vez de solo a
las filas donde las 3 series estaban simultáneamente en estado "alto".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from pbcrl.stochastic.entrenamiento import construir_datos  # noqa: E402
from pbcrl.stochastic.modelo import ConfigModeloEstocastico, ModeloEstocasticoAfluencias  # noqa: E402

SERIES = ("Saucio", "Neusa", "Sisga", "Tomine")


def calcular_residuos(modelo: ModeloEstocasticoAfluencias, covariables: pd.DataFrame, objetivo: pd.DataFrame) -> pd.DataFrame:
    """Residuo por serie y día: anomalía observada - anomalía predicha por el VARX.

    Usa la MISMA política de "reset a cero" para rezagos en estado bajo que
    sample()/fit() ya usan (ver modelo.py) — no una convención nueva.
    Devuelve NaN en los días donde esa serie está en estado bajo (no aplica
    "residuo" a un día que el hurdle, no el VARX, decide).
    """
    cov_cols = list(modelo.config.covariables)
    obj_cols = list(modelo.config.series_objetivo)
    datos = covariables[cov_cols].join(objetivo[obj_cols], how="inner").dropna()

    meses = datos.index.month.to_numpy()
    cov_arr = datos[cov_cols].to_numpy(dtype=float)
    umbral = modelo.config.umbral_bajo_m3s

    k = len(obj_cols)
    n = len(datos)
    anomalia = np.full((n, k), np.nan)
    for j, serie in enumerate(obj_cols):
        valores = datos[serie].to_numpy(dtype=float)
        bajo = valores <= umbral
        g = np.full(n, np.nan)
        g[~bajo] = np.log1p(valores[~bajo] - umbral)
        media_por_dia = modelo._media_g[serie][meses - 1]
        anomalia[~bajo, j] = g[~bajo] - media_por_dia[~bajo]

    residuos = np.full((n, k), np.nan)
    for t in range(1, n):
        rezago = anomalia[t - 1]
        rezago_efectivo = np.nan_to_num(rezago, nan=0.0)
        prediccion = modelo._intercepto_varx + modelo._phi @ rezago_efectivo + modelo._b_exog @ cov_arr[t]
        residuos[t] = anomalia[t] - prediccion  # NaN se propaga donde anomalia[t] es NaN

    return pd.DataFrame(residuos, index=datos.index, columns=[f"residuo_{s}" for s in obj_cols])


def analizar_estacionalidad(residuo: pd.Series, nombre: str) -> None:
    residuo = residuo.dropna()
    print(f"\n--- {nombre}: patrón estacional del residuo ---")
    print(f"Días válidos: {len(residuo)} | media global: {residuo.mean():.4f} | std global: {residuo.std():.4f}")

    tabla = pd.DataFrame({"residuo": residuo, "mes": residuo.index.month, "anio": residuo.index.year})

    # Patrón promedio por mes (pooled, los 13 años juntos)
    promedio_mes = tabla.groupby("mes")["residuo"].agg(["mean", "std", "count"])
    print("\nResiduo medio por mes (pooled, todos los años):")
    print(promedio_mes.round(4).to_string())

    # Consistencia año a año: media del residuo por (año, mes), luego dispersión
    # ENTRE años para cada mes.
    media_anio_mes = tabla.groupby(["anio", "mes"])["residuo"].mean().unstack("mes")
    media_entre_anios = media_anio_mes.mean(axis=0)
    std_entre_anios = media_anio_mes.std(axis=0)
    señal_ruido = (media_entre_anios.abs() / std_entre_anios).replace([np.inf], np.nan)

    resumen = pd.DataFrame({
        "media_entre_anios": media_entre_anios,
        "std_entre_anios": std_entre_anios,
        "señal_ruido_abs": señal_ruido,
    })
    print("\nConsistencia AÑO A AÑO por mes (¿la media de cada año, para ese mes, es "
          "parecida entre años, o dispersa?):")
    print(resumen.round(3).to_string())
    print(f"\nMeses con |señal/ruido| > 1 (media entre años supera su propia dispersión "
          f"entre años -> patrón consistente): "
          f"{resumen.index[resumen['señal_ruido_abs'] > 1].tolist()}")
    print(f"Mediana de |señal/ruido| sobre los 12 meses: {resumen['señal_ruido_abs'].median():.3f}")


def analizar_concentracion_eventos(residuo: pd.Series, nombre: str) -> None:
    residuo = residuo.dropna()
    print(f"\n--- {nombre}: ¿concentrado en pocos días (evento) o difuso (estacional)? ---")
    abs_res = residuo.abs().sort_values(ascending=False)
    total = (residuo ** 2).sum()
    for frac in (0.001, 0.01, 0.05):
        n_dias = max(1, int(len(residuo) * frac))
        top = abs_res.iloc[:n_dias]
        top_var = (residuo.loc[top.index] ** 2).sum()
        print(f"  Top {100*frac:.1f}% de días ({n_dias} días) concentran "
              f"{100*top_var/total:.1f}% de la suma de residuo^2")

    # Evento de Neusa 2022 ya identificado (jul-nov 2022) — ¿domina el residuo?
    evento_2022 = residuo[(residuo.index.year == 2022) & (residuo.index.month.isin([7, 8, 9, 10, 11]))]
    if not evento_2022.empty:
        var_evento = (evento_2022 ** 2).sum()
        print(f"  Días del evento de vertimiento 2022 (jul-nov) presentes: {len(evento_2022)}, "
              f"concentran {100*var_evento/total:.1f}% de la suma de residuo^2")


def main() -> None:
    covariables, objetivo = construir_datos()
    print(f"Ajustando modelo fresco sobre TODA la ventana ({len(covariables)} días, "
          f"{covariables.index.min().date()} -> {covariables.index.max().date()})...")
    modelo = ModeloEstocasticoAfluencias(ConfigModeloEstocastico()).fit(covariables, objetivo)

    residuos = calcular_residuos(modelo, covariables, objetivo)

    print("\n" + "=" * 78)
    print("(4) VARIANZA DEL RESIDUO: Neusa vs. Sisga vs. Tominé")
    print("=" * 78)
    varianzas = residuos.var().sort_values(ascending=False)
    print(varianzas.round(4).to_string())
    print(f"\nRazón varianza Neusa / Sisga: {varianzas['residuo_Neusa']/varianzas['residuo_Sisga']:.2f}")
    print(f"Razón varianza Neusa / Tomine: {varianzas['residuo_Neusa']/varianzas['residuo_Tomine']:.2f}")

    print("\n" + "=" * 78)
    print("(1)-(2)-(3) ANÁLISIS DE ESTACIONALIDAD Y CONCENTRACIÓN — Neusa "
          "(con Sisga y Tominé como referencia)")
    print("=" * 78)
    for serie in ["Neusa", "Sisga", "Tomine"]:
        analizar_estacionalidad(residuos[f"residuo_{serie}"], serie)
        analizar_concentracion_eventos(residuos[f"residuo_{serie}"], serie)

    print("\n" + "=" * 78)
    print("VERIFICACIÓN: ¿el patrón estacional de Neusa sobrevive SIN el evento 2022?")
    print("=" * 78)
    residuo_neusa = residuos["residuo_Neusa"].dropna()
    excluir_2022 = (residuo_neusa.index.year == 2022) & (residuo_neusa.index.month.isin([7, 8, 9, 10, 11]))
    residuo_neusa_sin_evento = residuo_neusa[~excluir_2022]
    analizar_estacionalidad(residuo_neusa_sin_evento, "Neusa (SIN evento 2022 jul-nov)")


if __name__ == "__main__":
    main()

"""Validación cruzada temporal por bloques del modelo estocástico de afluencias.

Pregunta que responde: ¿el modelo (VARX desestacionalizado + hurdle,
`pbcrl.stochastic.modelo`) generaliza a regímenes ENSO distintos de los que
entrenó, o solo replica el régimen dominante del entrenamiento? Se diagnosticó
(sesión anterior) que el sesgo de sobreestimación en meses húmedos aparece
mucho más fuerte fuera de muestra (validación 2023-2025, dominada por El Niño
2023-24) que dentro de muestra — evidencia de sobreajuste/régimen atípico, no
de un sesgo de retransformación logarítmica. Este script lo pone a prueba de
forma sistemática con 5 bloques en vez de un solo período de validación.

DESVIACIÓN DEL DISEÑO SOLICITADO (documentada explícitamente):
El diseño original pedía, para cada bloque, entrenar con "todo lo demás"
(los datos antes Y después del bloque, combinados). NO se implementó así:
`ModeloEstocasticoAfluencias.fit()` construye los rezagos autorregresivos del
VARX por POSICIÓN de fila, no por fecha — si se concatenan el tramo anterior
y el tramo posterior al bloque excluido, la fila del último día antes del
hueco y la del primer día después del hueco quedarían adyacentes en el array,
y el modelo las trataría como si fueran días consecutivos reales (un rezago
de 1 día espurio, que contamina Phi). Corregir eso requiere tocar
`modelo.py`, que la instrucción explícita de esta tarea es NO tocar todavía.

En su lugar, cada bloque se entrena con una VENTANA EXPANSIVA de un solo
tramo contiguo:
  - Bloques 2, 3, 4, 5 (tienen historia previa): entrenan con TODO lo anterior
    al bloque (desde el inicio de la ventana), con un margen de 30 días antes
    del bloque.
  - Bloque 1 (no tiene historia previa): entrena con TODO lo posterior al
    bloque (único caso "hacia atrás"), con el mismo margen de 30 días.
Esto no es solo un rodeo al problema de contigüidad: para la pregunta de
"¿generaliza a un régimen no visto?", una ventana expansiva (entrenar solo con
el pasado, validar en un futuro nunca visto) es un test más estricto y más
limpio que mezclar pasado y futuro alrededor del bloque — el régimen de
entrenamiento queda inequívocamente definido, sin mezclar climatologías de
ambos lados.

MÉTRICAS POR BLOQUE:
  - Régimen ENSO del bloque de validación Y de su conjunto de entrenamiento:
    RONI medio, distribución de fases (Niño/Niña/Neutral).
  - Sesgo de medias mensuales (m³/s, promedio de |sesgo| sobre 12 meses × 4
    series), simulado (media de 30 semillas) vs. histórico del bloque.
  - Error de correlación cruzada (diferencia media absoluta entre la matriz
    de correlación simulada y la histórica del bloque, sobre los 6 pares
    únicos de las 4 series).
  - "Distancia climática" = |RONI medio del bloque - RONI medio del
    entrenamiento de ese pliegue| — para responder la pregunta clave: ¿el
    error de validación crece con la distancia climática entre el bloque y
    su entrenamiento?
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

N_BLOQUES = 5
MARGEN_DIAS = 30
N_SEMILLAS = 30
COV_COLS = list(ConfigModeloEstocastico().covariables)
OBJ_COLS = list(ConfigModeloEstocastico().series_objetivo)


def definir_bloques(indice: pd.DatetimeIndex, n_bloques: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Divide la ventana completa en n_bloques contiguos de tamaño ~igual (por día calendario)."""
    inicio, fin = indice.min(), indice.max()
    fechas_corte = pd.date_range(inicio, fin, periods=n_bloques + 1)
    bloques = []
    for i in range(n_bloques):
        b_inicio = fechas_corte[i] if i == 0 else fechas_corte[i] + pd.Timedelta(days=1)
        b_fin = fechas_corte[i + 1]
        bloques.append((b_inicio, b_fin))
    return bloques


def _regimen_roni(roni: pd.Series) -> dict:
    return {
        "roni_medio": round(float(roni.mean()), 3),
        "roni_std": round(float(roni.std()), 3),
    }


def _fase_desde_roni(roni: pd.Series, umbral: float = 0.5) -> pd.Series:
    return pd.cut(roni, bins=[-np.inf, -umbral, umbral, np.inf], labels=["Nina", "Neutral", "Nino"])


def construir_pliegue(
    covariables: pd.DataFrame,
    objetivo: pd.DataFrame,
    bloques: list[tuple[pd.Timestamp, pd.Timestamp]],
    i: int,
) -> dict:
    """Construye train/val para el bloque i, con ventana expansiva de un solo tramo."""
    b_inicio, b_fin = bloques[i]
    margen = pd.Timedelta(days=MARGEN_DIAS)

    val_cov = covariables.loc[b_inicio:b_fin]
    val_obj = objetivo.loc[b_inicio:b_fin]

    if i == 0:
        # Único caso sin historia previa: entrena con todo lo POSTERIOR al bloque.
        train_inicio = b_fin + margen
        train_cov = covariables.loc[train_inicio:]
        train_obj = objetivo.loc[train_inicio:]
        direccion = "posterior (único caso sin historia previa)"
    else:
        # Ventana expansiva: todo lo ANTERIOR al bloque, con margen.
        train_fin = b_inicio - margen
        train_cov = covariables.loc[:train_fin]
        train_obj = objetivo.loc[:train_fin]
        direccion = "anterior (ventana expansiva)"

    return {
        "indice_bloque": i + 1,
        "bloque_inicio": b_inicio,
        "bloque_fin": b_fin,
        "direccion_entrenamiento": direccion,
        "train_cov": train_cov,
        "train_obj": train_obj,
        "val_cov": val_cov,
        "val_obj": val_obj,
    }


def evaluar_pliegue(pliegue: dict) -> dict:
    datos_train = pliegue["train_cov"][COV_COLS].join(pliegue["train_obj"][OBJ_COLS], how="inner").dropna()
    datos_val = pliegue["val_cov"][COV_COLS].join(pliegue["val_obj"][OBJ_COLS], how="inner").dropna()

    cov_train, obj_train = datos_train[COV_COLS], datos_train[OBJ_COLS]
    cov_val, obj_val = datos_val[COV_COLS], datos_val[OBJ_COLS]

    modelo = ModeloEstocasticoAfluencias(ConfigModeloEstocastico()).fit(cov_train, obj_train)

    # --- Medias mensuales: simulado (media de N semillas) vs. historico del bloque ---
    medias_hist = obj_val.groupby(obj_val.index.month).mean()
    acumulado = None
    corrs_sim = []
    for s in range(N_SEMILLAS):
        muestra = modelo.sample(cov_val, semilla=2000 + s)
        medias = muestra.groupby(muestra.index.month).mean()
        acumulado = medias if acumulado is None else acumulado + medias
        corrs_sim.append(muestra.corr().to_numpy())
    medias_sim = acumulado / N_SEMILLAS

    sesgo_abs = (medias_sim - medias_hist).abs()
    sesgo_medio_abs_m3s = float(sesgo_abs.to_numpy().mean())

    corr_hist = obj_val.corr().to_numpy()
    corr_sim_media = np.mean(corrs_sim, axis=0)
    mascara_off_diag = ~np.eye(4, dtype=bool)
    error_corr = float(np.abs(corr_hist[mascara_off_diag] - corr_sim_media[mascara_off_diag]).mean())

    regimen_val = _regimen_roni(cov_val["roni"])
    regimen_train = _regimen_roni(cov_train["roni"])
    distancia_climatica = abs(regimen_val["roni_medio"] - regimen_train["roni_medio"])

    fases_val = _fase_desde_roni(cov_val["roni"]).value_counts(normalize=True).mul(100).round(1)
    fases_train = _fase_desde_roni(cov_train["roni"]).value_counts(normalize=True).mul(100).round(1)

    return {
        "bloque": pliegue["indice_bloque"],
        "periodo_bloque": f"{pliegue['bloque_inicio'].date()} -> {pliegue['bloque_fin'].date()}",
        "direccion_entrenamiento": pliegue["direccion_entrenamiento"],
        "dias_train": len(datos_train),
        "dias_val": len(datos_val),
        "roni_medio_val": regimen_val["roni_medio"],
        "roni_medio_train": regimen_train["roni_medio"],
        "distancia_climatica": round(distancia_climatica, 3),
        "fases_val_pct": fases_val.to_dict(),
        "fases_train_pct": fases_train.to_dict(),
        "sesgo_medio_abs_m3s": round(sesgo_medio_abs_m3s, 4),
        "error_correlacion": round(error_corr, 4),
        "medias_hist": medias_hist,
        "medias_sim": medias_sim,
    }


def main() -> None:
    covariables, objetivo = construir_datos()
    bloques = definir_bloques(covariables.index, N_BLOQUES)

    print("=" * 90)
    print(f"BLOQUES DEFINIDOS ({N_BLOQUES}, sobre {covariables.index.min().date()} -> "
          f"{covariables.index.max().date()})")
    print("=" * 90)
    for i, (b_inicio, b_fin) in enumerate(bloques):
        print(f"  Bloque {i+1}: {b_inicio.date()} -> {b_fin.date()}")
    print()

    resultados = []
    for i in range(N_BLOQUES):
        pliegue = construir_pliegue(covariables, objetivo, bloques, i)
        print(f"Evaluando bloque {i+1}/{N_BLOQUES} "
              f"(entrenamiento {pliegue['direccion_entrenamiento']}, "
              f"{len(pliegue['train_cov'])} días de train candidatos)...")
        resultado = evaluar_pliegue(pliegue)
        resultados.append(resultado)

    print("\n" + "=" * 90)
    print("RESUMEN POR BLOQUE")
    print("=" * 90)
    resumen = pd.DataFrame([{
        "bloque": r["bloque"],
        "periodo": r["periodo_bloque"],
        "dir_train": r["direccion_entrenamiento"].split(" ")[0],
        "dias_train": r["dias_train"],
        "dias_val": r["dias_val"],
        "RONI_val": r["roni_medio_val"],
        "RONI_train": r["roni_medio_train"],
        "distancia_climatica": r["distancia_climatica"],
        "sesgo_medio_abs_m3s": r["sesgo_medio_abs_m3s"],
        "error_correlacion": r["error_correlacion"],
    } for r in resultados])
    print(resumen.to_string(index=False))

    print("\n" + "=" * 90)
    print("DISTRIBUCIÓN DE FASES ENSO POR BLOQUE (validación vs. entrenamiento)")
    print("=" * 90)
    for r in resultados:
        print(f"\nBloque {r['bloque']} ({r['periodo_bloque']}):")
        print(f"  Validación:    {r['fases_val_pct']}")
        print(f"  Entrenamiento: {r['fases_train_pct']}")

    print("\n" + "=" * 90)
    print("PREGUNTA CLAVE: ¿el error crece con la distancia climática?")
    print("=" * 90)
    tabla_pregunta = resumen[["bloque", "distancia_climatica", "sesgo_medio_abs_m3s", "error_correlacion"]]
    tabla_pregunta = tabla_pregunta.sort_values("distancia_climatica")
    print(tabla_pregunta.to_string(index=False))
    corr_dist_sesgo = resumen["distancia_climatica"].corr(resumen["sesgo_medio_abs_m3s"])
    corr_dist_corr = resumen["distancia_climatica"].corr(resumen["error_correlacion"])
    print(f"\nCorrelación (n={N_BLOQUES}, muestra pequeña — leer como tendencia, no como "
          f"significancia estadística):")
    print(f"  distancia_climatica vs. sesgo_medio_abs_m3s: {corr_dist_sesgo:.3f}")
    print(f"  distancia_climatica vs. error_correlacion:   {corr_dist_corr:.3f}")

    print("\n" + "=" * 90)
    print("DETALLE: medias mensuales histórico vs. simulado, por bloque y serie")
    print("=" * 90)
    for r in resultados:
        print(f"\n--- Bloque {r['bloque']} ({r['periodo_bloque']}) ---")
        for serie in OBJ_COLS:
            comp = pd.DataFrame({
                "historico": r["medias_hist"][serie],
                "simulado": r["medias_sim"][serie],
            })
            comp["sesgo_abs"] = (comp["simulado"] - comp["historico"]).round(2)
            print(f"  {serie}:")
            print("  " + comp.round(2).to_string().replace("\n", "\n  "))


if __name__ == "__main__":
    main()

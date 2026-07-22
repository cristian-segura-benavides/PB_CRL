"""Ajusta el modelo estocástico de afluencias con los datos históricos ya
integrados por los loaders existentes (NO duplica ninguna lectura de datos).

Uso: ``python -m pbcrl.stochastic.entrenamiento`` desde la raíz del proyecto
(con el entorno virtual activo). Imprime un reporte de validación temporal y
guarda el modelo VARX ajustado en `src/pbcrl/stochastic/artefactos/` (carpeta
ignorada por git — ver .gitignore: un modelo entrenado con datos reales es un
artefacto derivado de datos, no código).

Covariables usadas: precipitación (promedio de los pluviómetros de Neusa,
Sisga y Tominé — no hay un pluviómetro propio de la cuenca de Saucío; se usa
este promedio como proxy de precipitación de cuenca, documentado aquí como
decisión de diseño) y RONI. Temperatura EXCLUIDA por decisión del usuario
(2026-07-22): no existe fuente integrada en el proyecto, y su efecto es de
segundo orden (evapotranspiración) frente a precipitación y ENSO. Ver
`modelo.ConfigModeloEstocastico` para cómo agregarla más adelante sin tocar
la lógica del modelo.

Series objetivo: caudal de Saucío (observado) y afluencias de Neusa/Sisga/
Tominé (estimadas por balance inverso, ya calculadas por los loaders).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from pbcrl.data_ingest.roni import cargar_roni  # noqa: E402
from pbcrl.data_ingest.saucio import cargar_saucio  # noqa: E402
from dashboard.data_loader import load_dashboard_context  # noqa: E402

from pbcrl.stochastic.analogos import ConfigAnalogos, RemuestreoAnalogos  # noqa: E402
from pbcrl.stochastic.modelo import ConfigModeloEstocastico, ModeloEstocasticoAfluencias  # noqa: E402

ARTEFACTOS_DIR = Path(__file__).parent / "artefactos"
AÑOS_VALIDACION = 2
N_SEMILLAS_VALIDACION = 30


def construir_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye (covariables, series_objetivo) reutilizando los loaders existentes."""
    contexts, _, _ = load_dashboard_context()
    saucio_df, _ = cargar_saucio()
    roni_df, _ = cargar_roni()

    precip_neusa = contexts["Neusa"].frame["precipitacion_mm"]
    precip_sisga = contexts["Sisga"].frame["precipitacion_mm"]
    precip_tomine = contexts["Tomine"].frame["precipitacion_mm"]
    # Proxy de precipitación de cuenca: promedio de los tres pluviómetros
    # (no hay pluviómetro propio de la cuenca de Saucío). Documentado en el
    # docstring del módulo.
    precip_cuenca = pd.concat([precip_neusa, precip_sisga, precip_tomine], axis=1).mean(axis=1)

    covariables = pd.DataFrame({
        "precipitacion_mm": precip_cuenca,
        "roni": roni_df["roni"],
    })

    series_objetivo = pd.DataFrame({
        "Saucio": saucio_df["caudal_m3s"],
        "Neusa": contexts["Neusa"].frame["afluencia_m3s"],
        "Sisga": contexts["Sisga"].frame["afluencia_m3s"],
        "Tomine": contexts["Tomine"].frame["afluencia_m3s"],
    })

    return covariables, series_objetivo


def dividir_temporal(
    covariables: pd.DataFrame, series_objetivo: pd.DataFrame, años_validacion: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split TEMPORAL (no aleatorio): los últimos `años_validacion` años como validación."""
    fin = covariables.index.max()
    corte = fin - pd.DateOffset(years=años_validacion)

    cov_train = covariables.loc[:corte]
    cov_val = covariables.loc[corte:].iloc[1:]  # evita duplicar el día de corte
    obj_train = series_objetivo.loc[:corte]
    obj_val = series_objetivo.loc[corte:].iloc[1:]

    print(f"Corte de validación: {corte.date()} (últimos {años_validacion} años)")
    print(f"Entrenamiento: {cov_train.index.min().date()} -> {cov_train.index.max().date()} "
          f"({len(cov_train)} días)")
    print(f"Validación:    {cov_val.index.min().date()} -> {cov_val.index.max().date()} "
          f"({len(cov_val)} días)\n")

    return cov_train, cov_val, obj_train, obj_val


def _correlacion(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr()


def _pct_bajo(df: pd.DataFrame, umbral: float = 0.0) -> pd.Series:
    return (df <= umbral).mean() * 100.0


def reporte_validacion(
    obj_val_historico: pd.DataFrame,
    modelo_varx: ModeloEstocasticoAfluencias,
    modelo_analogos: RemuestreoAnalogos,
    cov_val: pd.DataFrame,
) -> None:
    """Compara histórico vs. VARX simulado (promedio de N semillas) vs. análogos."""
    print("=" * 78)
    print("VALIDACIÓN — correlación cruzada de las 4 series objetivo")
    print("=" * 78)
    print("\nHistórico (validación real):")
    print(_correlacion(obj_val_historico).round(3).to_string())

    # Promedio de correlaciones sobre N semillas para el VARX (una sola muestra
    # es ruidosa; el promedio da una estimación más estable de lo que el
    # modelo reproduce EN EXPECTATIVA).
    corrs_varx = []
    for semilla in range(N_SEMILLAS_VALIDACION):
        muestra = modelo_varx.sample(cov_val, semilla=semilla)
        corrs_varx.append(_correlacion(muestra).to_numpy())
    corr_varx_media = pd.DataFrame(
        np.mean(corrs_varx, axis=0),
        index=obj_val_historico.columns,
        columns=obj_val_historico.columns,
    )
    print(f"\nVARX simulado (media de {N_SEMILLAS_VALIDACION} semillas):")
    print(corr_varx_media.round(3).to_string())

    muestra_analogos = modelo_analogos.sample(cov_val, semilla=0)
    print("\nAnálogos (línea base, una muestra):")
    print(_correlacion(muestra_analogos).round(3).to_string())

    print("\n" + "=" * 78)
    print("VALIDACIÓN — % de días en estado bajo (<= umbral) por serie")
    print("=" * 78)
    tabla_bajo = pd.DataFrame({
        "historico": _pct_bajo(obj_val_historico),
        "varx (semilla 0)": _pct_bajo(modelo_varx.sample(cov_val, semilla=0)),
        "analogos": _pct_bajo(muestra_analogos),
    })
    print(tabla_bajo.round(2).to_string())

    print("\n" + "=" * 78)
    print("VALIDACIÓN — medias mensuales (m³/s), histórico vs. VARX (semilla 0)")
    print("=" * 78)
    muestra_varx_0 = modelo_varx.sample(cov_val, semilla=0)
    for serie in obj_val_historico.columns:
        medias_hist = obj_val_historico[serie].groupby(obj_val_historico.index.month).mean()
        medias_varx = muestra_varx_0[serie].groupby(muestra_varx_0.index.month).mean()
        comparacion = pd.DataFrame({"historico": medias_hist, "varx": medias_varx})
        print(f"\n{serie}:")
        print(comparacion.round(2).to_string())


def main() -> None:
    covariables, series_objetivo = construir_datos()
    cov_train, cov_val, obj_train, obj_val = dividir_temporal(
        covariables, series_objetivo, AÑOS_VALIDACION
    )

    print("Ajustando modelo VARX desestacionalizado + hurdle (método final)...")
    modelo_varx = ModeloEstocasticoAfluencias(ConfigModeloEstocastico()).fit(cov_train, obj_train)
    print("Ajustando remuestreo por análogos (línea base de validación)...\n")
    modelo_analogos = RemuestreoAnalogos(ConfigAnalogos()).fit(cov_train, obj_train)

    # La validación necesita covariables y objetivo alineados sin NaN, igual que fit().
    datos_val = cov_val.join(obj_val, how="inner").dropna()
    cov_val_limpio = datos_val[list(ConfigModeloEstocastico().covariables)]
    obj_val_limpio = datos_val[list(ConfigModeloEstocastico().series_objetivo)]

    reporte_validacion(obj_val_limpio, modelo_varx, modelo_analogos, cov_val_limpio)

    ARTEFACTOS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_modelo = ARTEFACTOS_DIR / "modelo_afluencias_v1"
    modelo_varx.guardar(ruta_modelo)
    print(f"\nModelo VARX guardado en: {ruta_modelo}")
    print("(no versionado — ver .gitignore; es un artefacto derivado de datos reales)")


if __name__ == "__main__":
    main()

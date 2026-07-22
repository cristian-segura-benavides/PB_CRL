"""Cargador del índice RONI (Relative Oceanic Niño Index) como covariable diaria.

El RONI es la covariable macroclimática que usará el modelo estocástico
multivariado de afluencias (fase 1, aún por construir): junto con Saucío y las
afluencias de Neusa/Sisga/Tominé, busca capturar la señal ENSO en el sistema.

Fuente: NOAA Climate Prediction Center, tabla oficial de RONI (base climatológica
1991-2020, ERSSTv5). NOAA CPC adoptó el RONI como índice oficial de monitoreo de
ENSO en febrero de 2026, en reemplazo del ONI.

Estructura del dato original (mensual)
-----------------------------------------
El RONI es una MEDIA MÓVIL DE 3 MESES, etiquetada por "temporada" (DJF, JFM, FMA,
..., NDJ). Cada valor mensual se asigna al MES CENTRAL de su ventana de 3 meses:
DJF -> enero, JFM -> febrero, FMA -> marzo, etc. El CSV fuente ya trae la columna
``fecha`` anclada al día 1 del mes central correspondiente.

Conversión mensual -> diaria: INTERPOLACIÓN LINEAL (no broadcast)
---------------------------------------------------------------------
Las series del proyecto son diarias; el RONI es mensual. Se interpola LINEALMENTE
entre los valores mensuales (anclados al día 1 de cada mes), en vez de repetir
("broadcast") el mismo valor durante todo el mes. Razón: el RONI ya es una media
móvil de 3 meses, por naturaleza una señal suave y continua; el broadcast
introduciría escalones artificiales el día 1 de cada mes que no corresponden a la
física del índice (que no puede "saltar" de un valor a otro de un día para otro).
La interpolación lineal preserva esa suavidad entre los anclajes mensuales.

La columna categórica ``fase`` (Nino/Nina/Neutral, umbral ±0.5 sobre el RONI) NO se
interpola (no tiene sentido interpolar una categoría); se propaga hacia adelante
(forward-fill) desde el ancla mensual más reciente hasta la siguiente.

Advertencia de NOAA (revisión de valores recientes)
-------------------------------------------------------
NOAA advierte que los valores más recientes del RONI pueden ajustarse hasta dos
meses después de publicados, por el filtro de alta frecuencia que aplica ERSSTv5.
Esto NO afecta la ventana del proyecto (que cierra en 2025-05-04): los valores
usados aquí tienen, para esa fecha, meses de antigüedad frente a la fecha de
publicación del archivo fuente y ya están estabilizados.

Ventana: se recorta a VENTANA_INICIO/VENTANA_FIN (``data_contracts.ventana``), la
misma ventana definitiva del proyecto (2012-01-01 a 2025-05-04). El CSV mensual
cubre 2009-2025, con margen suficiente antes y después de la ventana para
interpolar sin extrapolar en ningún extremo.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pbcrl.data_contracts.ventana import VENTANA_FIN, VENTANA_INICIO

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RONI_CSV = ROOT_DIR / "info_CAR" / "RONI_2009_2025_mensual.csv"

FUENTE_RONI = "NOAA CPC — RONI oficial (base 1991-2020, ERSSTv5)"


@dataclass(frozen=True)
class DiagnosticoRoni:
    """Resumen trazable de la carga e interpolación del RONI."""

    dias_totales: int
    meses_ancla_totales: int
    meses_ancla_en_ventana: int
    rango_mensual_inicio: pd.Timestamp
    rango_mensual_fin: pd.Timestamp
    rango_diario_inicio: pd.Timestamp
    rango_diario_fin: pd.Timestamp


def _cargar_serie_mensual(ruta_csv: str | Path) -> pd.DataFrame:
    ruta = Path(ruta_csv)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el CSV del RONI en: {ruta}")

    crudo = pd.read_csv(ruta)
    faltantes = [c for c in ("fecha", "roni", "fase") if c not in crudo.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas esperadas en el CSV del RONI: {faltantes}")

    d = crudo[["fecha", "roni", "fase"]].copy()
    d["fecha"] = pd.to_datetime(d["fecha"], errors="coerce")
    d["roni"] = pd.to_numeric(d["roni"], errors="coerce")
    d = d.dropna(subset=["fecha"])
    d = d.set_index("fecha").sort_index()
    d = d[~d.index.duplicated(keep="first")]
    return d


def cargar_roni(
    ruta_csv: str | Path = DEFAULT_RONI_CSV,
    ventana_inicio: str = VENTANA_INICIO,
    ventana_fin: str = VENTANA_FIN,
) -> tuple[pd.DataFrame, DiagnosticoRoni]:
    """Carga el RONI mensual, lo interpola linealmente a diario y recorta a la ventana.

    Devuelve un DataFrame con DatetimeIndex diario continuo (columnas ``roni``
    interpolado linealmente y ``fase`` propagada hacia adelante desde el ancla
    mensual), junto con un diagnóstico de la carga.
    """
    mensual = _cargar_serie_mensual(ruta_csv)

    inicio = pd.Timestamp(ventana_inicio)
    fin = pd.Timestamp(ventana_fin)

    if inicio < mensual.index.min() or fin > mensual.index.max():
        raise ValueError(
            f"La ventana solicitada ({inicio.date()} a {fin.date()}) excede el rango "
            f"del CSV mensual del RONI ({mensual.index.min().date()} a "
            f"{mensual.index.max().date()}); no se puede interpolar sin extrapolar."
        )

    # Interpolacion lineal sobre el rango COMPLETO de anclas mensuales (no solo la
    # ventana), para no extrapolar en los bordes de la ventana solicitada.
    indice_diario_completo = pd.date_range(mensual.index.min(), mensual.index.max(), freq="D")
    diario = mensual.reindex(indice_diario_completo)
    diario.index.name = "fecha"
    diario["roni"] = diario["roni"].interpolate(method="time")
    diario["fase"] = diario["fase"].ffill()

    resultado = diario.loc[inicio:fin].copy()

    diagnostico = DiagnosticoRoni(
        dias_totales=len(resultado),
        meses_ancla_totales=len(mensual),
        meses_ancla_en_ventana=int(((mensual.index >= inicio) & (mensual.index <= fin)).sum()),
        rango_mensual_inicio=mensual.index.min(),
        rango_mensual_fin=mensual.index.max(),
        rango_diario_inicio=resultado.index.min(),
        rango_diario_fin=resultado.index.max(),
    )
    return resultado, diagnostico


def resumen_roni(df: pd.DataFrame, diagnostico: DiagnosticoRoni) -> str:
    """Resumen legible del RONI cargado para verificación rápida."""
    lineas = [
        "=== RONI diario (interpolado linealmente desde mensual) ===",
        f"Fuente: {FUENTE_RONI}",
        f"Rango diario: {diagnostico.rango_diario_inicio.date()} -> "
        f"{diagnostico.rango_diario_fin.date()} ({diagnostico.dias_totales} días)",
        f"Meses ancla usados en la ventana: {diagnostico.meses_ancla_en_ventana} "
        f"(de {diagnostico.meses_ancla_totales} disponibles en el CSV, "
        f"{diagnostico.rango_mensual_inicio.date()} -> {diagnostico.rango_mensual_fin.date()})",
        "",
        "Estadísticas básicas del RONI diario:",
        df["roni"].describe().round(3).to_string(),
        "",
        "Distribución de fase (días):",
        df["fase"].value_counts().to_string(),
    ]
    return "\n".join(lineas)

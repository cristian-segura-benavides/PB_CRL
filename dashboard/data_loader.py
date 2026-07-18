"""Carga, limpieza y diagnostico de datos para el dashboard de embalses.

Fuentes:
- Neusa, Sisga: volumen, cota, descarga y lluvia de info_CAR/20261065095_Embalses.xlsx
  (CAR); evaporacion de info_CAR/ae (49).xlsx (CAR).
- Tomine: volumen (util), cota, descarga, bombeo y lluvia de
  info_CAR/datos operativos Tomine_Enlaza.xlsx (Enlaza), via pbcrl.data_ingest.tomine.
  Evaporacion ERA5-Land (ver TOMINE_EVAPORACION_FUENTE mas abajo).
- Limites operativos: pbcrl.data_contracts.embalses.EMBALSES.

La limpieza se divide en dos capas (los tres embalses):
- Capa 1: correccion de saltos anomalos de volumen por interpolacion.
- Capa 2: acotamiento a cero de afluencias residuales negativas, con trazabilidad.
Tomine ademas incluye el termino de bombeo (entrada artificial) en el balance.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
import unicodedata

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pbcrl.data_contracts.embalses import (
    CONVENCION_VOLUMEN,
    EMBALSES,
    VOLUMEN_MUERTO_MM3,
    ParametrosEmbalse,
)
from pbcrl.data_ingest.tomine import cargar_tomine
from pbcrl.hydrology.balance import calcular_afluencia


VOLUME_FILE = ROOT_DIR / "info_CAR" / "20261065095_Embalses.xlsx"
EVAP_FILE = ROOT_DIR / "info_CAR" / "ae (49).xlsx"

# Enlaza confirmó por escrito (radicado ENL-002443-2026-S) que Tominé no cuenta con
# medición de evaporación ni evaporímetro propio. Se usa evaporación ERA5-Land
# (derivada del flujo de calor latente), validada en magnitud (~3.19 mm/día) contra la
# evaporación medida de los embalses vecinos Neusa y Sisga.
TOMINE_EVAPORACION_FUENTE = (
    "ERA5-Land (flujo de calor latente). Enlaza confirmó por escrito "
    "(radicado ENL-002443-2026-S) que Tominé no cuenta con medición de evaporación "
    "ni evaporímetro; se validó en magnitud (~3.19 mm/día) contra la evaporación "
    "medida de Neusa y Sisga."
)
TOMINE_VOLUMEN_FUENTE = "Enlaza - datos operativos Tomine_Enlaza.xlsx (hoja 'Tomine')"

EVAPORATION_CODES = {
    "Neusa": 2401537,
    "Sisga": 2120659,
}

DEFAULT_VOLUME_JUMP_FRACTION = 0.10
DEFAULT_NEGATIVE_WARNING_THRESHOLD_M3S = 2.0
SHORT_GAP_LIMIT = 3
# Margen (en metros) alrededor de las cotas operativas que define el rango fisico
# valido. Una cota fuera de [cota_min - margen, cota_max + margen] se considera
# corrupta (error de digitacion) y se corrige por interpolacion lineal.
# 15 m captura los errores de digitacion conocidos de Neusa/Sisga (p.ej. 2625.28 en
# Sisga, ~19 m bajo el minimo) sin marcar la operacion real ~0.5 m sobre el maximo
# nominal (cotas 2670.x de Sisga con embalse casi lleno). Es configurable.
DEFAULT_COTA_OUTLIER_MARGIN_M = 15.0


@dataclass(frozen=True)
class CleaningConfig:
    """Parametros configurables del pre y post procesamiento."""

    volume_jump_fraction: float = DEFAULT_VOLUME_JUMP_FRACTION
    negative_warning_threshold_m3s: float = DEFAULT_NEGATIVE_WARNING_THRESHOLD_M3S
    cota_outlier_margin_m: float = DEFAULT_COTA_OUTLIER_MARGIN_M


@dataclass(frozen=True)
class ReservoirDiagnostics:
    """Resumen trazable del antes, despues de la limpieza y despues del clamp."""

    raw_negative_days: int
    raw_negative_pct: float
    raw_negative_min_m3s: float | None
    raw_negative_mean_m3s: float | None
    raw_negative_median_m3s: float | None
    raw_negative_p90_abs_m3s: float | None
    volume_jump_days: int
    volume_jump_threshold_mm3: float
    after_layer1_negative_days: int
    after_layer1_negative_pct: float
    after_layer1_negative_min_m3s: float | None
    after_layer1_negative_mean_m3s: float | None
    after_layer1_negative_median_m3s: float | None
    after_layer1_negative_p90_abs_m3s: float | None
    clamped_negative_days: int
    clamped_negative_warning_days: int
    clamped_negative_warning_max_abs_m3s: float | None
    final_negative_days: int
    final_negative_pct: float
    cota_outliers_corrected: int = 0
    cota_outlier_margin_m: float = DEFAULT_COTA_OUTLIER_MARGIN_M
    volumen_convencion: str = CONVENCION_VOLUMEN
    volumen_muerto_mm3: float = 0.0
    volumen_util_clamp_cero: int = 0


@dataclass(frozen=True)
class ReservoirContext:
    """Serie final limpia y metadatos de un embalse."""

    name: str
    params: ParametrosEmbalse
    frame: pd.DataFrame
    raw_frame: pd.DataFrame
    diagnostics: ReservoirDiagnostics
    operational_status: str
    evaporation_source: str


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _normalize_reservoir_name(value: str) -> str:
    cleaned = _strip_accents(str(value)).strip().upper()
    if cleaned.startswith("NEUSA"):
        return "Neusa"
    if cleaned.startswith("SISGA"):
        return "Sisga"
    if cleaned.startswith("TOMINE"):
        return "Tomine"
    return cleaned.title()


@lru_cache(maxsize=1)
def _read_volume_source() -> pd.DataFrame:
    raw = pd.read_excel(VOLUME_FILE)
    raw["fecha"] = pd.to_datetime(raw["fecha"])
    raw["embalse"] = raw["embalse"].map(_normalize_reservoir_name)
    renamed = raw.rename(
        columns={
            "cota_m_s_n_m": "cota_m",
            "descarga_m3_s": "descarga_m3s",
            "lluvia_mm": "precipitacion_mm",
        }
    )
    return renamed[["fecha", "embalse", "cota_m", "volumen_mm3", "descarga_m3s", "precipitacion_mm"]]


@lru_cache(maxsize=1)
def _read_evaporation_source() -> pd.DataFrame:
    raw = pd.read_excel(EVAP_FILE)
    raw["FECHA"] = pd.to_datetime(raw["FECHA"])
    raw["PARAMETRO"] = raw["PARAMETRO"].map(_strip_accents).str.upper()
    raw["CODIGO"] = pd.to_numeric(raw["CODIGO"], errors="coerce").astype("Int64")

    evaporation = raw[raw["PARAMETRO"].str.contains("EVAPOR", na=False)].copy()
    evaporation = evaporation[["CODIGO", "FECHA", "DATO"]]
    evaporation = evaporation.rename(columns={"FECHA": "fecha", "DATO": "evaporacion_mm"})
    evaporation["embalse"] = evaporation["CODIGO"].map({code: name for name, code in EVAPORATION_CODES.items()})
    evaporation = evaporation.dropna(subset=["embalse"])
    evaporation["embalse"] = evaporation["embalse"].map(_normalize_reservoir_name)
    return evaporation[["fecha", "embalse", "evaporacion_mm"]]


def _convertir_volumen_a_util(frame: pd.DataFrame, nombre: str) -> tuple[pd.DataFrame, float, int]:
    """Convierte la columna de volumen de TOTAL a ÚTIL restando el volumen muerto.

    Los datos de la CAR (Neusa, Sisga) reportan volumen total; el proyecto trabaja en
    volumen util (ver embalses.CONVENCION_VOLUMEN). Se resta el volumen muerto del
    embalse; los valores que quedaran negativos (volumen por debajo del muerto) se
    acotan a 0, porque el volumen util no puede ser negativo, y se cuentan.

    Es REVERSIBLE: si CONVENCION_VOLUMEN != "util", no se resta nada. La decision de
    convencion util esta respaldada por fuentes oficiales (Enlaza/CAR) y PENDIENTE de
    validacion final con el asesor.

    NOTA: la afluencia por balance inverso es invariante a esta conversion (usa ΔV, que
    no cambia al restar una constante), salvo en los dias acotados a 0.

    Devuelve (frame_convertido, volumen_muerto_restado_mm3, dias_acotados_a_cero).
    """
    if CONVENCION_VOLUMEN != "util":
        return frame, 0.0, 0
    muerto_mm3 = float(VOLUMEN_MUERTO_MM3.get(nombre, 0.0))
    if muerto_mm3 == 0.0:
        return frame, 0.0, 0
    convertido = frame.copy()
    util = convertido["volumen_mm3"] - muerto_mm3
    dias_clamp = int((util < 0).sum())  # NaN no cuenta (NaN < 0 == False)
    convertido["volumen_mm3"] = util.clip(lower=0.0)
    return convertido, muerto_mm3, dias_clamp


def _empty_diagnostics(
    volume_jump_threshold_mm3: float,
    cota_outlier_margin_m: float = DEFAULT_COTA_OUTLIER_MARGIN_M,
) -> ReservoirDiagnostics:
    return ReservoirDiagnostics(
        raw_negative_days=0,
        raw_negative_pct=0.0,
        raw_negative_min_m3s=None,
        raw_negative_mean_m3s=None,
        raw_negative_median_m3s=None,
        raw_negative_p90_abs_m3s=None,
        volume_jump_days=0,
        volume_jump_threshold_mm3=volume_jump_threshold_mm3,
        after_layer1_negative_days=0,
        after_layer1_negative_pct=0.0,
        after_layer1_negative_min_m3s=None,
        after_layer1_negative_mean_m3s=None,
        after_layer1_negative_median_m3s=None,
        after_layer1_negative_p90_abs_m3s=None,
        clamped_negative_days=0,
        clamped_negative_warning_days=0,
        clamped_negative_warning_max_abs_m3s=None,
        final_negative_days=0,
        final_negative_pct=0.0,
        cota_outliers_corrected=0,
        cota_outlier_margin_m=cota_outlier_margin_m,
    )


def _interpolate_short_gaps(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = result[column].interpolate(method="time", limit=SHORT_GAP_LIMIT, limit_area="inside")
    return result


def _clean_cota_series(
    series: pd.Series,
    params: ParametrosEmbalse,
    margin_m: float = DEFAULT_COTA_OUTLIER_MARGIN_M,
) -> tuple[pd.Series, int]:
    """Corrige cotas fisicamente imposibles (errores de digitacion).

    Marca como corruptas las cotas fuera del rango valido
    [cota_min - margin_m, cota_max + margin_m] del embalse y las repone por
    interpolacion lineal (promedio del dia anterior y siguiente para valores
    aislados). Devuelve la serie corregida y el numero de valores corregidos.
    """
    lower_bound = params.cota_min_m - margin_m
    upper_bound = params.cota_max_m + margin_m
    outlier_mask = ((series < lower_bound) | (series > upper_bound)) & series.notna()
    n_corrected = int(outlier_mask.sum())
    if n_corrected == 0:
        return series, 0
    cleaned = series.mask(outlier_mask)
    cleaned = cleaned.interpolate(method="linear", limit_direction="both")
    return cleaned, n_corrected


def _series_negative_stats(series: pd.Series) -> dict[str, float | int | None]:
    negative = series[series < 0].dropna()
    total = int(series.notna().sum())
    if negative.empty or total == 0:
        return {
            "count": 0,
            "pct": 0.0,
            "min": None,
            "mean": None,
            "median": None,
            "p90_abs": None,
        }
    return {
        "count": int(negative.size),
        "pct": round(100.0 * negative.size / total, 6),
        "min": float(negative.min()),
        "mean": float(negative.mean()),
        "median": float(negative.median()),
        "p90_abs": float(negative.abs().quantile(0.90)),
    }


def _apply_volume_jump_correction(
    frame: pd.DataFrame,
    params: ParametrosEmbalse,
    config: CleaningConfig,
) -> tuple[pd.DataFrame, pd.Series, float]:
    threshold_mm3 = config.volume_jump_fraction * params.capacidad_max_mm3
    delta_volume = frame["volumen_mm3"].diff().abs()
    jump_mask = (delta_volume > threshold_mm3).fillna(False)

    cleaned = frame.copy()
    cleaned.loc[jump_mask, "volumen_mm3"] = pd.NA
    cleaned["volumen_mm3"] = cleaned["volumen_mm3"].interpolate(method="time", limit=SHORT_GAP_LIMIT, limit_area="inside")
    return cleaned, jump_mask, threshold_mm3


def _compute_afluence(
    frame: pd.DataFrame,
    params: ParametrosEmbalse,
    validar: bool = True,
    bombeo_mm3: pd.Series | None = None,
) -> pd.Series:
    """Calcula la afluencia por bloques validos, opcionalmente restando el bombeo.

    bombeo_mm3 es None para Neusa/Sisga (sin bombeo, comportamiento identico al previo).
    Solo Tominé pasa una serie real (ver _build_tomine_context).
    """
    required = ["cota_m", "volumen_mm3", "descarga_m3s", "precipitacion_mm", "evaporacion_mm"]
    working = _interpolate_short_gaps(frame, required)
    valid_rows = working[required].notna().all(axis=1)
    result = pd.Series(index=working.index, name="afluencia_m3s", dtype="float64")

    def _afluencia_bloque(block_index: pd.DatetimeIndex) -> pd.Series:
        bombeo_bloque = bombeo_mm3.loc[block_index] if bombeo_mm3 is not None else None
        return calcular_afluencia(
            working.loc[block_index, required], params, validar=validar, bombeo_mm3=bombeo_bloque
        )

    block_start: list[pd.Timestamp] = []
    for fecha, keep in valid_rows.items():
        if bool(keep):
            block_start.append(fecha)
            continue
        if len(block_start) >= 2:
            block_index = pd.DatetimeIndex(block_start)
            result.loc[block_index] = _afluencia_bloque(block_index)
        block_start = []

    if len(block_start) >= 2:
        block_index = pd.DatetimeIndex(block_start)
        result.loc[block_index] = _afluencia_bloque(block_index)

    return result


def _clamp_negative_afluence(series: pd.Series, warning_threshold_m3s: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    negative_mask = (series < 0).fillna(False)
    warning_mask = negative_mask & (series.abs() > warning_threshold_m3s)
    cleaned = series.copy()
    cleaned.loc[negative_mask] = 0.0
    return cleaned, negative_mask, warning_mask


def _build_raw_reservoir_frame(name: str) -> pd.DataFrame:
    volume_source = _read_volume_source()
    evaporation_source = _read_evaporation_source()

    reservoir = volume_source[volume_source["embalse"] == name].copy()
    if reservoir.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="fecha"))

    reservoir = reservoir.set_index("fecha").sort_index()
    full_index = pd.date_range(reservoir.index.min(), reservoir.index.max(), freq="D")
    reservoir = reservoir.reindex(full_index)
    reservoir.index.name = "fecha"

    evaporation = evaporation_source[evaporation_source["embalse"] == name].copy()
    evaporation = evaporation.set_index("fecha").sort_index().reindex(full_index)
    reservoir["evaporacion_mm"] = evaporation["evaporacion_mm"]

    return reservoir[["cota_m", "volumen_mm3", "descarga_m3s", "precipitacion_mm", "evaporacion_mm"]]


def _build_reservoir_context(name: str, params: ParametrosEmbalse, config: CleaningConfig) -> ReservoirContext:
    raw_frame = _build_raw_reservoir_frame(name)
    if raw_frame.empty:
        diagnostics = _empty_diagnostics(
            config.volume_jump_fraction * params.capacidad_max_mm3,
            cota_outlier_margin_m=config.cota_outlier_margin_m,
        )
        return ReservoirContext(
            name=name,
            params=params,
            frame=pd.DataFrame(index=pd.DatetimeIndex([], name="fecha")),
            raw_frame=raw_frame,
            diagnostics=diagnostics,
            operational_status="Datos operativos pendientes",
            evaporation_source="No disponible aun",
        )

    # Convención: la CAR reporta volumen TOTAL; el proyecto trabaja en volumen ÚTIL.
    # Se resta el volumen muerto (Neusa/Sisga) antes de cualquier limpieza.
    raw_frame, volumen_muerto_mm3, volumen_util_clamp = _convertir_volumen_a_util(raw_frame, name)

    baseline = _interpolate_short_gaps(
        raw_frame,
        ["volumen_mm3", "descarga_m3s", "precipitacion_mm", "evaporacion_mm"],
    )
    baseline["cota_m"], cota_outliers_corrected = _clean_cota_series(
        baseline["cota_m"], params, config.cota_outlier_margin_m
    )

    raw_afluence = _compute_afluence(baseline, params)

    corrected_volume_frame, jump_mask, threshold_mm3 = _apply_volume_jump_correction(baseline, params, config)
    after_layer1_afluence = _compute_afluence(corrected_volume_frame, params)
    final_afluence, clamped_mask, warning_mask = _clamp_negative_afluence(
        after_layer1_afluence,
        warning_threshold_m3s=config.negative_warning_threshold_m3s,
    )

    cleaned_frame = corrected_volume_frame.copy()
    cleaned_frame["afluencia_m3s"] = final_afluence
    cleaned_frame["afluencia_pre_clamp_m3s"] = after_layer1_afluence
    cleaned_frame["cota_m"], _ = _clean_cota_series(
        cleaned_frame["cota_m"], params, config.cota_outlier_margin_m
    )
    cleaned_frame["fuente_volumen"] = "CAR 20261065095_Embalses.xlsx"
    cleaned_frame["fuente_evaporacion"] = "CAR ae (49).xlsx"

    raw_stats = _series_negative_stats(raw_afluence)
    after_layer1_stats = _series_negative_stats(after_layer1_afluence)
    clamped_warning_values = after_layer1_afluence[warning_mask].abs()

    diagnostics = ReservoirDiagnostics(
        raw_negative_days=int(raw_stats["count"]),
        raw_negative_pct=float(raw_stats["pct"]),
        raw_negative_min_m3s=raw_stats["min"],
        raw_negative_mean_m3s=raw_stats["mean"],
        raw_negative_median_m3s=raw_stats["median"],
        raw_negative_p90_abs_m3s=raw_stats["p90_abs"],
        volume_jump_days=int(jump_mask.sum()),
        volume_jump_threshold_mm3=float(threshold_mm3),
        after_layer1_negative_days=int(after_layer1_stats["count"]),
        after_layer1_negative_pct=float(after_layer1_stats["pct"]),
        after_layer1_negative_min_m3s=after_layer1_stats["min"],
        after_layer1_negative_mean_m3s=after_layer1_stats["mean"],
        after_layer1_negative_median_m3s=after_layer1_stats["median"],
        after_layer1_negative_p90_abs_m3s=after_layer1_stats["p90_abs"],
        clamped_negative_days=int(clamped_mask.sum()),
        clamped_negative_warning_days=int(warning_mask.sum()),
        clamped_negative_warning_max_abs_m3s=float(clamped_warning_values.max()) if not clamped_warning_values.empty else None,
        final_negative_days=int((final_afluence < 0).sum()),
        final_negative_pct=float((final_afluence < 0).sum() / final_afluence.notna().sum() * 100.0) if final_afluence.notna().sum() else 0.0,
        cota_outliers_corrected=cota_outliers_corrected,
        cota_outlier_margin_m=config.cota_outlier_margin_m,
        volumen_convencion=CONVENCION_VOLUMEN,
        volumen_muerto_mm3=volumen_muerto_mm3,
        volumen_util_clamp_cero=volumen_util_clamp,
    )

    return ReservoirContext(
        name=name,
        params=params,
        frame=cleaned_frame,
        raw_frame=raw_frame,
        diagnostics=diagnostics,
        operational_status="Serie operativa disponible",
        evaporation_source="CAR ae (49).xlsx",
    )


def _build_tomine_context(params: ParametrosEmbalse, config: CleaningConfig) -> ReservoirContext:
    """Construye el contexto de Tominé: Excel de Enlaza + evaporación ERA5.

    Mismo tratamiento de afluencias negativas en dos capas que Neusa/Sisga (corrección
    de saltos de volumen -> capa 1; acotamiento de residuales negativos -> capa 2),
    con el término de bombeo restado en el balance (Tominé es el único de los tres
    embalses con entrada artificial por bombeo). A diferencia de Neusa/Sisga, la serie
    de Enlaza ya viene en volumen ÚTIL (ver data_ingest.tomine), por lo que NO se aplica
    _convertir_volumen_a_util.
    """
    try:
        tomine_frame, _ = cargar_tomine()
    except FileNotFoundError:
        diagnostics = _empty_diagnostics(
            config.volume_jump_fraction * params.capacidad_max_mm3,
            cota_outlier_margin_m=config.cota_outlier_margin_m,
        )
        return ReservoirContext(
            name="Tomine",
            params=params,
            frame=pd.DataFrame(index=pd.DatetimeIndex([], name="fecha")),
            raw_frame=pd.DataFrame(index=pd.DatetimeIndex([], name="fecha")),
            diagnostics=diagnostics,
            operational_status="Datos operativos pendientes",
            evaporation_source="No disponible aun",
        )

    bombeo_mm3 = tomine_frame["bombeo_mm3"]
    raw_frame = tomine_frame[
        ["cota_m", "volumen_mm3", "descarga_m3s", "precipitacion_mm", "evaporacion_mm"]
    ].copy()

    baseline = _interpolate_short_gaps(
        raw_frame,
        ["volumen_mm3", "descarga_m3s", "precipitacion_mm", "evaporacion_mm"],
    )
    baseline["cota_m"], cota_outliers_corrected = _clean_cota_series(
        baseline["cota_m"], params, config.cota_outlier_margin_m
    )

    raw_afluence = _compute_afluence(baseline, params, bombeo_mm3=bombeo_mm3)

    corrected_volume_frame, jump_mask, threshold_mm3 = _apply_volume_jump_correction(baseline, params, config)
    after_layer1_afluence = _compute_afluence(corrected_volume_frame, params, bombeo_mm3=bombeo_mm3)
    final_afluence, clamped_mask, warning_mask = _clamp_negative_afluence(
        after_layer1_afluence,
        warning_threshold_m3s=config.negative_warning_threshold_m3s,
    )

    cleaned_frame = corrected_volume_frame.copy()
    cleaned_frame["afluencia_m3s"] = final_afluence
    cleaned_frame["afluencia_pre_clamp_m3s"] = after_layer1_afluence
    cleaned_frame["cota_m"], _ = _clean_cota_series(
        cleaned_frame["cota_m"], params, config.cota_outlier_margin_m
    )
    cleaned_frame["bombeo_mm3"] = bombeo_mm3
    cleaned_frame["fuente_volumen"] = TOMINE_VOLUMEN_FUENTE
    cleaned_frame["fuente_evaporacion"] = TOMINE_EVAPORACION_FUENTE

    raw_stats = _series_negative_stats(raw_afluence)
    after_layer1_stats = _series_negative_stats(after_layer1_afluence)
    clamped_warning_values = after_layer1_afluence[warning_mask].abs()

    diagnostics = ReservoirDiagnostics(
        raw_negative_days=int(raw_stats["count"]),
        raw_negative_pct=float(raw_stats["pct"]),
        raw_negative_min_m3s=raw_stats["min"],
        raw_negative_mean_m3s=raw_stats["mean"],
        raw_negative_median_m3s=raw_stats["median"],
        raw_negative_p90_abs_m3s=raw_stats["p90_abs"],
        volume_jump_days=int(jump_mask.sum()),
        volume_jump_threshold_mm3=float(threshold_mm3),
        after_layer1_negative_days=int(after_layer1_stats["count"]),
        after_layer1_negative_pct=float(after_layer1_stats["pct"]),
        after_layer1_negative_min_m3s=after_layer1_stats["min"],
        after_layer1_negative_mean_m3s=after_layer1_stats["mean"],
        after_layer1_negative_median_m3s=after_layer1_stats["median"],
        after_layer1_negative_p90_abs_m3s=after_layer1_stats["p90_abs"],
        clamped_negative_days=int(clamped_mask.sum()),
        clamped_negative_warning_days=int(warning_mask.sum()),
        clamped_negative_warning_max_abs_m3s=float(clamped_warning_values.max()) if not clamped_warning_values.empty else None,
        final_negative_days=int((final_afluence < 0).sum()),
        final_negative_pct=float((final_afluence < 0).sum() / final_afluence.notna().sum() * 100.0) if final_afluence.notna().sum() else 0.0,
        cota_outliers_corrected=cota_outliers_corrected,
        cota_outlier_margin_m=config.cota_outlier_margin_m,
        volumen_convencion=CONVENCION_VOLUMEN,
        volumen_muerto_mm3=0.0,  # la serie de Enlaza ya viene en util; no se resta nada
        volumen_util_clamp_cero=0,
    )

    return ReservoirContext(
        name="Tomine",
        params=params,
        frame=cleaned_frame,
        raw_frame=raw_frame,
        diagnostics=diagnostics,
        operational_status="Serie operativa disponible",
        evaporation_source=TOMINE_EVAPORACION_FUENTE,
    )


def load_dashboard_context(
    volume_jump_fraction: float = DEFAULT_VOLUME_JUMP_FRACTION,
    negative_warning_threshold_m3s: float = DEFAULT_NEGATIVE_WARNING_THRESHOLD_M3S,
    cota_outlier_margin_m: float = DEFAULT_COTA_OUTLIER_MARGIN_M,
) -> tuple[dict[str, ReservoirContext], pd.Timestamp, pd.Timestamp]:
    """Carga los contextos limpios del dashboard con umbrales configurables."""
    config = CleaningConfig(
        volume_jump_fraction=volume_jump_fraction,
        negative_warning_threshold_m3s=negative_warning_threshold_m3s,
        cota_outlier_margin_m=cota_outlier_margin_m,
    )

    contexts: dict[str, ReservoirContext] = {}
    date_min = pd.NaT
    date_max = pd.NaT

    for name, params in EMBALSES.items():
        if name == "Tomine":
            context = _build_tomine_context(params, config)
        else:
            context = _build_reservoir_context(name, params, config)
        contexts[name] = context
        if not context.frame.empty:
            current_min = context.frame.index.min()
            current_max = context.frame.index.max()
            date_min = current_min if pd.isna(date_min) else min(date_min, current_min)
            date_max = current_max if pd.isna(date_max) else max(date_max, current_max)

    return contexts, pd.Timestamp(date_min), pd.Timestamp(date_max)


def diagnostics_report_table(contexts: dict[str, ReservoirContext]) -> pd.DataFrame:
    """Devuelve un reporte tabular de diagnostico antes y despues de la limpieza."""
    rows: list[dict[str, object]] = []
    for name in ["Neusa", "Sisga"]:
        diagnostics = contexts[name].diagnostics
        rows.extend(
            [
                {
                    "embalse": name,
                    "etapa": "crudo",
                    "dias_negativos": diagnostics.raw_negative_days,
                    "pct_negativos": diagnostics.raw_negative_pct,
                    "min_negativo_m3s": diagnostics.raw_negative_min_m3s,
                    "media_negativa_m3s": diagnostics.raw_negative_mean_m3s,
                    "mediana_negativa_m3s": diagnostics.raw_negative_median_m3s,
                    "p90_abs_negativo_m3s": diagnostics.raw_negative_p90_abs_m3s,
                    "saltos_volumen_corregidos": diagnostics.volume_jump_days,
                    "umbral_salto_volumen_mm3": diagnostics.volume_jump_threshold_mm3,
                    "negativos_acotados": 0,
                    "advertencias_acotadas": 0,
                },
                {
                    "embalse": name,
                    "etapa": "tras_capa1",
                    "dias_negativos": diagnostics.after_layer1_negative_days,
                    "pct_negativos": diagnostics.after_layer1_negative_pct,
                    "min_negativo_m3s": diagnostics.after_layer1_negative_min_m3s,
                    "media_negativa_m3s": diagnostics.after_layer1_negative_mean_m3s,
                    "mediana_negativa_m3s": diagnostics.after_layer1_negative_median_m3s,
                    "p90_abs_negativo_m3s": diagnostics.after_layer1_negative_p90_abs_m3s,
                    "saltos_volumen_corregidos": diagnostics.volume_jump_days,
                    "umbral_salto_volumen_mm3": diagnostics.volume_jump_threshold_mm3,
                    "negativos_acotados": 0,
                    "advertencias_acotadas": 0,
                },
                {
                    "embalse": name,
                    "etapa": "final",
                    "dias_negativos": diagnostics.final_negative_days,
                    "pct_negativos": diagnostics.final_negative_pct,
                    "min_negativo_m3s": None,
                    "media_negativa_m3s": None,
                    "mediana_negativa_m3s": None,
                    "p90_abs_negativo_m3s": None,
                    "saltos_volumen_corregidos": diagnostics.volume_jump_days,
                    "umbral_salto_volumen_mm3": diagnostics.volume_jump_threshold_mm3,
                    "negativos_acotados": diagnostics.clamped_negative_days,
                    "advertencias_acotadas": diagnostics.clamped_negative_warning_days,
                },
            ]
        )
    return pd.DataFrame(rows)


def operational_limits_table() -> pd.DataFrame:
    rows = []
    for name, params in EMBALSES.items():
        rows.append(
            {
                "embalse": name,
                "cota_min_m": params.cota_min_m,
                "cota_max_m": params.cota_max_m,
                "capacidad_min_mm3": params.capacidad_min_mm3,
                "capacidad_max_mm3": params.capacidad_max_mm3,
            }
        )
    return pd.DataFrame(rows)


def diagnostics_note(contexts: dict[str, ReservoirContext]) -> str:
    """Nota breve visible para mostrar cuantos dias fueron corregidos/acotados."""
    parts = []
    for name in ["Neusa", "Sisga"]:
        diag = contexts[name].diagnostics
        parts.append(
            f"{name}: Capa 1 corrigio {diag.volume_jump_days} saltos; Capa 2 acoto {diag.clamped_negative_days} dias negativos"
            f" ({diag.clamped_negative_warning_days} por encima del umbral de advertencia)."
        )
    return " ".join(parts)

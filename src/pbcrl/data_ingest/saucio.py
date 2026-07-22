"""Cargador del caudal diario de Saucío (caudal natural hacia el punto El Sol).

Empalma dos fuentes con un CORTE LIMPIO en la frontera (sin promediar el solape
2021-2022):
- CAR (estación SAUCIO, código 2120719, escala diaria, tipo MEDIOS,
  ``info_CAR/20261065095_Estaciones.csv``): se usa hasta donde llega su cobertura,
  2022-12-31 inclusive.
- Enlaza (hoja oculta "S-PF-PT" del Excel de Tominé, columna
  "CAUDAL SAUCIO (m3/s)"): se usa desde 2023-01-01 en adelante, hasta el fin de la
  ventana del proyecto.

Justificación del corte limpio (sin promediar)
-----------------------------------------------
El diagnóstico del solape 2021-2022 (729 días con ambas series) mostró que son
consistentes entre sí: mismas unidades (m³/s), sin desfase temporal (correlación
máxima exactamente en lag=0, 0.9496), y sin sesgo sistemático (regresión
Enlaza ~ 1.031·CAR − 0.063, pendiente ≈1, intercepto despreciable frente a los
caudales típicos). Con esa consistencia confirmada, promediar el solape no aporta
precisión adicional y produciría una serie que no es exactamente ninguna de las
dos fuentes; se prefiere el corte limpio, documentando de qué fuente proviene
cada tramo (columna ``fuente_caudal``).

Día sospechoso (NO corregido)
-------------------------------
El 2021-06-21 mostró la mayor discrepancia del solape (CAR=20.03 vs Enlaza=34.52
m³/s, diferencia 14.49). Se marca en el diagnóstico (``DiagnosticoSaucio``) como
día sospechoso, pero NO se elimina ni se corrige: es un único día en toda la
serie y cae fuera del tramo CAR usado en el empalme (el corte usa CAR hasta
2022-12-31 completo, así que ese día en particular sí queda con el valor CAR).

Huecos conocidos (NO rellenados)
-----------------------------------
- CAR (tramo 2012-2022): 3 días puntuales faltantes (2017-11-30, 2018-04-30,
  2022-05-01).
- Enlaza (tramo 2023 en adelante): hueco interno de 90 días, 2025-01-20 a
  2025-04-19 (la serie retoma datos reales del 2025-04-20 al 2025-05-04, el
  hueco NO llega hasta el final de la ventana).
Se reportan en ``DiagnosticoSaucio.huecos_totales``; este loader no los rellena.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pbcrl.data_contracts.ventana import VENTANA_FIN, VENTANA_INICIO

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CAR_ESTACIONES_CSV = ROOT_DIR / "info_CAR" / "20261065095_Estaciones.csv"
DEFAULT_TOMINE_XLSX = ROOT_DIR / "info_CAR" / "datos operativos Tomine_Enlaza.xlsx"
ENLAZA_SHEET = "S-PF-PT"
ENLAZA_COLUMNA_CAUDAL = "CAUDAL SAUCIO (m3/s)"

CAR_ESTACION = "SAUCIO"
CAR_ESCALA = "DIARIO"
CAR_TIPO = "MEDIOS"

# Corte del empalme: CAR se usa hasta el día anterior a esta fecha (inclusive
# 2022-12-31); Enlaza se usa desde esta fecha en adelante.
CORTE_EMPALME = "2023-01-01"

DIA_SOSPECHOSO = pd.Timestamp("2021-06-21")


@dataclass(frozen=True)
class DiagnosticoSaucio:
    """Resumen trazable del empalme CAR/Enlaza para el caudal de Saucío."""

    dias_car: int
    dias_enlaza: int
    dias_totales: int
    corte_empalme: pd.Timestamp
    huecos_totales: int
    huecos_tramo_car: int
    huecos_tramo_enlaza: int
    dia_sospechoso: pd.Timestamp
    dia_sospechoso_car_m3s: float | None
    dia_sospechoso_enlaza_m3s: float | None
    dia_sospechoso_fuente_usada: str


def _cargar_serie_car(ruta_csv: str | Path) -> pd.Series:
    """Carga la serie diaria de caudales medios de Saucío desde el CSV de la CAR."""
    ruta = Path(ruta_csv)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el CSV de estaciones de la CAR en: {ruta}")

    crudo = pd.read_csv(ruta)
    car = crudo[
        (crudo["ESTACION"] == CAR_ESTACION)
        & (crudo["ESCALA"] == CAR_ESCALA)
        & (crudo["TIPO"] == CAR_TIPO)
    ].copy()
    car["FECHA"] = pd.to_datetime(car["FECHA"], errors="coerce")
    car["DATO"] = pd.to_numeric(car["DATO"], errors="coerce")
    car = car.dropna(subset=["FECHA"])
    serie = car.set_index("FECHA")["DATO"].sort_index()
    serie = serie[~serie.index.duplicated(keep="first")]
    serie.name = "caudal_m3s"
    return serie


def _cargar_serie_enlaza(ruta_excel: str | Path) -> pd.Series:
    """Carga la serie diaria de caudal de Saucío desde la hoja oculta de Enlaza."""
    ruta = Path(ruta_excel)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el Excel de Enlaza en: {ruta}")

    crudo = pd.read_excel(ruta, sheet_name=ENLAZA_SHEET)
    if ENLAZA_COLUMNA_CAUDAL not in crudo.columns:
        raise ValueError(
            f"No se encontró la columna '{ENLAZA_COLUMNA_CAUDAL}' en la hoja '{ENLAZA_SHEET}'"
        )
    d = crudo[["FECHA", ENLAZA_COLUMNA_CAUDAL]].copy()
    d["FECHA"] = pd.to_datetime(d["FECHA"], errors="coerce")
    d[ENLAZA_COLUMNA_CAUDAL] = pd.to_numeric(d[ENLAZA_COLUMNA_CAUDAL], errors="coerce")
    d = d.dropna(subset=["FECHA"])
    serie = d.set_index("FECHA")[ENLAZA_COLUMNA_CAUDAL].sort_index()
    serie = serie[~serie.index.duplicated(keep="first")]
    serie.name = "caudal_m3s"
    return serie


def cargar_saucio(
    ruta_car_csv: str | Path = DEFAULT_CAR_ESTACIONES_CSV,
    ruta_enlaza_xlsx: str | Path = DEFAULT_TOMINE_XLSX,
    ventana_inicio: str = VENTANA_INICIO,
    ventana_fin: str = VENTANA_FIN,
    corte_empalme: str = CORTE_EMPALME,
) -> tuple[pd.DataFrame, DiagnosticoSaucio]:
    """Carga y empalma el caudal diario de Saucío (CAR + Enlaza) para la ventana dada.

    Devuelve un DataFrame con DatetimeIndex diario continuo (columnas
    ``caudal_m3s`` y ``fuente_caudal``) sobre la ventana solicitada, más un
    diagnóstico del empalme. Los huecos NO se rellenan (quedan como NaN,
    reportados en el diagnóstico).
    """
    car = _cargar_serie_car(ruta_car_csv)
    enlaza = _cargar_serie_enlaza(ruta_enlaza_xlsx)

    inicio = pd.Timestamp(ventana_inicio)
    fin = pd.Timestamp(ventana_fin)
    corte = pd.Timestamp(corte_empalme)

    tramo_car = car.loc[inicio : corte - pd.Timedelta(days=1)]
    tramo_enlaza = enlaza.loc[corte:fin]

    combinado = pd.concat([tramo_car, tramo_enlaza])
    fuente = pd.Series(
        ["CAR"] * len(tramo_car) + ["Enlaza"] * len(tramo_enlaza),
        index=combinado.index,
        name="fuente_caudal",
    )

    full_index = pd.date_range(inicio, fin, freq="D")
    combinado = combinado.reindex(full_index)
    combinado.index.name = "fecha"
    fuente = fuente.reindex(full_index)
    fuente.index.name = "fecha"

    resultado = pd.DataFrame({"caudal_m3s": combinado, "fuente_caudal": fuente})

    huecos_car = int(tramo_car.reindex(pd.date_range(inicio, corte - pd.Timedelta(days=1), freq="D")).isna().sum())
    huecos_enlaza = int(tramo_enlaza.reindex(pd.date_range(corte, fin, freq="D")).isna().sum())

    dia_sosp_car = car.get(DIA_SOSPECHOSO)
    dia_sosp_enlaza = enlaza.get(DIA_SOSPECHOSO)
    fuente_usada = "CAR" if DIA_SOSPECHOSO < corte else "Enlaza"

    diagnostico = DiagnosticoSaucio(
        dias_car=int(len(tramo_car)),
        dias_enlaza=int(len(tramo_enlaza)),
        dias_totales=len(resultado),
        corte_empalme=corte,
        huecos_totales=int(resultado["caudal_m3s"].isna().sum()),
        huecos_tramo_car=huecos_car,
        huecos_tramo_enlaza=huecos_enlaza,
        dia_sospechoso=DIA_SOSPECHOSO,
        dia_sospechoso_car_m3s=float(dia_sosp_car) if dia_sosp_car is not None else None,
        dia_sospechoso_enlaza_m3s=float(dia_sosp_enlaza) if dia_sosp_enlaza is not None else None,
        dia_sospechoso_fuente_usada=fuente_usada,
    )
    return resultado, diagnostico


def resumen_saucio(df: pd.DataFrame, diagnostico: DiagnosticoSaucio) -> str:
    """Resumen legible del empalme de Saucío para verificación rápida."""
    lineas = [
        "=== Caudal de Saucío (CAR + Enlaza, empalmado) ===",
        f"Rango: {df.index.min().date()} -> {df.index.max().date()} ({diagnostico.dias_totales} días)",
        f"Corte de empalme: {diagnostico.corte_empalme.date()} "
        f"(CAR hasta el día anterior, Enlaza desde ahí)",
        f"Días de fuente CAR: {diagnostico.dias_car}",
        f"Días de fuente Enlaza: {diagnostico.dias_enlaza}",
        f"Huecos totales (no rellenados): {diagnostico.huecos_totales} "
        f"(CAR: {diagnostico.huecos_tramo_car}, Enlaza: {diagnostico.huecos_tramo_enlaza})",
        f"Día sospechoso {diagnostico.dia_sospechoso.date()} (NO corregido): "
        f"CAR={diagnostico.dia_sospechoso_car_m3s} m³/s, "
        f"Enlaza={diagnostico.dia_sospechoso_enlaza_m3s} m³/s "
        f"(se usó {diagnostico.dia_sospechoso_fuente_usada})",
        "",
        "Estadísticas básicas:",
        df["caudal_m3s"].describe().round(3).to_string(),
    ]
    return "\n".join(lineas)

"""Cargador de la serie operativa diaria de Tominé (Excel de Enlaza).

Análogo al cargador de la CAR para Neusa/Sisga: lee el Excel crudo, lo recorta a la
ventana de análisis, corrige los problemas detectados en la auditoría y traduce las
columnas al contrato de datos del proyecto (``data_contracts.ESQUEMA_EMBALSE``),
añadiendo la columna de bombeo.

Correcciones aplicadas (auditoría 2015-2025)
--------------------------------------------
1. Fechas duplicadas exactas (2021-11-15 y 2023-01-01): se conserva una sola fila.
2. Detección de saltos anómalos de volumen: se marcan los días cuyo cambio diario
   supera ``volume_jump_fraction`` de la capacidad total y se corrigen por
   interpolación temporal. El único salto conocido (2012-01-01) queda fuera de la
   ventana 2015-2025; la detección se conserva por robustez.
3. NaN de bombeo: se tratan como cero (el bombeo es un evento poco frecuente).

Notas de datos
--------------
- Convención de volumen: la serie de Enlaza (columna 'Aforo (Volumen Mm3)') ya viene en
  VOLUMEN ÚTIL (0 en la cota mínima operativa), que es la convención del proyecto
  (ver data_contracts.embalses.CONVENCION_VOLUMEN). Por eso NO se le resta el volumen
  muerto, a diferencia de Neusa/Sisga (CAR, en total). No requiere conversión.
- Evaporación: el Excel de Enlaza NO reporta evaporación para Tominé. Se toma de la
  serie ERA5-Land (flujo de calor latente), archivo
  ``info_CAR/Serie_Evaporacion_Tomine_2010_2025_CORRECTA.csv``, validada en magnitud
  (~3.19 mm/día, ~1164 mm/año, coherente con la evaporación medida de los embalses
  vecinos de la CAR). Se alinea a la ventana diaria del loader.
- Bombeo: 'Bombeo (m3)' es el volumen diario bombeado desde el canal Achury hacia el
  embalse. Se convierte de m³/día a Mm³/día (``bombeo_mm3``) para ser coherente con
  ``volumen_mm3``. Es un aporte (entrada), no una descarga.
- Ceros de descarga: en 2015-2025 son días reales sin descarga; NO se interpolan.

ADVERTENCIA — el balance hídrico de Tominé todavía NO debe ejecutarse
--------------------------------------------------------------------
Aunque la evaporación ya es real (ERA5), quedan DOS pendientes antes de correr el
balance inverso para Tominé, so pena de obtener afluencias falsas:
  1. Incluir el término de BOMBEO en el balance (Tominé recibe agua bombeada desde el
     canal Achury; el balance actual de Neusa/Sisga no contempla ese aporte).
  2. Confirmar con el asesor la CONVENCIÓN DE VOLUMEN (útil vs total): la serie de
     Tominé está en "útil" y difiere de embalses.py/curvas.py ("total") en el volumen
     muerto (9.90 Mm³). Ver notas de memoria del proyecto.
Este loader solo prepara y valida la serie; NO ejecuta el balance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pbcrl.data_contracts import EMBALSES, validar_dataframe_embalse

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_TOMINE_XLSX = ROOT_DIR / "info_CAR" / "datos operativos Tomine_Enlaza.xlsx"
DEFAULT_TOMINE_EVAP_CSV = ROOT_DIR / "info_CAR" / "Serie_Evaporacion_Tomine_2010_2025_CORRECTA.csv"
TOMINE_SHEET = "Tomine"
EVAP_FUENTE = "ERA5-Land (flujo de calor latente)"

VENTANA_INICIO = "2015-01-01"
VENTANA_FIN = "2025-12-31"

DEFAULT_VOLUME_JUMP_FRACTION = 0.10

# Excel de Enlaza -> contrato del proyecto. Las llaves deben coincidir exactamente
# con los encabezados de la hoja 'Tomine' (ojo con los espacios dobles).
_COLUMNAS: dict[str, str] = {
    "FECHA": "fecha",
    "COTA A LAS 23:30 HORAS (msnm)": "cota_m",
    "Aforo (Volumen Mm3)": "volumen_mm3",
    "Descarga promedio día (m3/s)": "descarga_m3s",
    "Bombeo (m3)": "bombeo_m3",
    "Lluvias totales  Pluviómetro (mm)": "precipitacion_mm",
}

_COLUMNAS_NUMERICAS = ["cota_m", "volumen_mm3", "descarga_m3s", "bombeo_m3", "precipitacion_mm"]

# Columnas del resultado: contrato + bombeo convertido a Mm³/día.
_COLUMNAS_SALIDA = [
    "cota_m",
    "volumen_mm3",
    "descarga_m3s",
    "precipitacion_mm",
    "evaporacion_mm",
    "bombeo_mm3",
]


@dataclass(frozen=True)
class DiagnosticoTomine:
    """Resumen trazable de la limpieza aplicada a la serie de Tominé."""

    filas_ventana_crudas: int
    fechas_duplicadas_eliminadas: int
    saltos_volumen_corregidos: int
    umbral_salto_volumen_mm3: float
    nan_bombeo_a_cero: int
    dias_descarga_cero: int
    dias_resultado: int
    rango_inicio: pd.Timestamp
    rango_fin: pd.Timestamp
    evaporacion_fuente: str = ""
    evaporacion_dias_rellenados: int = 0


def _corregir_saltos_volumen(volumen: pd.Series, umbral_mm3: float) -> tuple[pd.Series, int]:
    """Marca los saltos diarios de volumen > umbral y los repone por interpolación."""
    delta = volumen.diff().abs()
    mask = delta > umbral_mm3
    if not mask.any():
        return volumen, 0
    corregido = volumen.mask(mask)
    corregido = corregido.interpolate(method="time", limit_direction="both")
    return corregido, int(mask.sum())


def _cargar_evaporacion_era5(ruta_csv: str | Path, indice: pd.DatetimeIndex) -> tuple[pd.Series, int]:
    """Carga la evaporación ERA5 (mm/día) y la alinea al índice diario del loader.

    El CSV cubre 2010-2025 con columnas 'fecha' y 'evap_mm'. Se reindexa a ``indice``
    y se rellena cualquier día faltante por interpolación temporal (y ffill/bfill en los
    bordes; p.ej. falta 2025-12-31). Devuelve la serie alineada y cuántos días se
    rellenaron dentro de la ventana.
    """
    ruta = Path(ruta_csv)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el CSV de evaporación ERA5 en: {ruta}")

    evap = pd.read_csv(ruta)
    faltantes = [c for c in ("fecha", "evap_mm") if c not in evap.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV de evaporación: {faltantes}")

    evap["fecha"] = pd.to_datetime(evap["fecha"], errors="coerce")
    serie = (
        evap.dropna(subset=["fecha"])
        .drop_duplicates(subset=["fecha"], keep="first")
        .set_index("fecha")["evap_mm"]
        .sort_index()
        .astype("float64")
    )
    alineada = serie.reindex(indice)
    dias_rellenados = int(alineada.isna().sum())
    if dias_rellenados:
        alineada = alineada.interpolate(method="time", limit_direction="both")
        alineada = alineada.ffill().bfill()
    alineada.index.name = "fecha"
    return alineada, dias_rellenados


def cargar_tomine(
    ruta_excel: str | Path = DEFAULT_TOMINE_XLSX,
    ventana_inicio: str = VENTANA_INICIO,
    ventana_fin: str = VENTANA_FIN,
    volume_jump_fraction: float = DEFAULT_VOLUME_JUMP_FRACTION,
    ruta_evaporacion: str | Path = DEFAULT_TOMINE_EVAP_CSV,
    validar: bool = True,
) -> tuple[pd.DataFrame, DiagnosticoTomine]:
    """Carga y limpia la serie operativa diaria de Tominé para la ventana dada.

    Devuelve un DataFrame con DatetimeIndex diario continuo que cumple el contrato de
    datos del proyecto (más la columna ``bombeo_mm3``), junto con un diagnóstico de la
    limpieza aplicada.
    """
    ruta = Path(ruta_excel)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el Excel de Tominé en: {ruta}")

    crudo = pd.read_excel(ruta, sheet_name=TOMINE_SHEET)
    faltantes = [c for c in _COLUMNAS if c not in crudo.columns]
    if faltantes:
        raise ValueError(
            f"Faltan columnas esperadas en la hoja '{TOMINE_SHEET}': {faltantes}"
        )

    df = crudo[list(_COLUMNAS)].rename(columns=_COLUMNAS)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])
    for col in _COLUMNAS_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    inicio = pd.Timestamp(ventana_inicio)
    fin = pd.Timestamp(ventana_fin)
    df = df[(df["fecha"] >= inicio) & (df["fecha"] <= fin)].copy()
    filas_ventana_crudas = len(df)

    # 1. Duplicados exactos de fecha: quita filas idénticas y, por robustez, cualquier
    #    fecha repetida remanente conservando la primera aparición.
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["fecha"], keep="first")
    fechas_duplicadas_eliminadas = filas_ventana_crudas - len(df)

    df = df.set_index("fecha").sort_index()

    # Índice diario continuo sobre el rango observado (defensivo frente a huecos).
    idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(idx)
    df.index.name = "fecha"

    huecos = df[["cota_m", "volumen_mm3", "descarga_m3s", "precipitacion_mm"]].isna().any(axis=1)
    if huecos.any():
        fechas_hueco = df.index[huecos].strftime("%Y-%m-%d").tolist()
        raise ValueError(
            f"La serie de Tominé tiene {int(huecos.sum())} día(s) sin dato tras el "
            f"recorte a la ventana: {fechas_hueco[:10]}"
            + (" ..." if len(fechas_hueco) > 10 else "")
        )

    # 2. Saltos anómalos de volumen (0 esperados en 2015-2025).
    umbral_salto = volume_jump_fraction * EMBALSES["Tomine"].capacidad_max_mm3
    df["volumen_mm3"], saltos = _corregir_saltos_volumen(df["volumen_mm3"], umbral_salto)

    # 3. NaN de bombeo -> 0 (evento poco frecuente); convertir m³/día a Mm³/día.
    nan_bombeo = int(df["bombeo_m3"].isna().sum())
    df["bombeo_m3"] = df["bombeo_m3"].fillna(0.0)
    df["bombeo_mm3"] = df["bombeo_m3"] / 1e6

    # Evaporación real ERA5-Land alineada a la ventana diaria (reemplaza el marcador 0.0).
    # NOTA: tener evaporación real NO habilita correr el balance de Tominé; ver la
    # ADVERTENCIA del docstring (faltan el término de bombeo y la convención de volumen).
    df["evaporacion_mm"], evap_dias_rellenados = _cargar_evaporacion_era5(ruta_evaporacion, df.index)

    resultado = df[_COLUMNAS_SALIDA].astype("float64")

    dias_descarga_cero = int((resultado["descarga_m3s"] == 0).sum())

    if validar:
        # El contrato valida solo las columnas del esquema; bombeo_mm3 es adicional.
        validar_dataframe_embalse(resultado, nombre_embalse="Tomine")

    diagnostico = DiagnosticoTomine(
        filas_ventana_crudas=filas_ventana_crudas,
        fechas_duplicadas_eliminadas=fechas_duplicadas_eliminadas,
        saltos_volumen_corregidos=saltos,
        umbral_salto_volumen_mm3=float(umbral_salto),
        nan_bombeo_a_cero=nan_bombeo,
        dias_descarga_cero=dias_descarga_cero,
        dias_resultado=len(resultado),
        rango_inicio=resultado.index.min(),
        rango_fin=resultado.index.max(),
        evaporacion_fuente=EVAP_FUENTE,
        evaporacion_dias_rellenados=evap_dias_rellenados,
    )
    return resultado, diagnostico


def resumen_tomine(df: pd.DataFrame, diagnostico: DiagnosticoTomine) -> str:
    """Devuelve un resumen legible de la serie cargada para verificación rápida."""
    lineas = [
        "=== Serie operativa de Tominé (2015-2025) ===",
        f"Rango: {diagnostico.rango_inicio.date()} -> {diagnostico.rango_fin.date()}",
        f"Días (índice diario continuo): {diagnostico.dias_resultado}",
        f"Filas crudas en ventana: {diagnostico.filas_ventana_crudas}",
        f"Fechas duplicadas eliminadas: {diagnostico.fechas_duplicadas_eliminadas}",
        f"Saltos de volumen corregidos: {diagnostico.saltos_volumen_corregidos} "
        f"(umbral {diagnostico.umbral_salto_volumen_mm3:.2f} Mm³)",
        f"NaN de bombeo puestos a cero: {diagnostico.nan_bombeo_a_cero}",
        f"Días con descarga = 0 (reales, sin interpolar): {diagnostico.dias_descarga_cero}",
        f"Evaporación: {diagnostico.evaporacion_fuente} "
        f"(días rellenados en bordes: {diagnostico.evaporacion_dias_rellenados})",
        "",
        "Estadísticas básicas:",
        df.describe().round(3).to_string(),
        "",
        "ADVERTENCIA: el balance de Tominé NO debe correrse aún (falta el término de "
        "bombeo y confirmar la convención de volumen). Ver docstring del módulo.",
    ]
    return "\n".join(lineas)

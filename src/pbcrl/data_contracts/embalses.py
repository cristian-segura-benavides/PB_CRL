"""
Parámetros físicos de cada embalse.

Fuentes:
  - Neusa : Manual de Operación Embalse del Neusa, CAR (versión vigente).
  - Sisga : Manual de Operación Embalse del Sisga, CAR (versión vigente).
  - Tominé: batimetría oficial 2021, GEB/Enel. Curva completa en data_contracts/curvas.py.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ParametrosEmbalse:
    """Parámetros físicos fijos de un embalse.

    Atributos
    ---------
    nombre : str
        Identificador del embalse.
    capacidad_max_mm3 : float
        Capacidad máxima de almacenamiento [Mm³].
    capacidad_min_mm3 : float
        Volumen del embalse muerto (mínimo operativo) [Mm³].
    area_espejo_km2 : float
        Área del espejo de agua a nivel máximo [km²].
        Se usa para convertir láminas de precipitación/evaporación a volumen.
    cota_min_m : float
        Cota mínima operativa (nivel mínimo) [m.s.n.m.].
    cota_max_m : float
        Cota máxima operativa (nivel de aguas máximas / rebose) [m.s.n.m.].
    descarga_max_m3s : float
        Descarga máxima por compuertas/torre de toma [m³/s].
    """

    nombre: str
    capacidad_max_mm3: float
    capacidad_min_mm3: float
    area_espejo_km2: float
    cota_min_m: float
    cota_max_m: float
    descarga_max_m3s: float


# ---------------------------------------------------------------------------
# Parámetros de cada embalse
# ---------------------------------------------------------------------------

EMBALSES: dict[str, ParametrosEmbalse] = {
    # Fuente: Manual de Operación Embalse del Neusa, CAR.
    "Neusa": ParametrosEmbalse(
        nombre="Neusa",
        capacidad_max_mm3=102.0,    # volumen máximo total ~102.7 Mm³ según manual
        capacidad_min_mm3=7.0,      # volumen muerto según manual
        area_espejo_km2=8.34,       # 834 ha según manual
        cota_min_m=2950.0,          # nivel mínimo de operación
        cota_max_m=2974.5,          # nivel de aguas máximas
        descarga_max_m3s=16.0,      # capacidad de la torre de toma según manual
    ),
    # Fuente: Manual de Operación Embalse del Sisga, CAR.
    "Sisga": ParametrosEmbalse(
        nombre="Sisga",
        capacidad_max_mm3=90.1,     # volumen útil según manual
        capacidad_min_mm3=4.2,      # volumen del embalse muerto según manual
        area_espejo_km2=6.37,       # 637 ha (área inundable; relevante para evaporación)
        cota_min_m=2644.63,         # nivel mínimo de operación
        cota_max_m=2670.35,         # nivel de aguas máximas
        descarga_max_m3s=15.0,      # sin dato documentado que lo contradiga
    ),
    # Fuente: batimetría oficial Tominé 2021, GEB/Enel (curva cota-volumen en data_contracts/curvas.py).
    # cota_min_m y cota_max_m son puntos ancla exactos de la curva batimétrica oficial:
    #   9.90 Mm³  → 2566.63 m  (volumen muerto oficial; punto ancla de la tabla)
    #   699.43 Mm³ → 2598.38 m (volumen total oficial / rebosadero; punto ancla de la tabla)
    # Estos valores se usan para la penalización por proximidad al mínimo; la conversión
    # cota↔volumen usa directamente la curva (no la interpolación lineal entre estos extremos).
    "Tomine": ParametrosEmbalse(
        nombre="Tomine",
        capacidad_max_mm3=699.43,   # volumen total oficial, batimetría 2021
        capacidad_min_mm3=9.90,     # volumen muerto oficial, batimetría 2021
        area_espejo_km2=38.0,       # ajustado de 39 a 38, coherente con procesamiento satelital
        cota_min_m=2566.63,         # derivado de 9.90 Mm³ (volumen muerto oficial)
        cota_max_m=2598.38,         # derivado de 699.43 Mm³ (volumen total oficial)
        descarga_max_m3s=40.0,      # PENDIENTE verificar con fuente oficial
    ),
}

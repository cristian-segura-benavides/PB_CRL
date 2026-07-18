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
# Convención de volumen del proyecto (punto central y reversible)
# ---------------------------------------------------------------------------
# DECISIÓN (2026-07-17, CONFIRMADA CON EL ASESOR 2026-07-18, definitiva): el proyecto
# trabaja en VOLUMEN ÚTIL — el cero es el volumen muerto (nivel mínimo operativo) y el
# volumen se cuenta desde el mínimo hacia arriba.
#
# Respaldo de fuentes oficiales: Enlaza reporta Tominé en útil (su curva y serie
# empiezan en 0 a la cota mínima); la CAR razona en útil para la operación; y el
# volumen útil está oficialmente definido para estos embalses. La decisión sigue siendo
# REVERSIBLE en el código (aunque ya no está pendiente de validación): para volver a
# "total", cambie CONVENCION_VOLUMEN a "total" (el loader deja de restar el volumen
# muerto) y revierta los parámetros/curvas a la convención total; no hay que rehacer
# el código.
CONVENCION_VOLUMEN = "util"  # "util" | "total"

# Volumen muerto por embalse [Mm³] (almacenamiento bajo el nivel mínimo operativo).
# Se usa para convertir series reportadas en volumen TOTAL a volumen ÚTIL restándolo.
#   - Neusa, Sisga (datos CAR): vienen en TOTAL -> se convierten restando el muerto.
#   - Tominé (datos Enlaza): ya viene en ÚTIL -> NO se resta.
# Fuentes: Neusa 7.0 (Manual de Operación CAR); Sisga 4.7 (batimetría CAR 2004, 5% de
#          la capacidad 94.3 Mm³); Tominé 9.90 (batimetría oficial 2021, GEB/Enel).
VOLUMEN_MUERTO_MM3: dict[str, float] = {
    "Neusa": 7.0,
    "Sisga": 4.7,
    "Tomine": 9.90,
}


# ---------------------------------------------------------------------------
# Parámetros de cada embalse (en la convención declarada arriba: VOLUMEN ÚTIL)
# ---------------------------------------------------------------------------

EMBALSES: dict[str, ParametrosEmbalse] = {
    # Fuente: Manual de Operación Embalse del Neusa, CAR. Convención: volumen ÚTIL.
    "Neusa": ParametrosEmbalse(
        nombre="Neusa",
        capacidad_max_mm3=95.3,     # capacidad útil ≈ 102.3 total − 7.0 muerto (manual CAR)
        capacidad_min_mm3=0.0,      # mínimo útil = 0 (el volumen muerto es el cero)
        area_espejo_km2=8.34,       # 834 ha según manual
        cota_min_m=2950.0,          # nivel mínimo de operación
        cota_max_m=2974.5,          # nivel de aguas máximas
        descarga_max_m3s=16.0,      # capacidad de la torre de toma según manual
    ),
    # Fuente: Manual de Operación Embalse del Sisga, CAR. Convención: volumen ÚTIL.
    "Sisga": ParametrosEmbalse(
        nombre="Sisga",
        capacidad_max_mm3=89.6,     # capacidad útil ≈ 94.3 total − 4.7 muerto (batimetría CAR 2004)
        capacidad_min_mm3=0.0,      # mínimo útil = 0 (el volumen muerto es el cero)
        area_espejo_km2=6.37,       # 637 ha (área inundable; relevante para evaporación)
        cota_min_m=2644.63,         # nivel mínimo de operación
        cota_max_m=2670.35,         # nivel de aguas máximas
        descarga_max_m3s=15.0,      # sin dato documentado que lo contradiga
    ),
    # Fuente: batimetría oficial Tominé 2021, GEB/Enel (curva cota-volumen en data_contracts/curvas.py).
    # Convención: volumen ÚTIL (la serie de Enlaza ya viene en útil). Puntos ancla de la
    # curva (tras referenciar el datum al volumen muerto de 9.90 Mm³):
    #     0.00 Mm³  → 2566.63 m  (mínimo operativo = cero útil)
    #   689.53 Mm³ → 2598.38 m  (capacidad útil = 699.43 total − 9.90 muerto)
    # cota_min_m/cota_max_m se usan en la penalización por proximidad al mínimo; la
    # conversión cota↔volumen usa directamente la curva.
    "Tomine": ParametrosEmbalse(
        nombre="Tomine",
        capacidad_max_mm3=689.53,   # capacidad útil = 699.43 total − 9.90 muerto (batimetría 2021)
        capacidad_min_mm3=0.0,      # mínimo útil = 0 (el volumen muerto es el cero)
        area_espejo_km2=38.0,       # ajustado de 39 a 38, coherente con procesamiento satelital
        cota_min_m=2566.63,         # nivel mínimo operativo (cero útil)
        cota_max_m=2598.38,         # nivel de aguas máximas (capacidad útil)
        descarga_max_m3s=40.0,      # PENDIENTE verificar con fuente oficial
    ),
}

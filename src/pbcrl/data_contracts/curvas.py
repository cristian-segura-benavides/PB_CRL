"""
Curvas cota-volumen para cada embalse.

La relación cota-volumen de un embalse real NO es lineal: el área del espejo
cambia con la altura, por lo que la curva depende de la batimetría del vaso.

Diseño
------
- Cada embalse puede tener una `CurvaCotaVolumen` con sus puntos batimétricos reales.
- Si no tiene curva, las funciones de módulo usan el fallback lineal provisional
  (interpolación entre cota_min/cota_max de ParametrosEmbalse).
- El registro `CURVAS` es el único lugar donde se declara qué embalses tienen curva real.

Estado actual (2026-07-01)
--------------------------
  Tominé : curva real — batimetría oficial 2021, GEB/Enel.
  Neusa  : PENDIENTE — curva de la CAR no disponible aún.
  Sisga  : PENDIENTE — curva de la CAR no disponible aún.

Fuera de rango
--------------
`numpy.interp` acota al valor extremo más cercano cuando el argumento cae fuera
del rango de la tabla. Esta decisión es conservadora: evita extrapolaciones físicamente
dudosas y es coherente con el comportamiento de los embalses (no pueden llenarse más
allá de la capacidad máxima ni vaciarse más allá del mínimo operativo).
"""
from __future__ import annotations

import numpy as np

from pbcrl.data_contracts.embalses import ParametrosEmbalse


# ---------------------------------------------------------------------------
# Clase de curva
# ---------------------------------------------------------------------------

class CurvaCotaVolumen:
    """Tabla de puntos (cota_m, volumen_mm3) para interpolación bidireccional.

    Los puntos deben estar ordenados por cota ascendente (y por tanto también
    por volumen ascendente, dado que la curva es monótona creciente).

    Parámetros
    ----------
    nombre : str
        Identificador del embalse (solo informativo).
    cotas_m : sequence[float]
        Cotas en m.s.n.m., ordenadas de menor a mayor.
    volumenes_mm3 : sequence[float]
        Volúmenes en Mm³, correspondientes a cada cota, también ascendentes.
    """

    def __init__(
        self,
        nombre: str,
        cotas_m: list[float],
        volumenes_mm3: list[float],
    ) -> None:
        self.nombre = nombre
        self._cotas = np.asarray(cotas_m, dtype=float)
        self._volumenes = np.asarray(volumenes_mm3, dtype=float)

        if len(self._cotas) != len(self._volumenes):
            raise ValueError(
                f"[{nombre}] cotas_m y volumenes_mm3 deben tener la misma longitud."
            )
        if not np.all(np.diff(self._cotas) > 0):
            raise ValueError(f"[{nombre}] cotas_m debe ser estrictamente creciente.")
        if not np.all(np.diff(self._volumenes) > 0):
            raise ValueError(f"[{nombre}] volumenes_mm3 debe ser estrictamente creciente.")

    def cota_a_volumen(self, cota_m: float) -> float:
        """Interpola el volumen [Mm³] a partir de una cota [m.s.n.m.].

        Acota al rango de la tabla si el valor cae fuera (sin extrapolación).
        """
        return float(np.interp(cota_m, self._cotas, self._volumenes))

    def volumen_a_cota(self, volumen_mm3: float) -> float:
        """Interpola la cota [m.s.n.m.] a partir de un volumen [Mm³].

        Acota al rango de la tabla si el valor cae fuera (sin extrapolación).
        """
        return float(np.interp(volumen_mm3, self._volumenes, self._cotas))

    @property
    def cota_min_m(self) -> float:
        return float(self._cotas[0])

    @property
    def cota_max_m(self) -> float:
        return float(self._cotas[-1])

    @property
    def volumen_min_mm3(self) -> float:
        return float(self._volumenes[0])

    @property
    def volumen_max_mm3(self) -> float:
        return float(self._volumenes[-1])

    def __len__(self) -> int:
        return len(self._cotas)


# ---------------------------------------------------------------------------
# Curva de Tominé
# Fuente: batimetría oficial Tominé 2021, GEB/Enel.
#         Extremos anclados a volumen muerto (30 Mm³) y nivel máximo (690 Mm³).
# ---------------------------------------------------------------------------

CURVA_TOMINE = CurvaCotaVolumen(
    nombre="Tomine",
    cotas_m=[
        2559.254, 2560.256, 2561.392, 2563.062, 2564.532,
        2566.136, 2566.630, 2567.673, 2569.009, 2570.078,
        2571.281, 2572.483, 2573.552, 2574.421, 2575.290,
        2576.158, 2577.294, 2578.363, 2579.365, 2580.234,
        2581.102, 2581.904, 2582.840, 2583.909, 2584.911,
        2585.713, 2586.448, 2587.116, 2587.851, 2588.452,
        2589.187, 2589.855, 2590.457, 2591.125, 2591.860,
        2592.461, 2593.129, 2593.597, 2594.198, 2594.800,
        2595.401, 2596.069, 2598.380,
    ],
    volumenes_mm3=[
          0.759,   1.533,   2.311,   3.850,   4.636,
          9.171,   9.900,  17.449,  26.471,  34.737,
         46.003,  59.516,  72.277,  82.786,  95.541,
        109.795, 129.299, 147.304, 165.306, 183.305,
        201.305, 218.553, 238.052, 263.547, 286.044,
        307.038, 325.034, 343.028, 362.522, 380.514,
        403.005, 422.497, 439.740, 460.731, 481.723,
        503.461, 522.953, 540.193, 561.182, 579.175,
        600.913, 623.401, 699.430,
    ],
)

# ---------------------------------------------------------------------------
# Registro: nombre del embalse → curva real
# Neusa y Sisga: PENDIENTE — curva CAR no disponible (usar fallback lineal)
# ---------------------------------------------------------------------------

CURVAS: dict[str, CurvaCotaVolumen] = {
    "Tomine": CURVA_TOMINE,
    # "Neusa":  CURVA_NEUSA,   # PENDIENTE: reemplazar cuando llegue la curva de la CAR
    # "Sisga":  CURVA_SISGA,   # PENDIENTE: reemplazar cuando llegue la curva de la CAR
}


# ---------------------------------------------------------------------------
# Funciones de despacho: usan la curva real si existe, fallback lineal si no
# ---------------------------------------------------------------------------

def volumen_a_cota(
    volumen_mm3: float,
    nombre: str,
    params: ParametrosEmbalse,
) -> float:
    """Convierte volumen a cota para el embalse indicado.

    Si el embalse tiene curva batimétrica real en el registro `CURVAS`,
    usa interpolación sobre esa tabla.  Si no (Neusa, Sisga), usa el
    fallback lineal provisional basado en cota_min/cota_max de ParametrosEmbalse.

    Fallback lineal PROVISIONAL para Neusa y Sisga:
    pendiente de reemplazar cuando lleguen las curvas de la CAR.

    Parámetros
    ----------
    volumen_mm3 : float   Volumen [Mm³].
    nombre : str          Nombre del embalse ('Neusa', 'Sisga', 'Tomine').
    params : ParametrosEmbalse

    Retorna
    -------
    float   Cota [m.s.n.m.].
    """
    curva = CURVAS.get(nombre)
    if curva is not None:
        return curva.volumen_a_cota(volumen_mm3)

    # Fallback lineal (PROVISIONAL — Neusa y Sisga sin curva batimétrica)
    rango_vol = params.capacidad_max_mm3 - params.capacidad_min_mm3
    fraccion = (volumen_mm3 - params.capacidad_min_mm3) / rango_vol
    fraccion = max(0.0, min(1.0, fraccion))
    return params.cota_min_m + fraccion * (params.cota_max_m - params.cota_min_m)


def cota_a_volumen(
    cota_m: float,
    nombre: str,
    params: ParametrosEmbalse,
) -> float:
    """Convierte cota a volumen para el embalse indicado.

    Misma lógica de despacho que `volumen_a_cota`: curva real si existe,
    fallback lineal provisional si no.

    Fallback lineal PROVISIONAL para Neusa y Sisga:
    pendiente de reemplazar cuando lleguen las curvas de la CAR.

    Parámetros
    ----------
    cota_m : float        Cota [m.s.n.m.].
    nombre : str          Nombre del embalse.
    params : ParametrosEmbalse

    Retorna
    -------
    float   Volumen [Mm³].
    """
    curva = CURVAS.get(nombre)
    if curva is not None:
        return curva.cota_a_volumen(cota_m)

    # Fallback lineal (PROVISIONAL — Neusa y Sisga sin curva batimétrica)
    rango_cota = params.cota_max_m - params.cota_min_m
    fraccion = (cota_m - params.cota_min_m) / rango_cota
    fraccion = max(0.0, min(1.0, fraccion))
    return params.capacidad_min_mm3 + fraccion * (params.capacidad_max_mm3 - params.capacidad_min_mm3)

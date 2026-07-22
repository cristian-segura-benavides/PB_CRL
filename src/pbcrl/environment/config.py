"""
Configuración del entorno de simulación.

Todos los parámetros están en un único lugar para facilitar su ajuste.
Los valores marcados como PROVISIONAL deben validarse con el asesor de tesis
antes de usarlos en experimentos formales.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pbcrl.data_contracts.caudal_ecologico import q_eco_m3s as _calcular_q_eco_vmf


@dataclass
class ConfigEntorno:
    """Parámetros configurables del entorno de simulación.

    Atributos
    ---------
    calcular_q_eco_m3s : Callable[[int], float]
        Función que devuelve el caudal ecológico mínimo requerido en el punto
        de control El Sol [m³/s], dado el mes calendario (1-12) del paso
        actual. Por defecto usa el umbral VMF fijado en
        `data_contracts.caudal_ecologico.q_eco_m3s` (ver ese módulo para el
        método, el origen de los datos y las salvedades documentadas).
        Configurable sin tocar el resto del entorno: para un umbral fijo, usar
        `data_contracts.caudal_ecologico.umbral_fijo_m3s(valor)`; para otro
        método (normativo, Q95, Tennant), pasar cualquier `Callable[[int],
        float]` equivalente.

    sisga_descenso_umbral_cm : float
        Rata de descenso de nivel del Sisga sin penalización [cm/día].
        Fuente: Manual de Operación Embalse del Sisga, CAR.

    sisga_descenso_maximo_cm : float
        Rata de descenso a partir de la cual la penalización del Sisga es máxima [cm/día].
        PROVISIONAL: umbral superior extraído del manual CAR.

    peso_sisga : float
        Peso de la penalización del Sisga. PROVISIONAL.

    peso_neusa : float
        Peso de la penalización del Neusa. PROVISIONAL.

    peso_tomine : float
        Peso de la penalización del Tominé. PROVISIONAL.
    """

    # --- Caudal ecológico ---
    calcular_q_eco_m3s: Callable[[int], float] = _calcular_q_eco_vmf

    # --- Penalización Sisga: rata de descenso de nivel ---
    # Fuente umbrales: Manual de Operación Embalse del Sisga, CAR.
    sisga_descenso_umbral_cm: float = 15.0    # descenso permitido sin penalización
    sisga_descenso_maximo_cm: float = 45.0    # penalización = 1 a partir de aquí — PROVISIONAL
    peso_sisga: float = 0.7                   # PROVISIONAL

    # --- Penalización Neusa: proximidad al nivel mínimo ---
    peso_neusa: float = 1.0                   # PROVISIONAL

    # --- Penalización Tominé: flexibilidad (gran amortiguador) ---
    peso_tomine: float = 0.5                  # PROVISIONAL

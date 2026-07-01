"""
Configuración del entorno de simulación.

Todos los parámetros están en un único lugar para facilitar su ajuste.
Los valores marcados como PROVISIONAL deben validarse con el asesor de tesis
antes de usarlos en experimentos formales.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfigEntorno:
    """Parámetros configurables del entorno de simulación.

    Atributos
    ---------
    q_eco_m3s : float
        Caudal ecológico mínimo requerido en el punto de control El Sol [m³/s].
        PROVISIONAL: valor inicial de referencia.

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
    q_eco_m3s: float = 2.0                    # PROVISIONAL

    # --- Penalización Sisga: rata de descenso de nivel ---
    # Fuente umbrales: Manual de Operación Embalse del Sisga, CAR.
    sisga_descenso_umbral_cm: float = 15.0    # descenso permitido sin penalización
    sisga_descenso_maximo_cm: float = 45.0    # penalización = 1 a partir de aquí — PROVISIONAL
    peso_sisga: float = 0.7                   # PROVISIONAL

    # --- Penalización Neusa: proximidad al nivel mínimo ---
    peso_neusa: float = 1.0                   # PROVISIONAL

    # --- Penalización Tominé: flexibilidad (gran amortiguador) ---
    peso_tomine: float = 0.5                  # PROVISIONAL

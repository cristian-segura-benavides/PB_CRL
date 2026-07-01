"""
Funciones de penalización por embalse.

Cada función devuelve un valor en [0, peso], donde:
    0    = operación dentro de los límites aceptables
    peso = violación máxima del criterio

Todas las penalizaciones normalizadas están acotadas a [0, 1] antes de
multiplicar por el peso, garantizando que el resultado final esté en [0, peso].

TODOS LOS VALORES NUMÉRICOS SON PROVISIONALES y deben validarse con el asesor.
"""
from __future__ import annotations

from pbcrl.data_contracts.embalses import ParametrosEmbalse
from pbcrl.environment.config import ConfigEntorno


def pen_descenso_nivel_sisga(
    cota_anterior_m: float,
    cota_actual_m: float,
    config: ConfigEntorno,
) -> float:
    """Penalización del Sisga por rata de descenso de nivel (cm/día).

    Regla (fuente: Manual de Operación Embalse del Sisga, CAR):
        - Descenso ≤ umbral (15 cm/día): sin penalización.
        - Descenso ∈ (umbral, máximo): penalización proporcional.
        - Descenso ≥ máximo (45 cm/día): penalización máxima.

    Normalización:
        pen_norm = clamp((descenso_cm − umbral) / (máximo − umbral), 0, 1)

    Puntos de referencia con valores por defecto (umbral=15, máximo=45):
        descenso = 15 cm/día → pen = 0
        descenso = 30 cm/día → pen = 0.5 × peso_sisga
        descenso = 45 cm/día → pen = 1.0 × peso_sisga

    Solo se penaliza el descenso (cota_anterior > cota_actual);
    el ascenso no genera penalización por este criterio.

    Parámetros
    ----------
    cota_anterior_m : float
        Cota al inicio del paso [m.s.n.m.].
    cota_actual_m : float
        Cota al final del paso [m.s.n.m.].
    config : ConfigEntorno
        Configuración con umbrales y peso.

    Retorna
    -------
    float
        Penalización en [0, peso_sisga].
    """
    descenso_cm = max(0.0, (cota_anterior_m - cota_actual_m) * 100.0)

    rango = config.sisga_descenso_maximo_cm - config.sisga_descenso_umbral_cm
    pen_norm = (descenso_cm - config.sisga_descenso_umbral_cm) / rango
    pen_norm = max(0.0, min(1.0, pen_norm))

    return config.peso_sisga * pen_norm


def pen_proximidad_minimo(
    cota_actual_m: float,
    params: ParametrosEmbalse,
    peso: float,
) -> float:
    """Penalización por proximidad al nivel mínimo operativo.

    Usada por Neusa (abastece acueductos; fuente: CAR ABC de embalses)
    y Tominé (mismo concepto, menor severidad).

    Normalización:
        pen_norm = clamp((cota_max − cota_actual) / (cota_max − cota_min), 0, 1)

    Puntos de referencia:
        cota = cota_max  → pen_norm = 0   (embalse lleno)
        cota = cota_min  → pen_norm = 1   (embalse en nivel mínimo)

    Parámetros
    ----------
    cota_actual_m : float
        Cota actual del embalse [m.s.n.m.].
    params : ParametrosEmbalse
        Parámetros del embalse (define cota_min_m y cota_max_m).
    peso : float
        Peso de la penalización (peso_neusa o peso_tomine según el embalse).

    Retorna
    -------
    float
        Penalización en [0, peso].
    """
    rango_cota = params.cota_max_m - params.cota_min_m
    pen_norm = (params.cota_max_m - cota_actual_m) / rango_cota
    pen_norm = max(0.0, min(1.0, pen_norm))

    return peso * pen_norm

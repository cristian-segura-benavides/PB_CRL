"""Construcción de las restricciones lineales del shield a partir del estado.

No reimplementa nada que ya exista: importa la extracción de Tibitóc de
`data_contracts.captaciones`, el umbral de caudal ecológico de
`data_contracts.caudal_ecologico`, y la relación cota-volumen de
`data_contracts.curvas` (curva real si existe, fallback lineal si no — el
mismo despacho que usa el resto del proyecto).

TRES NIVELES DE RESTRICCIÓN, TODAS LINEALES EN a = (Q_Neusa, Q_Sisga, Q_Tomine)
--------------------------------------------------------------------------------

Nivel 1 — cajas individuales (desacopladas):
    0 <= Q_i <= descarga_max_m3s(i)      (capacidad de la torre de toma/compuertas,
                                           ya en data_contracts.embalses.EMBALSES)

Nivel 1b — rata de descenso de Sisga (solo Sisga, restricción dura del manual):
    Se traduce a un límite SUPERIOR dinámico sobre Q_Sisga (linealización
    alrededor del volumen actual, sin vertimiento activo — válida en el
    régimen de descenso que esta restricción regula):

        V_fin ~= V_ini + (afluencia_Sisga - Q_Sisga)*0.0864
                        + (precip_Sisga - evap_Sisga)*area_Sisga*1e-3    [Mm³]
        descenso_cm = 100 * pendiente_cota_volumen * (V_ini - V_fin) <= tasa_max_cm

    `pendiente_cota_volumen` [m/Mm³] se estima con diferencia finita centrada
    sobre `data_contracts.curvas.volumen_a_cota` (recoge automáticamente la
    curva real si algún día se agrega para Sisga, sin tocar este módulo).
    `tasa_max_cm` = 15.0 (mismo valor que `ConfigEntorno.sisga_descenso_umbral_cm`,
    usado aquí como límite DURO del shield — distinto de su uso como umbral de
    penalización BLANDA en `environment.penalizaciones`, donde 15 cm es donde
    la penalización empieza a crecer, no un techo absoluto).

    Despejando Q_Sisga, la restricción es EQUIVALENTE a un límite superior:
        Q_Sisga <= [tasa_max_cm + 100*pendiente*0.0864*afluencia
                    + 100*pendiente*(precip-evap)*area*1e-3] / (100*pendiente*0.0864)
    Se combina con el límite de capacidad (Nivel 1) tomando el mínimo de los dos.

Nivel 2 — caudal ecológico conjunto (acopla las tres variables de decisión):
    Q_ElSol(a) = max(0, Q_bocatoma(a) - Q_extraccion_nominal)  >= Q_eco(mes)
    con Q_bocatoma(a) = Q_Saucío(t) + Q_Neusa + Q_Sisga + Q_Tomine.

    El max(0, ·) de la cota física de extracción (data_contracts.captaciones)
    parece introducir una no linealidad, pero NO LA INTRODUCE en la práctica:
    como Q_eco(mes) > 0 siempre (los 12 valores de EFR_VMF_M3S son positivos),
    la rama "max=0" nunca puede satisfacer Q_ElSol >= Q_eco (0 no es >= a un
    positivo). Por lo tanto:

        max(0, Q_bocatoma - nominal) >= Q_eco   <=>   Q_bocatoma - nominal >= Q_eco

    (verificación por casos en el docstring de `proyeccion.py` / README.md).
    La restricción lineal resultante:

        Q_Neusa + Q_Sisga + Q_Tomine >= Q_eco(mes) + Q_extraccion_nominal - Q_Saucío(t)

    Se usa el valor NOMINAL de la extracción (data_contracts.captaciones.
    caudal_tibitoc_nominal), no la extracción acotada por `calcular_extraccion_
    tibitoc` de hidráulica.py — precisamente porque la equivalencia de arriba
    ya absorbe el efecto de la cota física, y aplicar el min() otra vez sobre
    una cantidad que depende de la propia acción sería circular.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pbcrl.data_contracts.captaciones import ESCENARIO_HISTORICO, caudal_tibitoc_nominal
from pbcrl.data_contracts.caudal_ecologico import q_eco_m3s
from pbcrl.data_contracts.curvas import volumen_a_cota
from pbcrl.data_contracts.embalses import EMBALSES

NOMBRES_EMBALSES: tuple[str, ...] = ("Neusa", "Sisga", "Tomine")

_M3S_A_MM3_DIA = 86_400.0 / 1e6  # 0.0864
_MM_KM2_A_MM3 = 1e-3

# Límite DURO del shield para la rata de descenso de Sisga [cm/día].
# NO se importa de environment.config.ConfigEntorno.sisga_descenso_umbral_cm a
# propósito: shield/ está diseñado para ser aislado (ver README.md), y
# environment.config solo es importable a través del paquete environment, cuyo
# __init__.py importa entorno.py — que a su vez importa este módulo cuando el
# shield está conectado. Importar ConfigEntorno aquí crea un ciclo real
# (environment -> shield -> environment). Mismo valor numérico que
# ConfigEntorno.sisga_descenso_umbral_cm (15.0, manual de operación CAR) pero
# con un rol distinto: allá es donde EMPIEZA la penalización blanda; acá es un
# techo que el shield nunca deja cruzar. Si ese valor cambia en config.py,
# actualizar también aquí.
TASA_MAX_DESCENSO_SISGA_CM: float = 15.0


@dataclass
class EstadoShield:
    """Estado necesario para construir las restricciones del shield en un paso.

    Atributos
    ---------
    volumen_mm3 : dict[str, float]
        Volumen ACTUAL (antes de la acción) de Neusa, Sisga y Tomine [Mm³].
    afluencia_m3s : dict[str, float]
        Afluencia natural del día a cada embalse [m³/s].
    precipitacion_mm, evaporacion_mm : dict[str, float]
        Lámina sobre el espejo de cada embalse [mm/día].
    caudal_saucio_m3s : float
        Caudal natural de Saucío del día [m³/s] — dato, no se optimiza.
    mes : int
        Mes calendario (1-12), determina el umbral VMF del día.
    caudal_tibitoc_nominal_m3s : float
        Caudal nominal que busca captar Tibitóc [m³/s]. Por defecto, el
        escenario histórico (4.5 m³/s) — pasar
        `data_contracts.captaciones.caudal_tibitoc_nominal(escenario)` para
        usar el ampliado u otro.
    """

    volumen_mm3: dict[str, float]
    afluencia_m3s: dict[str, float]
    precipitacion_mm: dict[str, float]
    evaporacion_mm: dict[str, float]
    caudal_saucio_m3s: float
    mes: int
    caudal_tibitoc_nominal_m3s: float = field(
        default_factory=lambda: caudal_tibitoc_nominal(ESCENARIO_HISTORICO)
    )


@dataclass
class RestriccionesLineales:
    """Representación del conjunto factible: caja [lo, hi]³ ∩ {c·a <= d}.

    Atributos
    ---------
    nombres : tuple[str, ...]
        Orden de las componentes de `a` (Neusa, Sisga, Tomine).
    lo, hi : np.ndarray (3,)
        Límites de la caja (Nivel 1 + Nivel 1b ya combinados en `hi`).
    hi_fuente : dict[str, str]
        Para cada embalse, qué restricción determinó su límite superior:
        "capacidad_toma" o "rata_descenso" (solo puede ser esto último Sisga).
    c, d : np.ndarray (3,), float
        Restricción conjunta del Nivel 2, en la forma c·a <= d.
    detalle_el_sol : dict
        Términos usados para construir la restricción conjunta (auditable).
    """

    nombres: tuple[str, ...]
    lo: np.ndarray
    hi: np.ndarray
    hi_fuente: dict[str, str]
    c: np.ndarray
    d: float
    detalle_el_sol: dict


def _pendiente_cota_volumen(volumen_mm3: float, nombre: str, params, paso: float = 0.01) -> float:
    """Pendiente local dCota/dVolumen [m/Mm³], por diferencia finita centrada.

    Usa `curvas.volumen_a_cota` (curva real si existe, fallback lineal si no)
    — no duplica ninguna fórmula, solo evalúa la función existente en dos
    puntos cercanos.
    """
    v_hi = min(volumen_mm3 + paso, params.capacidad_max_mm3)
    v_lo = max(volumen_mm3 - paso, params.capacidad_min_mm3)
    if v_hi <= v_lo:
        return 0.0
    cota_hi = volumen_a_cota(v_hi, nombre, params)
    cota_lo = volumen_a_cota(v_lo, nombre, params)
    return (cota_hi - cota_lo) / (v_hi - v_lo)


def _limite_descenso_sisga_m3s(estado: EstadoShield) -> float:
    """Cota superior dinámica de Q_Sisga por la rata de descenso (Nivel 1b).

    Ver el docstring del módulo para la derivación completa. Devuelve un
    valor >= 0 (nunca negativo: si el término libre da negativo, el límite
    efectivo es 0 — no hay margen para soltar nada sin violar la rata, lo
    cual el shield debe respetar, no ignorar).
    """
    params = EMBALSES["Sisga"]
    v_ini = estado.volumen_mm3["Sisga"]
    pendiente = _pendiente_cota_volumen(v_ini, "Sisga", params)

    if pendiente <= 0:
        # Curva degenerada en este punto (no debería ocurrir con datos reales):
        # sin información de pendiente, no se impone límite adicional.
        return params.descarga_max_m3s

    afluencia = estado.afluencia_m3s["Sisga"]
    precip = estado.precipitacion_mm["Sisga"]
    evap = estado.evaporacion_mm["Sisga"]

    coef = 100.0 * pendiente * _M3S_A_MM3_DIA  # cm por cada m3/s de Q_Sisga
    termino_libre = (
        TASA_MAX_DESCENSO_SISGA_CM
        + 100.0 * pendiente * afluencia * _M3S_A_MM3_DIA
        + 100.0 * pendiente * (precip - evap) * params.area_espejo_km2 * _MM_KM2_A_MM3
    )
    return max(0.0, termino_libre / coef)


def construir_restricciones(estado: EstadoShield) -> RestriccionesLineales:
    """Construye el conjunto factible (caja + restricción conjunta) del paso actual."""
    lo = np.zeros(3)
    hi_capacidad = np.array([EMBALSES[n].descarga_max_m3s for n in NOMBRES_EMBALSES])

    limite_sisga_descenso = _limite_descenso_sisga_m3s(estado)
    hi = hi_capacidad.copy()
    hi_fuente = {n: "capacidad_toma" for n in NOMBRES_EMBALSES}
    idx_sisga = NOMBRES_EMBALSES.index("Sisga")
    if limite_sisga_descenso < hi[idx_sisga]:
        hi[idx_sisga] = limite_sisga_descenso
        hi_fuente["Sisga"] = "rata_descenso"

    q_eco = q_eco_m3s(estado.mes)
    suma_minima_requerida = q_eco + estado.caudal_tibitoc_nominal_m3s - estado.caudal_saucio_m3s
    # suma(a) >= suma_minima_requerida  <=>  -suma(a) <= -suma_minima_requerida
    c = np.array([-1.0, -1.0, -1.0])
    d = -suma_minima_requerida

    detalle = {
        "q_eco_m3s": q_eco,
        "caudal_tibitoc_nominal_m3s": estado.caudal_tibitoc_nominal_m3s,
        "caudal_saucio_m3s": estado.caudal_saucio_m3s,
        "suma_minima_requerida_m3s": suma_minima_requerida,
        "limite_sisga_capacidad_m3s": float(hi_capacidad[idx_sisga]),
        "limite_sisga_descenso_m3s": limite_sisga_descenso,
    }

    return RestriccionesLineales(
        nombres=NOMBRES_EMBALSES,
        lo=lo,
        hi=hi,
        hi_fuente=hi_fuente,
        c=c,
        d=d,
        detalle_el_sol=detalle,
    )

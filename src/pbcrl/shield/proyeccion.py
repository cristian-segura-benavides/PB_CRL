"""Shield de proyección cuadrática — la contribución central de PB-CRL.

    a*_t = argmin_a ||a - â_t||²   sujeto a   g_k(s_t, a) <= 0  para todo k

Intercepta la acción PROPUESTA por el agente (â_t: los tres caudales de
suministro para Neusa, Sisga y Tominé) y la proyecta sobre el conjunto de
acciones factibles del paso actual, ANTES del recorte físico de
`environment.hidraulica.recortar_suministro`. Son dos mecanismos distintos:
este módulo corrige la INTENCIÓN del agente para no violar restricciones del
sistema (incluida la conjunta de los tres embalses); el recorte físico
verifica DESPUÉS si esa acción ya corregida es alcanzable con el agua
realmente disponible en cada embalse. No se toca `hidraulica.py` ni
`hydrology/balance.py`.

MÉTODO DE SOLUCIÓN — bisección sobre el multiplicador de KKT, no una
librería de QP
--------------------------------------------------------------------------
El conjunto factible es una caja en R³ intersectada con UNA sola restricción
lineal adicional (Nivel 2 de `restricciones.py`; el Nivel 1b ya se pliega
dentro de la caja como un límite superior dinámico de Sisga — ver
`restricciones.construir_restricciones`). Para "caja ∩ un semiespacio", el
problema

    min (1/2)||a-â||²   s.a.  lo<=a<=hi,  c·a <= d

tiene una estructura de KKT muy simple: para un multiplicador λ>=0 fijo, el
minimizador sin restricción conjunta es a_i(λ) = clip(â_i - λ·c_i, lo_i, hi_i)
(cada componente se proyecta a la caja de forma independiente). La función
dual φ(λ) = c·a(λ) es monótona no creciente en λ (propiedad estándar de la
proyección sobre un semiespacio), así que basta con bisección de una sola
variable para encontrar el λ que hace c·a(λ) = d exactamente (o confirmar que
ni el punto más favorable de la caja alcanza a satisfacer la restricción, en
cuyo caso el conjunto factible es VACÍO — se reporta, no se fuerza).

Se implementó así (en vez de una librería de QP como `qpsolvers`/`cvxopt`) porque:
  (a) el proyecto no tiene ninguna dependencia de optimización todavía, y con
      solo 3 variables y una restricción de acople, agregar una librería
      externa sería desproporcionado frente a un método analítico simple y
      verificable con pruebas unitarias exactas;
  (b) la bisección sobre un problema convexo unidimensional converge de forma
      garantizada y con tolerancia controlable, sin depender de un solver de
      terceros ni de sus supuestos de factibilidad numérica.
Si en el futuro el shield gana más restricciones de acople simultáneas, este
método deja de alcanzar (ya no hay un solo λ) y ahí sí se justificaría una
librería de QP general.

VERIFICACIÓN DE LA LINEALIDAD DEL NIVEL 2 (caudal ecológico conjunto)
--------------------------------------------------------------------------
Q_ElSol(a) = max(0, Q_bocatoma(a) - nominal), con Q_bocatoma(a) lineal en a.
Para Q_eco > 0 (siempre cierto: los 12 valores de EFR_VMF_M3S son positivos):

    Q_ElSol(a) >= Q_eco
    <=> max(0, Q_bocatoma(a)-nominal) >= Q_eco
    <=> Q_bocatoma(a)-nominal >= Q_eco   (la rama "max=0" no puede ser >= Q_eco>0)

Verificación por casos:
  - Q_bocatoma(a)-nominal >= Q_eco: Q_ElSol=Q_bocatoma-nominal>=Q_eco. Ambas formas de acuerdo.
  - 0 <= Q_bocatoma(a)-nominal < Q_eco: Q_ElSol=Q_bocatoma-nominal<Q_eco. Ambas violan.
  - Q_bocatoma(a)-nominal < 0: Q_ElSol=max(0,negativo)=0<Q_eco. Ambas violan.
La equivalencia es exacta; el min() de la cota física no introduce no linealidad
en la restricción, solo en la trayectoria de Q_ElSol fuera de la región relevante.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pbcrl.shield.restricciones import (
    EstadoShield,
    RestriccionesLineales,
    construir_restricciones,
)

_TOL = 1e-9
_MAX_ITER_BISECCION = 100
_LAMBDA_MAX = 1.0e6


@dataclass
class DiagnosticoShield:
    """Resultado de proyectar una acción propuesta.

    Atributos
    ---------
    accion_propuesta, accion_proyectada : dict[str, float]
        â_t y a*_t, por embalse.
    violaciones_previas : dict[str, bool]
        Qué restricciones violaba â_t ANTES de proyectar (diagnóstico pedido).
    restricciones_activas : dict[str, bool]
        Qué restricciones quedan exactamente en el borde en a*_t.
    factible : bool
        False si el conjunto factible del paso resultó VACÍO (ni el punto
        más favorable de la caja satisface la restricción conjunta) — en ese
        caso `accion_proyectada` es el punto que MENOS viola la restricción
        conjunta dentro de la caja, no una solución válida; hay que
        reportarlo, no ocultarlo.
    detalle : dict
        Términos numéricos de la restricción conjunta y los límites de caja
        efectivos, para auditoría.
    """

    accion_propuesta: dict[str, float]
    accion_proyectada: dict[str, float]
    violaciones_previas: dict[str, bool]
    restricciones_activas: dict[str, bool]
    factible: bool
    detalle: dict


def _proyectar_caja_mas_semiespacio(
    a_hat: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    c: np.ndarray,
    d: float,
    tol: float = _TOL,
) -> tuple[np.ndarray, float, bool, bool]:
    """Proyecta a_hat (norma euclidiana) sobre {a: lo<=a<=hi, c.a<=d}.

    Retorna (a_proyectada, lambda_optimo, restriccion_activa, factible).
    Ver el docstring del módulo para la derivación del método.
    """

    def a_de_lambda(lam: float) -> np.ndarray:
        return np.clip(a_hat - lam * c, lo, hi)

    a_caja = np.clip(a_hat, lo, hi)
    if c @ a_caja <= d + tol:
        return a_caja, 0.0, False, True

    # Punto más favorable posible dentro de la caja para esta restricción
    # (límite de a(lambda) cuando lambda -> infinito, componente a componente).
    a_extremo = np.where(c < 0, hi, np.where(c > 0, lo, a_caja))
    if c @ a_extremo > d + tol:
        # Ni el extremo más favorable alcanza: conjunto factible VACÍO.
        return a_extremo, float("inf"), True, False

    lam_lo, lam_hi = 0.0, _LAMBDA_MAX
    for _ in range(_MAX_ITER_BISECCION):
        lam_mid = (lam_lo + lam_hi) / 2.0
        if c @ a_de_lambda(lam_mid) > d + tol:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
    return a_de_lambda(lam_hi), lam_hi, True, True


def proyectar(estado: EstadoShield, accion_propuesta: dict[str, float]) -> DiagnosticoShield:
    """Proyecta la acción propuesta sobre el conjunto factible del paso actual.

    Parámetros
    ----------
    estado : EstadoShield
        Estado del sistema necesario para construir las restricciones del
        paso (ver `restricciones.EstadoShield`).
    accion_propuesta : dict[str, float]
        â_t: caudal de suministro propuesto por embalse, claves 'Neusa',
        'Sisga', 'Tomine' [m³/s]. Componentes ausentes se toman como 0.0.

    Retorna
    -------
    DiagnosticoShield
    """
    restricciones: RestriccionesLineales = construir_restricciones(estado)
    nombres = restricciones.nombres
    a_hat = np.array([accion_propuesta.get(n, 0.0) for n in nombres], dtype=float)

    violaciones_previas: dict[str, bool] = {}
    for i, n in enumerate(nombres):
        violaciones_previas[f"caja_{n}"] = bool(
            a_hat[i] < restricciones.lo[i] - _TOL or a_hat[i] > restricciones.hi[i] + _TOL
        )
    violaciones_previas["caudal_ecologico_conjunto"] = bool(
        restricciones.c @ a_hat > restricciones.d + _TOL
    )

    a_proy, lam, activa_conjunta, factible = _proyectar_caja_mas_semiespacio(
        a_hat, restricciones.lo, restricciones.hi, restricciones.c, restricciones.d
    )

    restricciones_activas: dict[str, bool] = {}
    for i, n in enumerate(nombres):
        restricciones_activas[f"caja_{n}_inferior"] = bool(abs(a_proy[i] - restricciones.lo[i]) < 1e-6)
        etiqueta_superior = f"caja_{n}_superior ({restricciones.hi_fuente[n]})"
        restricciones_activas[etiqueta_superior] = bool(abs(a_proy[i] - restricciones.hi[i]) < 1e-6)
    restricciones_activas["caudal_ecologico_conjunto"] = bool(activa_conjunta)

    detalle = dict(restricciones.detalle_el_sol)
    detalle["lambda_caudal_ecologico"] = lam
    detalle["hi_efectivo_m3s"] = dict(zip(nombres, restricciones.hi.tolist()))
    detalle["hi_fuente"] = dict(restricciones.hi_fuente)

    return DiagnosticoShield(
        accion_propuesta=dict(accion_propuesta),
        accion_proyectada=dict(zip(nombres, a_proy.tolist())),
        violaciones_previas=violaciones_previas,
        restricciones_activas=restricciones_activas,
        factible=factible,
        detalle=detalle,
    )

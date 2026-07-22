"""Umbral de caudal ecológico en el punto de control El Sol: VMF (Variable
Monthly Flow), fijado como umbral definitivo del entorno de simulación.

MÉTODO — VMF (Pastor et al. 2014; usado por Gerten et al. 2013 para
operacionalizar el límite planetario del agua dulce de Rockström et al. a
escala de cuenca)
--------------------------------------------------------------------------
Para cada mes calendario m, el requerimiento de caudal ambiental (EFR) preserva
un porcentaje del caudal medio de ESE mes (MMF_m), según su relación con el
caudal medio anual (MAF):

    MMF_m <= 0.4 * MAF                  -> mes de caudal BAJO   -> EFR_m = 0.60 * MMF_m
    0.4 * MAF < MMF_m <= 0.8 * MAF       -> mes INTERMEDIO       -> EFR_m = 0.45 * MMF_m
    MMF_m > 0.8 * MAF                    -> mes de caudal ALTO   -> EFR_m = 0.30 * MMF_m

ORIGEN DE LOS DATOS (decisión metodológica, 2026-07-22)
--------------------------------------------------------------------------
MAF y los 12 MMF se calculan sobre Q_natural = Q_Saucío + afluencia_Neusa +
afluencia_Sisga + afluencia_Tominé (afluencias por balance hídrico inverso,
NO descargas reguladas — ver hydrology.balance), sobre la ventana completa del
proyecto (2012-01-01 a 2025-05-04, ver data_contracts.ventana), incluyendo el
evento de vertimiento de Neusa de jul-nov 2022 (ver más abajo). Cálculo
reproducible en scratch_vmf/calcular_vmf_v2.py.

    MAF = 12.432 m³/s

    mes  MMF (m³/s)  régimen      fracción  EFR (m³/s)
    ---  ----------  -----------  --------  ----------
    ene   3.864      bajo         0.60      2.32
    feb   3.549      bajo         0.60      2.13
    mar   6.090      intermedio   0.45      2.74
    abr   9.768      intermedio   0.45      4.40
    may  12.975      alto         0.30      3.89
    jun  21.766      alto         0.30      6.53
    jul  25.019      alto         0.30      7.51
    ago  19.822      alto         0.30      5.95
    sep  12.429      alto         0.30      3.73
    oct  15.093      alto         0.30      4.53
    nov  12.673      alto         0.30      3.80
    dic   5.551      intermedio   0.45      2.50

DECISIÓN SOBRE EL EVENTO DE VERTIMIENTO DE NEUSA (jul-nov 2022)
--------------------------------------------------------------------------
Neusa registró volumen por encima de su capacidad máxima útil (95.3 Mm³)
durante 135 días consecutivos (2022-07-02 a 2022-11-13), con una trayectoria
suave de ascenso-pico-descenso (pico 2022-10-28, +13.7% sobre capacidad) —
coincidente con el episodio de La Niña de 2022 en Colombia, documentado
ampliamente. Esto infla la afluencia estimada de Neusa en esos meses (el
término de vertimiento entra sumando en el balance inverso, ver
hydrology.balance).

DECISIÓN: el umbral se calcula sobre la serie COMPLETA, incluyendo estos meses.
Justificación: el VMF caracteriza el régimen NATURAL del río; el evento de 2022
fue un fenómeno hidrológico real (no un artefacto de instrumentación — la
trayectoria suave lo respalda), y excluirlo recortaría la variabilidad natural
que el método intenta capturar. Lo que está en duda es la magnitud exacta del
VOLUMEN VERTIDO, no la ocurrencia del evento; y el vertimiento en sí no entra
en Q_natural (que usa afluencias, no descargas ni vertimientos observados
directamente — aunque sí afecta la afluencia ESTIMADA de Neusa vía el balance
inverso). La sensibilidad de la clasificación mensual a este período se
verificó explícitamente: excluir jul-nov 2022 SÍ cambia el régimen de 2 de 12
meses (octubre, por efecto directo; abril, por el desplazamiento del MAF de
referencia; MAF cae 12.5% sin el evento). Esa sensibilidad queda cubierta por
el análisis de robustez ±20% sobre el umbral (ver scratch_vmf/
calcular_vmf_v3_validacion.py, Pieza 2): la conclusión cualitativa del proyecto
(el escenario ampliado de extracción transgrede sustancialmente más que el
histórico) se mantiene en las tres variantes (EFR × 0.8 / 1.0 / 1.2).

PENDIENTE: esta decisión (incluir el evento de 2022 sin ajustar) y la magnitud
del volumen vertido reportado quedan pendientes de validación con el asesor
(Sebastian Hernández-Suárez) y de verificación con la CAR. Ver NOTAS.md.

VALIDACIÓN ESTACIONAL (referencia cruzada, no absoluta)
--------------------------------------------------------------------------
El patrón mensual de Q_natural se comparó contra Saucío (caudal MEDIDO, aguas
arriba de los tres embalses, sin regulación): correlación de Spearman
rho=0.916; la clasificación de régimen (bajo/intermedio/alto) coincide en 8 de
12 meses. Marzo y abril difieren por estar cerca del límite 0.4/0.8 en ambas
series; septiembre y octubre muestran una discrepancia real no explicada
(Q_natural los clasifica "alto", Saucío los clasifica "intermedio"). Ver
scratch_vmf/calcular_vmf_v3_validacion.py, Pieza 1.
"""
from __future__ import annotations

from typing import Callable

MAF_M3S: float = 12.432

MMF_M3S: dict[int, float] = {
    1: 3.864, 2: 3.549, 3: 6.090, 4: 9.768, 5: 12.975, 6: 21.766,
    7: 25.019, 8: 19.822, 9: 12.429, 10: 15.093, 11: 12.673, 12: 5.551,
}

REGIMEN_MES: dict[int, str] = {
    1: "bajo", 2: "bajo", 3: "intermedio", 4: "intermedio",
    5: "alto", 6: "alto", 7: "alto", 8: "alto", 9: "alto", 10: "alto",
    11: "alto", 12: "intermedio",
}

# EFR (Environmental Flow Requirement) = umbral de caudal ecológico VMF por mes.
# Valores fijados: fracción de MMF_M3S según REGIMEN_MES (60/45/30%), redondeados
# a dos decimales (ver tabla en el docstring del módulo).
EFR_VMF_M3S: dict[int, float] = {
    1: 2.32, 2: 2.13, 3: 2.74, 4: 4.40, 5: 3.89, 6: 6.53,
    7: 7.51, 8: 5.95, 9: 3.73, 10: 4.53, 11: 3.80, 12: 2.50,
}


def q_eco_m3s(mes: int) -> float:
    """Umbral de caudal ecológico VMF para el mes calendario dado [m³/s].

    Parámetros
    ----------
    mes : int
        Mes calendario, 1 (enero) a 12 (diciembre).

    Retorna
    -------
    float
        EFR del mes (ver EFR_VMF_M3S y el docstring del módulo).

    Raises
    ------
    ValueError
        Si `mes` no está en el rango 1-12.
    """
    try:
        return EFR_VMF_M3S[mes]
    except KeyError as exc:
        raise ValueError(f"Mes inválido: {mes!r}. Debe ser un entero entre 1 y 12.") from exc


def umbral_fijo_m3s(valor: float) -> Callable[[int], float]:
    """Construye un umbral CONSTANTE (mismo valor todos los meses).

    Permite revertir el entorno a un umbral fijo (p. ej. el 2.0 m³/s
    provisional anterior, o cualquier otro valor de comparación — normativo,
    Q95, Tennant) sin tocar `environment.entorno`: basta con pasar
    `ConfigEntorno(calcular_q_eco_m3s=umbral_fijo_m3s(valor))`.

    Parámetros
    ----------
    valor : float
        Caudal ecológico constante [m³/s], igual para los 12 meses.

    Retorna
    -------
    Callable[[int], float]
        Función compatible con `ConfigEntorno.calcular_q_eco_m3s` que ignora
        el mes y siempre devuelve `valor`.
    """
    def _umbral(mes: int) -> float:
        return valor
    return _umbral

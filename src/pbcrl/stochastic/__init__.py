"""Modelo estocástico multivariado de afluencias (fase 1 de Sebastian).

Módulo aislado: VARX desestacionalizado con componente hurdle (`modelo.py`,
el método final) y remuestreo por análogos (`analogos.py`, línea base de
validación). Ver `modelo.py` para la justificación completa y
`entrenamiento.py` para el script de ajuste con los datos históricos.
"""
from .analogos import ConfigAnalogos, RemuestreoAnalogos
from .modelo import ConfigModeloEstocastico, ModeloEstocasticoAfluencias

__all__ = [
    "ConfigAnalogos",
    "RemuestreoAnalogos",
    "ConfigModeloEstocastico",
    "ModeloEstocasticoAfluencias",
]

"""Shield de proyección cuadrática (contribución central de PB-CRL).

Módulo aislado: no toca `environment.entorno` ni `environment.hidraulica` ni
`hydrology.balance`. Conectarlo al entorno de simulación es un paso
posterior, una vez validado — ver `proyeccion.py` para el método y
`restricciones.py` para las tres restricciones (cajas, rata de descenso de
Sisga, caudal ecológico conjunto).
"""
from .proyeccion import DiagnosticoShield, proyectar
from .restricciones import (
    EstadoShield,
    NOMBRES_EMBALSES,
    RestriccionesLineales,
    TASA_MAX_DESCENSO_SISGA_CM,
    construir_restricciones,
)

__all__ = [
    "DiagnosticoShield",
    "proyectar",
    "EstadoShield",
    "NOMBRES_EMBALSES",
    "RestriccionesLineales",
    "TASA_MAX_DESCENSO_SISGA_CM",
    "construir_restricciones",
]

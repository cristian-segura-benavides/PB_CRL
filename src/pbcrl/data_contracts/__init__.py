"""Contratos de datos: esquemas, constantes de embalses y validación."""
from .schemas import ESQUEMA_EMBALSE, validar_dataframe_embalse
from .embalses import EMBALSES, ParametrosEmbalse

__all__ = [
    "ESQUEMA_EMBALSE",
    "validar_dataframe_embalse",
    "EMBALSES",
    "ParametrosEmbalse",
]

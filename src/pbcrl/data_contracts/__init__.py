"""Contratos de datos: esquemas, constantes de embalses y validación."""
from .schemas import ESQUEMA_EMBALSE, validar_dataframe_embalse
from .embalses import EMBALSES, ParametrosEmbalse
from .curvas import CURVAS, CURVA_TOMINE, CurvaCotaVolumen, cota_a_volumen, volumen_a_cota

__all__ = [
    "ESQUEMA_EMBALSE",
    "validar_dataframe_embalse",
    "EMBALSES",
    "ParametrosEmbalse",
    "CURVAS",
    "CURVA_TOMINE",
    "CurvaCotaVolumen",
    "cota_a_volumen",
    "volumen_a_cota",
]

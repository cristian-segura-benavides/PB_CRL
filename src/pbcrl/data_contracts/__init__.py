"""Contratos de datos: esquemas, constantes de embalses y validación."""
from .schemas import ESQUEMA_EMBALSE, validar_dataframe_embalse
from .embalses import EMBALSES, ParametrosEmbalse
from .curvas import CURVAS, CURVA_TOMINE, CurvaCotaVolumen, cota_a_volumen, volumen_a_cota
from .ventana import VENTANA_INICIO, VENTANA_FIN, VENTANA_INICIO_TS, VENTANA_FIN_TS
from .captaciones import (
    ESCENARIO_HISTORICO,
    ESCENARIO_AMPLIADO,
    CAUDAL_TIBITOC_HISTORICO_M3S,
    CAUDAL_TIBITOC_AMPLIADO_M3S,
    CAUDAL_TIBITOC_ESCENARIOS,
    caudal_tibitoc_nominal,
)

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
    "VENTANA_INICIO",
    "VENTANA_FIN",
    "VENTANA_INICIO_TS",
    "VENTANA_FIN_TS",
    "ESCENARIO_HISTORICO",
    "ESCENARIO_AMPLIADO",
    "CAUDAL_TIBITOC_HISTORICO_M3S",
    "CAUDAL_TIBITOC_AMPLIADO_M3S",
    "CAUDAL_TIBITOC_ESCENARIOS",
    "caudal_tibitoc_nominal",
]

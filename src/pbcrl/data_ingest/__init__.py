"""Ingesta de series operativas reales al contrato de datos del proyecto."""
from .tomine import (
    DEFAULT_TOMINE_XLSX,
    DiagnosticoTomine,
    cargar_tomine,
    resumen_tomine,
)
from .saucio import (
    DEFAULT_CAR_ESTACIONES_CSV,
    DiagnosticoSaucio,
    cargar_saucio,
    resumen_saucio,
)
from .roni import (
    DEFAULT_RONI_CSV,
    DiagnosticoRoni,
    cargar_roni,
    resumen_roni,
)

__all__ = [
    "DEFAULT_TOMINE_XLSX",
    "DiagnosticoTomine",
    "cargar_tomine",
    "resumen_tomine",
    "DEFAULT_CAR_ESTACIONES_CSV",
    "DiagnosticoSaucio",
    "cargar_saucio",
    "resumen_saucio",
    "DEFAULT_RONI_CSV",
    "DiagnosticoRoni",
    "cargar_roni",
    "resumen_roni",
]

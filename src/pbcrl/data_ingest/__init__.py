"""Ingesta de series operativas reales al contrato de datos del proyecto."""
from .tomine import (
    DEFAULT_TOMINE_XLSX,
    DiagnosticoTomine,
    cargar_tomine,
    resumen_tomine,
)

__all__ = [
    "DEFAULT_TOMINE_XLSX",
    "DiagnosticoTomine",
    "cargar_tomine",
    "resumen_tomine",
]

"""Ventana temporal única y DEFINITIVA del proyecto: período común a los tres
embalses (Neusa, Sisga, Tominé) y a la estación Saucío (caudal natural hacia El Sol).

Definida una sola vez aquí; los loaders de cada componente y el tablero importan
estas constantes en vez de tener fechas hardcodeadas cada uno.

Justificación de los bordes (definitiva, decisión del usuario 2026-07-20)
----------------------------------------------------------------------------
Inicio (2012-01-01):
  (a) La evaporación MEDIDA de Neusa (CAR) arranca el 2011-01-01; antes de esa
      fecha no hay evaporación real medida para Neusa (Sisga sí tiene desde 1995).
  (b) Los ceros de descarga de Tominé son sospechosos entre 2000 y 2011 (24%-100%
      de días con descarga cero por año, con 2011 al 100% — un año completo en
      cero, casi con certeza ausencia de registro, no descarga real). La
      proporción cae a 5.7% en 2012 y 1.1% en 2013, un quiebre claro.
  (c) Con la ventana arrancando exactamente en 2012-01-01, el salto anómalo de
      volumen de Tominé del 2012-01-01 (~138 Mm³ en un día, transición con el
      2011-12-31) queda fuera de la cadena de comparación día a día y ya no
      requiere corrección.
  Empezar en 2012 elimina las tres dudas de raíz para los tres embalses a la vez.

Fin (2025-05-04):
  Es donde termina la cobertura de la estación Saucío en su fuente más reciente
  (Enlaza, hoja "S-PF-PT"). Saucío es el caudal natural que alimenta el punto de
  control ecológico El Sol: sin él no se puede evaluar la restricción del caudal
  ecológico. Se prefirió una ventana donde TODOS los componentes del sistema (los
  tres embalses + Saucío) estén completos y sin huecos al final, en vez de una
  ventana más larga (hasta 2025-12-31, límite de la evaporación ERA5 de Tominé)
  con esa salvedad.

Nota sobre la lluvia
---------------------
La lluvia NO restringe esta ventana. Los tres embalses tienen lluvia MEDIDA en
sitio: Neusa y Sisga en el Excel de la CAR, Tominé en su propio pluviómetro
(Excel Enlaza). Un eventual pipeline satelital (p. ej. GEE/CHIRPS) queda como
respaldo o validación cruzada, nunca como fuente que acote el balance.
"""
from __future__ import annotations

import pandas as pd

VENTANA_INICIO = "2012-01-01"
VENTANA_FIN = "2025-05-04"

VENTANA_INICIO_TS = pd.Timestamp(VENTANA_INICIO)
VENTANA_FIN_TS = pd.Timestamp(VENTANA_FIN)

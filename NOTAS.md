1. los parámetros de Tominé están pendientes de verificar con batimetría SURER
2. los costos diferenciados son provisionales, validar con Sebastian
3. volumen_a_cota como lineal y provisional, con una nota explícita de que hay que reemplazarlo por la curva batimétrica real (la tabla cota-volumen)

NOTAS — Proyecto PB-CRL (dinámica de cuenca alta del río Bogotá)


Memoria viva del proyecto. Actualizar cada vez que se tome una decisión o llegue un dato.
Para retomar en un chat nuevo de Claude Code: "lee NOTAS.md y continuamos con X".




1. Qué es el proyecto

Framework PB-CRL (Planetary Boundary-Constrained Reinforcement Learning): RL con
restricciones para operar coordinadamente tres embalses del Sistema Agregado Norte
(Neusa, Sisga, Tominé) garantizando el caudal ecológico en el punto de control El Sol
(aguas abajo de Tibitóc). Primer paper de la tesis, instancia el límite planetario del agua.
Objetivo de publicación: Earth's Future. Contribución central: el shield de proyección
cuadrática que fuerza el cumplimiento del límite.


2. Estado actual del código (construido y probado)


Convención de volumen: RESUELTA y definitiva, confirmada con el asesor. Los tres
embalses en volumen ÚTIL (cero = volumen muerto). Punto central y reversible:
data_contracts/embalses.py, constante CONVENCION_VOLUMEN. Volúmenes muertos restados:
Neusa 7.0 (manual CAR), Sisga 4.7 (batimetría CAR 2004), Tominé 9.90 (batimetría 2021).
Capacidades útiles resultantes: Neusa 95.3, Sisga 89.6, Tominé 689.53 Mm³.
Balance hídrico inverso (hydrology/balance.py): conservación de masa exacta (1e-9).
Incluye término de BOMBEO opcional (por defecto cero; Neusa/Sisga sin cambio). Solo
Tominé lo usa: es el único embalse con entrada artificial por bombeo (canal Achurí).
Contrato de datos (data_contracts/schemas.py): esquema + validación por embalse.
Parámetros de embalses (data_contracts/embalses.py): valores reales con fuente, en
volumen útil.
Curvas cota-volumen (data_contracts/curvas.py): curva real de Tominé (batimetría 2021,
desplazada a volumen útil), fallback lineal para Neusa/Sisga.
Generador sintético (synthetic/): datos de prueba.
Entorno (environment/): acople a El Sol, cota física, penalizaciones diferenciadas,
detección (no forzado aún) de violación del caudal ecológico.
Cargador de Tominé (data_ingest/tomine.py): serie operativa 2015-2025 desde Excel
Enlaza (cota, volumen útil, descarga, bombeo, lluvia) + evaporación ERA5-Land.
Tablero de exploración (dashboard/, Streamlit + Plotly): series de volumen y afluencia
de los TRES embalses — Tominé integrado esta sesión, incluyendo su balance con bombeo.
Límites activables, slider de fechas + calendario sincronizado, tratamiento en dos
capas de afluencias negativas con reporte de diagnóstico, cotas corruptas de la CAR
corregidas (Neusa 4, Sisga 4), y marcadores de días con bombeo sobre la línea de
volumen de Tominé.
87/87 pruebas pasan (core del modelo). El tablero es módulo aparte del core.



3. Qué falta construir


Shield de proyección cuadrática (contribución central). PENDIENTE reunión Sebastian.
Agente de RL. PENDIENTE reunión.
Modelo estocástico multivariado de afluencias (fase 1 de Sebastian): un modelo,
4 salidas (Saucío + afluencias Neusa/Sisga/Tominé), covariables precip/temp/RONI.
Integrar datos reales al entorno: reemplazar sintético por las series de la CAR.



4. Decisiones provisionales — VALIDAR CON SEBASTIAN


Penalizaciones diferenciadas (conceptos distintos, escala común 0-1 × peso):

Sisga: rata de descenso > 15 cm/día (manual CAR). Umbral "muy malo" = 45 cm/día. Peso 0.7.
Neusa: proximidad al mínimo (abastece acueductos, CAR ABC embalses). Peso 1.0.
Tominé: flexibilidad (amortiguador). Peso 0.5.
Los pesos NO suman 1 a propósito: califican independientemente, no reparten un total.



Acople a El Sol: suma simple de descargas + caudal natural, sin tiempo de viaje.
Caudal ecológico: Q_eco = 2.0 m³/s (valor inicial, configurable).
Restricción de Sisga: sobre la cota (fiel al manual), no sobre volumen.


4b. Preguntas hidrológicas abiertas para Sebastian


Afluencias negativas: el balance da 5.67% de días negativos en Neusa (cola hasta
-32.69 m³/s) y 8.53% en Sisga (cola hasta -5.62 m³/s). NO vienen de saltos de volumen
(los datos de volumen son limpios). Sospecha: salidas no registradas en la serie de
descarga de la CAR (¿captaciones de acueducto? ¿vertimientos por aliviadero?).
PREGUNTA: ¿qué incluye exactamente la columna descarga_m3_s de la CAR?
pywr: revisado. Decisión: NO como núcleo del control (rompería la transparencia del
entorno y la limpieza del shield); SÍ como posible línea base de comparación (benchmark).



5. Datos: recibido / en espera / cerrado

Recibido — primer envío CAR (radicado 20261625095)


Neusa y Sisga: cota, volumen, descarga, lluvia diaria 2009–2026. Calidad excelente (~1% faltantes).
Saucío: caudal medio diario 1970–2022 (corta antes de El Niño 2023-24, ~12% faltantes).
Manuales de operación Neusa y Sisga (PDF): de aquí salen los límites operativos.


Recibido — segundo envío CAR (radicado 20261071858)


Evaporación medida en sitio: Represa Neusa (2011–2026, 5% faltantes, 0–7.8 mm/día) y
Represa Sisga (1995–2026, 17% faltantes). RESUELVE el dilema de ET satelital (MODIS/ERA5)
para Neusa y Sisga: usar esta medición real como principal.
Canal Achurí / Tominé (Puente Sesquilé, 2120863): caudal 2013–2026, 1% faltantes,
0.12–47.4 m³/s. Excelente calidad. Ventana indirecta al flujo de Tominé.
Puente Tocancipá (2120792): caudal 1970–2026 pero 35% faltantes. Respaldo del control.
Niveles de estaciones limnimétricas (Embalse Neusa, Embalse Sisga, Tocancipá): OJO,
vienen en cm de lámina (lectura de mira), NO en cota msnm. Requieren conversión antes de usar.


Recibido — batimetría Tominé (vía Sebastian)


Batimetría oficial 2021 (SURER, GEB/Enel): capacidades, cotas, curva cota-volumen
(digitalizada con WebPlotDigitizer, anclada a valores oficiales).


Recibido — Enlaza (operación interna de Tominé)


Tominé: cota, volumen (Aforo, en volumen útil), descarga, bombeo y lluvia diaria
2015-2025, vía Excel "datos operativos Tomine_Enlaza.xlsx". Integrado al cargador
(data_ingest/tomine.py) y al tablero esta sesión.
Evaporación de Tominé: RESUELTA y definitiva. Enlaza confirmó por escrito (radicado
ENL-002443-2026-S) que Tominé NO tiene medición de evaporación ni evaporímetro propio.
Se usa evaporación ERA5-Land (flujo de calor latente), ~3.19 mm/día (~1164 mm/año),
validada en magnitud contra la evaporación medida de Neusa y Sisga.
HALLAZGO: el Excel de Enlaza trae 14 hojas, varias ocultas por defecto:
  - "Aforo": curva cota-volumen OFICIAL de Tominé (no digitalizada). Pendiente menor:
    adoptarla en curvas.py en reemplazo de la curva digitalizada actual (ver #6).
  - Series de Neusa y Sisga: sirven de VALIDACIÓN CRUZADA contra la CAR, no como fuente
    del modelo (la CAR sigue siendo la fuente única, serie más larga 2009-2026).
  - Saucío hasta 2025: RESUELVE el hueco que dejaba la serie de la CAR (que corta en
    dic-2022, antes de El Niño). Revisar si conviene usarla para extender esa serie.


En espera


Curvas cota-volumen de Neusa y Sisga: pedidas a CAR, NO vinieron en el 2º envío. Insistir
o digitalizar si aparecen en algún documento.


Cerrado (definitivo)


El Sol: la CAR confirmó POR ESCRITO que no pertenece a su red. Única vía = IDEAM/DHIME.
Estación localizada: SOL EL (21207780), activa, limnimétrica. PENDIENTE descargar de DHIME
y verificar si da caudal o solo nivel.



6. Deuda técnica de datos


Descarga máxima de Tominé (40 m³/s): sin fuente. Verificar con Enlaza.
Curva de Tominé: digitalizada aproximada (pendiente menor). Ya se tiene la tabla oficial
exacta (hoja "Aforo" del Excel Enlaza, en volumen útil); adoptarla en curvas.py daría
mayor precisión — se detectó ~7 Mm³ de discrepancia con la digitalizada en el interior
de la curva (los puntos ancla sí coinciden).
Curvas de Neusa/Sisga: fallback lineal hasta recibir las reales.
Niveles limnimétricos de la CAR: en cm, requieren curva de gastos o relación mira-cota.
Afluencias negativas: tratadas en el tablero (acotadas a cero + reporte), pero la causa
de fondo (salidas no registradas) está sin resolver para el modelo. Ver 4b.
Rango de sanidad de cota (schemas.py): genérico (2600–3100), mejorable a rango por embalse.



7. Fuentes de datos (para citar después)


Neusa, Sisga (parámetros y series): Manuales de operación + series, CAR, radicados
20261625095 y 20261071858.
Funciones de embalses (Neusa abastecimiento+regulación; Sisga solo regulación): CAR, ABC de los embalses.
Tominé (curva, capacidades, cotas): Batimetría oficial 2021, GEB/Enel (SURER).
Tominé (operación diaria: cota, volumen, descarga, bombeo, lluvia): Enlaza, Excel
"datos operativos Tomine_Enlaza.xlsx".
Tominé (evaporación): ERA5-Land (flujo de calor latente). Ausencia de evaporímetro
confirmada por Enlaza por escrito, radicado ENL-002443-2026-S.
Estaciones: Catálogo de Estaciones CAR 2020; Catálogo Nacional IDEAM (datos.gov.co).
El Sol: portal DHIME/IDEAM (dhime.ideam.gov.co).



8. Reglas de trabajo del proyecto


Construir contra interfaces con datos sintéticos; los datos reales se integran al final.
El repositorio guarda CÓDIGO, no datos: los .xlsx/.csv/.pdf van en .gitignore, viven en disco.
Un chat de Claude Code por bloque de trabajo encadenado; abrir nuevo al cambiar de frente.
La curva en curvas.py es la fuente de verdad; los escalares de embalses.py se derivan
de ella y se mantienen coherentes (hay pruebas que lo verifican).
No modificar hydrology/balance.py (validado): limpieza de datos va en el pipeline, no en el core.
pywr: solo como benchmark, nunca como núcleo del control.

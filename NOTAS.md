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
Entorno (environment/): acople a El Sol CORREGIDO con topología real y cota física
sobre la extracción de Tibitóc (ver 4e), penalizaciones diferenciadas, detección (no
forzado aún) de violación del caudal ecológico.
Ventana temporal DEFINITIVA del proyecto: 2012-01-01 a 2025-05-04, para los TRES
embalses Y la estación Saucío. Punto central: data_contracts/ventana.py
(VENTANA_INICIO, VENTANA_FIN); los loaders de Neusa/Sisga (dashboard/data_loader.py),
Tominé (data_ingest/tomine.py) y Saucío (data_ingest/saucio.py) importan de ahí, sin
fechas hardcodeadas. Justificación de los bordes:
  - Inicio 2012-01-01: (a) evaporación medida de Neusa (CAR) arranca en 2011 (antes
    no hay dato real); (b) ceros de descarga de Tominé son sospechosos 2000-2011
    (24%-100% de días en cero por año, 2011 al 100%) y caen a 5.7% en 2012; (c) con
    la ventana arrancando justo ahí, el salto real de volumen de Tominé del
    2012-01-01 (~138 Mm³ en un día) queda fuera de la cadena de diffs y ya no
    requiere corrección.
  - Fin 2025-05-04: donde termina la cobertura de la estación Saucío (fuente Enlaza).
    Saucío es el caudal natural hacia El Sol (punto de control del caudal ecológico);
    sin él no se puede evaluar esa restricción. Se prefirió esta ventana (con los
    4 componentes completos al final) sobre una más larga (hasta 2025-12-31, límite
    de la evaporación ERA5 de Tominé) que habría dejado a Saucío con un hueco final.
  - La lluvia NO restringe la ventana (medida en sitio en los tres embalses).
Cargador de Tominé (data_ingest/tomine.py): serie operativa desde Excel Enlaza (cota,
volumen útil, descarga, bombeo, lluvia) + evaporación ERA5-Land.
Cargador de Saucío (data_ingest/saucio.py): EMPALME de dos fuentes con corte limpio
(sin promediar el solape 2021-2022): CAR hasta 2022-12-31, Enlaza (hoja oculta
"S-PF-PT") desde 2023-01-01. Diagnóstico del solape (729 días) que respaldó el corte
limpio: mismas unidades (m³/s), sin desfase temporal (correlación máxima en lag=0,
0.9496), sin sesgo sistemático (regresión Enlaza ~ 1.031·CAR − 0.063). Un día
sospechoso (2021-06-21: CAR=20.03 vs Enlaza=34.52 m³/s) se deja documentado pero SIN
corregir (cae en el tramo CAR usado, un único día en toda la serie).
Tablero de exploración (dashboard/, Streamlit + Plotly): series de volumen y afluencia
de los TRES embalses, con la ventana definitiva. Límites activables, slider de fechas +
calendario sincronizado, tratamiento en dos capas de afluencias negativas con reporte
de diagnóstico, cotas corruptas de la CAR corregidas (Neusa 4, Sisga 4), y marcadores
de días con bombeo sobre la línea de volumen de Tominé.
Cargador del RONI (data_ingest/roni.py): covariable macroclimática (ENSO) para el
futuro modelo estocástico de afluencias. Fuente mensual NOAA CPC (RONI oficial desde
feb-2026, reemplaza al ONI), interpolada linealmente a diaria (NO broadcast) y
recortada a la ventana definitiva. Ver sección 5 para el detalle de la conversión.
Captación de Tibitóc (data_contracts/captaciones.py + entorno.py): extracción
modelada como escenario configurable con cota física (ver 4e).
Caudal ecológico (data_contracts/caudal_ecologico.py): umbral VMF mensual FIJADO
en el entorno, reemplaza el valor provisional de 2.0 m³/s (ver 4f). 140/140
pruebas pasan (core del modelo). El tablero es módulo aparte del core.



3. Qué falta construir


Shield de proyección cuadrática (contribución central). PENDIENTE reunión Sebastian.
Agente de RL. PENDIENTE reunión.
Modelo estocástico multivariado de afluencias (fase 1 de Sebastian): un modelo,
4 salidas (Saucío + afluencias Neusa/Sisga/Tominé), covariables precip/temp/RONI.
RONI ya integrado como covariable diaria (data_ingest/roni.py); falta el modelo en sí.
Integrar datos reales al entorno: reemplazar sintético por las series de la CAR.



4. Decisiones provisionales — VALIDAR CON SEBASTIAN


Penalizaciones diferenciadas (conceptos distintos, escala común 0-1 × peso):

Sisga: rata de descenso > 15 cm/día (manual CAR). Umbral "muy malo" = 45 cm/día. Peso 0.7.
Neusa: proximidad al mínimo (abastece acueductos, CAR ABC embalses). Peso 1.0.
Tominé: flexibilidad (amortiguador). Peso 0.5.
Los pesos NO suman 1 a propósito: califican independientemente, no reparten un total.



Acople a El Sol: CORREGIDO (ver 4e) — ya no es "suma simple de descargas + caudal
natural". Sin tiempo de viaje (se mantiene esa simplificación).
Caudal ecológico: FIJADO como umbral VMF mensual (ver 4f) — ya NO es el valor fijo
provisional de 2.0 m³/s. Pendiente de validación FINAL con Sebastian (método ya
estándar en la literatura; la decisión específica sobre el evento de Neusa 2022
es la pieza que más necesita su visto bueno).
Restricción de Sisga: sobre la cota (fiel al manual), no sobre volumen.


4b. Preguntas hidrológicas abiertas para Sebastian


Afluencias negativas (recalculado con la ventana 2012-2025): 5.49% de días negativos
en Neusa (cola hasta -32.69 m³/s), 8.65% en Sisga (cola hasta -5.62 m³/s), 9.72% en
Tominé (cola hasta -71.64 m³/s, ya con el término de bombeo restado). NO vienen de
saltos de volumen (los datos de volumen son limpios). Sospecha: salidas no registradas
en la serie de descarga de la CAR (¿captaciones de acueducto? ¿vertimientos por
aliviadero?).
PREGUNTA: ¿qué incluye exactamente la columna descarga_m3_s de la CAR?
pywr: revisado. Decisión: NO como núcleo del control (rompería la transparencia del
entorno y la limpieza del shield); SÍ como posible línea base de comparación (benchmark).


4c. Huecos de datos detectados dentro de la ventana definitiva 2012-01-01/2025-05-04


Neusa y Sisga comparten un hueco real de 61 días en volumen/descarga/precipitación:
2019-11-01 a 2019-12-31 (probable interrupción de estación CAR que afectó a ambos
embalses a la vez). PERSISTE en la ventana definitiva (está en medio del período, no
en un borde). La limpieza actual del tablero solo interpola huecos cortos (≤3 días);
este hueco de 61 días queda sin llenar en el frame final (NaN), lo mismo que varios
huecos de evaporación más largos (hasta 32 días en Neusa, 62 en Sisga — Sisga con
292 días de evaporación en NaN en total dentro de la ventana, Neusa con 266).
Efecto: 1.81% de días sin afluencia calculable en Neusa, 2.44% en Sisga, 0.02% en
Tominé (sin huecos crudos en esta ventana). Saucío (ver 4d) tiene 93 días en NaN
propios (3 en el tramo CAR, 90 en el tramo Enlaza). PENDIENTE decidir si se rellenan
esos huecos largos o se dejan como NaN (documentado, no oculto) — no se han tocado.


4d. Empalme de Saucío — APLICADO (data_ingest/saucio.py)


Caudal de Saucío (caudal natural hacia El Sol) empalmado con CORTE LIMPIO (sin
promediar el solape 2021-2022): CAR (estación 2120719, escala diaria, tipo medios)
hasta 2022-12-31 inclusive; Enlaza (hoja oculta "S-PF-PT", columna
"CAUDAL SAUCIO (m3/s)") desde 2023-01-01 hasta el fin de la ventana (2025-05-04).
Se usó el corte limpio en vez de promediar porque el diagnóstico del solape
(2021-2022, 729 días con ambas series) mostró series consistentes: mismas unidades
(m³/s), sin desfase temporal (correlación máxima en lag=0, 0.9496), sin sesgo
sistemático (regresión Enlaza ~ 1.031·CAR − 0.063, pendiente ≈1). Promediar no habría
aportado precisión y habría mezclado una serie que no es ninguna de las dos fuentes.
Día sospechoso 2021-06-21 (CAR=20.03 vs Enlaza=34.52 m³/s, mayor discrepancia del
solape): documentado en el diagnóstico (DiagnosticoSaucio), NO corregido ni
eliminado — cae en el tramo CAR usado, es un único día en 4873.
Resultado del empalme: 4873 días (2012-01-01 a 2025-05-04), 4015 de fuente CAR + 765
de fuente Enlaza + 93 huecos sin rellenar (3 en CAR: 2017-11-30, 2018-04-30,
2022-05-01; 90 en Enlaza: hueco interno 2025-01-20 a 2025-04-19 — la serie SÍ retoma
dato real del 2025-04-20 al 2025-05-04, el hueco no llega hasta el final).
Los cuatro componentes del sistema (Neusa, Sisga, Tominé, Saucío) coinciden
EXACTAMENTE en 4873 días, mismo rango — la ventana definitiva logró su objetivo.


4e. Topología El Sol y captación de Tibitóc — CORREGIDA e IMPLEMENTADA


TOPOLOGÍA CONFIRMADA (reemplaza la vieja "suma simple", que ignoraba la
extracción y podía dar caudales negativos):
  Saucío -> confluencia Sisga -> confluencia Tominé -> confluencia Neusa ->
  bocatoma Tibitóc (extracción) -> El Sol
Las CUATRO entradas llegan a la bocatoma ANTES de la extracción:
  Q_bocatoma(t) = Q_Saucío(t) + Q_desc_Sisga(t) + Q_desc_Tominé(t) + Q_desc_Neusa(t)
  Q_extraccion(t) = min(Q_Tibitoc_escenario, Q_bocatoma(t))   # COTA FÍSICA
  Q_ElSol(t) = Q_bocatoma(t) - Q_extraccion(t)                # nunca negativo
Sin términos de afluencia lateral ni Q_otros: no hay datos que los sustenten y la
ecuación cierra sin ellos.
TOPOLOGÍA RESUELTA (2026-07-22): la correcta es la del diagrama del asesor —
Neusa confluye AGUAS ARRIBA de la bocatoma de Tibitóc, como queda implementado
arriba. La frase del documento de consultoría EAAB (contrato
1-02-25300-1221-2013) que decía que Tocancipá "no [registra] el aporte del
Neusa, pues descarga al río Bogotá, aguas abajo de la Planta de Tibitóc" es un
ERROR DE REDACCIÓN de ese documento: el propio documento se contradice en dos
puntos si se lee con cuidado:
  - Describe que las compuertas de El Espino están 200 m aguas abajo de la
    desembocadura del río Neusa, y que su función es represar el río Bogotá
    para permitir a contraflujo el ingreso del agua del Neusa a la planta de
    Tibitóc. Si esa infraestructura existe para llevar agua del Neusa a la
    planta, la desembocadura está necesariamente aguas ARRIBA de la bocatoma.
  - Explica que el Neusa se descartó como fuente de abastecimiento por el
    deterioro de su calidad tras recibir la afluencia del río Checua. Si el
    Neusa entrara aguas abajo de la planta, su calidad no podría afectar el
    agua ya captada.
Ambos pasajes del mismo documento EAAB solo tienen sentido si Neusa está aguas
arriba de la bocatoma — consistente con la topología del asesor, no con la
frase aislada que la contradice. Se conserva esta referencia para que quede
trazable si alguien vuelve a consultar el documento original y se encuentra con
la misma frase aparentemente contradictoria.

Implementado en:
  - data_contracts/captaciones.py: escenarios de extracción + cota física, con
    las fuentes documentadas.
  - environment/hidraulica.py: calcular_extraccion_tibitoc() (función pura).
  - environment/entorno.py: ForzantesExternos.caudal_tibitoc_m3s (nuevo, default
    escenario histórico); EstadoSistema.caudal_bocatoma_m3s (nuevo);
    ResultadoPaso.caudal_extraccion_m3s / cota_fisica_activada /
    deficit_extraccion_m3s (nuevo, diagnóstico por paso).

Escenarios de extracción de Tibitóc (NO hay serie pública de captación; no hay
estación aguas abajo de la bocatoma para estimarla por diferencia — Puente
Tocancipá, que se había considerado, está AGUAS ARRIBA de la bocatoma):
  - HISTÓRICO (4.5 m³/s constante, default): derivación promedio del informe de
    recorrido del río Bogotá, CAR. VALIDACIÓN CRUZADA: con este escenario el
    modelo da una media de 6.34 m³/s en El Sol sobre la ventana del proyecto,
    coherente con los 6.57 m³/s que la CAR reporta aguas abajo de la
    desembocadura del Neusa — usando el dato de extracción de la CAR se
    reproduce (sin ajustar nada) el caudal que la misma CAR mide aguas abajo.
    Corresponde a la operación DURANTE la ventana de análisis (2012-2025-05-04).
  - AMPLIADO (8.0 m³/s constante): caudal tratado tras la optimización reciente
    de la planta (nuevos trenes de tratamiento). Representa la operación HACIA
    la que evoluciona el sistema, NO la operación histórica de la ventana.
  SALVEDAD METODOLÓGICA: ambos escenarios tienen respaldo documental, pero NO
  representan el mismo período — no promediarlos ni tratarlos como
  intercambiables. Se evalúan ambos para sensibilidad, no como alternativas
  igualmente representativas del mismo momento.

Análisis de sensibilidad hecho ANTES de implementar (ver historial de chat, no
en el repo): con la cota física, cero días negativos por construcción en los
dos escenarios. Días con cota activada (el río no alcanza a dar el nominal):
histórico 0.25% de los días (casi nunca limita), ampliado 15.87% (~1 de 6 días).
Un tercer supuesto de la Resolución 760/2011 (8.0/9.56 m³/s estacional) se
evaluó pero NO se incluyó como escenario permanente del código: activaba la
cota 24.52% del tiempo, el más exigente de los tres.



4f. Umbral de caudal ecológico — VMF FIJADO en el entorno (2026-07-22)


MÉTODO: VMF (Variable Monthly Flow, Pastor et al. 2014), usado por Gerten et al.
(2013) para operacionalizar el límite planetario del agua dulce de Rockström a
escala de cuenca. Por mes calendario, preserva un % del caudal medio de ESE mes
(MMF) según su relación con el caudal medio anual (MAF): 60% si MMF ≤ 0.4·MAF
(bajo), 45% si intermedio, 30% si MMF > 0.8·MAF (alto).

ORIGEN DE LOS DATOS: MAF y los 12 MMF se calculan sobre Q_natural = Q_Saucío +
afluencia_Neusa + afluencia_Sisga + afluencia_Tominé (afluencias por balance
inverso, NO descargas reguladas), ventana completa 2012-01-01/2025-05-04.
MAF = 12.432 m³/s.

    mes  MMF (m³/s)  régimen      EFR (m³/s)
    ene   3.864      bajo         2.32
    feb   3.549      bajo         2.13
    mar   6.090      intermedio   2.74
    abr   9.768      intermedio   4.40
    may  12.975      alto         3.89
    jun  21.766      alto         6.53
    jul  25.019      alto         7.51
    ago  19.822      alto         5.95
    sep  12.429      alto         3.73
    oct  15.093      alto         4.53
    nov  12.673      alto         3.80
    dic   5.551      intermedio   2.50

IMPLEMENTADO en data_contracts/caudal_ecologico.py: función q_eco_m3s(mes),
diccionarios MAF_M3S/MMF_M3S/REGIMEN_MES/EFR_VMF_M3S, y umbral_fijo_m3s(valor)
para revertir a un umbral constante (o sustituir por otro método) sin tocar
environment/entorno.py. ConfigEntorno.calcular_q_eco_m3s (antes q_eco_m3s fijo)
es ahora un Callable[[int], float]; por defecto usa el VMF. ForzantesExternos
gana el campo `mes` (requerido, sin default) para que el entorno sepa qué
umbral aplicar en cada paso — la violación se sigue DETECTANDO, no forzando
(el forzado es el shield, aún por construir). ResultadoPaso reporta
q_eco_aplicado_m3s explícitamente (auditable sin recalcular).

TRES SALVEDADES DECLARADAS (releer antes de citar el umbral en el entregable):

1. Las afluencias se estiman por balance hídrico inverso; su validación es de
   consistencia interna (cierre del balance con residuo 3.75% incluyendo
   vertimiento y bombeo — NO es validación física independiente, es casi una
   identidad algebraica) más el anclaje del patrón estacional a Saucío (caudal
   MEDIDO, aguas arriba de los tres embalses): Spearman ρ=0.916, clasificación
   VMF coincidente en 8/12 meses. Marzo/abril difieren por estar cerca del
   límite 0.4/0.8 en ambas series; septiembre/octubre muestran una discrepancia
   real no explicada (Q_natural los ve "alto", Saucío los ve "intermedio"). NO
   existe aforo directo de afluencias para validación absoluta.

2. El evento de vertimiento de Neusa (jul-nov 2022, 135 días consecutivos,
   trayectoria suave ascenso-pico-descenso, coincidente con La Niña 2022):
   identificado y caracterizado; la magnitud del volumen vertido depende de la
   formulación del cálculo y de la cota real del aliviadero, PENDIENTE de
   verificación con la CAR. DECISIÓN: el umbral se calcula sobre la serie
   COMPLETA, incluyendo estos meses — excluir un evento hidrológico real
   recortaría la variabilidad natural que el VMF intenta capturar. VERIFICADO
   que SÍ afecta la clasificación (contrario a la expectativa inicial):
   excluir jul-nov 2022 cambia el régimen de 2 de 12 meses (octubre, efecto
   directo; abril, por el desplazamiento del MAF de referencia — su propia MMF
   no cambia). MAF cae 12.5% sin el evento. Esta sensibilidad queda cubierta
   por el análisis de robustez ±20% (ver más abajo).

3. La extracción de Tibitóc se modela por escenarios documentados (histórico
   4.5 m³/s, ampliado 8.0 m³/s) con cota física, ante la ausencia de serie
   pública de captación (ver 4e).

ANÁLISIS DE SENSIBILIDAD (EFR × 0.8 / 1.0 / 1.2), ambos escenarios de extracción:

    escenario   x0.8    x1.0    x1.2   (% días en violación)
    histórico   18.13   27.13   35.66
    ampliado    58.20   66.75   74.46

Racha máxima: histórico 30→37→73 días; ampliado 84→132→156 días. CONCLUSIÓN
ROBUSTA: en las tres variantes, ampliado transgrede 2.1–3.2x más días que
histórico y su racha máxima es 2–3x mayor — la incertidumbre de ±20% en el
umbral no cambia la conclusión cualitativa del proyecto.

PENDIENTE PARA EL ASESOR: validar la decisión de incluir el evento de Neusa
2022 sin ajustar, y la magnitud real del volumen vertido (verificar con la
CAR). Scripts fuente en scratch_vmf/ (calcular_vmf_v2.py, cálculo del umbral;
calcular_vmf_v3_validacion.py, las tres piezas de cierre: validación
estacional, sensibilidad, verificación del caveat de Neusa 2022).


5. Datos: recibido / en espera / cerrado

Recibido — primer envío CAR (radicado 20261625095)


Neusa y Sisga: cota, volumen, descarga, lluvia diaria 2009–2026. Calidad excelente (~1% faltantes).
Saucío: caudal medio diario 1970–2022 (corta antes de El Niño 2023-24, ~12% faltantes).
YA EMPALMADO con Enlaza para extenderlo hasta 2025-05-04 — ver 4d y data_ingest/saucio.py.
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
(Excel trae desde el año 2000), vía Excel "datos operativos Tomine_Enlaza.xlsx".
Integrado al cargador (data_ingest/tomine.py) y al tablero. Ventana DEFINITIVA:
2012-01-01 a 2025-05-04 (ver sección 2; antes se probó con 2009-07-01 y 2010-01-01
como inicio, y con 2025-12-31 como fin, hasta fijar los bordes definitivos).
Salto real de volumen del 2012-01-01 (~138 Mm³ en un día): con la ventana actual
arrancando justo ahí, la transición anómala queda fuera de la ventana y no requiere
corrección (ver sección 2).
Evaporación de Tominé: RESUELTA y definitiva. Enlaza confirmó por escrito (radicado
ENL-002443-2026-S) que Tominé NO tiene medición de evaporación ni evaporímetro propio.
Se usa evaporación ERA5-Land (flujo de calor latente), ~3.19 mm/día (~1164 mm/año),
validada en magnitud contra la evaporación medida de Neusa y Sisga. Serie ERA5 en
Serie_Evaporacion_Tomine_2009_2025.csv (extiende la anterior, 2010-2025, un año hacia
atrás; idéntica en el período que se solapa — se verificó diferencia 0.0).
HALLAZGO: el Excel de Enlaza trae 14 hojas, varias ocultas por defecto:
  - "Aforo": curva cota-volumen OFICIAL de Tominé (no digitalizada). Pendiente menor:
    adoptarla en curvas.py en reemplazo de la curva digitalizada actual (ver #6).
  - Series de Neusa y Sisga: sirven de VALIDACIÓN CRUZADA contra la CAR, no como fuente
    del modelo (la CAR sigue siendo la fuente única, serie más larga 2009-2026).
  - Saucío: RESUELVE el hueco que dejaba la serie de la CAR (que corta en dic-2022,
    antes de El Niño). OJO: esta hoja en realidad llega solo hasta 2025-05-04 (no
    dic-2025 como se pensó inicialmente), con ~90 días faltantes dispersos (todos
    en 2025). YA EMPALMADO con la CAR (corte limpio, sin promediar) — ver 4d y
    data_ingest/saucio.py. Este límite de cobertura (2025-05-04) es, de hecho, el
    que fija el fin de la ventana definitiva del proyecto (ver sección 2).


Recibido — RONI (covariable macroclimática ENSO)


RONI (Relative Oceanic Niño Index), mensual, 2009-2025 (RONI_2009_2025_mensual.csv).
Fuente: NOAA Climate Prediction Center, tabla oficial (base 1991-2020, ERSSTv5). NOAA
CPC adoptó el RONI como índice oficial de monitoreo ENSO en feb-2026, en reemplazo
del ONI. Integrado en data_ingest/roni.py como covariable diaria para el futuro
modelo estocástico de afluencias (no se usa en el balance ni en el entorno).
CONVERSIÓN MENSUAL -> DIARIA: interpolación LINEAL entre los valores mensuales
(anclados al día 1 del mes central de cada media móvil de 3 meses: DJF->enero,
JFM->febrero, etc.), NO broadcast. El RONI ya es una media móvil de 3 meses, señal
suave por naturaleza; el broadcast introduciría escalones artificiales el día 1 de
cada mes sin sentido físico. La columna categórica "fase" (Niño/Niña/Neutral,
umbral ±0.5) se propaga hacia adelante (ffill) desde el ancla mensual, no se
interpola. Recortado a la ventana definitiva del proyecto (2012-01-01/2025-05-04):
4873 días, igual que los demás componentes.
Verificado el evento El Niño 2023-24: el ancla de julio 2023 (JJA=0.6) ya supera el
umbral 0.5; bajo interpolación lineal estricta el cruce diario cae el 2023-06-17
(no en julio), consecuencia esperada de interpolar entre el ancla de junio (0.4) y
julio (0.6), no un error. El pico de 1.5 se sostiene plano del 2023-11-01 al
2023-12-01 (anclas OND y NDJ, ambas 1.5).
Advertencia de NOAA: los valores más recientes del RONI pueden ajustarse hasta dos
meses después de publicados (filtro de alta frecuencia de ERSSTv5). No afecta la
ventana del proyecto (cierra 2025-05-04, ya estabilizada).


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
Rango de sanidad de cota (schemas.py): genérico (2550–3100; CORREGIDO 2026-07-22,
NOTAS.md tenía 2600 por error de transcripción — el código siempre usó 2550, valor
necesario para no rechazar la cota mínima de Tominé, ~2566.63), mejorable a rango
por embalse.



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
Captación Tibitóc, escenario histórico (4.5 m³/s): informe de recorrido del río
Bogotá, CAR. PENDIENTE DE TRAZABILIDAD (sin radicado, fecha ni página; completar
antes de publicar, igual que el resto de fuentes débiles marcadas en el proyecto).
Validado cruzado contra el caudal medido por la CAR aguas abajo de la
desembocadura del Neusa (6.57 m³/s, ver 4e) — ese valor de referencia también
está PENDIENTE DE TRAZABILIDAD, y la validación presupone (sin verificar) que
ese punto equivale al punto de control El Sol.
Captación Tibitóc, escenario ampliado (8.0 m³/s): caudal tratado tras
optimización reciente de la planta (reportado, sin radicado formal aún).



8. Reglas de trabajo del proyecto


Construir contra interfaces con datos sintéticos; los datos reales se integran al final.
El repositorio guarda CÓDIGO, no datos: los .xlsx/.csv/.pdf van en .gitignore, viven en disco.
Un chat de Claude Code por bloque de trabajo encadenado; abrir nuevo al cambiar de frente.
La curva en curvas.py es la fuente de verdad; los escalares de embalses.py deben
mantenerse coherentes con ella. CORREGIDO (2026-07-22, auditoría adversarial): hasta
esta fecha esta afirmación era FALSA — no existía ninguna prueba que comparara los
extremos de la curva contra embalses.py (test_curvas.py comparaba contra literales
hardcodeados, coincidencia solo por sincronización manual). Ahora sí existe:
test_curvas.py::TestPuntosAnclaTomIne (incluye test_extremos_de_la_curva_coinciden_con_embalses_py)
compara explícitamente CURVA_TOMINE contra EMBALSES["Tomine"] y falla si se desincronizan.
No modificar hydrology/balance.py (validado): limpieza de datos va en el pipeline, no en el core.
pywr: solo como benchmark, nunca como núcleo del control.

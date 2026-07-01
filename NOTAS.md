1. los parámetros de Tominé están pendientes de verificar con batimetría SURER
2. los costos diferenciados son provisionales, validar con Sebastian
3. volumen_a_cota como lineal y provisional, con una nota explícita de que hay que reemplazarlo por la curva batimétrica real (la tabla cota-volumen)

1. Qué es el proyecto (2-3 líneas).
Una frase que recuerde el objetivo: framework PB-CRL de RL con restricciones para operar tres embalses del Sistema Agregado Norte garantizando el caudal ecológico en El Sol. Sirve para que cualquier chat nuevo entienda el contexto en segundos.
2. Estado actual: qué está construido y probado.
La lista de módulos terminados con su estado. Balance hídrico (conservación de masa validada), entorno (acople a El Sol, cota física, penalizaciones diferenciadas), parámetros de embalses con fuente, curva cota-volumen de Tominé. Y el número de pruebas que pasan (78). Esto le dice a un chat nuevo "esto ya existe, no lo reconstruyas".
3. Qué falta construir.
El shield de proyección y el agente de RL, marcados como pendientes de la reunión con Sebastian. Así queda claro dónde retomar.
4. Decisiones provisionales que Sebastian debe validar.
Esta es la sección más importante para no perder el hilo. Lista explícita de lo que decidiste tú como "primera versión razonable" y que él puede cambiar:

Penalizaciones diferenciadas por embalse (Sisga: rata de descenso 15 cm/día; Neusa: proximidad al mínimo por abastecimiento; Tominé: flexibilidad).
Pesos relativos 1.0 (Neusa) / 0.7 (Sisga) / 0.5 (Tominé).
Umbral de descenso "muy malo" de Sisga en 45 cm/día.
Acople a El Sol como suma simple, sin tiempo de viaje del agua.
Caudal ecológico Q_eco = 2.0 m³/s (valor inicial).

5. Datos: qué tienes y qué está pendiente.
El estado de las fuentes, que es donde más fácil te pierdes:

Recibido: Neusa y Sisga (cota, volumen, descarga, lluvia, 2009-2026, CAR); Saucío caudal hasta 2022 (CAR); curva batimétrica de Tominé (digitalizada del PDF SURER 2021).
Solicitado y en espera: correo ampliado a CAR (El Sol, Saucío 2023-2025, evaporación, descargas Sisga, canal Achurí); solicitud a Enlaza/GEB (operación interna de Tominé); ambas por PQR y correo.
Pendiente de conseguir: curvas cota-volumen de Neusa y Sisga (van con la respuesta de la CAR); tabla cm-a-cm exacta de Tominé (en el informe completo de Enlaza, no en la presentación).

6. Parámetros con fuente incierta (deuda técnica de datos).
Los que aún no tienen respaldo firme y hay que verificar:

Descarga máxima de Tominé (40 m³/s): sin fuente documentada.
Curva de Tominé: digitalizada aproximada, reemplazable por tabla oficial cm-a-cm.
Curvas de Neusa/Sisga: fallback lineal provisional hasta que lleguen las reales.
Rango de sanidad de cota en schemas.py: genérico (2600-3100), mejorable a rango por embalse.

7. Fuentes de datos (referencia rápida).
De dónde salió cada cosa, para citar después: manuales de operación CAR (Neusa, Sisga) del radicado 20261625095; batimetría Tominé 2021 (GEB/Enel); catálogo de estaciones CAR; portal DHIME/IDEAM para El Sol.
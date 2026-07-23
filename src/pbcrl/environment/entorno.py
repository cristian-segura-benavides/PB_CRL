"""
Entorno de simulación para la operación coordinada de los tres embalses.

Flujo de un paso de simulación
-------------------------------
1. El agente proporciona los caudales suministrados por cada embalse (acciones).
1b. (Opcional, ConfigEntorno.con_shield=True) La acción propuesta pasa PRIMERO
    por el shield de proyección cuadrática (shield.proyeccion.proyectar), que
    la corrige a la más cercana que respeta las restricciones del sistema
    (cajas, rata de descenso de Sisga, caudal ecológico conjunto) — ver
    shield/README.md. Esto ocurre ANTES del recorte físico del paso 2: son dos
    mecanismos distintos, el shield corrige la INTENCIÓN, el recorte físico
    verifica DESPUÉS si esa acción ya corregida es alcanzable con el agua
    realmente disponible ese día en cada embalse.
2. El entorno recorta las acciones (ya corregidas por el shield si aplica) a
   lo físicamente posible (recortar_suministro).
3. El entorno aplica el balance hídrico hacia adelante en cada embalse (paso_embalse).
4. Se calcula el caudal en la bocatoma de Tibitóc y el caudal en El Sol, según la
   topología confirmada (ver data_contracts.captaciones):
       Saucío -> Sisga -> Tominé -> Neusa -> bocatoma (extracción) -> El Sol
       Q_bocatoma  = Q_natural (Saucío) + Σ(suministro_real + vertimiento) de los
                     tres embalses
       Q_extraccion = min(Q_Tibitoc_nominal, Q_bocatoma)   # cota física
       Q_sol        = Q_bocatoma - Q_extraccion             # siempre >= 0
   Q_Tibitoc_nominal viene de ForzantesExternos.caudal_tibitoc_m3s (constante de
   escenario por defecto, o serie real día a día si se obtiene).
5. Se detecta si se viola el caudal ecológico mínimo (Q_sol < Q_eco(mes)), con
   Q_eco dado por ConfigEntorno.calcular_q_eco_m3s (por defecto, el umbral VMF
   mensual de data_contracts.caudal_ecologico — ver ese módulo para el método
   y las salvedades documentadas).
6. Se calculan las penalizaciones por embalse.
7. Se devuelve el nuevo estado, las penalizaciones, y un dict de información adicional.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pbcrl.data_contracts.captaciones import CAUDAL_TIBITOC_HISTORICO_M3S
from pbcrl.data_contracts.curvas import volumen_a_cota
from pbcrl.data_contracts.embalses import EMBALSES, ParametrosEmbalse
from pbcrl.environment.config import ConfigEntorno
from pbcrl.environment.hidraulica import (
    calcular_extraccion_tibitoc,
    paso_embalse,
    recortar_suministro,
)
from pbcrl.environment.penalizaciones import (
    pen_descenso_nivel_sisga,
    pen_proximidad_minimo,
)
from pbcrl.shield.proyeccion import DiagnosticoShield, proyectar
from pbcrl.shield.restricciones import EstadoShield

# Orden canónico de los embalses (debe mantenerse consistente en todo el código)
_NOMBRES_EMBALSES: tuple[str, ...] = ("Neusa", "Sisga", "Tomine")

# Tolerancia numérica para la detección de violación del caudal ecológico. El
# shield (cuando está conectado) puede dejar Q_sol EXACTAMENTE en el borde de
# Q_eco (esa es la naturaleza de una restricción activa en una proyección);
# recalcular Q_sol desde Q_bocatoma-Q_extracción pasa por una secuencia de
# operaciones de punto flotante distinta a la que usó el shield para fijar la
# acción, así que puede quedar ~1e-15 por debajo del umbral sin que exista
# ningún déficit físico real. Una comparación estricta (`<` sin tolerancia)
# marcaría eso como violación por ruido de redondeo, no por un problema real
# — encontrado al conectar el shield al entorno (ver NOTAS.md).
_TOL_VIOLACION_ECOLOGICA_M3S = 1e-9


# ---------------------------------------------------------------------------
# Dataclasses de entrada y salida
# ---------------------------------------------------------------------------

@dataclass
class EstadoSistema:
    """Estado observable del sistema en un instante dado.

    Atributos
    ---------
    volumen_mm3 : dict[str, float]
        Volumen almacenado en cada embalse [Mm³].
    cota_m : dict[str, float]
        Cota de cada embalse [m.s.n.m.]. Derivada del volumen con curva lineal (PROVISIONAL).
    caudal_natural_m3s : float
        Caudal natural aportado por Saucío [m³/s]. Primer término de la bocatoma.
    caudal_bocatoma_m3s : float
        Caudal disponible en la bocatoma de Tibitóc, ANTES de la extracción [m³/s]:
        Saucío + descargas (suministro real + vertimiento) de Sisga, Tominé y Neusa.
    caudal_sol_m3s : float
        Caudal en el punto de control El Sol [m³/s]: caudal_bocatoma_m3s menos la
        extracción real de Tibitóc (acotada por el caudal disponible). Nunca negativo.
    """

    volumen_mm3: dict[str, float]
    cota_m: dict[str, float]
    caudal_natural_m3s: float
    caudal_bocatoma_m3s: float
    caudal_sol_m3s: float


@dataclass
class ForzantesExternos:
    """Entradas exógenas requeridas en cada paso de simulación.

    Representan variables no controlables por el agente (clima, hidrología natural).
    En producción vendrán de series de tiempo reales o del modelo estocástico.

    Atributos
    ---------
    afluencia_m3s : dict[str, float]
        Afluencia natural a cada embalse [m³/s].
    precipitacion_mm : dict[str, float]
        Precipitación sobre el espejo de cada embalse [mm/día].
    evaporacion_mm : dict[str, float]
        Evaporación sobre el espejo de cada embalse [mm/día].
    caudal_natural_m3s : float
        Caudal natural (no regulado) que llega a la bocatoma [m³/s]. Estación
        Saucío en datos reales.
    mes : int
        Mes calendario del paso actual (1-12). Determina el umbral de caudal
        ecológico aplicado (ver ConfigEntorno.calcular_q_eco_m3s — por
        defecto, el VMF mensual de data_contracts.caudal_ecologico). Requerido
        explícitamente (sin valor por defecto) para evitar evaluar la
        violación ecológica contra el mes equivocado por un descuido silencioso.
    caudal_tibitoc_m3s : float
        Caudal NOMINAL que la Planta de Tibitóc busca captar en este paso [m³/s]
        (ver data_contracts.captaciones). La extracción real se acota por el
        caudal disponible en la bocatoma (nunca deja a El Sol en negativo).
        Por defecto usa el escenario HISTÓRICO (4.5 m³/s constante); para el
        escenario AMPLIADO u otro, pasar
        ``data_contracts.captaciones.caudal_tibitoc_nominal(escenario)``, o una
        serie real día a día si se obtiene — no requiere cambios en el entorno.
    """

    afluencia_m3s: dict[str, float]
    precipitacion_mm: dict[str, float]
    evaporacion_mm: dict[str, float]
    caudal_natural_m3s: float
    mes: int
    caudal_tibitoc_m3s: float = CAUDAL_TIBITOC_HISTORICO_M3S


@dataclass
class ResultadoPaso:
    """Resultado de un paso de simulación.

    Atributos
    ---------
    estado : EstadoSistema
        Nuevo estado del sistema tras aplicar las acciones y los forzantes.
    penalizaciones : dict[str, float]
        Penalización por embalse en [0, peso_embalse].
    suministro_real_m3s : dict[str, float]
        Caudal suministrado real tras aplicar los recortes físicos [m³/s].
    vertimiento_m3s : dict[str, float]
        Caudal vertido por el aliviadero de cada embalse [m³/s].
    violacion_ecologica : bool
        True si el caudal en El Sol es inferior al caudal ecológico mínimo del
        mes (ver q_eco_aplicado_m3s).
    q_eco_aplicado_m3s : float
        Umbral de caudal ecológico efectivamente usado en este paso [m³/s]
        (ConfigEntorno.calcular_q_eco_m3s aplicado al mes de ForzantesExternos).
        Se reporta explícitamente para que sea auditable sin recalcularlo.
    caudal_extraccion_m3s : float
        Extracción REAL de Tibitóc en este paso [m³/s]: min(caudal nominal del
        escenario/serie, caudal disponible en la bocatoma). Nunca supera el
        caudal disponible.
    cota_fisica_activada : bool
        True si el caudal disponible en la bocatoma fue menor que el caudal
        nominal solicitado (la planta no pudo captar su caudal nominal completo).
    deficit_extraccion_m3s : float
        caudal_tibitoc_m3s (nominal) - caudal_extraccion_m3s (real) en este paso.
        Cero cuando la cota física no se activa. Para reportar el "déficit medio"
        de un escenario, promediar este campo (o solo sobre los pasos con
        cota_fisica_activada=True) a lo largo de una corrida.
    deficit_volumen_mm3 : dict[str, float]
        Masa "conjurada" por embalse al activarse el clamp inferior de volumen en
        `paso_embalse` (evaporación mayor al agua disponible cerca del volumen
        muerto) [Mm³]. Cero en condiciones normales. NO es una entrada física real;
        es la magnitud del ajuste. Acumulable a lo largo de una corrida para medir
        cuánta masa fue "perdonada" por el clamp (ver hidraulica.paso_embalse).
    diagnostico_shield : DiagnosticoShield | None
        Diagnóstico del shield en este paso (acción propuesta vs. proyectada,
        qué restricciones estaban violadas/activas, si el conjunto factible
        era vacío) — ver shield.proyeccion.DiagnosticoShield. None si
        ConfigEntorno.con_shield es False (el shield no se ejecutó este paso).
    """

    estado: EstadoSistema
    penalizaciones: dict[str, float]
    suministro_real_m3s: dict[str, float]
    vertimiento_m3s: dict[str, float]
    violacion_ecologica: bool
    q_eco_aplicado_m3s: float
    caudal_extraccion_m3s: float
    cota_fisica_activada: bool
    deficit_extraccion_m3s: float
    deficit_volumen_mm3: dict[str, float]
    diagnostico_shield: "DiagnosticoShield | None" = None


# ---------------------------------------------------------------------------
# Entorno principal
# ---------------------------------------------------------------------------

class EntornoEmbalses:
    """Entorno de simulación para la operación coordinada de Neusa, Sisga y Tominé.

    Parámetros
    ----------
    config : ConfigEntorno | None
        Configuración de penalizaciones y caudal ecológico.
        Si es None se usa la configuración por defecto.
    embalses : dict[str, ParametrosEmbalse] | None
        Parámetros físicos de los embalses.
        Si es None se usan los valores de data_contracts.embalses.EMBALSES.
    """

    def __init__(
        self,
        config: ConfigEntorno | None = None,
        embalses: dict[str, ParametrosEmbalse] | None = None,
    ) -> None:
        self.config = config or ConfigEntorno()
        self.embalses = embalses or {k: EMBALSES[k] for k in _NOMBRES_EMBALSES}
        self._estado: EstadoSistema | None = None
        # Guardamos la cota del paso anterior para calcular la rata de descenso del Sisga
        self._cota_anterior_m: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Estado actual (solo lectura)
    # ------------------------------------------------------------------

    @property
    def estado(self) -> EstadoSistema:
        if self._estado is None:
            raise RuntimeError("El entorno no ha sido inicializado. Llama reset() primero.")
        return self._estado

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        volumenes_iniciales_mm3: dict[str, float] | None = None,
        caudal_natural_inicial_m3s: float = 1.0,
    ) -> EstadoSistema:
        """Reinicia el entorno al estado inicial.

        Parámetros
        ----------
        volumenes_iniciales_mm3 : dict[str, float] | None
            Volumen inicial de cada embalse [Mm³].
            Si es None, se inicializa al 70 % de la capacidad útil de cada embalse.
        caudal_natural_inicial_m3s : float
            Caudal natural inicial en El Sol [m³/s].

        Retorna
        -------
        EstadoSistema
            Estado inicial del sistema.
        """
        if volumenes_iniciales_mm3 is None:
            volumenes_iniciales_mm3 = {
                nombre: (
                    params.capacidad_min_mm3
                    + 0.70 * (params.capacidad_max_mm3 - params.capacidad_min_mm3)
                )
                for nombre, params in self.embalses.items()
            }

        cotas = {
            nombre: volumen_a_cota(vol, nombre, self.embalses[nombre])
            for nombre, vol in volumenes_iniciales_mm3.items()
        }

        self._cota_anterior_m = dict(cotas)

        # Simplificación de reset (sin extracción/descargas aún): bocatoma = El Sol = natural.
        self._estado = EstadoSistema(
            volumen_mm3=dict(volumenes_iniciales_mm3),
            cota_m=cotas,
            caudal_natural_m3s=caudal_natural_inicial_m3s,
            caudal_bocatoma_m3s=caudal_natural_inicial_m3s,
            caudal_sol_m3s=caudal_natural_inicial_m3s,
        )
        return self._estado

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        acciones_m3s: dict[str, float],
        forzantes: ForzantesExternos,
    ) -> ResultadoPaso:
        """Avanza la simulación un paso de tiempo diario.

        Parámetros
        ----------
        acciones_m3s : dict[str, float]
            Caudal que el agente desea suministrar por cada embalse [m³/s].
            Claves: 'Neusa', 'Sisga', 'Tomine'.
        forzantes : ForzantesExternos
            Entradas exógenas del paso actual (afluencia, precipitación,
            evaporación, caudal natural en El Sol).

        Retorna
        -------
        ResultadoPaso
            Nuevo estado, penalizaciones, caudales reales, e indicador de violación.

        Raises
        ------
        RuntimeError
            Si reset() no ha sido llamado previamente.
        """
        if self._estado is None:
            raise RuntimeError("Llama reset() antes de step().")

        # --- Shield de proyección (opcional) — corrige la INTENCIÓN antes del
        # recorte físico. Ver el flujo documentado en el encabezado del módulo.
        diagnostico_shield: DiagnosticoShield | None = None
        acciones_efectivas = acciones_m3s
        if self.config.con_shield:
            estado_shield = EstadoShield(
                volumen_mm3=dict(self._estado.volumen_mm3),
                afluencia_m3s=dict(forzantes.afluencia_m3s),
                precipitacion_mm=dict(forzantes.precipitacion_mm),
                evaporacion_mm=dict(forzantes.evaporacion_mm),
                caudal_saucio_m3s=forzantes.caudal_natural_m3s,
                mes=forzantes.mes,
                caudal_tibitoc_nominal_m3s=forzantes.caudal_tibitoc_m3s,
            )
            diagnostico_shield = proyectar(estado_shield, acciones_m3s)
            acciones_efectivas = diagnostico_shield.accion_proyectada

        nuevos_volumenes: dict[str, float] = {}
        suministro_real: dict[str, float] = {}
        vertimiento_mm3: dict[str, float] = {}
        deficit_volumen_mm3: dict[str, float] = {}

        # --- Paso hidráulico para cada embalse ---
        for nombre, params in self.embalses.items():
            vol_actual = self._estado.volumen_mm3[nombre]

            # 1. Recortar la acción (ya corregida por el shield si aplica) a
            # lo físicamente posible
            q_real = recortar_suministro(
                suministro_pedido_m3s=acciones_efectivas.get(nombre, 0.0),
                volumen_actual_mm3=vol_actual,
                params=params,
            )
            suministro_real[nombre] = q_real

            # 2. Aplicar balance hídrico hacia adelante
            v_nuevo, vert_mm3, deficit_mm3 = paso_embalse(
                volumen_actual_mm3=vol_actual,
                afluencia_m3s=forzantes.afluencia_m3s.get(nombre, 0.0),
                suministro_m3s=q_real,
                precipitacion_mm=forzantes.precipitacion_mm.get(nombre, 0.0),
                evaporacion_mm=forzantes.evaporacion_mm.get(nombre, 0.0),
                params=params,
            )
            nuevos_volumenes[nombre] = v_nuevo
            vertimiento_mm3[nombre] = vert_mm3
            deficit_volumen_mm3[nombre] = deficit_mm3

        # --- Nuevas cotas ---
        nuevas_cotas = {
            nombre: volumen_a_cota(vol, nombre, self.embalses[nombre])
            for nombre, vol in nuevos_volumenes.items()
        }

        # --- Caudal en la bocatoma de Tibitóc (antes de la extracción) ---
        # Topología: Saucío -> Sisga -> Tominé -> Neusa -> bocatoma (extracción) -> El Sol.
        # Q_bocatoma = Q_natural (Saucío) + Σ(suministro_real + vertimiento) de los tres
        # embalses. Factor de conversión de vertimiento: Mm³/día → m³/s.
        _MM3_DIA_A_M3S = 1e6 / 86_400.0

        q_bocatoma = forzantes.caudal_natural_m3s
        for nombre in self.embalses:
            q_bocatoma += suministro_real[nombre]
            q_bocatoma += vertimiento_mm3[nombre] * _MM3_DIA_A_M3S

        # --- Extracción de Tibitóc, acotada por el caudal disponible ---
        # La planta no puede captar más agua de la que el río trae en la bocatoma;
        # esta cota es la que garantiza Q_sol >= 0 (ver data_contracts.captaciones).
        q_extraccion = calcular_extraccion_tibitoc(q_bocatoma, forzantes.caudal_tibitoc_m3s)
        cota_activada = q_bocatoma < forzantes.caudal_tibitoc_m3s
        deficit_extraccion = forzantes.caudal_tibitoc_m3s - q_extraccion

        # --- Caudal en El Sol ---
        q_sol = q_bocatoma - q_extraccion

        # --- Detección de violación del caudal ecológico ---
        # Umbral dependiente del mes (por defecto, VMF — ver data_contracts.caudal_ecologico).
        q_eco = self.config.calcular_q_eco_m3s(forzantes.mes)
        violacion = q_sol < q_eco - _TOL_VIOLACION_ECOLOGICA_M3S

        # --- Penalizaciones ---
        penalizaciones: dict[str, float] = {}

        # Sisga: rata de descenso de nivel
        penalizaciones["Sisga"] = pen_descenso_nivel_sisga(
            cota_anterior_m=self._cota_anterior_m["Sisga"],
            cota_actual_m=nuevas_cotas["Sisga"],
            config=self.config,
        )

        # Neusa: proximidad al nivel mínimo
        penalizaciones["Neusa"] = pen_proximidad_minimo(
            cota_actual_m=nuevas_cotas["Neusa"],
            params=self.embalses["Neusa"],
            peso=self.config.peso_neusa,
        )

        # Tominé: flexibilidad (mismo concepto que Neusa, menor peso)
        penalizaciones["Tomine"] = pen_proximidad_minimo(
            cota_actual_m=nuevas_cotas["Tomine"],
            params=self.embalses["Tomine"],
            peso=self.config.peso_tomine,
        )

        # --- Actualizar estado interno ---
        self._cota_anterior_m = dict(nuevas_cotas)
        self._estado = EstadoSistema(
            volumen_mm3=nuevos_volumenes,
            cota_m=nuevas_cotas,
            caudal_natural_m3s=forzantes.caudal_natural_m3s,
            caudal_bocatoma_m3s=q_bocatoma,
            caudal_sol_m3s=q_sol,
        )

        # Vertimiento en m³/s para el resultado (más útil externamente)
        vertimiento_m3s = {
            nombre: vert * _MM3_DIA_A_M3S for nombre, vert in vertimiento_mm3.items()
        }

        return ResultadoPaso(
            estado=self._estado,
            penalizaciones=penalizaciones,
            suministro_real_m3s=suministro_real,
            vertimiento_m3s=vertimiento_m3s,
            violacion_ecologica=violacion,
            q_eco_aplicado_m3s=q_eco,
            caudal_extraccion_m3s=q_extraccion,
            cota_fisica_activada=cota_activada,
            deficit_extraccion_m3s=deficit_extraccion,
            deficit_volumen_mm3=deficit_volumen_mm3,
            diagnostico_shield=diagnostico_shield,
        )

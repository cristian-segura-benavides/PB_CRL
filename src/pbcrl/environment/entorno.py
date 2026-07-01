"""
Entorno de simulación para la operación coordinada de los tres embalses.

Flujo de un paso de simulación
-------------------------------
1. El agente proporciona los caudales suministrados por cada embalse (acciones).
2. El entorno recorta las acciones a lo físicamente posible (recortar_suministro).
3. El entorno aplica el balance hídrico hacia adelante en cada embalse (paso_embalse).
4. Se calcula el caudal total en El Sol:
       Q_sol = Σ(suministro_real + vertimiento) de los tres embalses + Q_natural
5. Se detecta si se viola el caudal ecológico mínimo (Q_sol < Q_eco).
6. Se calculan las penalizaciones por embalse.
7. Se devuelve el nuevo estado, las penalizaciones, y un dict de información adicional.

NOTA: El mecanismo que *fuerza* el cumplimiento del caudal ecológico (shield de
proyección) NO está implementado aquí; se añadirá en el próximo módulo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pbcrl.data_contracts.embalses import EMBALSES, ParametrosEmbalse
from pbcrl.environment.config import ConfigEntorno
from pbcrl.environment.hidraulica import (
    paso_embalse,
    recortar_suministro,
    volumen_a_cota,
)
from pbcrl.environment.penalizaciones import (
    pen_descenso_nivel_sisga,
    pen_proximidad_minimo,
)

# Orden canónico de los embalses (debe mantenerse consistente en todo el código)
_NOMBRES_EMBALSES: tuple[str, ...] = ("Neusa", "Sisga", "Tomine")


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
        Caudal natural en El Sol (proveniente de la cuenca sin regulación) [m³/s].
        En datos reales corresponde a la estación Saucío.
    caudal_sol_m3s : float
        Caudal total en el punto de control El Sol [m³/s].
    """

    volumen_mm3: dict[str, float]
    cota_m: dict[str, float]
    caudal_natural_m3s: float
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
        Caudal natural (no regulado) en El Sol [m³/s]. Estación Saucío en datos reales.
    """

    afluencia_m3s: dict[str, float]
    precipitacion_mm: dict[str, float]
    evaporacion_mm: dict[str, float]
    caudal_natural_m3s: float


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
        True si el caudal en El Sol es inferior al caudal ecológico mínimo.
    """

    estado: EstadoSistema
    penalizaciones: dict[str, float]
    suministro_real_m3s: dict[str, float]
    vertimiento_m3s: dict[str, float]
    violacion_ecologica: bool


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
            nombre: volumen_a_cota(vol, self.embalses[nombre])
            for nombre, vol in volumenes_iniciales_mm3.items()
        }

        self._cota_anterior_m = dict(cotas)

        self._estado = EstadoSistema(
            volumen_mm3=dict(volumenes_iniciales_mm3),
            cota_m=cotas,
            caudal_natural_m3s=caudal_natural_inicial_m3s,
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

        nuevos_volumenes: dict[str, float] = {}
        suministro_real: dict[str, float] = {}
        vertimiento_mm3: dict[str, float] = {}

        # --- Paso hidráulico para cada embalse ---
        for nombre, params in self.embalses.items():
            vol_actual = self._estado.volumen_mm3[nombre]

            # 1. Recortar la acción a lo físicamente posible
            q_real = recortar_suministro(
                suministro_pedido_m3s=acciones_m3s.get(nombre, 0.0),
                volumen_actual_mm3=vol_actual,
                params=params,
            )
            suministro_real[nombre] = q_real

            # 2. Aplicar balance hídrico hacia adelante
            v_nuevo, vert_mm3 = paso_embalse(
                volumen_actual_mm3=vol_actual,
                afluencia_m3s=forzantes.afluencia_m3s.get(nombre, 0.0),
                suministro_m3s=q_real,
                precipitacion_mm=forzantes.precipitacion_mm.get(nombre, 0.0),
                evaporacion_mm=forzantes.evaporacion_mm.get(nombre, 0.0),
                params=params,
            )
            nuevos_volumenes[nombre] = v_nuevo
            vertimiento_mm3[nombre] = vert_mm3

        # --- Nuevas cotas ---
        nuevas_cotas = {
            nombre: volumen_a_cota(vol, self.embalses[nombre])
            for nombre, vol in nuevos_volumenes.items()
        }

        # --- Caudal en El Sol ---
        # Q_sol = Σ(suministro_real + vertimiento_m3s) para cada embalse + Q_natural
        # Factor de conversión de vertimiento: Mm³/día → m³/s
        _MM3_DIA_A_M3S = 1e6 / 86_400.0

        q_sol = forzantes.caudal_natural_m3s
        for nombre in self.embalses:
            q_sol += suministro_real[nombre]
            q_sol += vertimiento_mm3[nombre] * _MM3_DIA_A_M3S

        # --- Detección de violación del caudal ecológico ---
        violacion = q_sol < self.config.q_eco_m3s

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
        )

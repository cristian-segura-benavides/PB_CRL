"""
Pruebas del entorno de simulación (environment/).

Cobertura:
  (a) Reset: estado inicial coherente con los parámetros.
  (b) Paso con acciones válidas: balance de masa correcto.
  (c) Recorte físico: suministro se reduce cuando no hay volumen suficiente.
  (d) Detección de violación ecológica: caso con y sin violación.
  (e) Penalizaciones siempre en [0, peso].
  (f) Penalización Sisga en los puntos exactos: 15, 30 y 45 cm/día.
"""
from __future__ import annotations

import pytest

from pbcrl.data_contracts.captaciones import (
    CAUDAL_TIBITOC_AMPLIADO_M3S,
    CAUDAL_TIBITOC_HISTORICO_M3S,
    ESCENARIO_AMPLIADO,
    ESCENARIO_HISTORICO,
    caudal_tibitoc_nominal,
)
from pbcrl.data_contracts.caudal_ecologico import q_eco_m3s, umbral_fijo_m3s
from pbcrl.data_contracts.embalses import EMBALSES, ParametrosEmbalse
from pbcrl.environment.config import ConfigEntorno
from pbcrl.environment.entorno import EntornoEmbalses, ForzantesExternos
from pbcrl.data_contracts.curvas import volumen_a_cota
from pbcrl.environment.hidraulica import (
    _M3S_A_MM3_DIA,
    _MM_KM2_A_MM3,
    calcular_extraccion_tibitoc,
    recortar_suministro,
)
from pbcrl.environment.penalizaciones import pen_descenso_nivel_sisga

# ---------------------------------------------------------------------------
# Helpers de prueba
# ---------------------------------------------------------------------------

def _forzantes_neutros(q_natural: float = 1.0, mes: int = 6) -> ForzantesExternos:
    """Forzantes con afluencia, lluvia y evaporación cero: solo el natural llega a El Sol.

    `mes` por defecto es junio (6), un mes cualquiera sin significado especial
    para las pruebas que no dependen del umbral de caudal ecológico.
    """
    return ForzantesExternos(
        afluencia_m3s={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
        precipitacion_mm={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
        evaporacion_mm={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
        caudal_natural_m3s=q_natural,
        mes=mes,
    )


def _acciones_cero() -> dict[str, float]:
    return {"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0}


# ---------------------------------------------------------------------------
# (a) Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_volumenes_iniciales_por_defecto(self):
        """Sin pasar volúmenes, el entorno arranca al 70 % de la capacidad útil."""
        env = EntornoEmbalses()
        estado = env.reset()
        for nombre, params in EMBALSES.items():
            if nombre not in ("Neusa", "Sisga", "Tomine"):
                continue
            vol_esperado = (
                params.capacidad_min_mm3
                + 0.70 * (params.capacidad_max_mm3 - params.capacidad_min_mm3)
            )
            assert abs(estado.volumen_mm3[nombre] - vol_esperado) < 1e-9

    def test_volumenes_personalizados(self):
        """Los volúmenes pasados a reset() se reflejan en el estado inicial."""
        env = EntornoEmbalses()
        vols = {"Neusa": 50.0, "Sisga": 30.0, "Tomine": 300.0}
        estado = env.reset(volumenes_iniciales_mm3=vols)
        for nombre, vol in vols.items():
            assert abs(estado.volumen_mm3[nombre] - vol) < 1e-9

    def test_cotas_coherentes_con_volumenes(self):
        """Las cotas en el estado inicial son coherentes con los volúmenes."""
        env = EntornoEmbalses()
        estado = env.reset()
        for nombre, params in env.embalses.items():
            cota_esperada = volumen_a_cota(estado.volumen_mm3[nombre], nombre, params)
            assert abs(estado.cota_m[nombre] - cota_esperada) < 1e-9

    def test_caudal_natural_inicial(self):
        """El caudal natural inicial se refleja en el estado."""
        env = EntornoEmbalses()
        estado = env.reset(caudal_natural_inicial_m3s=5.0)
        assert abs(estado.caudal_natural_m3s - 5.0) < 1e-9

    def test_step_sin_reset_lanza_error(self):
        """Llamar step() antes de reset() debe lanzar RuntimeError."""
        env = EntornoEmbalses()
        with pytest.raises(RuntimeError):
            env.step(_acciones_cero(), _forzantes_neutros())


# ---------------------------------------------------------------------------
# (b) Balance de masa en un paso
# ---------------------------------------------------------------------------

class TestBalanceMasa:
    def test_volumen_decrece_con_suministro(self):
        """Con suministro positivo y sin afluencia, el volumen disminuye."""
        env = EntornoEmbalses()
        vols_ini = {"Neusa": 80.0, "Sisga": 50.0, "Tomine": 400.0}
        env.reset(volumenes_iniciales_mm3=vols_ini)

        suministro = 5.0  # m³/s constante para todos
        resultado = env.step(
            acciones_m3s={"Neusa": suministro, "Sisga": suministro, "Tomine": suministro},
            forzantes=_forzantes_neutros(),
        )

        for nombre in ("Neusa", "Sisga", "Tomine"):
            assert resultado.estado.volumen_mm3[nombre] < vols_ini[nombre]

    def test_balance_neusa_exacto(self):
        """Verifica el balance de masa exacto para el Neusa en condiciones controladas.

        Con afluencia=0, precipitación=0, evaporación=0 y suministro=Q:
            V(t) = V(t-1) - Q * 0.0864   [Mm³]
        """
        env = EntornoEmbalses()
        v_ini = 80.0
        env.reset(volumenes_iniciales_mm3={"Neusa": v_ini, "Sisga": 50.0, "Tomine": 400.0})

        q_neusa = 3.0  # m³/s
        resultado = env.step(
            acciones_m3s={"Neusa": q_neusa, "Sisga": 0.0, "Tomine": 0.0},
            forzantes=_forzantes_neutros(),
        )

        v_esperado = v_ini - q_neusa * _M3S_A_MM3_DIA
        assert abs(resultado.estado.volumen_mm3["Neusa"] - v_esperado) < 1e-9

    def test_afluencia_incrementa_volumen(self):
        """Con afluencia y sin suministro, el volumen aumenta."""
        env = EntornoEmbalses()
        v_ini = {"Neusa": 50.0, "Sisga": 30.0, "Tomine": 200.0}
        env.reset(volumenes_iniciales_mm3=v_ini)

        forzantes = ForzantesExternos(
            afluencia_m3s={"Neusa": 10.0, "Sisga": 5.0, "Tomine": 20.0},
            precipitacion_mm={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
            evaporacion_mm={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
            caudal_natural_m3s=1.0,
            mes=6,
        )
        resultado = env.step(_acciones_cero(), forzantes)

        for nombre in ("Neusa", "Sisga", "Tomine"):
            assert resultado.estado.volumen_mm3[nombre] > v_ini[nombre]

    def test_embalse_lleno_genera_vertimiento(self):
        """Si el volumen supera la capacidad máxima por afluencia, hay vertimiento."""
        env = EntornoEmbalses()
        params_neusa = EMBALSES["Neusa"]
        # Iniciar casi al máximo
        v_ini = params_neusa.capacidad_max_mm3 - 0.1
        env.reset(volumenes_iniciales_mm3={
            "Neusa": v_ini, "Sisga": 30.0, "Tomine": 200.0
        })

        # Afluencia grande que desborda el Neusa
        forzantes = ForzantesExternos(
            afluencia_m3s={"Neusa": 20.0, "Sisga": 0.0, "Tomine": 0.0},
            precipitacion_mm={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
            evaporacion_mm={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
            caudal_natural_m3s=1.0,
            mes=6,
        )
        resultado = env.step(_acciones_cero(), forzantes)

        assert resultado.vertimiento_m3s["Neusa"] > 0.0
        assert abs(resultado.estado.volumen_mm3["Neusa"] - params_neusa.capacidad_max_mm3) < 1e-9

    def test_deficit_volumen_en_evaporacion_extrema(self):
        """Si la evaporación supera el agua disponible cerca del mínimo (sequía
        extrema), el volumen se acota al mínimo y el déficit de masa que el clamp
        tuvo que cubrir queda reportado explícitamente en deficit_volumen_mm3, en
        vez de perderse sin dejar rastro (ver hidraulica.paso_embalse)."""
        env = EntornoEmbalses()
        params_neusa = EMBALSES["Neusa"]
        # Embalse casi vacío: solo 0.05 Mm³ sobre el volumen muerto.
        agua_disponible = 0.05
        v_ini = params_neusa.capacidad_min_mm3 + agua_disponible
        env.reset(volumenes_iniciales_mm3={"Neusa": v_ini, "Sisga": 50.0, "Tomine": 400.0})

        # Sin afluencia ni lluvia; evaporación en el techo de sanidad de schemas.py
        # (20 mm/día) — no es un valor absurdo, es físicamente plausible.
        forzantes = ForzantesExternos(
            afluencia_m3s={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
            precipitacion_mm={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
            evaporacion_mm={"Neusa": 20.0, "Sisga": 0.0, "Tomine": 0.0},
            caudal_natural_m3s=1.0,
            mes=6,
        )
        resultado = env.step(_acciones_cero(), forzantes)

        # El volumen nunca baja del mínimo.
        assert abs(resultado.estado.volumen_mm3["Neusa"] - params_neusa.capacidad_min_mm3) < 1e-9

        # El déficit reportado es exactamente la evaporación que no pudo cubrirse
        # con el agua disponible (evaporación total - agua que sí había).
        evap_mm3 = 20.0 * params_neusa.area_espejo_km2 * _MM_KM2_A_MM3
        deficit_esperado = evap_mm3 - agua_disponible
        assert deficit_esperado > 0.0, "el escenario debe forzar el clamp inferior"
        assert abs(resultado.deficit_volumen_mm3["Neusa"] - deficit_esperado) < 1e-9

        # Los embalses sin estrés no reportan déficit.
        assert resultado.deficit_volumen_mm3["Sisga"] == 0.0
        assert resultado.deficit_volumen_mm3["Tomine"] == 0.0


# ---------------------------------------------------------------------------
# (c) Recorte físico del suministro
# ---------------------------------------------------------------------------

class TestRecorteFisico:
    def test_suministro_cero_en_embalse_al_minimo(self):
        """Si el volumen es igual al mínimo, no puede suministrarse nada."""
        params = EMBALSES["Neusa"]
        q_real = recortar_suministro(
            suministro_pedido_m3s=10.0,
            volumen_actual_mm3=params.capacidad_min_mm3,
            params=params,
        )
        assert q_real == 0.0

    def test_suministro_recortado_por_volumen_insuficiente(self):
        """Con poco volumen, el suministro real es menor al pedido."""
        params = EMBALSES["Sisga"]
        # Solo 0.1 Mm³ sobre el mínimo
        volumen = params.capacidad_min_mm3 + 0.1
        q_max_posible = 0.1 / _M3S_A_MM3_DIA  # m³/s

        q_real = recortar_suministro(
            suministro_pedido_m3s=100.0,  # pide mucho más de lo disponible
            volumen_actual_mm3=volumen,
            params=params,
        )
        assert abs(q_real - q_max_posible) < 1e-6

    def test_suministro_no_supera_capacidad_compuertas(self):
        """El suministro real no puede superar descarga_max_m3s."""
        params = EMBALSES["Tomine"]
        q_real = recortar_suministro(
            suministro_pedido_m3s=9999.0,  # absurdamente grande
            volumen_actual_mm3=params.capacidad_max_mm3,
            params=params,
        )
        assert q_real <= params.descarga_max_m3s

    def test_suministro_negativo_se_recorta_a_cero(self):
        """Un suministro negativo (sin sentido físico) se recorta a cero."""
        params = EMBALSES["Neusa"]
        q_real = recortar_suministro(
            suministro_pedido_m3s=-5.0,
            volumen_actual_mm3=80.0,
            params=params,
        )
        assert q_real == 0.0

    def test_entorno_recorta_suministro_en_step(self):
        """El entorno recorta el suministro cuando el embalse está al mínimo."""
        env = EntornoEmbalses()
        params_sisga = EMBALSES["Sisga"]
        # Sisga exactamente al volumen mínimo
        env.reset(volumenes_iniciales_mm3={
            "Neusa": 50.0,
            "Sisga": params_sisga.capacidad_min_mm3,
            "Tomine": 200.0,
        })
        resultado = env.step(
            acciones_m3s={"Neusa": 0.0, "Sisga": 10.0, "Tomine": 0.0},
            forzantes=_forzantes_neutros(),
        )
        assert resultado.suministro_real_m3s["Sisga"] == 0.0


# ---------------------------------------------------------------------------
# (d) Detección de violación del caudal ecológico
# ---------------------------------------------------------------------------

class TestViolacionEcologica:
    def _env_con_config(self, q_eco: float) -> EntornoEmbalses:
        """Entorno con un umbral CONSTANTE (no el VMF mensual por defecto).

        Usa `umbral_fijo_m3s`, el mecanismo documentado para revertir al
        comportamiento anterior sin tocar `environment.entorno` — ver
        `data_contracts.caudal_ecologico`.
        """
        config = ConfigEntorno(calcular_q_eco_m3s=umbral_fijo_m3s(q_eco))
        return EntornoEmbalses(config=config)

    def test_violacion_detectada(self):
        """Q_sol < Q_eco debe reportarse como violación."""
        q_eco = 5.0
        env = self._env_con_config(q_eco)
        # Iniciar con volúmenes en el mínimo: no puede suministrar nada
        env.reset(volumenes_iniciales_mm3={
            "Neusa": EMBALSES["Neusa"].capacidad_min_mm3,
            "Sisga": EMBALSES["Sisga"].capacidad_min_mm3,
            "Tomine": EMBALSES["Tomine"].capacidad_min_mm3,
        })
        # Sin suministro ni afluencia, Q_sol = Q_natural < Q_eco
        resultado = env.step(
            acciones_m3s=_acciones_cero(),
            forzantes=_forzantes_neutros(q_natural=1.0),  # 1 < 5
        )
        assert resultado.violacion_ecologica is True
        assert abs(resultado.q_eco_aplicado_m3s - q_eco) < 1e-9

    def test_sin_violacion(self):
        """Q_sol ≥ Q_eco no debe reportarse como violación.

        El flujo debe superar Q_eco con holgura DESPUÉS de la extracción de
        Tibitóc (escenario histórico por defecto, 4.5 m³/s), no antes.
        """
        q_eco = 2.0
        env = self._env_con_config(q_eco)
        env.reset(volumenes_iniciales_mm3={
            "Neusa": 80.0, "Sisga": 50.0, "Tomine": 400.0
        })
        # Bocatoma = 3 (natural) + 5+3+5 (suministros) = 16; tras extraer 4.5 -> 11.5 > 2.
        resultado = env.step(
            acciones_m3s={"Neusa": 5.0, "Sisga": 3.0, "Tomine": 5.0},
            forzantes=_forzantes_neutros(q_natural=3.0),
        )
        assert resultado.violacion_ecologica is False
        assert abs(resultado.q_eco_aplicado_m3s - q_eco) < 1e-9


# ---------------------------------------------------------------------------
# (d.2) Umbral VMF mensual por defecto (data_contracts.caudal_ecologico)
# ---------------------------------------------------------------------------

class TestUmbralVMFPorDefecto:
    """El entorno, SIN configuración explícita, debe usar el umbral VMF del mes
    (q_eco_m3s de data_contracts.caudal_ecologico), no un valor fijo."""

    def _step_con_mes(self, env: EntornoEmbalses, q_natural: float, mes: int):
        env.reset(volumenes_iniciales_mm3={
            "Neusa": EMBALSES["Neusa"].capacidad_min_mm3,
            "Sisga": EMBALSES["Sisga"].capacidad_min_mm3,
            "Tomine": EMBALSES["Tomine"].capacidad_min_mm3,
        })
        return env.step(
            acciones_m3s=_acciones_cero(),
            forzantes=_forzantes_neutros(q_natural=q_natural, mes=mes),
        )

    def test_umbral_aplicado_coincide_con_q_eco_m3s(self):
        """q_eco_aplicado_m3s debe ser exactamente q_eco_m3s(mes) para cada mes."""
        env = EntornoEmbalses()
        for mes in range(1, 13):
            resultado = self._step_con_mes(env, q_natural=1.0, mes=mes)
            assert abs(resultado.q_eco_aplicado_m3s - q_eco_m3s(mes)) < 1e-9

    def test_mismo_caudal_distinta_violacion_segun_el_mes(self):
        """Un mismo Q_sol puede violar en un mes de umbral alto (julio, EFR=7.51)
        y no violar en un mes de umbral bajo (enero, EFR=2.32) — evidencia
        directa de que el entorno usa el umbral correspondiente al mes, no un
        valor fijo.

        Con volúmenes al mínimo (suministro=0) y sin afluencia, Q_bocatoma =
        caudal_natural_m3s; la extracción histórica (4.5 m³/s) se resta
        siempre que Q_bocatoma la supere. Se fija caudal_natural_m3s para que
        Q_sol resultante sea exactamente 5.0 m³/s (entre 2.32 y 7.51).
        """
        env = EntornoEmbalses()
        q_sol_deseado = 5.0  # entre 2.32 (enero) y 7.51 (julio)
        q_natural = q_sol_deseado + CAUDAL_TIBITOC_HISTORICO_M3S  # 9.5

        resultado_enero = self._step_con_mes(env, q_natural=q_natural, mes=1)
        resultado_julio = self._step_con_mes(env, q_natural=q_natural, mes=7)

        assert abs(resultado_enero.estado.caudal_sol_m3s - q_sol_deseado) < 1e-9
        assert abs(resultado_julio.estado.caudal_sol_m3s - q_sol_deseado) < 1e-9
        assert resultado_enero.violacion_ecologica is False
        assert resultado_julio.violacion_ecologica is True

    def test_q_bocatoma_coincide_con_suma(self):
        """Q_bocatoma es exactamente la suma de suministros + vertimientos + natural.

        Q_bocatoma es el caudal disponible ANTES de la extracción de Tibitóc
        (topología: Saucío -> Sisga -> Tominé -> Neusa -> bocatoma -> El Sol).
        """
        env = EntornoEmbalses()
        env.reset(volumenes_iniciales_mm3={
            "Neusa": 80.0, "Sisga": 50.0, "Tomine": 400.0
        })
        q_natural = 3.0
        acciones = {"Neusa": 2.0, "Sisga": 1.0, "Tomine": 1.5}

        resultado = env.step(acciones, _forzantes_neutros(q_natural=q_natural))

        q_bocatoma_esperado = (
            q_natural
            + resultado.suministro_real_m3s["Neusa"]
            + resultado.suministro_real_m3s["Sisga"]
            + resultado.suministro_real_m3s["Tomine"]
            + resultado.vertimiento_m3s["Neusa"]
            + resultado.vertimiento_m3s["Sisga"]
            + resultado.vertimiento_m3s["Tomine"]
        )
        assert abs(resultado.estado.caudal_bocatoma_m3s - q_bocatoma_esperado) < 1e-9

    def test_q_sol_es_bocatoma_menos_extraccion(self):
        """Q_sol = Q_bocatoma - Q_extraccion, con la extracción acotada."""
        env = EntornoEmbalses()
        env.reset(volumenes_iniciales_mm3={
            "Neusa": 80.0, "Sisga": 50.0, "Tomine": 400.0
        })
        q_natural = 3.0
        acciones = {"Neusa": 2.0, "Sisga": 1.0, "Tomine": 1.5}

        resultado = env.step(acciones, _forzantes_neutros(q_natural=q_natural))

        q_bocatoma = resultado.estado.caudal_bocatoma_m3s
        q_extraccion_esperada = min(CAUDAL_TIBITOC_HISTORICO_M3S, q_bocatoma)
        assert abs(resultado.caudal_extraccion_m3s - q_extraccion_esperada) < 1e-9
        assert abs(resultado.estado.caudal_sol_m3s - (q_bocatoma - q_extraccion_esperada)) < 1e-9


# ---------------------------------------------------------------------------
# (e) Penalizaciones siempre en [0, peso]
# ---------------------------------------------------------------------------

class TestRangoPenalizaciones:
    def test_penalizaciones_dentro_de_rango(self):
        """Las penalizaciones de cada embalse están siempre en [0, peso]."""
        config = ConfigEntorno()
        pesos = {"Neusa": config.peso_neusa, "Sisga": config.peso_sisga, "Tomine": config.peso_tomine}

        env = EntornoEmbalses(config=config)
        env.reset()

        for _ in range(20):
            resultado = env.step(
                acciones_m3s={"Neusa": 5.0, "Sisga": 3.0, "Tomine": 10.0},
                forzantes=ForzantesExternos(
                    afluencia_m3s={"Neusa": 2.0, "Sisga": 1.5, "Tomine": 8.0},
                    precipitacion_mm={"Neusa": 5.0, "Sisga": 3.0, "Tomine": 4.0},
                    evaporacion_mm={"Neusa": 2.0, "Sisga": 1.0, "Tomine": 1.5},
                    caudal_natural_m3s=1.0,
                    mes=6,
                ),
            )
            for nombre, pen in resultado.penalizaciones.items():
                peso = pesos[nombre]
                assert 0.0 <= pen <= peso + 1e-9, (
                    f"Penalización de {nombre} = {pen:.4f} fuera de [0, {peso}]"
                )


# ---------------------------------------------------------------------------
# (f) Penalización Sisga en los casos exactos: 15, 30 y 45 cm/día
# ---------------------------------------------------------------------------

class TestPenalizacionSisga:
    """Verifica los tres puntos de la función de penalización del Sisga."""

    def _pen_sisga(self, descenso_cm: float) -> float:
        """Calcula la penalización del Sisga para un descenso dado."""
        config = ConfigEntorno()
        params = EMBALSES["Sisga"]
        cota_ant = params.cota_max_m
        # Convertir descenso en cm a metros y calcular la cota resultante
        cota_act = cota_ant - descenso_cm / 100.0
        return pen_descenso_nivel_sisga(cota_ant, cota_act, config)

    def test_descenso_en_umbral_pen_cero(self):
        """Descenso = 15 cm/día → penalización = 0."""
        pen = self._pen_sisga(15.0)
        assert abs(pen - 0.0) < 1e-9

    def test_descenso_a_mitad_pen_media(self):
        """Descenso = 30 cm/día → penalización = 0.5 × peso_sisga."""
        config = ConfigEntorno()
        pen = self._pen_sisga(30.0)
        assert abs(pen - 0.5 * config.peso_sisga) < 1e-9

    def test_descenso_en_maximo_pen_llena(self):
        """Descenso = 45 cm/día → penalización = peso_sisga."""
        config = ConfigEntorno()
        pen = self._pen_sisga(45.0)
        assert abs(pen - config.peso_sisga) < 1e-9

    def test_descenso_mayor_al_maximo_acotado(self):
        """Descenso > 45 cm/día → penalización = peso_sisga (acotada, no supera peso)."""
        config = ConfigEntorno()
        pen = self._pen_sisga(100.0)
        assert abs(pen - config.peso_sisga) < 1e-9

    def test_sin_descenso_pen_cero(self):
        """Ascenso de nivel (cota aumenta) → penalización = 0."""
        config = ConfigEntorno()
        params = EMBALSES["Sisga"]
        # Cota actual mayor que la anterior (ascenso)
        pen = pen_descenso_nivel_sisga(params.cota_min_m, params.cota_max_m, config)
        assert pen == 0.0


# ---------------------------------------------------------------------------
# (g) Cota física sobre la extracción de Tibitóc
#
# Topología: Saucío -> Sisga -> Tominé -> Neusa -> bocatoma (extracción) -> El Sol.
# La extracción real nunca debe superar el caudal disponible en la bocatoma, y
# por construcción Q_ElSol nunca debe ser negativo (ver data_contracts.captaciones).
# ---------------------------------------------------------------------------

class TestExtraccionTibitoc:
    def test_escenarios_devuelven_los_valores_documentados(self):
        assert caudal_tibitoc_nominal(ESCENARIO_HISTORICO) == CAUDAL_TIBITOC_HISTORICO_M3S
        assert caudal_tibitoc_nominal(ESCENARIO_AMPLIADO) == CAUDAL_TIBITOC_AMPLIADO_M3S

    def test_escenario_desconocido_lanza_error(self):
        with pytest.raises(ValueError):
            caudal_tibitoc_nominal("no_existe")

    @pytest.mark.parametrize("bocatoma,nominal", [
        (10.0, 4.5),   # bocatoma sobra: extraccion = nominal completo
        (2.0, 4.5),    # bocatoma escasa: extraccion se acota al disponible
        (0.0, 8.0),    # bocatoma seca: extraccion = 0
        (10.0, 0.0),   # nominal nulo: extraccion = 0
    ])
    def test_extraccion_nunca_supera_lo_disponible(self, bocatoma, nominal):
        """Q_extraccion = min(nominal, bocatoma), nunca mayor que la bocatoma."""
        extraccion = calcular_extraccion_tibitoc(bocatoma, nominal)
        assert extraccion <= bocatoma + 1e-9
        assert extraccion <= nominal + 1e-9
        assert extraccion >= 0.0

    def test_q_sol_nunca_negativo_en_el_entorno(self):
        """Con bocatoma menor que el nominal, Q_sol se acota a 0, nunca negativo."""
        env = EntornoEmbalses()
        # Volumenes minimos: suministro=0; sin afluencia: vertimiento=0.
        env.reset(volumenes_iniciales_mm3={
            "Neusa": EMBALSES["Neusa"].capacidad_min_mm3,
            "Sisga": EMBALSES["Sisga"].capacidad_min_mm3,
            "Tomine": EMBALSES["Tomine"].capacidad_min_mm3,
        })
        forzantes = ForzantesExternos(
            afluencia_m3s={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
            precipitacion_mm={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
            evaporacion_mm={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
            caudal_natural_m3s=1.0,  # bocatoma = 1.0 m3/s
            mes=6,
            caudal_tibitoc_m3s=CAUDAL_TIBITOC_AMPLIADO_M3S,  # nominal = 8.0 > bocatoma
        )
        resultado = env.step(_acciones_cero(), forzantes)

        assert resultado.estado.caudal_sol_m3s >= 0.0
        assert abs(resultado.estado.caudal_sol_m3s - 0.0) < 1e-9
        assert resultado.cota_fisica_activada is True
        assert abs(resultado.caudal_extraccion_m3s - resultado.estado.caudal_bocatoma_m3s) < 1e-9
        assert abs(resultado.deficit_extraccion_m3s - (8.0 - resultado.estado.caudal_bocatoma_m3s)) < 1e-9

    def test_cota_no_activada_cuando_bocatoma_sobra(self):
        """Con bocatoma mayor que el nominal, la extraccion es el nominal completo."""
        env = EntornoEmbalses()
        env.reset(volumenes_iniciales_mm3={
            "Neusa": 80.0, "Sisga": 50.0, "Tomine": 400.0
        })
        forzantes = ForzantesExternos(
            afluencia_m3s={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
            precipitacion_mm={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
            evaporacion_mm={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
            caudal_natural_m3s=20.0,
            mes=6,
            caudal_tibitoc_m3s=CAUDAL_TIBITOC_HISTORICO_M3S,
        )
        resultado = env.step(_acciones_cero(), forzantes)

        assert resultado.cota_fisica_activada is False
        assert abs(resultado.caudal_extraccion_m3s - CAUDAL_TIBITOC_HISTORICO_M3S) < 1e-9
        assert abs(resultado.deficit_extraccion_m3s - 0.0) < 1e-9

    @pytest.mark.parametrize("q_natural", [0.0, 0.5, 1.0, 3.0, 4.5, 4.999, 5.0, 8.0, 20.0])
    def test_q_sol_nunca_negativo_barrido(self, q_natural):
        """Barrido de caudales de bocatoma: Q_sol nunca negativo, para ambos escenarios."""
        for escenario in (ESCENARIO_HISTORICO, ESCENARIO_AMPLIADO):
            env = EntornoEmbalses()
            env.reset(volumenes_iniciales_mm3={
                "Neusa": EMBALSES["Neusa"].capacidad_min_mm3,
                "Sisga": EMBALSES["Sisga"].capacidad_min_mm3,
                "Tomine": EMBALSES["Tomine"].capacidad_min_mm3,
            })
            forzantes = ForzantesExternos(
                afluencia_m3s={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
                precipitacion_mm={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
                evaporacion_mm={"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0},
                caudal_natural_m3s=q_natural,
                mes=6,
                caudal_tibitoc_m3s=caudal_tibitoc_nominal(escenario),
            )
            resultado = env.step(_acciones_cero(), forzantes)
            assert resultado.estado.caudal_sol_m3s >= -1e-9, (
                f"Q_sol negativo con escenario={escenario}, q_natural={q_natural}"
            )

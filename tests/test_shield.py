"""Pruebas del shield de proyección cuadrática (src/pbcrl/shield/).

Cobertura:
  (a) Acción ya factible -> la proyección no la altera (idéntica, no solo "cercana").
  (b) Acción que viola una caja individual -> se recorta a ese límite.
  (c) Acción que viola la rata de descenso de Sisga -> se ajusta solo esa componente.
  (d) Acción que viola el caudal ecológico conjunto -> la corrección se reparte
      entre los tres embalses (no recorta arbitrariamente uno solo).
  (e) Conjunto factible vacío -> se detecta y se reporta explícitamente.
"""
from __future__ import annotations

import numpy as np
import pytest

from pbcrl.data_contracts.captaciones import ESCENARIO_HISTORICO, caudal_tibitoc_nominal
from pbcrl.data_contracts.caudal_ecologico import q_eco_m3s
from pbcrl.data_contracts.curvas import volumen_a_cota
from pbcrl.data_contracts.embalses import EMBALSES
from pbcrl.shield.restricciones import EstadoShield, construir_restricciones
from pbcrl.shield.proyeccion import proyectar

_CERO = {"Neusa": 0.0, "Sisga": 0.0, "Tomine": 0.0}


def _estado_base(
    v_neusa: float = 50.0,
    v_sisga: float = 50.0,
    v_tomine: float = 300.0,
    afluencia: dict | None = None,
    precip: dict | None = None,
    evap: dict | None = None,
    caudal_saucio: float = 5.0,
    mes: int = 1,
    tibitoc_nominal: float | None = None,
) -> EstadoShield:
    kwargs = {}
    if tibitoc_nominal is not None:
        kwargs["caudal_tibitoc_nominal_m3s"] = tibitoc_nominal
    return EstadoShield(
        volumen_mm3={"Neusa": v_neusa, "Sisga": v_sisga, "Tomine": v_tomine},
        afluencia_m3s=dict(afluencia or _CERO),
        precipitacion_mm=dict(precip or _CERO),
        evaporacion_mm=dict(evap or _CERO),
        caudal_saucio_m3s=caudal_saucio,
        mes=mes,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# (a) Acción ya factible: no se altera
# ---------------------------------------------------------------------------

class TestAccionFactibleNoSeAltera:
    def test_accion_factible_queda_identica(self):
        # mes=1 -> Q_eco=2.32; nominal historico=4.5; saucio=5.0 ->
        # suma_minima_requerida = 2.32+4.5-5.0 = 1.82, muy poco exigente.
        estado = _estado_base(mes=1, caudal_saucio=5.0)
        accion = {"Neusa": 3.0, "Sisga": 2.0, "Tomine": 10.0}

        diag = proyectar(estado, accion)

        assert diag.accion_proyectada["Neusa"] == pytest.approx(accion["Neusa"], abs=1e-12)
        assert diag.accion_proyectada["Sisga"] == pytest.approx(accion["Sisga"], abs=1e-12)
        assert diag.accion_proyectada["Tomine"] == pytest.approx(accion["Tomine"], abs=1e-12)
        assert diag.factible is True
        assert not any(diag.violaciones_previas.values())
        assert diag.restricciones_activas["caudal_ecologico_conjunto"] is False


# ---------------------------------------------------------------------------
# (b) Violación de caja individual
# ---------------------------------------------------------------------------

class TestViolacionCajaIndividual:
    def test_neusa_se_recorta_a_capacidad_maxima(self):
        estado = _estado_base(mes=1, caudal_saucio=5.0)
        accion = {"Neusa": 25.0, "Sisga": 2.0, "Tomine": 10.0}  # Neusa > 16.0

        diag = proyectar(estado, accion)

        assert diag.violaciones_previas["caja_Neusa"] is True
        assert diag.accion_proyectada["Neusa"] == pytest.approx(EMBALSES["Neusa"].descarga_max_m3s)
        # Las otras componentes no debían cambiar (sin acople activo en este caso).
        assert diag.accion_proyectada["Sisga"] == pytest.approx(2.0)
        assert diag.accion_proyectada["Tomine"] == pytest.approx(10.0)

    def test_tomine_se_recorta_a_capacidad_maxima(self):
        estado = _estado_base(mes=1, caudal_saucio=5.0)
        accion = {"Neusa": 3.0, "Sisga": 2.0, "Tomine": 55.0}  # Tomine > 40.0

        diag = proyectar(estado, accion)

        assert diag.violaciones_previas["caja_Tomine"] is True
        assert diag.accion_proyectada["Tomine"] == pytest.approx(EMBALSES["Tomine"].descarga_max_m3s)

    def test_accion_negativa_se_recorta_a_cero(self):
        estado = _estado_base(mes=1, caudal_saucio=5.0)
        accion = {"Neusa": -3.0, "Sisga": 2.0, "Tomine": 10.0}

        diag = proyectar(estado, accion)

        assert diag.violaciones_previas["caja_Neusa"] is True
        assert diag.accion_proyectada["Neusa"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# (c) Violación de la rata de descenso de Sisga: se ajusta SOLO esa componente
# ---------------------------------------------------------------------------

class TestViolacionRataDescensoSisga:
    def _limite_esperado(self, estado: EstadoShield) -> float:
        """Recalcula el límite dinámico de forma independiente (no reusa
        restricciones._limite_descenso_sisga_m3s), para que la prueba no sea
        circular con la implementación."""
        params = EMBALSES["Sisga"]
        v = estado.volumen_mm3["Sisga"]
        paso = 0.01
        cota_hi = volumen_a_cota(v + paso, "Sisga", params)
        cota_lo = volumen_a_cota(v - paso, "Sisga", params)
        pendiente = (cota_hi - cota_lo) / (2 * paso)
        tasa_max_cm = 15.0
        coef = 100.0 * pendiente * (86_400.0 / 1e6)
        return tasa_max_cm / coef  # afluencia=precip=evap=0 en este escenario

    def test_sisga_se_ajusta_al_limite_de_descenso_y_solo_esa_componente(self):
        estado = _estado_base(v_sisga=50.0, mes=1, caudal_saucio=5.0)
        limite_esperado = self._limite_esperado(estado)
        assert limite_esperado < EMBALSES["Sisga"].descarga_max_m3s, (
            "el escenario debe hacer que la rata de descenso ate antes que la capacidad"
        )

        accion = {"Neusa": 8.0, "Sisga": 10.0, "Tomine": 20.0}  # Sisga > limite dinamico
        diag = proyectar(estado, accion)

        assert diag.violaciones_previas["caja_Sisga"] is True
        assert diag.accion_proyectada["Sisga"] == pytest.approx(limite_esperado, rel=1e-6)
        assert diag.restricciones_activas["caja_Sisga_superior (rata_descenso)"] is True
        # Neusa y Tomine no debian moverse: sin violacion de caja propia ni acople activo.
        assert diag.accion_proyectada["Neusa"] == pytest.approx(8.0)
        assert diag.accion_proyectada["Tomine"] == pytest.approx(20.0)
        assert diag.restricciones_activas["caudal_ecologico_conjunto"] is False


# ---------------------------------------------------------------------------
# (d) Violación del caudal ecológico conjunto: se reparte entre los tres
# ---------------------------------------------------------------------------

class TestViolacionCaudalEcologicoConjunto:
    def test_correccion_se_reparte_equitativamente(self):
        # mes=7 -> Q_eco=7.51 (el mas exigente); nominal historico=4.5; saucio=1.0
        # suma_minima_requerida = 7.51 + 4.5 - 1.0 = 11.01
        estado = _estado_base(v_sisga=50.0, mes=7, caudal_saucio=1.0)
        accion = {"Neusa": 2.0, "Sisga": 2.0, "Tomine": 2.0}  # suma=6.0 < 11.01

        diag = proyectar(estado, accion)

        assert diag.violaciones_previas["caudal_ecologico_conjunto"] is True
        assert diag.factible is True
        assert diag.restricciones_activas["caudal_ecologico_conjunto"] is True

        suma_proyectada = sum(diag.accion_proyectada.values())
        assert suma_proyectada == pytest.approx(11.01, abs=1e-6)

        # La correccion (lambda) debe repartirse EN PARTES IGUALES entre las
        # tres componentes -- es la solucion de minima norma para un aumento
        # de suma con coeficientes simetricos, no un recorte arbitrario de una.
        deltas = {n: diag.accion_proyectada[n] - accion[n] for n in accion}
        valores = list(deltas.values())
        assert valores[0] == pytest.approx(valores[1], abs=1e-6)
        assert valores[1] == pytest.approx(valores[2], abs=1e-6)
        assert all(v > 0 for v in valores), "las tres componentes deben subir, no solo una"

    def test_no_se_reparte_si_ya_es_factible(self):
        """Contraste: con margen suficiente, la corrección NO debe activarse."""
        estado = _estado_base(v_sisga=50.0, mes=1, caudal_saucio=5.0)
        accion = {"Neusa": 3.0, "Sisga": 2.0, "Tomine": 10.0}  # suma=15 >> 1.82

        diag = proyectar(estado, accion)

        assert diag.violaciones_previas["caudal_ecologico_conjunto"] is False
        assert diag.accion_proyectada == pytest.approx(accion)


# ---------------------------------------------------------------------------
# (e) Conjunto factible vacío: se detecta y se reporta
# ---------------------------------------------------------------------------

class TestConjuntoFactibleVacio:
    def test_infactibilidad_detectada_y_reportada(self):
        # Exigencia deliberadamente imposible: nominal "inventado" de 1000 m3/s.
        estado = _estado_base(
            v_sisga=50.0, mes=7, caudal_saucio=0.0, tibitoc_nominal=1000.0
        )
        accion = {"Neusa": 8.0, "Sisga": 8.0, "Tomine": 20.0}

        diag = proyectar(estado, accion)

        assert diag.factible is False
        assert diag.restricciones_activas["caudal_ecologico_conjunto"] is True
        # La accion proyectada debe ser el extremo mas favorable de la caja
        # (todas al maximo posible), aun sin alcanzar a satisfacer la restriccion.
        r = construir_restricciones(estado)
        for i, n in enumerate(r.nombres):
            assert diag.accion_proyectada[n] == pytest.approx(r.hi[i], abs=1e-6)
        assert sum(diag.accion_proyectada.values()) < r.detalle_el_sol["suma_minima_requerida_m3s"]


# ---------------------------------------------------------------------------
# Verificación de la construcción de restricciones en sí (unidades, consistencia)
# ---------------------------------------------------------------------------

class TestConstruirRestricciones:
    def test_suma_minima_requerida_coincide_con_formula(self):
        estado = _estado_base(mes=6, caudal_saucio=3.0)
        r = construir_restricciones(estado)
        esperado = q_eco_m3s(6) + caudal_tibitoc_nominal(ESCENARIO_HISTORICO) - 3.0
        assert r.detalle_el_sol["suma_minima_requerida_m3s"] == pytest.approx(esperado)

    def test_cajas_inferiores_son_cero(self):
        estado = _estado_base()
        r = construir_restricciones(estado)
        assert np.all(r.lo == 0.0)

    def test_limite_sisga_nunca_supera_capacidad_de_toma(self):
        estado = _estado_base(v_sisga=50.0)
        r = construir_restricciones(estado)
        idx = r.nombres.index("Sisga")
        assert r.hi[idx] <= EMBALSES["Sisga"].descarga_max_m3s + 1e-9

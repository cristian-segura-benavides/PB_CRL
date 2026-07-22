"""
Pruebas del módulo de curvas cota-volumen (data_contracts/curvas.py).

Cobertura:
  (a) La curva de Tominé es monótona: mayor cota → mayor volumen.
  (b) Puntos ancla coherentes con embalses.py (volumen ÚTIL): la curva debe dar
      exactamente capacidad_min_mm3/capacidad_max_mm3 en cota_min_m/cota_max_m.
      Estas pruebas comparan contra EMBALSES["Tomine"], NO contra literales
      hardcodeados: si curvas.py y embalses.py se desincronizan (p.ej. al
      adoptar la tabla oficial "Aforo" pendiente, ver NOTAS.md sección 6), fallan.
      Ver también NOTAS.md sección 8 (regla "la curva es la fuente de verdad").
  (c) cota_a_volumen y volumen_a_cota son inversas consistentes (round-trip).
  (d) Neusa y Sisga (sin curva) usan el fallback lineal sin romperse.
  (e) Acotación fuera de rango: sin extrapolación fuera de la tabla.
  (f) Validación de construcción: la clase rechaza tablas no monótonas.
"""
from __future__ import annotations

import numpy as np
import pytest

from pbcrl.data_contracts.curvas import (
    CURVA_TOMINE,
    CURVAS,
    CurvaCotaVolumen,
    cota_a_volumen,
    volumen_a_cota,
)
from pbcrl.data_contracts.embalses import EMBALSES


# ---------------------------------------------------------------------------
# (a) Monotonía de la curva de Tominé
# ---------------------------------------------------------------------------

class TestMonotoniaCurvTomine:
    def test_cotas_estrictamente_crecientes(self):
        """Las cotas de la tabla están en orden estrictamente creciente."""
        cotas = CURVA_TOMINE._cotas
        assert np.all(np.diff(cotas) > 0)

    def test_volumenes_estrictamente_crecientes(self):
        """Los volúmenes de la tabla están en orden estrictamente creciente."""
        vols = CURVA_TOMINE._volumenes
        assert np.all(np.diff(vols) > 0)

    def test_mayor_cota_mayor_volumen(self):
        """Muestreo denso: a mayor cota, siempre mayor volumen."""
        cotas_prueba = np.linspace(
            CURVA_TOMINE.cota_min_m + 0.01,
            CURVA_TOMINE.cota_max_m - 0.01,
            200,
        )
        vols = np.array([CURVA_TOMINE.cota_a_volumen(c) for c in cotas_prueba])
        assert np.all(np.diff(vols) > 0)


# ---------------------------------------------------------------------------
# (b) Puntos ancla coherentes con embalses.py (NO literales hardcodeados: esta es
#     la prueba de coherencia entre curvas.py y embalses.py que garantiza que no
#     se desincronicen — ver NOTAS.md secciones 3 y 8).
# ---------------------------------------------------------------------------

class TestPuntosAnclaTomIne:
    def test_ancla_inferior(self):
        """cota_min_m de embalses.py -> capacidad_min_mm3 de embalses.py, vía la curva."""
        params = EMBALSES["Tomine"]
        vol = CURVA_TOMINE.cota_a_volumen(params.cota_min_m)
        assert abs(vol - params.capacidad_min_mm3) < 1e-9

    def test_ancla_superior(self):
        """cota_max_m de embalses.py -> capacidad_max_mm3 de embalses.py, vía la curva."""
        params = EMBALSES["Tomine"]
        vol = CURVA_TOMINE.cota_a_volumen(params.cota_max_m)
        assert abs(vol - params.capacidad_max_mm3) < 1e-9

    def test_ancla_inversa_inferior(self):
        """capacidad_min_mm3 de embalses.py -> cota_min_m de embalses.py, vía la curva."""
        params = EMBALSES["Tomine"]
        cota = CURVA_TOMINE.volumen_a_cota(params.capacidad_min_mm3)
        assert abs(cota - params.cota_min_m) < 1e-9

    def test_ancla_inversa_superior(self):
        """capacidad_max_mm3 de embalses.py -> cota_max_m de embalses.py, vía la curva."""
        params = EMBALSES["Tomine"]
        cota = CURVA_TOMINE.volumen_a_cota(params.capacidad_max_mm3)
        assert abs(cota - params.cota_max_m) < 1e-9

    def test_extremo_superior_de_la_tabla_coincide_con_embalses_py(self):
        """El último punto de la tabla batimétrica (cota_max_m/volumen_max_mm3,
        el nivel de aguas máximas) debe coincidir EXACTAMENTE con embalses.py.

        NOTA: el extremo INFERIOR de la tabla NO se compara aquí a propósito: la
        tabla se extiende deliberadamente por debajo de cota_min_m (zona de
        volumen muerto, con volumen útil negativo por construcción, ver el
        comentario en curvas.py), así que CURVA_TOMINE.cota_min_m /
        volumen_min_mm3 NO son el mínimo operativo — no deben coincidir con
        embalses.py. La coherencia en el mínimo ya la garantizan
        test_ancla_inferior/test_ancla_inversa_inferior arriba, vía interpolación
        en la cota operativa (no en el extremo bruto de la tabla).
        """
        params = EMBALSES["Tomine"]
        assert abs(CURVA_TOMINE.cota_max_m - params.cota_max_m) < 1e-9
        assert abs(CURVA_TOMINE.volumen_max_mm3 - params.capacidad_max_mm3) < 1e-9


# ---------------------------------------------------------------------------
# (c) Inversas consistentes (round-trip)
# ---------------------------------------------------------------------------

class TestRoundTripTomine:
    @pytest.mark.parametrize("cota_ini", [2560.0, 2570.0, 2580.0, 2590.0, 2596.0])
    def test_cota_a_vol_a_cota(self, cota_ini: float):
        """cota → volumen → cota devuelve la cota original con tolerancia 1e-6 m."""
        vol = CURVA_TOMINE.cota_a_volumen(cota_ini)
        cota_rec = CURVA_TOMINE.volumen_a_cota(vol)
        assert abs(cota_rec - cota_ini) < 1e-6, (
            f"Round-trip fallido: cota_ini={cota_ini}, cota_rec={cota_rec}"
        )

    @pytest.mark.parametrize("vol_ini", [10.0, 100.0, 300.0, 500.0, 650.0])
    def test_vol_a_cota_a_vol(self, vol_ini: float):
        """volumen → cota → volumen devuelve el volumen original con tolerancia 1e-6 Mm³."""
        cota = CURVA_TOMINE.volumen_a_cota(vol_ini)
        vol_rec = CURVA_TOMINE.cota_a_volumen(cota)
        assert abs(vol_rec - vol_ini) < 1e-6, (
            f"Round-trip fallido: vol_ini={vol_ini}, vol_rec={vol_rec}"
        )


# ---------------------------------------------------------------------------
# (d) Fallback lineal para Neusa y Sisga
# ---------------------------------------------------------------------------

class TestFallbackLineal:
    @pytest.mark.parametrize("nombre", ["Neusa", "Sisga"])
    def test_fallback_no_lanza(self, nombre: str):
        """Neusa y Sisga (sin curva real) no lanzan excepción al convertir."""
        params = EMBALSES[nombre]
        vol_medio = (params.capacidad_min_mm3 + params.capacidad_max_mm3) / 2
        cota = volumen_a_cota(vol_medio, nombre, params)
        vol_rec = cota_a_volumen(cota, nombre, params)
        # El valor devuelto debe ser un float finito
        assert np.isfinite(cota)
        assert np.isfinite(vol_rec)

    @pytest.mark.parametrize("nombre", ["Neusa", "Sisga"])
    def test_fallback_en_extremos(self, nombre: str):
        """En los extremos, el fallback lineal devuelve cota_min y cota_max."""
        params = EMBALSES[nombre]
        cota_en_min = volumen_a_cota(params.capacidad_min_mm3, nombre, params)
        cota_en_max = volumen_a_cota(params.capacidad_max_mm3, nombre, params)
        assert abs(cota_en_min - params.cota_min_m) < 1e-9
        assert abs(cota_en_max - params.cota_max_m) < 1e-9

    @pytest.mark.parametrize("nombre", ["Neusa", "Sisga"])
    def test_fallback_monotonico(self, nombre: str):
        """El fallback lineal es monótono creciente."""
        params = EMBALSES[nombre]
        vols = np.linspace(params.capacidad_min_mm3, params.capacidad_max_mm3, 50)
        cotas = np.array([volumen_a_cota(v, nombre, params) for v in vols])
        assert np.all(np.diff(cotas) >= 0)

    @pytest.mark.parametrize("nombre", ["Neusa", "Sisga"])
    def test_neusa_sisga_no_en_registro(self, nombre: str):
        """Neusa y Sisga no deben estar en el registro de curvas reales."""
        assert nombre not in CURVAS

    def test_tomine_en_registro(self):
        """Tominé sí debe estar en el registro de curvas reales."""
        assert "Tomine" in CURVAS


# ---------------------------------------------------------------------------
# (e) Acotación fuera de rango (sin extrapolación)
# ---------------------------------------------------------------------------

class TestAcotacionFueraDeRango:
    def test_cota_bajo_minimo_da_volumen_minimo(self):
        """Una cota por debajo del mínimo de la tabla devuelve el volumen mínimo."""
        vol = CURVA_TOMINE.cota_a_volumen(CURVA_TOMINE.cota_min_m - 10.0)
        assert abs(vol - CURVA_TOMINE.volumen_min_mm3) < 1e-9

    def test_cota_sobre_maximo_da_volumen_maximo(self):
        """Una cota por encima del máximo de la tabla devuelve el volumen máximo."""
        vol = CURVA_TOMINE.cota_a_volumen(CURVA_TOMINE.cota_max_m + 10.0)
        assert abs(vol - CURVA_TOMINE.volumen_max_mm3) < 1e-9

    def test_volumen_bajo_minimo_da_cota_minima(self):
        """Un volumen por debajo del mínimo de la tabla devuelve la cota mínima.

        En convención útil la tabla llega hasta un volumen mínimo negativo (zona de
        volumen muerto); se usa un valor claramente por debajo de ese mínimo.
        """
        cota = CURVA_TOMINE.volumen_a_cota(CURVA_TOMINE.volumen_min_mm3 - 10.0)
        assert abs(cota - CURVA_TOMINE.cota_min_m) < 1e-9

    def test_volumen_sobre_maximo_da_cota_maxima(self):
        """Un volumen por encima del máximo de la tabla devuelve la cota máxima."""
        cota = CURVA_TOMINE.volumen_a_cota(9999.0)
        assert abs(cota - CURVA_TOMINE.cota_max_m) < 1e-9


# ---------------------------------------------------------------------------
# (f) Validación de construcción
# ---------------------------------------------------------------------------

class TestValidacionConstruccion:
    def test_rechaza_cotas_no_monotanas(self):
        """CurvaCotaVolumen rechaza una tabla con cotas no estrictamente crecientes."""
        with pytest.raises(ValueError, match="creciente"):
            CurvaCotaVolumen(
                nombre="Test",
                cotas_m=[100.0, 90.0, 110.0],   # no monotona
                volumenes_mm3=[1.0, 2.0, 3.0],
            )

    def test_rechaza_longitudes_distintas(self):
        """CurvaCotaVolumen rechaza arrays de longitud diferente."""
        with pytest.raises(ValueError, match="longitud"):
            CurvaCotaVolumen(
                nombre="Test",
                cotas_m=[100.0, 110.0, 120.0],
                volumenes_mm3=[1.0, 2.0],         # longitud distinta
            )

    def test_numero_de_puntos_tomine(self):
        """La curva de Tominé tiene exactamente 43 puntos (los de la batimetría oficial)."""
        assert len(CURVA_TOMINE) == 43

"""Pruebas del umbral de caudal ecológico VMF (data_contracts.caudal_ecologico).

Cobertura:
  (a) Los 12 valores EFR documentados, exactos.
  (b) Mes inválido lanza ValueError.
  (c) umbral_fijo_m3s devuelve un valor constante, independiente del mes.
"""
from __future__ import annotations

import pytest

from pbcrl.data_contracts.caudal_ecologico import (
    EFR_VMF_M3S,
    MAF_M3S,
    q_eco_m3s,
    umbral_fijo_m3s,
)


class TestValoresEFR:
    """Los 12 valores documentados del umbral VMF, exactos."""

    VALORES_ESPERADOS = {
        1: 2.32, 2: 2.13, 3: 2.74, 4: 4.40, 5: 3.89, 6: 6.53,
        7: 7.51, 8: 5.95, 9: 3.73, 10: 4.53, 11: 3.80, 12: 2.50,
    }

    @pytest.mark.parametrize("mes,valor_esperado", sorted(VALORES_ESPERADOS.items()))
    def test_valor_mensual_correcto(self, mes, valor_esperado):
        assert q_eco_m3s(mes) == pytest.approx(valor_esperado, abs=1e-9)

    def test_diccionario_completo(self):
        """EFR_VMF_M3S tiene exactamente 12 entradas, meses 1-12."""
        assert set(EFR_VMF_M3S.keys()) == set(range(1, 13))

    def test_maf_documentado(self):
        assert MAF_M3S == pytest.approx(12.432, abs=1e-6)


class TestMesInvalido:
    @pytest.mark.parametrize("mes_invalido", [0, 13, -1, 100])
    def test_mes_fuera_de_rango_lanza_error(self, mes_invalido):
        with pytest.raises(ValueError):
            q_eco_m3s(mes_invalido)


class TestUmbralFijo:
    def test_umbral_fijo_constante_en_todos_los_meses(self):
        """umbral_fijo_m3s(valor) devuelve el mismo valor sin importar el mes."""
        umbral = umbral_fijo_m3s(2.0)
        for mes in range(1, 13):
            assert umbral(mes) == pytest.approx(2.0, abs=1e-9)

    def test_umbral_fijo_ignora_mes_invalido(self):
        """A diferencia de q_eco_m3s, el umbral fijo no valida el mes (es constante)."""
        umbral = umbral_fijo_m3s(3.5)
        assert umbral(0) == pytest.approx(3.5, abs=1e-9)
        assert umbral(99) == pytest.approx(3.5, abs=1e-9)

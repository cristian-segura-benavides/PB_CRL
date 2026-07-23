"""Genera el archivo de resultados día a día del shield (ambos escenarios de
Tibitóc, CON y SIN shield) que consume la pestaña "VMF y Shield de Protección"
del dashboard.

Se corre UNA sola vez (o cuando cambien los datos fuente); el dashboard NO
recalcula el shield en cada carga, solo lee este CSV. Reutiliza
`construir_serie_diaria` de scratch_shield/simulacion_historica_con_shield.py
(no duplica la carga de datos) y hace el mismo rollout continuo ya verificado
(NOTAS.md 4i), esta vez guardando el detalle por día en vez de solo el
resumen agregado.

Corre CADA escenario dos veces (con_shield=True y con_shield=False), con las
MISMAS acciones (descargas históricas observadas) y forzantes en ambos casos
— para que el dashboard pueda mostrar, con un checkbox, qué habría pasado sin
el shield (ver la comparación aislada en
scratch_shield/simulacion_estocastica_con_shield.py, que hace lo mismo con
forzantes generadas por el modelo estocástico).

Salida: scratch_shield/resultados_shield_dashboard.csv (no versionado — ver
.gitignore, *.csv). Columnas: fecha, escenario, con_shield, mes, shield_actuo,
violacion_real, correccion_neusa_m3s, correccion_sisga_m3s,
correccion_tomine_m3s.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from pbcrl.data_contracts.captaciones import (  # noqa: E402
    CAUDAL_TIBITOC_AMPLIADO_M3S,
    CAUDAL_TIBITOC_HISTORICO_M3S,
)
from pbcrl.environment.config import ConfigEntorno  # noqa: E402
from pbcrl.environment.entorno import EntornoEmbalses, ForzantesExternos  # noqa: E402

from scratch_shield.simulacion_historica_con_shield import (  # noqa: E402
    NOMBRES,
    construir_serie_diaria,
)

SALIDA_CSV = Path(__file__).parent / "resultados_shield_dashboard.csv"


def correr_y_registrar(
    datos: pd.DataFrame, escenario: str, tibitoc_nominal: float, con_shield: bool
) -> pd.DataFrame:
    volumenes_iniciales = {n: float(datos.iloc[0][f"volumen_{n}"]) for n in NOMBRES}

    config = ConfigEntorno(con_shield=con_shield)
    env = EntornoEmbalses(config=config)
    env.reset(
        volumenes_iniciales_mm3=volumenes_iniciales,
        caudal_natural_inicial_m3s=float(datos.iloc[0]["caudal_saucio_m3s"]),
    )

    registros: list[dict] = []
    for fecha, fila in datos.iloc[1:].iterrows():
        forzantes = ForzantesExternos(
            afluencia_m3s={n: float(fila[f"afluencia_{n}"]) for n in NOMBRES},
            precipitacion_mm={n: float(fila[f"precip_{n}"]) for n in NOMBRES},
            evaporacion_mm={n: float(fila[f"evap_{n}"]) for n in NOMBRES},
            caudal_natural_m3s=float(fila["caudal_saucio_m3s"]),
            mes=fecha.month,
            caudal_tibitoc_m3s=tibitoc_nominal,
        )
        accion_propuesta = {n: float(fila[f"descarga_{n}"]) for n in NOMBRES}
        resultado = env.step(accion_propuesta, forzantes)

        diag = resultado.diagnostico_shield
        if diag is not None:
            actuo = any(diag.violaciones_previas.values())
            correccion = {
                "correccion_neusa_m3s": abs(diag.accion_proyectada["Neusa"] - diag.accion_propuesta["Neusa"]),
                "correccion_sisga_m3s": abs(diag.accion_proyectada["Sisga"] - diag.accion_propuesta["Sisga"]),
                "correccion_tomine_m3s": abs(diag.accion_proyectada["Tomine"] - diag.accion_propuesta["Tomine"]),
            }
        else:
            actuo = False
            correccion = {
                "correccion_neusa_m3s": 0.0,
                "correccion_sisga_m3s": 0.0,
                "correccion_tomine_m3s": 0.0,
            }

        registros.append({
            "fecha": fecha,
            "escenario": escenario,
            "con_shield": con_shield,
            "mes": fecha.month,
            "shield_actuo": actuo,
            "violacion_real": bool(resultado.violacion_ecologica),
            **correccion,
        })

    return pd.DataFrame(registros)


def main() -> None:
    datos = construir_serie_diaria()
    print(f"Días con datos completos: {len(datos)}")

    bloques = []
    for escenario, nominal in [
        ("historico", CAUDAL_TIBITOC_HISTORICO_M3S),
        ("ampliado", CAUDAL_TIBITOC_AMPLIADO_M3S),
    ]:
        for con_shield in (True, False):
            etiqueta = "CON shield" if con_shield else "SIN shield"
            print(f"Corriendo escenario {escenario} ({etiqueta}, nominal={nominal} m3/s)...")
            df = correr_y_registrar(datos, escenario, nominal, con_shield)
            bloques.append(df)
            pct_actua = 100 * df["shield_actuo"].mean()
            pct_viola = 100 * df["violacion_real"].mean()
            print(f"  shield actuó: {pct_actua:.2f}%  |  violación real: {pct_viola:.2f}%")

    resultado = pd.concat(bloques, ignore_index=True)
    resultado.to_csv(SALIDA_CSV, index=False)
    print(f"\nGuardado: {SALIDA_CSV} ({len(resultado)} filas)")


if __name__ == "__main__":
    main()

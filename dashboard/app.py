"""Tablero de exploracion de embalses para PB-CRL.

Neusa, Sisga: Excel CAR 20261065095_Embalses.xlsx (volumen) + ae (49).xlsx (evaporacion).
Tominé: Excel Enlaza (volumen, cota, descarga, bombeo, lluvia) + evaporacion ERA5-Land.
Afluencias: balance hidrico inverso (incluye termino de bombeo para Tominé).
"""
from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pbcrl.data_contracts.caudal_ecologico import EFR_VMF_M3S, MAF_M3S, MMF_M3S, REGIMEN_MES

from dashboard import data_loader as data_loader_module

DEFAULT_VOLUME_JUMP_FRACTION = getattr(data_loader_module, "DEFAULT_VOLUME_JUMP_FRACTION", 0.10)
DEFAULT_NEGATIVE_WARNING_THRESHOLD_M3S = getattr(data_loader_module, "DEFAULT_NEGATIVE_WARNING_THRESHOLD_M3S", 2.0)
load_dashboard_context = data_loader_module.load_dashboard_context
operational_limits_table = data_loader_module.operational_limits_table
load_shield_results = data_loader_module.load_shield_results

MESES_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
SHIELD_ESCENARIOS = {
    "historico": "Histórico (4.5 m³/s)",
    "ampliado": "Ampliado (8.0 m³/s)",
}

# --- Flags de presentación (ajuste visual, no de cálculo) ---
# Ninguno borra código: solo controlan qué se RENDERIZA hoy. Volver a True
# restaura la vista completa sin tocar el resto del archivo.
MOSTRAR_TAB_DIAGNOSTICO = False  # oculto para la reunión de hoy
MOSTRAR_EXPLICACION_SHIELD = False  # oculto para la reunión de hoy


st.set_page_config(page_title="PB-CRL Dashboard", layout="wide")

COLORS = {
    "Neusa": "#1f77b4",
    "Sisga": "#2ca02c",
    "Tomine": "#ff7f0e",
}


@st.cache_data(show_spinner="Recalculando limpieza de series…")
def _load_context(volume_jump_fraction: float, negative_warning_threshold_m3s: float):
    try:
        return load_dashboard_context(
            volume_jump_fraction=volume_jump_fraction,
            negative_warning_threshold_m3s=negative_warning_threshold_m3s,
        )
    except TypeError:
        return load_dashboard_context()


def _date_range_controls(min_date: pd.Timestamp, max_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Slider y calendario sincronizados para elegir el rango de fechas, con reset."""
    full_slider = (min_date.to_pydatetime(), max_date.to_pydatetime())
    full_cal = (min_date.date(), max_date.date())
    if "rango_slider" not in st.session_state:
        st.session_state["rango_slider"] = full_slider
    if "rango_cal" not in st.session_state:
        st.session_state["rango_cal"] = full_cal

    def _sync_from_slider() -> None:
        start, end = st.session_state["rango_slider"]
        st.session_state["rango_cal"] = (start.date(), end.date())

    def _sync_from_calendar() -> None:
        value = st.session_state["rango_cal"]
        # date_input devuelve una tupla de 1 elemento mientras el usuario elige el fin.
        if isinstance(value, (tuple, list)) and len(value) == 2:
            start, end = value
            st.session_state["rango_slider"] = (
                datetime.combine(start, time.min),
                datetime.combine(end, time.min),
            )

    def _reset() -> None:
        st.session_state["rango_slider"] = full_slider
        st.session_state["rango_cal"] = full_cal

    st.sidebar.slider(
        "Rango de fechas",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        format="YYYY-MM-DD",
        key="rango_slider",
        on_change=_sync_from_slider,
    )
    st.sidebar.date_input(
        "Rango de fechas por Calendario",
        min_value=min_date.date(),
        max_value=max_date.date(),
        format="YYYY-MM-DD",
        key="rango_cal",
        on_change=_sync_from_calendar,
    )
    st.sidebar.button("Reset", width="stretch", on_click=_reset)

    start, end = st.session_state["rango_slider"]
    return pd.Timestamp(start), pd.Timestamp(end)


def _add_limit_line(fig: go.Figure, name: str, y_value: float, start: pd.Timestamp, end: pd.Timestamp, suffix: str, dash: str) -> None:
    fig.add_trace(
        go.Scatter(
            x=[start, end],
            y=[y_value, y_value],
            mode="lines",
            name=f"{name} {suffix}",
            line=dict(color=COLORS.get(name, "#666666"), dash=dash),
            hoverinfo="skip",
        )
    )


PUMPING_MARKER_COLOR = "#d62728"  # rojo: contrasta con el naranja de la linea de Tomine


def _add_pumping_markers(fig: go.Figure, frame: pd.DataFrame) -> None:
    """Marca con circulos los dias de bombeo (entrada artificial) sobre la linea de volumen.

    Solo aplica a Tominé (unico embalse con bombeo); son eventos dispersos, no rachas.
    """
    if "bombeo_mm3" not in frame.columns:
        return
    pumped = frame[frame["bombeo_mm3"] > 0]
    if pumped.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=pumped.index,
            y=pumped["volumen_mm3"],
            mode="markers",
            name="Días con bombeo (Tominé)",
            marker=dict(color=PUMPING_MARKER_COLOR, size=8, symbol="circle", line=dict(color="white", width=1)),
            customdata=(pumped["bombeo_mm3"] * 1e6),
            hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Bombeo: %{customdata:,.0f} m³<extra></extra>",
        )
    )


def _build_volume_figure(contexts, start: pd.Timestamp, end: pd.Timestamp, show_limits: bool) -> go.Figure:
    fig = go.Figure()
    for name in ["Neusa", "Sisga", "Tomine"]:
        context = contexts[name]
        frame = context.frame.loc[start:end]
        if frame.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["volumen_mm3"],
                mode="lines",
                name=f"{name} volumen",
                line=dict(color=COLORS[name], width=2),
            )
        )
        if name == "Tomine":
            _add_pumping_markers(fig, frame)
        if show_limits:
            _add_limit_line(fig, name, context.params.capacidad_min_mm3, start, end, "volumen minimo", "dash")
            _add_limit_line(fig, name, context.params.capacidad_max_mm3, start, end, "volumen maximo", "dot")

    fig.update_layout(
        title="Volumen historico por embalse",
        xaxis_title="Fecha",
        yaxis_title="Volumen (Mm³)",
        legend_title="Series",
        template="plotly_white",
        height=560,
    )
    fig.update_xaxes(range=[start, end])
    # Ancla el eje en 0: los volumenes minimos (~4-7 Mm³) son pequenos frente a los
    # volumenes reales (~20-110 Mm³), y con autorango quedarian pegados al borde inferior.
    fig.update_yaxes(rangemode="tozero")
    return fig


def _build_inflow_figure(contexts, start: pd.Timestamp, end: pd.Timestamp) -> go.Figure:
    fig = go.Figure()
    for name in ["Neusa", "Sisga", "Tomine"]:
        context = contexts[name]
        frame = context.frame.loc[start:end]
        if frame.empty or "afluencia_m3s" not in frame.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["afluencia_m3s"],
                mode="lines",
                name=f"{name} afluencia",
                line=dict(color=COLORS[name], width=2),
            )
        )

    fig.update_layout(
        title="Afluencia estimada por balance inverso",
        xaxis_title="Fecha",
        yaxis_title="Afluencia (m³/s)",
        legend_title="Series",
        template="plotly_white",
        height=480,
    )
    fig.update_xaxes(range=[start, end])
    return fig


REGIMEN_COLORES = {"bajo": "#d62728", "intermedio": "#ff7f0e", "alto": "#1f77b4"}


def _build_efr_figure() -> go.Figure:
    """Caudal ecológico VMF (EFR) por mes, coloreado por régimen (bajo/intermedio/alto)."""
    meses_idx = list(range(1, 13))
    efr = [EFR_VMF_M3S[m] for m in meses_idx]
    mmf = [MMF_M3S[m] for m in meses_idx]
    regimen = [REGIMEN_MES[m] for m in meses_idx]
    colores = [REGIMEN_COLORES[r] for r in regimen]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=MESES_ES,
            y=efr,
            marker_color=colores,
            customdata=list(zip(mmf, regimen)),
            hovertemplate="Mes: %{x}<br>EFR: %{y:.2f} m³/s<br>MMF: %{customdata[0]:.2f} m³/s"
            "<br>Régimen: %{customdata[1]}<extra></extra>",
            showlegend=False,
        )
    )
    # Trazas fantasma (sin datos visibles) solo para mostrar la leyenda de régimen.
    for nombre_regimen, color in REGIMEN_COLORES.items():
        fig.add_trace(
            go.Bar(x=[None], y=[None], marker_color=color, name=nombre_regimen.capitalize())
        )

    fig.update_layout(
        title="Caudal ecológico VMF (EFR) por mes",
        xaxis_title="Mes",
        yaxis_title="EFR (m³/s)",
        legend_title="Régimen",
        template="plotly_white",
        height=420,
    )
    return fig


# Correlación cruzada histórico vs. modelo estocástico, para los 6 pares de las
# 4 series objetivo. Valores YA CALCULADOS en la validación del modelo entrenado
# (pbcrl.stochastic.entrenamiento, reporte de validación con dummies de mes) —
# no se recalculan aquí. Actualizar a mano si el modelo se reentrena de nuevo.
CORRELACIONES_PARES = [
    "Saucío-Neusa", "Saucío-Sisga", "Saucío-Tomine",
    "Neusa-Sisga", "Neusa-Tomine", "Sisga-Tomine",
]
CORRELACIONES_HISTORICO = [0.233, 0.743, 0.595, 0.256, 0.306, 0.688]
CORRELACIONES_MODELO = [0.242, 0.709, 0.578, 0.091, 0.152, 0.588]


def _build_correlation_comparison_figure() -> go.Figure:
    """Correlación cruzada histórico vs. modelo, para los 6 pares de series."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=CORRELACIONES_PARES, y=CORRELACIONES_HISTORICO,
        name="Histórico observado", marker_color="#1f77b4",
    ))
    fig.add_trace(go.Bar(
        x=CORRELACIONES_PARES, y=CORRELACIONES_MODELO,
        name="Generado por el modelo", marker_color="#ff7f0e",
    ))
    fig.update_layout(
        title="Correlación cruzada entre series: histórico vs. modelo",
        xaxis_title="Par de series",
        yaxis_title="Correlación",
        barmode="group",
        template="plotly_white",
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _build_violation_comparison_figure(df_con: pd.DataFrame, df_sin: pd.DataFrame) -> go.Figure:
    """% de días con violación real por mes: CON shield vs. SIN shield."""
    pct_con = 100 * df_con.groupby("mes")["violacion_real"].mean().reindex(range(1, 13), fill_value=0)
    pct_sin = 100 * df_sin.groupby("mes")["violacion_real"].mean().reindex(range(1, 13), fill_value=0)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=MESES_ES, y=pct_sin.values, name="Sin shield", marker_color="#d62728"))
    fig.add_trace(go.Bar(x=MESES_ES, y=pct_con.values, name="Con shield", marker_color="#2ca02c"))
    fig.update_layout(
        title="% de días con violación real por mes: con vs. sin shield",
        xaxis_title="Mes",
        yaxis_title="Días con violación (%)",
        barmode="group",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _build_shield_monthly_figure(df_escenario: pd.DataFrame) -> go.Figure:
    """Días de intervención del shield por mes calendario (13 años agregados)."""
    conteo = (
        df_escenario[df_escenario["shield_actuo"]]
        .groupby("mes")
        .size()
        .reindex(range(1, 13), fill_value=0)
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(x=MESES_ES, y=conteo.values, marker_color="#d62728"))
    fig.update_layout(
        title="Días de intervención del shield por mes",
        xaxis_title="Mes",
        yaxis_title="Días con corrección",
        template="plotly_white",
        height=420,
    )
    return fig


def _build_shield_correction_figure(df_escenario: pd.DataFrame) -> go.Figure:
    """Magnitud media de corrección por embalse, solo días con intervención."""
    activos = df_escenario[df_escenario["shield_actuo"]]
    embalses = ["Neusa", "Sisga", "Tomine"]
    medias = [
        float(activos[f"correccion_{n.lower()}_m3s"].mean()) if not activos.empty else 0.0
        for n in embalses
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=embalses, y=medias, marker_color=[COLORS[n] for n in embalses]))
    fig.update_layout(
        title="Magnitud media de corrección por embalse",
        xaxis_title="Embalse",
        yaxis_title="Corrección media (m³/s)",
        template="plotly_white",
        height=420,
    )
    return fig


def _diagnostics_note(contexts) -> str:
    parts = []
    for name in ["Neusa", "Sisga", "Tomine"]:
        diag = getattr(contexts[name], "diagnostics", None)
        if diag is None:
            parts.append(f"{name}: diagnostico no disponible en esta sesion.")
            continue
        parts.append(
            f"{name}: Capa 1 corrigio {diag.volume_jump_days} saltos; Capa 2 acoto {diag.clamped_negative_days} dias negativos"
            f" ({diag.clamped_negative_warning_days} por encima del umbral de advertencia)."
        )
    return " ".join(parts)


def _diagnostics_report_table(contexts) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name in ["Neusa", "Sisga", "Tomine"]:
        diagnostics = getattr(contexts[name], "diagnostics", None)
        if diagnostics is None:
            rows.append(
                {
                    "embalse": name,
                    "etapa": "sin_diagnostico",
                    "dias_negativos": None,
                    "pct_negativos": None,
                    "min_negativo_m3s": None,
                    "media_negativa_m3s": None,
                    "mediana_negativa_m3s": None,
                    "p90_abs_negativo_m3s": None,
                    "saltos_volumen_corregidos": None,
                    "umbral_salto_volumen_mm3": None,
                    "negativos_acotados": None,
                    "advertencias_acotadas": None,
                }
            )
            continue
        rows.extend(
            [
                {
                    "embalse": name,
                    "etapa": "crudo",
                    "dias_negativos": diagnostics.raw_negative_days,
                    "pct_negativos": diagnostics.raw_negative_pct,
                    "min_negativo_m3s": diagnostics.raw_negative_min_m3s,
                    "media_negativa_m3s": diagnostics.raw_negative_mean_m3s,
                    "mediana_negativa_m3s": diagnostics.raw_negative_median_m3s,
                    "p90_abs_negativo_m3s": diagnostics.raw_negative_p90_abs_m3s,
                    "saltos_volumen_corregidos": diagnostics.volume_jump_days,
                    "umbral_salto_volumen_mm3": diagnostics.volume_jump_threshold_mm3,
                    "negativos_acotados": 0,
                    "advertencias_acotadas": 0,
                },
                {
                    "embalse": name,
                    "etapa": "tras_capa1",
                    "dias_negativos": diagnostics.after_layer1_negative_days,
                    "pct_negativos": diagnostics.after_layer1_negative_pct,
                    "min_negativo_m3s": diagnostics.after_layer1_negative_min_m3s,
                    "media_negativa_m3s": diagnostics.after_layer1_negative_mean_m3s,
                    "mediana_negativa_m3s": diagnostics.after_layer1_negative_median_m3s,
                    "p90_abs_negativo_m3s": diagnostics.after_layer1_negative_p90_abs_m3s,
                    "saltos_volumen_corregidos": diagnostics.volume_jump_days,
                    "umbral_salto_volumen_mm3": diagnostics.volume_jump_threshold_mm3,
                    "negativos_acotados": 0,
                    "advertencias_acotadas": 0,
                },
                {
                    "embalse": name,
                    "etapa": "final",
                    "dias_negativos": diagnostics.final_negative_days,
                    "pct_negativos": diagnostics.final_negative_pct,
                    "min_negativo_m3s": None,
                    "media_negativa_m3s": None,
                    "mediana_negativa_m3s": None,
                    "p90_abs_negativo_m3s": None,
                    "saltos_volumen_corregidos": diagnostics.volume_jump_days,
                    "umbral_salto_volumen_mm3": diagnostics.volume_jump_threshold_mm3,
                    "negativos_acotados": diagnostics.clamped_negative_days,
                    "advertencias_acotadas": diagnostics.clamped_negative_warning_days,
                },
            ]
        )
    return pd.DataFrame(rows)


DIAGRAMA_MODELO_HTML = """
<div style="display:flex; align-items:center; justify-content:center; gap:14px; flex-wrap:wrap; padding:24px 8px;">
  <div style="display:flex; flex-direction:column; gap:10px;">
    <div style="border:2px solid #1f77b4; border-radius:10px; padding:14px 18px; text-align:center; background:rgba(31,119,180,0.12); font-weight:600; min-width:170px;">
      🌧️ Precipitación
    </div>
    <div style="border:2px dashed #999999; border-radius:10px; padding:14px 18px; text-align:center; background:rgba(153,153,153,0.10); color:#999999; min-width:170px;">
      🌡️ Temperatura*<br><span style="font-size:0.72em;">*pendiente de integrar</span>
    </div>
    <div style="border:2px solid #1f77b4; border-radius:10px; padding:14px 18px; text-align:center; background:rgba(31,119,180,0.12); font-weight:600; min-width:170px;">
      🌊 RONI (El Niño / La Niña)
    </div>
  </div>
  <div style="font-size:2.2em; padding:0 6px;">→</div>
  <div style="border:3px solid #2ca02c; border-radius:14px; padding:30px 22px; text-align:center; background:rgba(44,160,44,0.14); font-weight:700; min-width:170px;">
    Modelo único<br><span style="font-size:0.72em; font-weight:400;">(las 4 salidas correlacionadas entre sí)</span>
  </div>
  <div style="font-size:2.2em; padding:0 6px;">→</div>
  <div style="display:flex; flex-direction:column; gap:10px;">
    <div style="border:2px solid #ff7f0e; border-radius:10px; padding:14px 18px; text-align:center; background:rgba(255,127,14,0.12); font-weight:600; min-width:170px;">
      Caudal Saucío
    </div>
    <div style="border:2px solid #ff7f0e; border-radius:10px; padding:14px 18px; text-align:center; background:rgba(255,127,14,0.12); font-weight:600; min-width:170px;">
      Afluencia Neusa
    </div>
    <div style="border:2px solid #ff7f0e; border-radius:10px; padding:14px 18px; text-align:center; background:rgba(255,127,14,0.12); font-weight:600; min-width:170px;">
      Afluencia Sisga
    </div>
    <div style="border:2px solid #ff7f0e; border-radius:10px; padding:14px 18px; text-align:center; background:rgba(255,127,14,0.12); font-weight:600; min-width:170px;">
      Afluencia Tominé
    </div>
  </div>
</div>
"""


def main() -> None:
    st.title("Tablero exploratorio de embalses")
    st.caption(
        "Volumen: Informacion CAR. Afluencias: balance hidrico inverso con evaporacion "
    )

    if st.sidebar.button("Recargar datos", width="stretch", help="Limpia la cache y vuelve a leer los Excel y los textos del data_loader."):
        st.cache_data.clear()
        st.rerun()

    with st.sidebar.expander("Parámetros técnicos avanzados", expanded=False):
        volume_jump_fraction = st.number_input(
            "Umbral de cambio volumetrico diario admisible (fraccion de la capacidad util)",
            min_value=0.01,
            max_value=0.50,
            value=float(DEFAULT_VOLUME_JUMP_FRACTION),
            step=0.01,
            format="%.2f",
            help="Capa 1: variacion diaria de volumen que se considera anomala y se corrige por "
            "interpolacion. Se expresa como fraccion de la capacidad maxima del embalse.",
        )
        negative_warning_threshold_m3s = st.number_input(
            "Umbral de alerta para residual de afluencia negativa (m³/s)",
            min_value=0.1,
            max_value=20.0,
            value=float(DEFAULT_NEGATIVE_WARNING_THRESHOLD_M3S),
            step=0.1,
            format="%.1f",
            help="Capa 2: magnitud de afluencia negativa (residual del balance inverso) por encima "
            "de la cual el dia se marca como alerta antes de acotarlo a cero.",
        )

    contexts, min_date, max_date = _load_context(volume_jump_fraction, negative_warning_threshold_m3s)
    if pd.isna(min_date) or pd.isna(max_date):
        st.error("No se pudieron cargar las series de Neusa y Sisga.")
        return

    show_limits = st.sidebar.checkbox("Mostrar limites operativos", value=True)
    st.sidebar.caption("Un Click para activar/desactivar y dos para aislar")
    start, end = _date_range_controls(min_date, max_date)

    # Orden: Series primero, luego VMF y Shield, luego el Modelo Estocástico.
    # Diagnostico se oculta hoy (MOSTRAR_TAB_DIAGNOSTICO=False) sin borrar su
    # código — ver el bloque `with tab_diagnostic:` más abajo, que solo se
    # renderiza si el flag está en True.
    if MOSTRAR_TAB_DIAGNOSTICO:
        tab_series, tab_shield, tab_stochastic, tab_diagnostic = st.tabs(
            ["Series", "VMF y Shield de Protección", "Modelo Estocástico", "Diagnostico"]
        )
    else:
        tab_series, tab_shield, tab_stochastic = st.tabs(
            ["Series", "VMF y Shield de Protección", "Modelo Estocástico"]
        )
        tab_diagnostic = None

    with tab_series:
        left, right = st.columns([2, 1], vertical_alignment="top")
        with left:
            st.plotly_chart(_build_volume_figure(contexts, start, end, show_limits), width="stretch")
            st.plotly_chart(_build_inflow_figure(contexts, start, end), width="stretch")

        with right:
            st.subheader("Estado de datos")
            status_rows = []
            for name in ["Neusa", "Sisga", "Tomine"]:
                context = contexts[name]
                serie_disponible = "Disponible" if not context.frame.empty else "Pendiente"
                afluencia_disponible = (
                    "Disponible"
                    if "afluencia_m3s" in context.frame.columns and not context.frame.empty
                    else "Pendiente"
                )
                status_rows.append(
                    {
                        "embalse": name,
                        "serie_volumen": serie_disponible,
                        "serie_afluencia": afluencia_disponible,
                        "estado": context.operational_status,
                    }
                )
            st.dataframe(pd.DataFrame(status_rows), width="stretch", hide_index=True)

            st.subheader("Limites operativos")
            st.dataframe(operational_limits_table(), width="stretch", hide_index=True)

            if MOSTRAR_TAB_DIAGNOSTICO:
                st.subheader("Nota de limpieza")
                st.info(_diagnostics_note(contexts))

            st.subheader("Fuentes por grafica")
            st.write("**Neusa, Sisga**")
            st.write("Volumen: Excel CAR 20261065095_Embalses.xlsx.")
            st.write("Evaporacion: Excel CAR ae (49).xlsx.")
            st.write("**Tominé**")
            st.write("Volumen, cota, descarga, bombeo, lluvia: Excel Enlaza (datos operativos Tomine_Enlaza.xlsx).")
            st.write(
                "Evaporacion: ERA5-Land (flujo de calor latente). Enlaza confirmo por escrito "
                "(radicado ENL-002443-2026-S) que Tominé no cuenta con medicion de evaporacion ni "
                "evaporimetro; se valido en magnitud (~3.19 mm/dia) contra la evaporacion medida de "
                "Neusa y Sisga."
            )
            st.write("**Todos los embalses**")
            st.write(
                "Afluencias: calculadas con pbcrl.hydrology.balance.calcular_afluencia "
                "(para Tominé incluye el termino de bombeo)."
            )

    if tab_diagnostic is not None:
        with tab_diagnostic:
            st.subheader("Diagnostico de afluencias negativas")
            st.write(
                "El reporte compara el estado crudo, el estado tras la limpieza de volumen y el estado final tras acotar residuales negativos."
            )
            st.dataframe(_diagnostics_report_table(contexts), width="stretch", hide_index=True)

            st.subheader("Resumen por embalse")
            summary_rows = []
            for name in ["Neusa", "Sisga", "Tomine"]:
                diag = getattr(contexts[name], "diagnostics", None)
                if diag is None:
                    summary_rows.append(
                        {
                            "embalse": name,
                            "negativos crudos": None,
                            "negativos tras capa 1": None,
                            "negativos acotados": None,
                            "advertencias acotadas": None,
                            "saltos de volumen corregidos": None,
                            "umbral salto volumen (Mm3)": None,
                            "umbral advertencia afluencia (m3/s)": negative_warning_threshold_m3s,
                        }
                    )
                    continue
                summary_rows.append(
                    {
                        "embalse": name,
                        "negativos crudos": diag.raw_negative_days,
                        "negativos tras capa 1": diag.after_layer1_negative_days,
                        "negativos acotados": diag.clamped_negative_days,
                        "advertencias acotadas": diag.clamped_negative_warning_days,
                        "saltos de volumen corregidos": diag.volume_jump_days,
                        "umbral salto volumen (Mm3)": diag.volume_jump_threshold_mm3,
                        "umbral advertencia afluencia (m3/s)": negative_warning_threshold_m3s,
                    }
                )
            st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

            st.caption(
                "Si el numero de negativos tras capa 1 sigue siendo alto, el problema no viene de saltos bruscos de volumen sino del residual del balance; por eso se acota a cero y se deja trazado."
            )

    with tab_shield:
        st.subheader("El límite planetario del agua dulce, operacionalizado (VMF)")
        st.write(
            "El VMF es el Variable Monthly Flow, y es el método de Pastor et al. (2014), "
            "usado a su vez por Gerten et al. (2013) para operacionalizar el límite planetario "
            "del agua dulce de Rockström a escala de cuenca. Lo empleo en vez de un "
            "caudal fijo o del caudal ambiental normativo, por que el normativo certifica "
            "cumplimiento regulatorio local pero no el respeto de un límite planetario, "
            "que es lo que esta tesis busca operacionalizar." 
            "  **EXPLICACIÓN**:El VMF preserva un "
            "porcentaje del caudal medio de CADA mes según su propio régimen: 60% en "
            "meses de caudal bajo, 45% en intermedios, 30% en altos, así el umbral "
            "se adapta a la estacionalidad real de la cuenca en vez de exigir lo "
            "mismo los doce meses del año."
        )
        st.plotly_chart(_build_efr_figure(), width="stretch")
        st.caption(
            f"MAF (caudal medio anual) = {MAF_M3S:.2f} m³/s, calculado en **El Sol** "
            "(el punto de control de toda la cuenca: Saucío + afluencia de Neusa + "
            "Sisga + Tominé, menos la extracción de Tibitóc) — no es un cálculo de "
            "Saucío solo. Ventana 2012-2025."
        )

        st.divider()
        st.subheader("Shield de proyección cuadrática")
        st.caption(
            "Resultados precalculados del rollout histórico completo (2012-01-02 a "
            "2025-05-04, 4606 días) — no se recalcula el shield en esta pestaña."
        )
        try:
            shield_df = load_shield_results()
        except FileNotFoundError as exc:
            st.warning(
                f"{exc}\n\nCorre `python scratch_shield/generar_resultados_dashboard.py` "
                "desde la raíz del proyecto para generarlo."
            )
        else:
            escenario = st.radio(
                "Escenario de extracción Tibitóc",
                options=list(SHIELD_ESCENARIOS.keys()),
                format_func=lambda v: SHIELD_ESCENARIOS[v],
                horizontal=True,
                key="shield_escenario",
            )
            df_con_shield = shield_df[
                (shield_df["escenario"] == escenario) & (shield_df["con_shield"])
            ]
            df_sin_shield = shield_df[
                (shield_df["escenario"] == escenario) & (~shield_df["con_shield"])
            ]

            pct_actua = 100 * df_con_shield["shield_actuo"].mean()
            pct_viola_con = 100 * df_con_shield["violacion_real"].mean()

            mostrar_comparacion = st.checkbox(
                "Comparar contra un escenario SIN shield",
                key="shield_comparar",
                help="Misma acción propuesta y las mismas forzantes, corridas otra vez "
                "con el shield desactivado — para ver qué tanto cambia el resultado.",
            )

            if mostrar_comparacion:
                pct_viola_sin = 100 * df_sin_shield["violacion_real"].mean()
                col1, col2, col3 = st.columns(3)
                col1.metric("Días donde el shield corrigió la acción", f"{pct_actua:.2f}%")
                col2.metric("Violación real — CON shield", f"{pct_viola_con:.2f}%")
                col3.metric(
                    "Violación real — SIN shield",
                    f"{pct_viola_sin:.2f}%",
                    delta=f"{pct_viola_con - pct_viola_sin:+.2f} pts vs. sin shield",
                    delta_color="inverse",
                )
                st.plotly_chart(
                    _build_violation_comparison_figure(df_con_shield, df_sin_shield),
                    width="stretch",
                )
            else:
                col1, col2 = st.columns(2)
                col1.metric("Días donde el shield corrigió la acción", f"{pct_actua:.2f}%")
                col2.metric("Días con violación REAL (tras el recorte físico)", f"{pct_viola_con:.2f}%")

            left, right = st.columns(2)
            with left:
                st.plotly_chart(_build_shield_monthly_figure(df_con_shield), width="stretch")
            with right:
                st.plotly_chart(_build_shield_correction_figure(df_con_shield), width="stretch")

            if MOSTRAR_EXPLICACION_SHIELD:
                st.info(
                    "**Dos cosas distintas.** Que 'el shield corrigió la acción' es una "
                    "garantía matemática: la acción propuesta se proyecta para que, en "
                    "teoría, el caudal en El Sol cumpla el umbral ecológico del mes — "
                    "esto ocurre siempre que hace falta, sin excepción. Que 'el "
                    "resultado físico violó el caudal ecológico' es otra cosa: ocurre "
                    "solo cuando el agua para cumplir esa corrección no está "
                    "físicamente disponible ese día en los embalses, y el recorte "
                    "físico (que actúa después del shield, de forma independiente) "
                    "entrega menos de lo corregido. Ningún shield puede resolver eso "
                    "por sí solo — no puede generar agua que no existe."
                )

    with tab_stochastic:
        st.subheader("Modelo estocástico de afluencias")
        st.write(
            "Un solo modelo que aprende del clima histórico y genera nuevos "
            "escenarios plausibles de caudal para los cuatro puntos de la cuenca "
            "al mismo tiempo — manteniendo la relación natural entre ellos (si "
            "llueve mucho, las cuatro series suben juntas; el modelo no las trata "
            "por separado)."
        )
        st.markdown(DIAGRAMA_MODELO_HTML, unsafe_allow_html=True)

        st.divider()
        st.subheader("¿Qué tan bien reproduce el modelo la relación real entre las series?")
        st.write(
            "Se compara la correlación observada en el histórico contra la que "
            "genera el modelo, para cada par de series. Mientras más parecidas "
            "sean las barras, mejor reproduce el modelo esa relación."
        )
        st.plotly_chart(_build_correlation_comparison_figure(), width="stretch")

        st.divider()
        st.success(
            "**Hallazgo:** la afluencia de Neusa muestra menor correlación con el "
            "clima que Sisga y Tominé — consistente con que Neusa abastece "
            "acueductos además de responder a la lluvia, mientras los otros dos "
            "embalses son principalmente reguladores."
        )


if __name__ == "__main__":
    main()

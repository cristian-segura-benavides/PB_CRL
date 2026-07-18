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

from dashboard import data_loader as data_loader_module

DEFAULT_VOLUME_JUMP_FRACTION = getattr(data_loader_module, "DEFAULT_VOLUME_JUMP_FRACTION", 0.10)
DEFAULT_NEGATIVE_WARNING_THRESHOLD_M3S = getattr(data_loader_module, "DEFAULT_NEGATIVE_WARNING_THRESHOLD_M3S", 2.0)
load_dashboard_context = data_loader_module.load_dashboard_context
operational_limits_table = data_loader_module.operational_limits_table


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


def main() -> None:
    st.title("Tablero exploratorio de embalses")
    st.caption(
        "Volumen: Informacion CAR. Afluencias: balance hidrico inverso con evaporacion "
    )

    if st.sidebar.button("Recargar datos", width="stretch", help="Limpia la cache y vuelve a leer los Excel y los textos del data_loader."):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.header("Parametros de depuracion de series")
    volume_jump_fraction = st.sidebar.number_input(
        "Umbral de cambio volumetrico diario admisible (fraccion de la capacidad util)",
        min_value=0.01,
        max_value=0.50,
        value=float(DEFAULT_VOLUME_JUMP_FRACTION),
        step=0.01,
        format="%.2f",
        help="Capa 1: variacion diaria de volumen que se considera anomala y se corrige por "
        "interpolacion. Se expresa como fraccion de la capacidad maxima del embalse.",
    )
    negative_warning_threshold_m3s = st.sidebar.number_input(
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

    tab_series, tab_diagnostic = st.tabs(["Series", "Diagnostico"])

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


if __name__ == "__main__":
    main()

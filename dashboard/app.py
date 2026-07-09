"""Tablero de exploracion de embalses para PB-CRL.

Volumen: Excel CAR 20261065095_Embalses.xlsx.
Afluencias: balance hidrico inverso con evaporacion de ae (49).xlsx.
Tominé: sin serie operativa diaria en esta entrega.
"""
from __future__ import annotations

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


def _date_slider(min_date: pd.Timestamp, max_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    selected = st.sidebar.slider(
        "Rango de fechas",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
        format="YYYY-MM-DD",
    )
    return pd.Timestamp(selected[0]), pd.Timestamp(selected[1])


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
    for name in ["Neusa", "Sisga"]:
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
    for name in ["Neusa", "Sisga"]:
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
    st.title("Tablero exploratorio de embalses PB-CRL")
    st.caption(
        "Volumen: Excel CAR 20261065095_Embalses.xlsx. Afluencias: balance hidrico inverso con evaporacion de ae (49).xlsx. "
        "Tominé no tiene aun serie operativa diaria."
    )

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
    st.sidebar.caption("Muestra u oculta series clicando su nombre en la leyenda de cada grafica (doble clic para aislar una).")
    start, end = _date_slider(min_date, max_date)

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
                status_rows.append(
                    {
                        "embalse": name,
                        "serie_volumen": "Disponible" if not context.frame.empty else "Pendiente",
                        "serie_afluencia": "Disponible" if name != "Tomine" else "Pendiente",
                        "estado": context.operational_status,
                    }
                )
            st.dataframe(pd.DataFrame(status_rows), width="stretch", hide_index=True)

            st.subheader("Limites operativos")
            st.dataframe(operational_limits_table(), width="stretch", hide_index=True)

            st.subheader("Nota de limpieza")
            st.info(_diagnostics_note(contexts))

            st.subheader("Fuentes por grafica")
            st.write("Volumen: Excel CAR 20261065095_Embalses.xlsx.")
            st.write("Afluencias: calculadas con pbcrl.hydrology.balance.calcular_afluencia.")
            st.write("Evaporacion: Excel CAR ae (49).xlsx.")
            st.write("Tominé: sin serie operativa diaria en esta entrega.")

    with tab_diagnostic:
        st.subheader("Diagnostico de afluencias negativas")
        st.write(
            "El reporte compara el estado crudo, el estado tras la limpieza de volumen y el estado final tras acotar residuales negativos."
        )
        st.dataframe(_diagnostics_report_table(contexts), width="stretch", hide_index=True)

        st.subheader("Resumen por embalse")
        summary_rows = []
        for name in ["Neusa", "Sisga"]:
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

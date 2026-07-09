"""Exporta un dashboard HTML autocontenido para explorar los embalses.

La salida es un archivo HTML que se puede abrir con doble clic en el navegador.
No requiere servidor local ni Streamlit.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dashboard.data_loader import load_dashboard_context, operational_limits_table


COLORS = {
    "Neusa": "#1f77b4",
    "Sisga": "#2ca02c",
    "Tomine": "#ff7f0e",
}

OUTPUT_HTML = ROOT_DIR / "dashboard" / "pbcrl_dashboard.html"


def _add_limit_trace(fig: go.Figure, name: str, y_value: float, start: pd.Timestamp, end: pd.Timestamp, suffix: str, dash: str) -> None:
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


def _build_volume_figure(contexts, start: pd.Timestamp, end: pd.Timestamp) -> go.Figure:
    fig = go.Figure()
    for name in ["Neusa", "Sisga"]:
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
        _add_limit_trace(fig, name, context.params.capacidad_min_mm3, start, end, "volumen minimo", "dash")
        _add_limit_trace(fig, name, context.params.capacidad_max_mm3, start, end, "volumen maximo", "dot")

    fig.update_layout(
        title="Volumen historico por embalse",
        xaxis_title="Fecha",
        yaxis_title="Volumen (Mm3)",
        legend_title="Series",
        template="plotly_white",
        height=560,
    )
    fig.update_xaxes(range=[start, end], rangeslider=dict(visible=True))
    return fig


def _build_inflow_figure(contexts, start: pd.Timestamp, end: pd.Timestamp) -> go.Figure:
    fig = go.Figure()
    for name in ["Neusa", "Sisga"]:
        context = contexts[name]
        frame = context.frame.loc[start:end]
        if frame.empty:
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
        yaxis_title="Afluencia (m3/s)",
        legend_title="Series",
        template="plotly_white",
        height=480,
    )
    fig.update_xaxes(range=[start, end], rangeslider=dict(visible=True))
    return fig


def _table_html(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, classes="table")


def build_html_report() -> str:
    contexts, min_date, max_date = load_dashboard_context()
    volume_fig = _build_volume_figure(contexts, min_date, max_date)
    inflow_fig = _build_inflow_figure(contexts, min_date, max_date)

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
    status_table = _table_html(pd.DataFrame(status_rows))
    limits_table = _table_html(operational_limits_table())

    volume_div = volume_fig.to_html(full_html=False, include_plotlyjs="cdn")
    inflow_div = inflow_fig.to_html(full_html=False, include_plotlyjs=False)

    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PB-CRL Dashboard</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --card: #e5e7eb;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #38bdf8;
    }}
    body {{ margin: 0; font-family: Arial, sans-serif; background: linear-gradient(180deg, #0f172a, #111827 45%, #f8fafc 45%, #f8fafc 100%); color: #0f172a; }}
    header {{ padding: 32px 24px 20px; color: var(--text); }}
    h1 {{ margin: 0 0 8px; font-size: 32px; }}
    p {{ margin: 0; max-width: 900px; color: var(--muted); line-height: 1.5; }}
    main {{ padding: 0 24px 32px; }}
    .grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; align-items: start; }}
    .panel {{ background: white; border-radius: 18px; box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12); padding: 18px; }}
    .stack {{ display: grid; gap: 20px; }}
    .section-title {{ margin: 0 0 12px; font-size: 18px; color: #0f172a; }}
    .note {{ color: #334155; font-size: 14px; line-height: 1.5; }}
    .table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    .table th, .table td {{ border-bottom: 1px solid #e2e8f0; padding: 10px 8px; text-align: left; vertical-align: top; }}
    .table th {{ background: #f8fafc; }}
    @media (max-width: 1100px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>PB-CRL Dashboard</h1>
    <p>Volumen: Excel CAR 20261065095_Embalses.xlsx. Afluencias: balance hídrico inverso con evaporación de ae (49).xlsx. Tominé se muestra como pendiente, sin inventar serie operativa.</p>
  </header>
  <main>
    <div class="grid">
      <div class="stack">
        <section class="panel">
          <h2 class="section-title">Volumen histórico</h2>
          {volume_div}
        </section>
        <section class="panel">
          <h2 class="section-title">Afluencia estimada</h2>
          {inflow_div}
        </section>
      </div>
      <aside class="stack">
        <section class="panel">
          <h2 class="section-title">Estado de datos</h2>
          {status_table}
        </section>
        <section class="panel">
          <h2 class="section-title">Límites operativos</h2>
          {limits_table}
        </section>
        <section class="panel">
          <h2 class="section-title">Fuentes</h2>
          <div class="note">
            <p>Volumen: Excel CAR 20261065095_Embalses.xlsx.</p>
            <p>Afluencias: pbcrl.hydrology.balance.calcular_afluencia.</p>
            <p>Evaporación: Excel CAR ae (49).xlsx.</p>
            <p>Tominé: sin serie operativa diaria en esta entrega.</p>
          </div>
        </section>
      </aside>
    </div>
  </main>
</body>
</html>"""


def main() -> None:
    OUTPUT_HTML.write_text(build_html_report(), encoding="utf-8")
    print(f"HTML generado en: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()

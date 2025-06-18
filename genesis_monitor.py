# mission_control.py
import os
import time
import logging
import dash
from dash import dcc, html, Input, Output, State
import dash_daq as daq
import plotly.graph_objs as go
import duckdb

logger = logging.getLogger(__name__)
import pandas as pd

# ——— Configuration ———
PARQUET_DIR = "snapshots/parquet_export"
HEARTBEATS_FILE = os.path.join(PARQUET_DIR, "heartbeats.parquet")
MUTATION_CONTEXT_FILE = os.path.join(PARQUET_DIR, "mutation_contexts.parquet")
SURVIVAL_DETAILS_FILE = os.path.join(PARQUET_DIR, "survival_details.parquet")
MUTATION_EPISODES_FILE = os.path.join(PARQUET_DIR, "mutation_episodes.parquet")
REFLECTIONS_FILE = os.path.join(PARQUET_DIR, "reflections.parquet")
MUTATION_METRICS_FILE = os.path.join(PARQUET_DIR, "mutation_metrics.parquet")
FATAL_EVENTS_FILE = os.path.join(PARQUET_DIR, "fatal_events.parquet")

REFRESH_INTERVAL = 5  # seconds

# ——— App Initialization ———
app = dash.Dash(__name__)
server = app.server
app.title = "🦰 Genesis Mission Control"

# ——— Data Loaders ———
def load_heartbeats():
    if not os.path.isfile(HEARTBEATS_FILE):
        return pd.DataFrame(columns=['ts', 'heartbeat', 'survival_score'])
    return duckdb.sql(f"SELECT * FROM '{HEARTBEATS_FILE}' ORDER BY ts").df()


def load_mutations():
    if not os.path.isfile(MUTATION_CONTEXT_FILE):
        return pd.DataFrame(columns=['ts', 'strategy', 'param', 'survival_change'])
    return duckdb.sql(f"SELECT * FROM '{MUTATION_CONTEXT_FILE}' ORDER BY ts").df()


def load_survival_details():
    if not os.path.isfile(SURVIVAL_DETAILS_FILE):
        return pd.DataFrame(columns=[
            'ts', 'heartbeat', 'gene_count',
            'cpu', 'memory', 'disk',
            'network', 'composite',
            'cpu_pct', 'disk_io'
        ])
    return duckdb.sql(f"""
        SELECT ts, heartbeat, gene_count,
               cpu, memory, disk,
               network, composite,
               COALESCE(cpu_pct, 0.0) AS cpu_pct,
               COALESCE(disk_io, 0.0) AS disk_io
        FROM '{SURVIVAL_DETAILS_FILE}'
        ORDER BY ts
    """).df()


def load_mutation_episodes():
    if not os.path.isfile(MUTATION_EPISODES_FILE):
        return pd.DataFrame(columns=[
            'ts', 'episode_id', 'strategies_applied', 'parameters_changed',
            'survival_before', 'survival_after', 'survival_change'
        ])
    return duckdb.sql(f"SELECT * FROM '{MUTATION_EPISODES_FILE}' ORDER BY ts").df()


def load_reflections():
    if not os.path.isfile(REFLECTIONS_FILE):
        return pd.DataFrame(columns=[
            'ts', 'heartbeat', 'mode', 'gene_count', 'heartbeat_interval',
            'survival_threshold', 'recent_survival', 'trend', 'most_used_strategy',
            'mutation_rate'
        ])
    return duckdb.sql(f"SELECT * FROM '{REFLECTIONS_FILE}' ORDER BY ts").df()


def load_mutation_metrics():
    if not os.path.isfile(MUTATION_METRICS_FILE):
        return pd.DataFrame(columns=['ts', 'target_rate', 'observed_rate'])
    return duckdb.sql(
        f"SELECT * FROM '{MUTATION_METRICS_FILE}' ORDER BY ts"
    ).df()


def load_fatal_events():
    if not os.path.isfile(FATAL_EVENTS_FILE):
        return pd.DataFrame(columns=['ts', 'description'])
    return duckdb.sql(f"SELECT * FROM '{FATAL_EVENTS_FILE}' ORDER BY ts").df()

# ——— Layout ———
app.layout = html.Div(style={'fontFamily': 'Arial', 'padding': '20px'}, children=[
    html.H1("🦰 Genesis Mission Control", style={'textAlign': 'center'}),

    dcc.Graph(id='heartbeat-graph'),

    html.Div([
        daq.Gauge(
            id='survival-gauge',
            label="Survival Score",
            min=0, max=1,
            showCurrentValue=True,
            color={"gradient": True, "ranges": {"green": [0.7, 1], "yellow": [0.5, 0.7], "red": [0, 0.5]}},
            style={'width': '300px', 'margin': 'auto'}
        ),
        html.Div(id='live-status', style={'textAlign': 'center', 'paddingTop': '10px', 'color': '#666'}),
    ]),

    html.Hr(),

    html.Div([
        html.H4("📊 Mutation Strategy Analytics"),
        html.Label("Filter by Strategy:"),
        dcc.Dropdown(id='strategy-filter', multi=True, placeholder="Select one or more strategies..."),
        dcc.Graph(id='avg-survival-change'),
        dcc.Graph(id='strategy-heatmap'),
    ], style={'paddingTop': '20px'}),

    html.Div([
        html.H4("📈 Resource Usage"),
        dcc.Graph(id='resource-graph'),
    ], style={'paddingTop': '20px'}),

    html.Div([
        html.H4("🔀 Parallel Coordinates (Health Metrics)"),
        dcc.Graph(id='parcoords'),
    ], style={'paddingTop': '20px'}),

    html.Div([
        html.H4("🔍 Correlation Matrix"),
        dcc.Graph(id='corr-heatmap'),
    ], style={'paddingTop': '20px'}),

    html.Div([
        html.H4("📜 Mutation Episode Impact"),
        dcc.Graph(id='mutation-episode-impact'),
    ], style={'paddingTop': '20px'}),

    html.Div([
        html.H4("📓 Self-Reflective Narration"),
        dcc.Graph(id='reflection-timeline'),
    ], style={'paddingTop': '20px'}),

    html.Div([
        html.H4("🧬 Mutation Rate Trend"),
        dcc.Graph(id='mutation-rate-trend'),
    ], style={'paddingTop': '20px'}),

    html.Div([
        html.H4("☠️ Fatal Events"),
        dcc.Graph(id='fatal-events-timeline'),
    ], style={'paddingTop': '20px'}),

    dcc.Interval(id='interval-refresh', interval=REFRESH_INTERVAL * 1000, n_intervals=0)
])

# ——— Callbacks ———
@app.callback(
    [
        Output('heartbeat-graph', 'figure'),
        Output('survival-gauge', 'value'),
        Output('live-status', 'children'),
        Output('strategy-filter', 'options'),
        Output('strategy-filter', 'value'),
        Output('avg-survival-change', 'figure'),
        Output('strategy-heatmap', 'figure'),
        Output('resource-graph', 'figure'),
        Output('parcoords', 'figure'),
        Output('corr-heatmap', 'figure'),
        Output('mutation-episode-impact', 'figure'),
        Output('reflection-timeline', 'figure'),
        Output('mutation-rate-trend', 'figure'),
        Output('fatal-events-timeline', 'figure')
    ],
    [Input('interval-refresh', 'n_intervals')],
    [State('strategy-filter', 'value')]
)
def update_dashboard(n, selected_strategies):
    heartbeats    = load_heartbeats()
    mutations     = load_mutations()
    resources     = load_survival_details()
    episodes      = load_mutation_episodes()
    reflections   = load_reflections()
    mutation_metrics = load_mutation_metrics()

    status = f"✅ Last refresh: {time.strftime('%H:%M:%S')}"

    # Heartbeat chart
    if not heartbeats.empty:
        hb_fig = go.Figure([go.Scatter(x=heartbeats['ts'], y=heartbeats['survival_score'], mode='lines+markers')])
        hb_fig.update_layout(title='Survival Score Over Time', xaxis_title='Timestamp', yaxis=dict(range=[0, 1]))
        last_val = heartbeats['survival_score'].iloc[-1]
    else:
        hb_fig = go.Figure().update_layout(title='No heartbeat data yet', yaxis=dict(range=[0, 1]))
        last_val = 0

    # Strategy dropdown & analytics
    all_strats = sorted(mutations['strategy'].dropna().unique().tolist())
    strat_opts = [{'label': s, 'value': s} for s in all_strats]
    default_strats = selected_strategies if selected_strategies else all_strats
    filt = mutations[mutations['strategy'].isin(default_strats)]

    # Avg survival change
    bar_fig = go.Figure()
    if not filt.empty:
        summary = filt.groupby('strategy')['survival_change'].mean().reset_index()
        bar_fig.add_trace(go.Bar(x=summary['strategy'], y=summary['survival_change']))
        bar_fig.update_layout(title='Average Survival Change by Strategy', yaxis_title='Δ Survival')
    else:
        bar_fig.update_layout(title='No Data for Selected Strategies')

    # Strategy heatmap
    heatmap_fig = go.Figure()
    if not filt.empty:
        heat_data = filt.groupby([filt['ts'].dt.date, 'strategy']).size().unstack(fill_value=0)
        heatmap_fig = go.Figure(data=go.Heatmap(z=heat_data.values, x=heat_data.columns, y=heat_data.index.astype(str), colorscale='Viridis'))
        heatmap_fig.update_layout(title='Strategy Usage Over Time', xaxis_title='Strategy', yaxis_title='Date')

    # Resource usage over time
    res_fig = go.Figure()
    if not resources.empty:
        res_fig.add_trace(go.Scatter(x=resources['ts'], y=resources['cpu_pct'], mode='lines', name='CPU%'))
        res_fig.add_trace(go.Scatter(x=resources['ts'], y=resources['memory'], mode='lines', name='Mem'))
        res_fig.add_trace(go.Scatter(x=resources['ts'], y=resources['disk'], mode='lines', name='Disk'))
        res_fig.update_layout(title='Resource Usage', xaxis_title='Timestamp')

    # Parallel coordinates
    parcoords_fig = go.Figure()
    if not resources.empty:
        parcoords_fig = go.Figure(data=go.Parcoords(
            dimensions=[
                dict(label='CPU', values=resources['cpu']),
                dict(label='Memory', values=resources['memory']),
                dict(label='Disk', values=resources['disk']),
                dict(label='Network', values=resources['network']),
                dict(label='Composite', values=resources['composite'])
            ]
        ))


    # Correlation heatmap
    corr_fig = go.Figure()
    if not resources.empty:
        corr = resources[['cpu','memory','disk','network','composite']].corr()
        corr_fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='RdBu', zmid=0))
        corr_fig.update_layout(title='Metric Correlation Matrix')

    # Mutation episode impact
    ep_fig = go.Figure()
    if not episodes.empty:
        ep_fig.add_trace(go.Scatter(x=episodes['ts'], y=episodes['survival_change'], mode='lines+markers', name='Δ Survival'))
        ep_fig.update_layout(title='Mutation Episode Survival Change', xaxis_title='Timestamp', yaxis_title='Δ Survival')

    # Reflection timeline
    refl_fig = go.Figure()
    if not reflections.empty:
        refl_fig.add_trace(go.Scatter(x=reflections['ts'], y=reflections['recent_survival'], mode='lines+markers', name='Survival'))
        refl_fig.update_layout(title='Self-Reflective Narration (Survival)', xaxis_title='Timestamp', yaxis_title='Survival Score')

    # Mutation rate trend
    mrt_fig = go.Figure()
    if not mutation_metrics.empty:
        mrt_fig.add_trace(go.Scatter(x=mutation_metrics['ts'], y=mutation_metrics['target_rate'], mode='lines+markers', name='Target Rate'))
        mrt_fig.add_trace(go.Scatter(x=mutation_metrics['ts'], y=mutation_metrics['observed_rate'], mode='lines+markers', name='Observed Rate'))
        mrt_fig.update_layout(title='Mutation Rate Trend', xaxis_title='Timestamp', yaxis_title='Rate')

    # Fatal events timeline
    events = load_fatal_events()
    fatal_fig = go.Figure()
    if not events.empty:
        fatal_fig.add_trace(go.Scatter(x=events['ts'], y=[1]*len(events), mode='markers', text=events['description']))
        fatal_fig.update_layout(title='Fatal Events', xaxis_title='Timestamp', yaxis=dict(showticklabels=False))

    return (
        hb_fig,
        last_val,
        status,
        strat_opts,
        default_strats,
        bar_fig,
        heatmap_fig,
        res_fig,
        parcoords_fig,
        corr_fig,
        ep_fig,
        refl_fig,
        mrt_fig,
        fatal_fig
    )

# ——— Main Runner ———
if __name__ == '__main__':
    from logging_config import configure_logging
    configure_logging()
    if not os.path.isdir(PARQUET_DIR):
        logger.warning(f"[WARN] Parquet directory not found: {PARQUET_DIR}")
    app.run_server(debug=True)

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import yaml

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])  # Changed theme to CYBORG for a modern look

app.layout = dbc.Container([
    html.H1("Capacity Disaggregation Configuration", className="text-center mt-4 mb-4", style={"color": "#F8F9FA"}),

    # Main row for side-by-side layout
    dbc.Row([
        # Solar Configuration Section
        dbc.Col(dbc.Card([
            dbc.CardHeader("Solar Configuration", style={"background-color": "#343a40", "color": "#F8F9FA"}),
            dbc.CardBody([
                dbc.Row([dbc.Col(html.Label("Max Capacity (GW)", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(dcc.Input(id="solar-max-capacity", type="number", value=5, className="form-control"), width=8)],
                         className="mb-3"),
                dbc.Row([dbc.Col(html.Label("Landuse Intensity", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(dcc.Input(id="solar-landuse-intensity", type="number", value=1.45, className="form-control"), width=8)],
                         className="mb-3"),
                dbc.Row([dbc.Col(html.Label("Atlite Panel Type", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(dcc.Dropdown(id="solar-atlite-panel",
                                               options=[{'label': 'CSi', 'value': 'CSi'},
                                                        {'label': 'CdTe', 'value': 'CdTe'}],
                                               value='CSi', className="form-control"), width=8)],
                         className="mb-3"),
                dbc.Row([dbc.Col(html.Label("Tracking Type", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(dcc.Dropdown(id="solar-tracking",
                                               options=[{'label': 'None', 'value': 'None'},
                                                        {'label': 'Horizontal', 'value': 'horizontal'},
                                                        {'label': 'Tilted Horizontal', 'value': 'tilted_horizontal'},
                                                        {'label': 'Vertical', 'value': 'vertical'},
                                                        {'label': 'Dual', 'value': 'dual'}],
                                               value='dual', className="form-control"), width=8)],
                         className="mb-3"),
            ])
        ], className="mb-4 shadow-lg"), width=6),

        # Wind Configuration Section
        dbc.Col(dbc.Card([
            dbc.CardHeader("Wind Configuration", style={"background-color": "#343a40", "color": "#F8F9FA"}),
            dbc.CardBody([
                dbc.Row([dbc.Col(html.Label("Max Capacity (GW)", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(dcc.Input(id="wind-max-capacity", type="number", value=15, className="form-control"), width=8)],
                         className="mb-3"),
                dbc.Row([dbc.Col(html.Label("Landuse Intensity", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(dcc.Input(id="wind-landuse-intensity", type="number", value=3, className="form-control"), width=8)],
                         className="mb-3"),
                dbc.Row([dbc.Col(html.Label("Capacity Factor Low", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(dcc.Input(id="wind-CF-low", type="number", value=0.2, className="form-control"), width=8)],
                         className="mb-3"),
                dbc.Row([dbc.Col(html.Label("Capacity Factor High", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(dcc.Input(id="wind-CF-high", type="number", value=1, className="form-control"), width=8)],
                         className="mb-3"),
            ])
        ], className="mb-4 shadow-lg"), width=6),

    ], className="mb-4"),

    # BESS and Transmission Configuration Section in a new row
    dbc.Row([
        # BESS Configuration Section
        dbc.Col(dbc.Card([
            dbc.CardHeader("BESS Configuration", style={"background-color": "#343a40", "color": "#F8F9FA"}),
            dbc.CardBody([
                dbc.Row([dbc.Col(html.Label("Max Capacity (GW)", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(dcc.Input(id="bess-max-capacity", type="number", value=10, className="form-control"), width=8)],
                         className="mb-3"),
                dbc.Row([dbc.Col(html.Label("Storage Discharge Duration (hrs)", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(dcc.Input(id="bess-storage-duration", type="number", value=5.5, className="form-control"), width=8)],
                         className="mb-3"),
            ])
        ], className="mb-4 shadow-lg"), width=6),

        # Transmission Configuration Section
        dbc.Col(dbc.Card([
            dbc.CardHeader("Transmission Configuration", style={"background-color": "#343a40", "color": "#F8F9FA"}),
            dbc.CardBody([
                dbc.Row([dbc.Col(html.Label("Grid Connection Cost per Km (M$/km)", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(dcc.Input(id="transmission-grid-cost", type="number", value=2.6, className="form-control"), width=8)],
                         className="mb-3"),
                dbc.Row([dbc.Col(html.Label("Transmission Line Rebuild Cost (M$/km)", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(dcc.Input(id="transmission-rebuild-cost", type="number", value=0.56, className="form-control"), width=8)],
                         className="mb-3"),
                dbc.Row([dbc.Col(html.Label("Proximity Filter (km)", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(dcc.Input(id="transmission-proximity-filter", type="number", value=100, className="form-control"), width=8)],
                         className="mb-3"),
            ])
        ], className="mb-4 shadow-lg"), width=6),
    ]),

    # Submit Button
    dbc.Button("Submit and Save Configuration", id="submit-button", n_clicks=0, color="primary", className="mt-3 mb-3"),
    
    # Output Message
    html.Div(id="output-config", className="mt-3", style={"color": "#F8F9FA"})
])

# Callback to generate the configuration dictionary and save to a YAML file
@app.callback(
    Output("output-config", "children"),
    Input("submit-button", "n_clicks"),
    State("solar-max-capacity", "value"),
    State("solar-landuse-intensity", "value"),
    State("solar-atlite-panel", "value"),
    State("solar-tracking", "value"),
    State("wind-max-capacity", "value"),
    State("wind-landuse-intensity", "value"),
    State("wind-CF-low", "value"),
    State("wind-CF-high", "value"),
    State("bess-max-capacity", "value"),
    State("bess-storage-duration", "value"),
    State("transmission-grid-cost", "value"),
    State("transmission-rebuild-cost", "value"),
    State("transmission-proximity-filter", "value")
)
def update_config(n_clicks, solar_max_cap, solar_land_intensity, solar_panel, solar_tracking,
                  wind_max_cap, wind_land_intensity, wind_CF_low, wind_CF_high,
                  bess_max_cap, bess_storage_duration,
                  tx_grid_cost, tx_rebuild_cost, tx_proximity_filter):
    
    # Constructing the configuration dictionary
    config_dict = {
        "capacity_disaggregation": {
            "solar": {
                "max_capacity": solar_max_cap,
                "landuse_intensity": solar_land_intensity,
                "atlite_panel": solar_panel,
                "tracking": solar_tracking
            },
            "wind": {
                "max_capacity": wind_max_cap,
                "landuse_intensity": wind_land_intensity,
                "CF_low": wind_CF_low,
                "CF_high": wind_CF_high
            },
            "bess": {
                "max_capacity": bess_max_cap,
                "storage_discharge_duration": bess_storage_duration
            },
            "transmission": {
                "grid_connection_cost_per_Km": tx_grid_cost,
                "tx_line_rebuild_cost": tx_rebuild_cost,
                "proximity_filter": tx_proximity_filter
            }
        }
    }

    # Save the configuration to a YAML file
    if n_clicks > 0:
        with open('config/user_config.yaml', 'w') as file:
            yaml.dump(config_dict, file)

    return f"Configuration saved successfully as 'config/user_config.yaml'"
    

if __name__ == "__main__":
    app.run_server(debug=True)

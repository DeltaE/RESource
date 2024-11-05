import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import yaml

# Define turbine sources globally for easy access


# InputComponents class to handle dynamic components
class InputComponents:
    @staticmethod
    def GetSlider(id, min_val, max_val, step, value, marks_interval, tooltip=True):
        return dcc.Slider(
            id=id,
            min=min_val,
            max=max_val,
            step=step,
            value=value,
            marks=InputComponents.generate_marks(min_val, max_val, marks_interval),
            tooltip={"placement": "bottom", "always_visible": tooltip}
        )

    @staticmethod
    def generate_marks(min_val, max_val, interval):
        return {i: str(i) for i in range(int(min_val), int(max_val) + 1, interval + 1)}

# Solar configuration class
class SolarConfig:
    def __init__(self):
        self.layout = dbc.Card([
            dbc.CardHeader("Solar Configuration", style={"background-color": "#343a40", "color": "#F8F9FA"}),
            dbc.CardBody([
                dbc.Row([dbc.Col(html.Label("Max Capacity (GW)", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(InputComponents.GetSlider("solar-max-capacity", 0, 20, 1, 5, 1), width=8)],
                         className="mb-3"),
                dbc.Row([dbc.Col(html.Label("Landuse Intensity (MW/km²)", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(InputComponents.GetSlider("solar-landuse-intensity", 0, 5, 0.1, 1.45, 2), width=8)],
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
        ], className="mb-4 shadow-lg")

# Wind configuration class
class WindConfig:
    TURBINE_SOURCES = {
    "atlite": [
        {'label': 'Enercon E82 3MW - Rotor Dia: 82m', 'value': 'Enercon_E82_3000kW'},
        {'label': 'Vestas V90 3MW - Rotor Dia: 90m', 'value': 'Vestas_V90_3MW'}
    ],
    "OEDB": [
        {'label': 'GE2.75 120 - Rotor Dia: 120m', 'value': 'GE2.75_120'},
        {'label': 'Senvion/REpower 3.2M114 - Rotor Dia: 114m', 'value': '3.2M114_NES'}
    ]
}
    def __init__(self):
        self.layout = dbc.Card([
            dbc.CardHeader("Wind Configuration", style={"background-color": "#343a40", "color": "#F8F9FA"}),
            dbc.CardBody([
                dbc.Row([dbc.Col(html.Label("Max Capacity (GW)", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(InputComponents.GetSlider("wind-max-capacity", 0, 30, 0.5, 15, 2), width=8)],
                         className="mb-3"),
                dbc.Row([dbc.Col(html.Label("Landuse Intensity (MW/km²)", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(InputComponents.GetSlider("wind-landuse-intensity", 0, 8, 0.1, 3, 1), width=8)],
                         className="mb-3"),
                dbc.Row([dbc.Col(html.Label("Capacity Factor Low", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(InputComponents.GetSlider("wind-CF-low", 0.1, 0.4, 0.05, 0.2, 1), width=8)],
                         className="mb-3"),
                dbc.Row([dbc.Col(html.Label("Capacity Factor High", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(InputComponents.GetSlider("wind-CF-high", 0.4, 1, 0.06, 0.4, 1), width=8)],
                         className="mb-3"),
                dbc.Row([dbc.Col(html.Label("Select Turbine Data Source", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(dcc.Dropdown(id="wind-turbine-source",
                                                options=[{'label': 'Atlite', 'value': 'atlite'},
                                                         {'label': 'OEDB', 'value': 'OEDB'}],
                                                value='OEDB', className="form-control"), width=8)],
                         className="mb-3"),
                dbc.Row([dbc.Col(html.Label("Select Turbine Model", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(dcc.Dropdown(id="wind-turbine-model",
                                                options=WindConfig.TURBINE_SOURCES['OEDB'],  # Default options based on default source
                                                value=WindConfig.TURBINE_SOURCES['OEDB'][0]['value'], className="form-control"), width=8)],
                         className="mb-3"),
            ])
        ], className="mb-4 shadow-lg")

# BESS configuration class
class BESSConfig:
    def __init__(self):
        self.layout = dbc.Card([
            dbc.CardHeader("BESS Configuration", style={"background-color": "#343a40", "color": "#F8F9FA"}),
            dbc.CardBody([
                dbc.Row([dbc.Col(html.Label("Max Capacity (GW)", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(InputComponents.GetSlider("bess-max-capacity", 1, 50, 0.5, 10, 5), width=8)],
                         className="mb-3"),
                dbc.Row([dbc.Col(html.Label("Storage Discharge Duration (hrs)", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(InputComponents.GetSlider("bess-storage-duration", 1, 12, 0.1, 5.5, 1), width=8)],
                         className="mb-3"),
            ])
        ], className="mb-4 shadow-lg")

# Transmission configuration class
class TransmissionConfig:
    def __init__(self):
        self.layout = dbc.Card([
            dbc.CardHeader("Transmission Configuration", style={"background-color": "#343a40", "color": "#F8F9FA"}),
            dbc.CardBody([
                dbc.Row([dbc.Col(html.Label("Grid Connection Cost per Km (M$/km)", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(InputComponents.GetSlider("transmission-grid-cost", 1, 5, 0.05, 2.6, 1), width=8)],
                         className="mb-3"),
                dbc.Row([dbc.Col(html.Label("Transmission Line Rebuild Cost (M$/km)", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(InputComponents.GetSlider("transmission-rebuild-cost", 1, 5, 0.05, 2.6, 1), width=8)],
                         className="mb-3"),
                dbc.Row([dbc.Col(html.Label("Proximity Filter (km)", style={"color": "#F8F9FA"}), width=4),
                          dbc.Col(InputComponents.GetSlider("transmission-proximity-filter", 1, 50, 1, 10, 5), width=8)],
                         className="mb-3"),
            ])
        ], className="mb-4 shadow-lg")
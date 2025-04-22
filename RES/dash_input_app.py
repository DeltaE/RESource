import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import yaml
from dash_input_config import SolarConfig, WindConfig, BESSConfig, TransmissionConfig

class CapacityDisaggregationApp:
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
        # >> Other themes
        # Bootstrap (default)
        # Darkly
        # Cosmo
        # Cyborg
        # Flatly
        # Journal
        # Litera
        # Lumen
        # Paper
        # Sandstone
        # Simplex
        # Slate
        # Spacelab
        # Superhero
        # United
        # Yeti


        # Instantiate configuration sections
        self.solar_config = SolarConfig()
        self.wind_config = WindConfig()
        self.bess_config = BESSConfig()
        self.transmission_config = TransmissionConfig()

        # Layout
        self.app.layout = dbc.Container([
            html.H1("Capacity Disaggregation Configuration", className="text-center mt-4 mb-4", style={"color": "#F8F9FA"}),

            # Layout Rows for Config Sections
            dbc.Row([dbc.Col(self.solar_config.layout, width=6), dbc.Col(self.wind_config.layout, width=6)], className="mb-4"),
            dbc.Row([dbc.Col(self.bess_config.layout, width=6), dbc.Col(self.transmission_config.layout, width=6)]),

            # Submit Button
            dbc.Button("Submit and Save Configuration", id="submit-button", n_clicks=0, color="primary", className="mt-3 mb-3"),

            # Output Message
            html.Div(id="output-config", className="mt-3", style={"color": "#F8F9FA"})
        ])

        # Callbacks
        self.register_callbacks()

    def register_callbacks(self):
        # Update turbine model options based on the selected source
        @self.app.callback(
            Output('wind-turbine-model', 'options'),
            Input('wind-turbine-source', 'value')
        )
        def update_turbine_options(source):
            return self.wind_config.TURBINE_SOURCES.get(source, [])

        # Update configuration when the submit button is clicked
        @self.app.callback(
            Output("output-config", "children"),
            Input("submit-button", "n_clicks"),
            State("solar-max-capacity", "value"),
            State("solar-landuse-intensity", "value"),
            State("solar-atlite-panel", "value"),
            State("solar-tracking", "value"),
            State("wind-max-capacity", "value"),
            State("wind-landuse-intensity", "value"),
            State("wind-turbine-source", "value"),
            State("wind-turbine-model", "options"),
            State("wind-CF-low", "value"),
            State("wind-CF-high", "value"),
            State("bess-max-capacity", "value"),
            State("bess-storage-duration", "value"),
            State("transmission-grid-cost", "value"),
            State("transmission-rebuild-cost", "value"),
            State("transmission-proximity-filter", "value")
        )
        def update_config(n_clicks,
                          solar_max_cap, solar_land_intensity, solar_panel, solar_tracking,
                          wind_max_cap, wind_land_intensity, wind_turbine_source, wind_turbine_model,
                          wind_CF_low, wind_CF_high,
                          bess_max_cap, bess_storage_duration,
                          tx_grid_cost, tx_rebuild_cost, tx_proximity_filter):
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
                        "wind-turbine-source": wind_turbine_source,
                        "wind-turbine-model": wind_turbine_model,
                        "capacity_factor_low": wind_CF_low,
                        "capacity_factor_high": wind_CF_high
                    },
                    "BESS": {
                        "max_capacity": bess_max_cap,
                        "storage_discharge_duration": bess_storage_duration
                    },
                    "transmission": {
                        "grid_connection_cost": tx_grid_cost,
                        "line_rebuild_cost": tx_rebuild_cost,
                        "proximity_filter": tx_proximity_filter
                    }
                }
            }

            # Save YAML
            with open("config.yaml", "w") as f:
                yaml.dump(config_dict, f)

            return "Configuration saved to config.yaml."

    def run(self):
        self.app.run_server(debug=True)

# Run the app
if __name__ == "__main__":
    app_instance = CapacityDisaggregationApp()
    app_instance.run()

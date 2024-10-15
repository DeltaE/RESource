import pandas as pd

class CellScorer:
    def __init__(self):
        pass  # No need to initialize with constants since values are in the DataFrame

    def calculate_total_cost(self, 
                             distance_to_grid_km: float, 
                             grid_connection_cost_per_km: float, 
                             tx_line_rebuild_cost: float, 
                             capex_tech: float) -> float:
        """
        Calculate the total cost, which includes the CAPEX and distance-based grid connection costs.
        """
        # Calculate distance-based cost
        add_to_grid_cost = (distance_to_grid_km * grid_connection_cost_per_km / 1.60934) * (tx_line_rebuild_cost / 1.60934)  # Convert to miles as our costs are given in $/miles (USA study)
        
        # Total cost is CAPEX plus distance cost
        total_cost = capex_tech + add_to_grid_cost  # in M$
        
        return total_cost

    def get_cell_score(self, 
                             cells: pd.DataFrame, 
                             CF_column: str = 'CF_mean') -> pd.DataFrame:
            """
            Calculate the potential LCOE score for each cell in the dataframe,
            reading cost parameters directly from the DataFrame.
            """
            dataframe = cells.copy()  # Use the input DataFrame for calculations
            print(">> Calculating Score for each Cell...")

            dataframe['lcoe'] = dataframe.apply(
                lambda x: self.calculate_total_cost(
                    x['nearest_station_distance_km'], 
                    x['grid_connection_cost_per_km'], 
                    x['tx_line_rebuild_cost'], 
                    x['capex']
                ) / (8760 * x[CF_column]),  # LCOE = Total Cost / Total Energy Produced
                axis=1 # LCOE in M$/MWh;
            )  
            dataframe['lcoe']=dataframe['lcoe']*1E3 # LCOE in $/kWh; lower lcoe indicates better cells

            scored_dataframe = dataframe.sort_values(by='lcoe', ascending=False).copy()  # Lower LCOE is better
            
            return scored_dataframe
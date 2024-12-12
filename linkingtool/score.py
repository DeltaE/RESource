import pandas as pd
from linkingtool.AttributesParser import AttributesParser
from dataclasses import dataclass

@dataclass
class CellScorer(AttributesParser):
    
    def __post_init__(self):
        super().__post_init__()
        
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
                        CF_column:str) -> pd.DataFrame:
            """
            Calculate the potential LCOE score for each cell in the dataframe,
            reading cost parameters directly from the DataFrame.
            
            ## Args:
            - **Cells** : Pandas DataFrame of grid cells.
            - **CF_column**: Column name from which Capacity Factor (CF) will be used to calculated Annual. Avg. Energy (MWh). User can have multiple CF_mean columns sourced from different data sources.
           
            """
            dataframe = cells.copy()  # Use the input DataFrame for calculations
            print(">> Calculating Score for each Cell...")

            dataframe[f'lcoe_{self.resource_type}'] = dataframe.apply(
                lambda x: self.calculate_total_cost(
                    x['nearest_station_distance_km'], 
                    x[f'grid_connection_cost_per_km_{self.resource_type}'], 
                    x[f'tx_line_rebuild_cost_{self.resource_type}'], 
                    x[f'capex_{self.resource_type}']
                ) / (8760 * x[CF_column]),  # LCOE = Total Cost / Total Energy Produced
                axis=1 # LCOE in M$/MWh;
            )  
            dataframe[f'lcoe_{self.resource_type}']=dataframe[f'lcoe_{self.resource_type}']*1E3 # LCOE in $/kWh; lower lcoe indicates better cells

            dataframe[f'LCOE_{self.resource_type}'] = dataframe.apply(lambda row: self.calc_LCOE_lambda(row), axis=1) # adopting NREL's method + some added costs
            
            # scored_dataframe = dataframe.sort_values(by=f'lcoe_{self.resource_type}', ascending=False).copy()  # Lower LCOE is better
            scored_dataframe = dataframe.sort_values(by=f'LCOE_{self.resource_type}', ascending=False).copy()  # Lower LCOE is better
            
            return scored_dataframe

    def calc_LCOE_lambda(self,
                         row):
        
        """ 
        # Method: 
        LCOE = (FCR * TCC + FOC + GCC + TRC) / AEP + VOC)
            - Total Capital cost, $ (TCC)
            - Fixed annual operating cost, $ (FOC)
            - Variable operating cost, $/kWh (VOC)
            - Fixed charge rate (FCR)
            - Annual electricity production, kWh (AEP)
            - Grid Connection Cost (GCC)
            - Transmission Line Rebuild Cost (TRC) 
            
        ### Ref: 
        - https://sam.nrel.gov/financial-models/lcoe-calculator.html
        - https://www.nrel.gov/docs/legosti/old/5173.pdf
        - https://www.nrel.gov/docs/fy07osti/40566.pdf
        
        """

        dtg = row['nearest_station_distance_km'] # km
        gcc_pu = row[f'grid_connection_cost_per_km_{self.resource_type}'] # m$/km
        gcc=dtg*gcc_pu/1.60934  # Convert to miles as our costs are given in m$/miles (USA study)
        trc=row[f'tx_line_rebuild_cost_{self.resource_type}']/ 1.60934 # m$/km
        tcc = row[f'capex_{self.resource_type}'] # m$/km
        foc = row[f'fom_{self.resource_type}'] * row[f'potential_capacity_{self.resource_type}'] # $/ MW * MW
        voc = row[f'vom_{self.resource_type}'] * row[f'potential_capacity_{self.resource_type}'] # $/ MW * MW
        fcr = row.get('FCR', 0.098) 
        aep = 8760 * row[f'{self.resource_type}_CF_mean'] * row[f'potential_capacity_{self.resource_type}'] # MWh
        
        if aep == 0: # some cells have no potentials
            return float(99999)  # handle the error 
        else:
            lcoe = ((fcr * tcc + gcc + trc + foc) / aep + voc)  # m$/MWh
            return lcoe  * 1E6 # LCOE in $/MWh      
        

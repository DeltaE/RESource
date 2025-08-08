import pandas as pd
from RES.AttributesParser import AttributesParser
from dataclasses import dataclass
import RES.utility as utils
print_level_base=2

@dataclass
class CellScorer(AttributesParser):
    """
    Economic evaluation and scoring system for renewable energy grid cells.
    
    This class implements Levelized Cost of Energy (LCOE) calculations to economically
    rank and score potential renewable energy sites. It integrates capital costs,
    operational expenses, grid connection costs, and capacity factors to provide
    comprehensive economic metrics for site comparison and selection.
    
    The scoring methodology follows NREL LCOE documentation and incorporates
    distance-based grid connection costs, technology-specific capital expenditures,
    and site-specific capacity factors to generate comparable economic indicators
    across different locations and technologies.
    
    Parameters
    ----------
    config_file_path : str or Path
        Configuration file containing economic parameters and assumptions
    region_short_code : str
        Region identifier for localized cost parameters
    resource_type : {'solar', 'wind'}
        Technology type for appropriate cost parameter selection
        
    Attributes
    ----------
    Inherits configuration parsing capabilities from AttributesParser
    
    Methods
    -------
    get_CRF(r, N)
        Calculate Capital Recovery Factor for annualized cost calculations
    calculate_total_cost(distance_to_grid_km, grid_connection_cost_per_km, 
                        tx_line_rebuild_cost, capex_tech, potential_capacity_mw)
        Compute total project costs including CAPEX and grid connection
    calculate_score(row, CF_column, CRF)
        Generate LCOE score for individual grid cells
    score_cells(cells_with_cf)
        Apply economic scoring to entire dataset of grid cells
    get_economic_rankings(scored_cells, top_percentile=10)
        Rank and filter cells by economic attractiveness
        
    Examples
    --------
    Basic economic scoring workflow:
    
    >>> from RES.score import CellScorer  
    >>> scorer = CellScorer(
    ...     config_file_path="config/config_BC.yaml",
    ...     region_short_code="BC",
    ...     resource_type="wind"
    ... )
    >>> scored_cells = scorer.score_cells(cells_with_capacity_factors)
    >>> top_sites = scorer.get_economic_rankings(scored_cells, top_percentile=15)
    
    Custom economic analysis:
    
    >>> # Calculate CRF for different financial scenarios
    >>> crf_conservative = scorer.get_CRF(r=0.08, N=25)  # 8% discount, 25 year life
    >>> crf_aggressive = scorer.get_CRF(r=0.06, N=30)    # 6% discount, 30 year life
    >>> 
    >>> # Apply scoring with custom parameters
    >>> for idx, row in cells.iterrows():
    ...     lcoe = scorer.calculate_score(row, 'CF_mean', crf_conservative)
    
    Notes
    -----
    LCOE Calculation Methodology:
    - Follows NREL Simple LCOE calculation framework
    - LCOE = (CAPEX × CRF + OPEX) / Annual Energy Production
    - Includes distance-based grid connection costs
    - Uses technology-specific cost parameters from configuration
    
    Cost Components:
    - Technology CAPEX ($/MW installed capacity)
    - Grid connection costs ($/km distance to transmission)
    - Transmission line rebuild costs ($/km)
    - Annual O&M expenses (% of CAPEX)
    - Financial parameters (discount rate, project lifetime)
    
    Economic Parameters:
    - Capital Recovery Factor (CRF) for cost annualization
    - Technology-specific cost assumptions
    - Regional cost multipliers and adjustments
    - Grid connection distance penalties
    
    Limitations:
    - Simplified LCOE model without detailed financial modeling
    - Grid connection costs based on straight-line distances
    - Does not account for economies of scale in large projects
    - Static cost assumptions without temporal price variations
    """
    
    def __post_init__(self):

        super().__post_init__()
        
    def get_CRF(self,
                r, 
                N):
        """
        Calculate Capital Recovery Factor (CRF) for annualized cost calculations.
        
        The CRF converts a present-value capital cost into a stream of equal
        annual payments over the project lifetime. This is essential for LCOE
        calculations as it allows comparison of projects with different capital
        costs and lifetimes on an annualized basis.
        
        Formula: CRF = [r × (1 + r)^N] / [(1 + r)^N - 1]
        
        Args:
            r (float): Discount rate (as decimal, e.g., 0.08 for 8%)
            N (int): Project lifetime in years
            
        Returns:
            float: Capital Recovery Factor
            
        Example:
            >>> scorer = CellScorer(**config)
            >>> crf = scorer.get_CRF(r=0.07, N=25)  # 7% discount, 25 years
            >>> print(f"CRF: {crf:.4f}")
            CRF: 0.0858
        """
        return (r * (1 + r) ** N) / ((1 + r) ** N - 1) if N > 0 else 0
        
    def calculate_total_cost(self, 
                             distance_to_grid_km: float, 
                             grid_connection_cost_per_km: float, 
                             tx_line_rebuild_cost: float, 
                             capex_tech: float,
                             potential_capacity_mw:float) -> float:
        """
        Calculate total project cost including CAPEX and distance-based grid connection costs.
        
        This method implements the cost calculation framework following NREL's Simple 
        Levelized Cost of Energy methodology. It combines technology capital expenditures
        with grid integration costs that scale with distance to existing transmission
        infrastructure.
        
        Cost Components:
        1. Technology CAPEX: Base capital cost for renewable energy installation
        2. Grid Connection Cost: Cost to connect to nearest transmission point
        3. Transmission Rebuild Cost: Infrastructure upgrade costs for grid integration
        
        Args:
            distance_to_grid_km (float): Distance to nearest grid connection point (km)
            grid_connection_cost_per_km (float): Cost per km for grid connection (M$/km)
            tx_line_rebuild_cost (float): Transmission line rebuild cost (M$/km)
            capex_tech (float): Technology-specific capital expenditure (M$/MW)
            potential_capacity_mw (float): Potential installed capacity (MW)
            
        Returns:
            float: Total project cost in millions of dollars (M$)
            
        Notes:
            - Converts km to miles for cost calculations (US-based cost data)
            - Grid connection costs scale linearly with distance
            - Method follows NREL Simple LCOE documentation framework
            
        Reference:
            https://www.nrel.gov/analysis/tech-lcoe-documentation.html
        """
        # Calculate distance-based cost
        add_to_grid_cost = (distance_to_grid_km * grid_connection_cost_per_km / 1.60934) * (tx_line_rebuild_cost / 1.60934)  # Convert to miles as our costs are given in $/miles (USA study)

        # Total cost is CAPEX plus distance cost
        total_cost = capex_tech*potential_capacity_mw + add_to_grid_cost  # in M$
        
        return total_cost
    
    def calculate_score(self,
                        row,
                        CF_column,
                        CRF) -> float:
        """
        Calculate the Levelized Cost of Energy (LCOE) score for an individual grid cell.
        
        This method computes the LCOE using the simplified NREL methodology, combining
        annualized capital costs with operational expenses and dividing by annual energy
        production. The resulting LCOE provides a standardized metric for comparing
        the economic attractiveness of different renewable energy sites.
        
        LCOE Formula:
        LCOE = (Total Cost × CRF + OPEX) / Annual Energy Production
        
        Where:
        - Total Cost includes CAPEX and grid connection costs
        - CRF converts capital costs to annual payments
        - Annual Energy = 8760 hours × Capacity Factor × Installed Capacity
        
        Args:
            row (pd.Series): DataFrame row containing cell-specific data including:
                - nearest_station_distance_km: Distance to grid connection
                - grid_connection_cost_per_km_{resource_type}: Grid connection cost
                - tx_line_rebuild_cost_{resource_type}: Transmission rebuild cost
                - capex_{resource_type}: Technology capital expenditure
                - potential_capacity_{resource_type}: Installable capacity
                - CF_column value: Capacity factor for the cell
            CF_column (str): Column name containing capacity factor data
            CRF (float): Capital Recovery Factor for cost annualization
            
        Returns:
            float: LCOE in $/MWh, or 999 if annual energy production is zero
            
        Examples:
            >>> # Calculate LCOE for a single grid cell
            >>> crf = scorer.get_CRF(r=0.07, N=25)
            >>> lcoe = scorer.calculate_score(grid_cell, 'CF_mean', crf)
            >>> print(f"LCOE: ${lcoe:.2f}/MWh")
            
        Notes:
            - Returns high penalty value (999) for cells with zero energy production
            - Converts from M$/MWh to $/MWh for standard reporting units
            - Incorporates site-specific capacity factors and grid integration costs
        """
        # Calculate the total cost
        total_cost = self.calculate_total_cost(
            row['nearest_station_distance_km'],  # km
            row[f'grid_connection_cost_per_km_{self.resource_type}'],  # m$/km
            row[f'tx_line_rebuild_cost_{self.resource_type}'],  # m$/km
            row[f'capex_{self.resource_type}'],
            row[f'potential_capacity_{self.resource_type}']# MW
        ) # mW
        
        annual_energy = 8760  * row[CF_column] * row[f'potential_capacity_{self.resource_type}']# Total energy produced in a year
        if annual_energy == 0: # some cells have no potentials
            return float('999')  # handle the error 
        else:
            # Calculate the LCOE

            lcoe = (total_cost * CRF / annual_energy)  # Avoid division by zero,  m$/MWh
        
            return lcoe * 1E6 # m$/MWh → $/MWh
        
    def get_cell_score(self, 
                    cells: pd.DataFrame,
                    CF_column:str,
                    interest_rate=0.03) -> pd.DataFrame:
        """
        Calculate LCOE scores for all grid cells in a DataFrame and return ranked results.
        
        This method applies economic scoring to an entire dataset of potential renewable
        energy sites, calculating LCOE for each cell and sorting results by economic
        attractiveness. It serves as the primary interface for batch economic analysis
        of renewable energy development opportunities.
        
        Processing Steps:
        1. Calculate Capital Recovery Factor from financial parameters
        2. Apply LCOE calculation to each grid cell
        3. Sort results by LCOE (ascending = most economically attractive first)
        4. Return scored and ranked DataFrame
        
        Args:
            cells (pd.DataFrame): DataFrame containing grid cells with required columns:
                - nearest_station_distance_km: Distance to transmission (km)
                - grid_connection_cost_per_km_{resource_type}: Connection cost (M$/km)
                - tx_line_rebuild_cost_{resource_type}: Rebuild cost (M$/km)
                - capex_{resource_type}: Technology CAPEX (M$/MW)
                - potential_capacity_{resource_type}: Installable capacity (MW)
                - Operational_life_{resource_type}: Project lifetime (years)
            CF_column (str): Column name containing capacity factor data
                          (e.g., 'CF_mean', 'wind_CF_mean', 'solar_CF_mean')
            interest_rate (float, optional): Discount rate for CRF calculation.
                                           Defaults to 0.03 (3%)
                                           
        Returns:
            pd.DataFrame: Input DataFrame with added LCOE column, sorted by economic
                         attractiveness (lowest LCOE first). Column name format:
                         'lcoe_{resource_type}' with values in $/MWh
                         
        Raises:
            KeyError: If required columns are missing from input DataFrame
            ValueError: If capacity factors or operational life contain invalid values
            
        Examples:
            >>> # Score wind energy sites using mean capacity factors
            >>> wind_cells = scorer.get_cell_score(
            ...     cells=grid_data, 
            ...     CF_column='wind_CF_mean',
            ...     interest_rate=0.07
            ... )
            >>> print(f"Best site LCOE: ${wind_cells.iloc[0]['lcoe_wind']:.2f}/MWh")
            
            >>> # Score solar sites with conservative financial assumptions  
            >>> solar_cells = scorer.get_cell_score(
            ...     cells=grid_data,
            ...     CF_column='solar_CF_mean', 
            ...     interest_rate=0.08
            ... )
            
        Notes:
            - Cells with zero annual energy production receive infinite LCOE values
            - Results are sorted ascending (lowest LCOE = most attractive)
            - Method handles edge cases like zero capacity factors gracefully
            - LCOE values are in $/MWh for standard industry comparison
        """
        dataframe = cells.copy()  # Use the input DataFrame for calculations
        utils.print_update(level=print_level_base+2,
                       message=f"{__name__}| Calculating score for cells...") 
        
        # Calculate the LCOE for each cell
        N=cells[f'Operational_life_{self.resource_type}'].iloc[0]
        CRF=self.get_CRF(interest_rate,N)
        dataframe[f'lcoe_{self.resource_type}'] = dataframe.apply(
            lambda x: self.calculate_total_cost(
                x['nearest_station_distance_km'], # km
                x[f'grid_connection_cost_per_km_{self.resource_type}'],  # m$/km
                x[f'tx_line_rebuild_cost_{self.resource_type}'],  # m$/km
                x[f'capex_{self.resource_type}'],
                x[f'potential_capacity_{self.resource_type}']# m$
            )*CRF / (8760 * x[CF_column]* x[f'potential_capacity_{self.resource_type}']) if (8760 * x[CF_column]) != 0 else float('inf'),  # LCOE = Total Cost / Total Energy Produced
            axis=1 # LCOE in M$/MWh;
        )  
        
        # dataframe[f'lcoe_{self.resource_type}']=dataframe[f'lcoe_{self.resource_type}']*1E3 # LCOE in $/kWh; lower lcoe indicates better cells
        dataframe[f'lcoe_{self.resource_type}'] = dataframe.apply(lambda row: self.calculate_score(row,CF_column,CRF), axis=1) # LCOE in $/MWh  # adopting NREL's method + some added costs
        scored_dataframe = dataframe.sort_values(by=f'lcoe_{self.resource_type}', ascending=False).copy()  # Lower LCOE is better
        
        # dataframe[f'LCOE_{self.resource_type}'] = dataframe.apply(lambda row: self.calc_LCOE_lambda_m2(row), axis=1) # LCOE in $/MWh  # adopting NREL's method + some added costs
        # scored_dataframe = dataframe.sort_values(by=f'LCOE_{self.resource_type}', ascending=False).copy()  # Lower LCOE is better
        
        return scored_dataframe

    def calc_LCOE_lambda_m1(self,
                         row):
        
        """ 
        # Method: 
        LCOE = [(FCR x TCC + FOC + GCC + TRC) / AEP + VOC)
            - Total Capital cost, $ (TCC)
            - Fixed annual operating cost, $ (FOC)
            - Variable operating cost, $/kWh (VOC)
            - Fixed charge rate (FCR)
            - Annual electricity production, kWh (AEP)
            - Grid Connection Cost (GCC)
            - Transmission Line Rebuild Cost (TRC) 
            
        ### Ref: 
        - https://atb.nrel.gov/electricity/2024/equations_&_variables
        - https://sam.nrel.gov/financial-models/lcoe-calculator.html
        - https://www.nrel.gov/docs/legosti/old/5173.pdf
        - https://www.nrel.gov/docs/fy07osti/40566.pdf
        
        """

        dtg = row['nearest_station_distance_km'] # km
        gcc_pu = row[f'grid_connection_cost_per_km_{self.resource_type}'] # m$/km
        gcc=dtg*gcc_pu/1.60934  # Convert to miles as our costs are given in m$/miles (USA study)
        trc=row[f'tx_line_rebuild_cost_{self.resource_type}']/ 1.60934 # m$/km
        tcc = row[f'capex_{self.resource_type}'] # m$/km
        
        foc = row[f'fom_{self.resource_type}'] * row[f'potential_capacity_{self.resource_type}'] # m$/ MW * MW
        voc = row[f'vom_{self.resource_type}'] * row[f'potential_capacity_{self.resource_type}'] # m$/ MW * MW
        
        fcr = row.get('FCR', 0.098) 
        aep = 8760 * row[f'{self.resource_type}_CF_mean'] * row[f'potential_capacity_{self.resource_type}'] # MWh
        
        if aep == 0: # some cells have no potentials
            return float(99999)  # handle the error 
        else:
            lcoe = ((fcr * tcc + gcc + trc + foc) / aep + voc)  # m$/MWh
            return lcoe  * 1E6 # LCOE in $/MWh      
        
    """
    'Fixed O&M', 'CFC', 'LCOE', 'CAPEX', 'CF', 'OCC', 'GCC',
       'Variable O&M', 'Heat Rate', 'Fuel', 'Additional OCC',
       'Heat Rate Penalty', 'Net Output Penalty', 'FCR', 'Inflation Rate',
       'Interest Rate Nominal', 'Rate of Return on Equity Nominal',
       'Calculated Interest Rate Real',
       'Interest During Construction - Nominal',
       'Calculated Rate of Return on Equity Real', 'Debt Fraction',
       'Tax Rate (Federal and State)', 'WACC Nominal', 'WACC Real', 'CRF'
      """
    def calc_LCOE_lambda_m2(self,
                         row):
        
        """ 
        # Method: 
        LCOE = [{(FCR x CAPEX) + FOM ) /  (CF x 8760) } + VOM + Fuel - PTC)
            
            - Fixed charge rate (FCR) :
                - Amount of revenue per dollar of investment required that must be collected annually from customers to pay the carrying charges on that investment.
            - CAPEX (m$)
                - expenditures required to achieve commercial operation of the generation plant.
                - ConFinFactor x (OCC +GCC)
                    - ConFinFactor: Conversion Factor for Capital Recovery. 
                        - The portion of all-in capital cost associated with construction period financing; 
                        - ConFinFactor = Σ(y=0 t0 C-1) FC-y x AI_y
                        - assumed to be 1.0 for simplification
                    - OCC ($/kW) : Overnight Capital Cost CAPEX if plant could be constructed overnight (i.e., excludes construction period financing); includes on-site electrical equipment (e.g., switchyard), a nominal-distance spur line (<1 mi), and necessary upgrades at a transmission substation. ($/kW)
                    - GCC ($/kW): Grid Connection Cost 
            - CF: Capacity Factor
            - Variable operation and maintenance (VOM), $/MWh (VOC)
            - Fuel: 
                - Fuel costs, converted to $/MWh, using heat rates.
                - Heat rate (MMBtu/MWh) * Fuel Costs($/MMBtu)
                - Zero for VREs
            - PTC ($/MWh) :  Production Tax Credit
                - a before-tax credit that reduces LCOE; credits are available for 10 years, so it must be adjusted for a 10-year CRF relative to the full CRF of the project. 
                - This formulation of the PTC accounts for a pre-tax LCOE and aligns with the equation used for the ProFinFactor.
                - PTC= {PTC_full/(1-TR)}x(CRF/CRF_10yrs)
                    - TR: Tax Rate
                    - CRF: Capital Recovery Factor
                        - ratio of a constant annuity to the present value of receiving that annuity for a given length of time
                        - CRF = WACC x[1/(1-(1+WACC)^-t)]
                            - WACC: Weighted Average Cost of Capital
                                - average expected rate that is paid to finance assets
                                - WACC = [ 1+ [1-DF] x [(1+RROE)(1=i)-1] + DF x [(1+IR)(1+i)-1] x [1-TR] ]/(1+i) -1

            
        ### Ref: 
        - https://atb.nrel.gov/electricity/2024/equations_&_variables
        
        """

        dtg = row['nearest_station_distance_km'] # km
        
        gcc_pu = row[f'grid_connection_cost_per_km_{self.resource_type}'] # m$/km
        gcc=dtg*gcc_pu/1.60934  # Convert to miles as our costs are given in m$/miles (USA study)
        
        trc=dtg*row[f'tx_line_rebuild_cost_{self.resource_type}']/ 1.60934 # m$=m$/km*km
        
        tcc = row[f'capex_{self.resource_type}'] * row[f'potential_capacity_{self.resource_type}'] # m$=m$/MW * MW
        
        foc = row[f'fom_{self.resource_type}'] * row[f'potential_capacity_{self.resource_type}'] # m$/ MW * MW
        voc = row[f'vom_{self.resource_type}']  # m$/ MW 
        
        fcr = row.get('FCR', 0.098) 
        aep = 8760 * row[f'{self.resource_type}_CF_mean'] * row[f'potential_capacity_{self.resource_type}'] # MWh
        
        if aep == 0: # some cells have no potentials
            return float('99999')  # handle the error 
        else:
            lcoe = ((fcr * (tcc + gcc + trc ) + foc) / aep + voc)  # m$/MWh
            return lcoe  * 1E6 # LCOE in $/MWh      
        
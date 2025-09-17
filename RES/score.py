import pandas as pd
from RES.AttributesParser import AttributesParser
from dataclasses import dataclass
import RES.utility as utils
PRINT_LEVEL_BASE:int=2

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
    get_cell_score(cells, CF_column, interest_rate=0.03)
        Apply economic scoring to entire dataset of grid cells
    calc_LCOE_lambda_m1(row)
        Alternative LCOE calculation method following NREL methodology
    calc_LCOE_lambda_m2(row)
        Enhanced LCOE calculation with detailed cost components
        
    Examples
    --------
    Basic economic scoring workflow:
    
    >>> from RES.score import CellScorer  
    >>> scorer = CellScorer(
    ...     config_file_path="config/config_CAN.yaml",
    ...     region_short_code="BC",
    ...     resource_type="wind"
    ... )
    >>> scored_cells = scorer.get_cell_score(cells_with_capacity_factors, 'CF_mean')
    >>> # Get top 10% of cells by LCOE
    >>> top_sites = scored_cells.head(int(len(scored_cells) * 0.1))
    
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
                r: float, 
                N: int)-> float:
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
        crf:float=(r * (1 + r) ** N) / ((1 + r) ** N - 1) if N > 0 else 0
        return crf
    
    def calculate_total_cost(self, 
                         distance_to_grid_km: float, 
                         grid_connection_cost_per_km: float, 
                         tx_line_rebuild_cost: float, 
                         capex_tech: float,
                         potential_capacity_mw: float) -> float:
        """
        Calculate total project cost with economies of scale for grid connection.
        """
        # Technology CAPEX scales linearly with capacity
        tech_capex = capex_tech * potential_capacity_mw
        
        # Grid connection with economies of scale
        base_grid_cost = (distance_to_grid_km * grid_connection_cost_per_km) + tx_line_rebuild_cost
        
        # Method 1: Simple scaling factor based on capacity tiers
        if potential_capacity_mw <= 50:  # Small projects
            grid_scaling_factor = 1.0
        elif potential_capacity_mw <= 200:  # Medium projects
            grid_scaling_factor = 0.85  # 15% reduction
        elif potential_capacity_mw <= 500:  # Large projects
            grid_scaling_factor = 0.70  # 30% reduction
        else:  # Very large projects (>500MW)
            grid_scaling_factor = 0.60  # 40% reduction
        
        grid_connection_cost = base_grid_cost * grid_scaling_factor
        
        total_cost = tech_capex + grid_connection_cost
        return total_cost


    def calculate_total_cost_smooth_scaling(self, 
                                        distance_to_grid_km: float, 
                                        grid_connection_cost_per_km: float, 
                                        tx_line_rebuild_cost: float, 
                                        capex_tech: float,
                                        potential_capacity_MW: float,
                                        reference_capacity_MW:float=100,
                                        scaling_exponent:float=0.8) -> float:
        """
        Calculate total cost with smooth economies of scale for grid connection.
        
        Args:
            distance_to_grid_km (float): Distance to nearest grid connection point (km)
            grid_connection_cost_per_km (float): Cost per km for grid connection (M$/km)
            tx_line_rebuild_cost (float): Transmission line rebuild cost (M$/km)
            capex_tech (float): Technology-specific capital expenditure (M$/MW)
            potential_capacity_MW (float): Potential installed capacity (MW)
            reference_capacity_MW (float, optional): Reference capacity for scaling. Defaults to 100 MW.
            scaling_exponent (float, optional): Exponent for scaling economies e.g. <1 means economies of scale. Defaults to 0.8.
        Returns:
            float: Total project cost in millions of dollars (M$)
        """
        # Technology CAPEX scales linearly
        tech_capex = capex_tech * potential_capacity_MW
        
        # Base grid connection cost
        base_grid_cost = (distance_to_grid_km * grid_connection_cost_per_km) + tx_line_rebuild_cost
        
        # Method 2: Smooth scaling using power law or logarithmic function
        # Power law scaling: cost per MW decreases as capacity increases

        
        if potential_capacity_MW <= reference_capacity_MW:
            capacity_scaling = 1.0
        else:
            capacity_scaling = (potential_capacity_MW / reference_capacity_MW) ** scaling_exponent
        
        # Apply scaling to grid costs
        grid_connection_cost = base_grid_cost * capacity_scaling
        
        total_cost = tech_capex + grid_connection_cost
        return total_cost


    def calculate_total_cost_transmission_sizing(self, 
                                            distance_to_grid_km: float, 
                                            grid_connection_cost_per_km: float, 
                                            tx_line_rebuild_cost: float, 
                                            capex_tech: float,
                                            potential_capacity_mw: float) -> float:
        """
        Calculate total cost with transmission line sizing based on capacity.
        More realistic approach considering actual transmission requirements.
        """
        # Technology CAPEX scales linearly
        tech_capex = capex_tech * potential_capacity_mw
        
        # Method 3: Transmission line sizing approach
        # Different voltage levels have different cost structures
        
        if potential_capacity_mw <= 50:  # Distribution voltage (25-35kV)
            voltage_multiplier = 1.0
            line_capacity_utilization = potential_capacity_mw / 50  # Utilization of line capacity
        elif potential_capacity_mw <= 200:  # Sub-transmission (69-138kV) 
            voltage_multiplier = 1.2  # Higher voltage = higher initial cost
            line_capacity_utilization = potential_capacity_mw / 200
        elif potential_capacity_mw <= 500:  # Transmission (230-345kV)
            voltage_multiplier = 1.5
            line_capacity_utilization = potential_capacity_mw / 500
        else:  # High voltage transmission (500kV+)
            voltage_multiplier = 2.0
            line_capacity_utilization = min(1.0, potential_capacity_mw / 1000)
        
        # Base cost adjusted for voltage level
        base_cost_adjusted = (distance_to_grid_km * grid_connection_cost_per_km * voltage_multiplier) + tx_line_rebuild_cost
        
        # Economies of scale: cost per MW decreases as utilization increases
        utilization_efficiency = 0.3 + 0.7 * line_capacity_utilization  # 30% fixed + 70% variable
        
        grid_connection_cost = base_cost_adjusted * utilization_efficiency
        
        total_cost = tech_capex + grid_connection_cost
        return total_cost


    def calculate_total_cost_shared_infrastructure(self, 
                                                distance_to_grid_km: float, 
                                                grid_connection_cost_per_km: float, 
                                                tx_line_rebuild_cost: float, 
                                                capex_tech: float,
                                                potential_capacity_mw: float,
                                                nearby_projects_mw: float = 0) -> float:
        """
        Calculate total cost considering potential for shared transmission infrastructure.
        This is most relevant for clustering applications.
        """
        # Technology CAPEX scales linearly
        tech_capex = capex_tech * potential_capacity_mw
        
        # Method 4: Shared infrastructure approach
        total_capacity = potential_capacity_mw + nearby_projects_mw
        
        # Base transmission cost
        base_grid_cost = (distance_to_grid_km * grid_connection_cost_per_km) + tx_line_rebuild_cost
        
        if nearby_projects_mw > 0:
            # Cost sharing: project pays proportional share of larger transmission line
            capacity_share = potential_capacity_mw / total_capacity
            
            # Economies of scale for the larger combined transmission
            if total_capacity <= 100:
                total_scaling_factor = 1.0
            elif total_capacity <= 300:
                total_scaling_factor = 0.8
            elif total_capacity <= 600:
                total_scaling_factor = 0.65
            else:
                total_scaling_factor = 0.5
            
            # This project's share of the optimized transmission cost
            grid_connection_cost = base_grid_cost * total_scaling_factor * capacity_share
            
        else:
            # No shared infrastructure - use individual project scaling
            if potential_capacity_mw <= 50:
                grid_scaling_factor = 1.0
            elif potential_capacity_mw <= 200:
                grid_scaling_factor = 0.85
            else:
                grid_scaling_factor = 0.7
                
            grid_connection_cost = base_grid_cost * grid_scaling_factor
        
        total_cost = tech_capex + grid_connection_cost
        return total_cost
    
    # def calculate_total_cost(self, 
    #                      distance_to_grid_km: float, 
    #                      grid_connection_cost_per_km: float, 
    #                      tx_line_rebuild_cost: float, 
    #                      capex_tech: float,
    #                      potential_capacity_mw: float) -> float:
    #     """
    #     Calculate total project cost including CAPEX and distance-based grid connection costs.
        
    #     Args:
    #         distance_to_grid_km (float): Distance to nearest grid connection point (km)
    #         grid_connection_cost_per_km (float): Cost per km for grid connection (M$/km)
    #         tx_line_rebuild_cost (float): Transmission line rebuild cost (M$/km)
    #         capex_tech (float): Technology-specific capital expenditure (M$/MW)
    #         potential_capacity_mw (float): Potential installed capacity (MW)
            
    #     Returns:
    #         float: Total project cost in millions of dollars (M$)
    #     """
        
    #     # Technology CAPEX scales with capacity
    #     tech_capex = capex_tech * potential_capacity_mw
        
    #     # Grid connection costs - could scale with project size for larger projects
    #     # Option 1: Fixed connection cost (current approach)
    #     grid_connection_cost = (distance_to_grid_km * grid_connection_cost_per_km) + tx_line_rebuild_cost
        
    #     # Option 2: Scale grid costs with capacity for very large projects
    #     # capacity_factor = min(1.0, potential_capacity_mw / 100.0)  # Scale factor for projects >100MW
    #     # grid_connection_cost = ((distance_to_grid_km * grid_connection_cost_per_km) + tx_line_rebuild_cost) * capacity_factor
        
    #     total_cost = tech_capex + grid_connection_cost
        
    #     return total_cost
    
    def calculate_score(self,
                    row: pd.Series,
                    node_distance_col: str,
                    CF_column: str,
                    CRF: float) -> float:
        """
        Calculate the Levelized Cost of Energy (LCOE) score for an individual grid cell.
        
        LCOE Formula:
        LCOE = (CAPEX × CRF + FOM + VOM × Annual_Energy) / Annual_Energy
        
        Args:
            row (pd.Series): DataFrame row containing cell-specific data
            node_distance_col (str): Column name for distance to grid connection
            CF_column (str): Column name containing capacity factor data
            CRF (float): Capital Recovery Factor for cost annualization
            
        Returns:
            float: LCOE in $/MWh, or 999999 if annual energy production is zero
        """
        
        # Get capacity and annual energy
        capacity = row[f'potential_capacity_{self.resource_type}']  # MW
        capacity_factor = row[CF_column]
        annual_energy = 8760 * capacity_factor * capacity  # MWh/year
        
        # Handle zero energy production
        if annual_energy == 0:
            return float('999999')
        
        # Calculate total capital cost
        total_capex = self.calculate_total_cost_smooth_scaling(
            row[node_distance_col],  # km
            row[f'grid_connection_cost_per_km_{self.resource_type}'],  # M$/km
            row[f'tx_line_rebuild_cost_{self.resource_type}'],  # M$/km
            row[f'capex_{self.resource_type}'],  # M$/MW
            capacity  # MW
        )  # Total in M$
        
        # Calculate O&M costs
        # Fixed O&M: typically in M$/MW/year
        fom_annual = row[f'fom_{self.resource_type}'] * capacity  # M$/year
        
        # Variable O&M: typically in M$/MWh
        vom_annual = row[f'vom_{self.resource_type}'] * annual_energy  # M$/year
        
        # Calculate LCOE in M$/MWh
        lcoe_millions = ((total_capex * CRF) + fom_annual + vom_annual) / annual_energy
        
        # Convert to $/MWh
        lcoe_dollars = lcoe_millions * 1E6
        
        return lcoe_dollars
    # Debug version to help identify the issue
    def calculate_score_debug(self,
                            row: pd.Series,
                            node_distance_col: str,
                            CF_column: str,
                            CRF: float) -> dict:
        """
        Debug version that returns breakdown of LCOE components.
        Use this to identify why larger sites get lower scores.
        """
        
        capacity = row[f'potential_capacity_{self.resource_type}']  # MW
        capacity_factor = row[CF_column]
        annual_energy = 8760 * capacity_factor * capacity  # MWh/year
        
        if annual_energy == 0:
            return {'lcoe': 999999, 'reason': 'zero_energy'}
        
        # CAPEX breakdown
        tech_capex = row[f'capex_{self.resource_type}'] * capacity  # M$
        grid_cost = (row[node_distance_col] * 
                    row[f'grid_connection_cost_per_km_{self.resource_type}'] + 
                    row[f'tx_line_rebuild_cost_{self.resource_type}'])  # M$
        total_capex = tech_capex + grid_cost
        
        # O&M costs
        fom_annual = row[f'fom_{self.resource_type}'] * capacity  # M$/year
        vom_annual = row[f'vom_{self.resource_type}'] * annual_energy  # M$/year
        
        # LCOE components
        capex_component = (total_capex * CRF) / annual_energy * 1E6  # $/MWh
        fom_component = fom_annual / annual_energy * 1E6  # $/MWh
        vom_component = vom_annual / annual_energy * 1E6  # $/MWh
        
        total_lcoe = capex_component + fom_component + vom_component
        
        return {
        'lcoe': total_lcoe,
        'capacity_mw': capacity,
        'annual_energy_mwh': annual_energy,
        'capex_component_per_mwh': capex_component,
        'fom_component_per_mwh': fom_component,
        'vom_component_per_mwh': vom_component,
        'tech_capex_m': tech_capex,
        'grid_cost_m': grid_cost,
        'capex_per_mw': row[f'capex_{self.resource_type}'],
        'distance_km': row[node_distance_col]
    }
        
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

        
        node_distance_col = utils.get_available_column(dataframe, ['nearest_station_distance_km', 'nearest_distance'])
        
        required_columns = [
            node_distance_col,
            f'grid_connection_cost_per_km_{self.resource_type}',
            f'tx_line_rebuild_cost_{self.resource_type}',
            f'capex_{self.resource_type}',
            f'potential_capacity_{self.resource_type}',
            f'Operational_life_{self.resource_type}',
            CF_column
        ]
        # Check that all required columns were found
        missing_columns = [col for col in required_columns if col is None or col not in dataframe.columns]
        if missing_columns:
            raise AssertionError("Missing required columns or alternatives not found")
        
        utils.print_update(level=PRINT_LEVEL_BASE+2,
                       message=f"{__name__}| Calculating score for cells...") 
        
        # Calculate the LCOE for each cell
        N:int=cells[f'Operational_life_{self.resource_type}'].iloc[0]
        CRF:float=self.get_CRF(interest_rate,N)
        
        # Calculate LCOE using the dedicated calculate_score method
        dataframe[f'lcoe_{self.resource_type}'] = dataframe.apply(
            lambda row: self.calculate_score_normalized(row, node_distance_col,CF_column, CRF), 
            axis=1
        )
        
        dataframe[f'lcoe_actualCap_{self.resource_type}'] = dataframe.apply(
            lambda row: self.calculate_score(row, node_distance_col,CF_column, CRF), 
            axis=1
        )
        
        scored_dataframe:pd.DataFrame = dataframe.sort_values(by=f'lcoe_actualCap_{self.resource_type}', ascending=True).copy()  # Lower LCOE is better (ascending=True)
        
        # dataframe[f'LCOE_{self.resource_type}'] = dataframe.apply(lambda row: self.calc_LCOE_lambda_m2(row), axis=1) # LCOE in $/MWh  # adopting NREL's method + some added costs
        # scored_dataframe = dataframe.sort_values(by=f'LCOE_{self.resource_type}', ascending=False).copy()  # Lower LCOE is better
        utils.print_update(level=PRINT_LEVEL_BASE+2,
               message=f"{__name__}|✓ Scores calculated and sorted for {self.resource_type} resources in {len(scored_dataframe)} cells. ") 
        return scored_dataframe



    # def calc_LCOE_lambda_m1(self,
    #                      row):
        
    #     """ 
    #     # Method: 
    #     LCOE = [(FCR x TCC + FOC + GCC + TRC) / AEP + VOC)
    #         - Total Capital cost, $ (TCC)
    #         - Fixed annual operating cost, $ (FOC)
    #         - Variable operating cost, $/kWh (VOC)
    #         - Fixed charge rate (FCR)
    #         - Annual electricity production, kWh (AEP)
    #         - Grid Connection Cost (GCC)
    #         - Transmission Line Rebuild Cost (TRC) 
            
    #     ### Ref: 
    #     - https://atb.nrel.gov/electricity/2024/equations_&_variables
    #     - https://sam.nrel.gov/financial-models/lcoe-calculator.html
    #     - https://www.nrel.gov/docs/legosti/old/5173.pdf
    #     - https://www.nrel.gov/docs/fy07osti/40566.pdf
        
    #     """

    #     dtg = row['nearest_station_distance_km'] # km
    #     gcc_pu = row[f'grid_connection_cost_per_km_{self.resource_type}'] # m$/km
    #     gcc=dtg*gcc_pu/1.60934  # Convert to miles as our costs are given in m$/miles (USA study)
    #     trc=row[f'tx_line_rebuild_cost_{self.resource_type}']/ 1.60934 # m$/km
    #     tcc = row[f'capex_{self.resource_type}'] # m$/km
        
    #     foc = row[f'fom_{self.resource_type}'] * row[f'potential_capacity_{self.resource_type}'] # m$/ MW * MW
    #     voc = row[f'vom_{self.resource_type}'] * row[f'potential_capacity_{self.resource_type}'] # m$/ MW * MW
        
    #     fcr = row.get('FCR', 0.098) 
    #     aep = 8760 * row[f'{self.resource_type}_CF_mean'] * row[f'potential_capacity_{self.resource_type}'] # MWh
        
    #     if aep == 0: # some cells have no potentials
    #         return float(99999)  # handle the error 
    #     else:
    #         lcoe = ((fcr * tcc + gcc + trc + foc) / aep + voc)  # m$/MWh
    #         return lcoe  * 1E6 # LCOE in $/MWh      
        
    # """
    # 'Fixed O&M', 'CFC', 'LCOE', 'CAPEX', 'CF', 'OCC', 'GCC',
    #    'Variable O&M', 'Heat Rate', 'Fuel', 'Additional OCC',
    #    'Heat Rate Penalty', 'Net Output Penalty', 'FCR', 'Inflation Rate',
    #    'Interest Rate Nominal', 'Rate of Return on Equity Nominal',
    #    'Calculated Interest Rate Real',
    #    'Interest During Construction - Nominal',
    #    'Calculated Rate of Return on Equity Real', 'Debt Fraction',
    #    'Tax Rate (Federal and State)', 'WACC Nominal', 'WACC Real', 'CRF'
    #   """
    # def calc_LCOE_lambda_m2(self,
    #                      row):
        
    #     """ 
    #     # Method: 
    #     LCOE = [{(FCR x CAPEX) + FOM ) /  (CF x 8760) } + VOM + Fuel - PTC)
            
    #         - Fixed charge rate (FCR) :
    #             - Amount of revenue per dollar of investment required that must be collected annually from customers to pay the carrying charges on that investment.
    #         - CAPEX (m$)
    #             - expenditures required to achieve commercial operation of the generation plant.
    #             - ConFinFactor x (OCC +GCC)
    #                 - ConFinFactor: Conversion Factor for Capital Recovery. 
    #                     - The portion of all-in capital cost associated with construction period financing; 
    #                     - ConFinFactor = Σ(y=0 t0 C-1) FC-y x AI_y
    #                     - assumed to be 1.0 for simplification
    #                 - OCC ($/kW) : Overnight Capital Cost CAPEX if plant could be constructed overnight (i.e., excludes construction period financing); includes on-site electrical equipment (e.g., switchyard), a nominal-distance spur line (<1 mi), and necessary upgrades at a transmission substation. ($/kW)
    #                 - GCC ($/kW): Grid Connection Cost 
    #         - CF: Capacity Factor
    #         - Variable operation and maintenance (VOM), $/MWh (VOC)
    #         - Fuel: 
    #             - Fuel costs, converted to $/MWh, using heat rates.
    #             - Heat rate (MMBtu/MWh) * Fuel Costs($/MMBtu)
    #             - Zero for VREs
    #         - PTC ($/MWh) :  Production Tax Credit
    #             - a before-tax credit that reduces LCOE; credits are available for 10 years, so it must be adjusted for a 10-year CRF relative to the full CRF of the project. 
    #             - This formulation of the PTC accounts for a pre-tax LCOE and aligns with the equation used for the ProFinFactor.
    #             - PTC= {PTC_full/(1-TR)}x(CRF/CRF_10yrs)
    #                 - TR: Tax Rate
    #                 - CRF: Capital Recovery Factor
    #                     - ratio of a constant annuity to the present value of receiving that annuity for a given length of time
    #                     - CRF = WACC x[1/(1-(1+WACC)^-t)]
    #                         - WACC: Weighted Average Cost of Capital
    #                             - average expected rate that is paid to finance assets
    #                             - WACC = [ 1+ [1-DF] x [(1+# It looks like the
    #                             # code you provided is
    #                             # not valid Python
    #                             # code. It seems to be
    #                             # a random string of
    #                             # characters. Can you
    #                             # please provide more
    #                             # context or clarify
    #                             # what you are trying
    #                             # to achieve with this
    #                             # code snippet?
    #                             RROE)(1+i)-1] + DF x [(1+IR)(1+i)-1] x [1-TR] ]/(1+i) -1

            
    #     ### Ref: 
    #     - https://atb.nrel.gov/electricity/2024/equations_&_variables
        
    #     """

    #     dtg = row['nearest_station_distance_km'] # km
        
    #     gcc_pu = row[f'grid_connection_cost_per_km_{self.resource_type}'] # m$/km
    #     gcc=dtg*gcc_pu/1.60934  # Convert to miles as our costs are given in m$/miles (USA study)
        
    #     trc=dtg*row[f'tx_line_rebuild_cost_{self.resource_type}']/ 1.60934 # m$=m$/km*km
        
    #     tcc = row[f'capex_{self.resource_type}'] * row[f'potential_capacity_{self.resource_type}'] # m$=m$/MW * MW
        
    #     foc = row[f'fom_{self.resource_type}'] * row[f'potential_capacity_{self.resource_type}'] # m$/ MW * MW
    #     voc = row[f'vom_{self.resource_type}']  # m$/ MW 
        
    #     fcr = row.get('FCR', 0.098) 
    #     aep = 8760 * row[f'{self.resource_type}_CF_mean'] * row[f'potential_capacity_{self.resource_type}'] # MWh
        
    #     if aep == 0: # some cells have no potentials
    #         return float('99999')  # handle the error 
    #     else:
    #         lcoe = ((fcr * (tcc + gcc + trc ) + foc) / aep + voc)  # m$/MWh
    #         return lcoe  * 1E6 # LCOE in $/MWh
    def calculate_score_normalized(self,
                                row: pd.Series,
                                node_distance_col: str,
                                CF_column: str,
                                CRF: float,
                                reference_capacity: float = 1.0) -> float:
        """
        Calculate LCOE using a normalized capacity for fair comparison in clustering.
        
        Args:
            reference_capacity (float): Fixed capacity to use for cost calculations (MW)
        """
        
        # Use fixed reference capacity instead of actual capacity
        capacity_factor = row[CF_column]
        annual_energy = 8760 * capacity_factor * reference_capacity  # MWh/year
        
        if annual_energy == 0:
            return float('999999')
        
        # Calculate costs using reference capacity
        total_capex = self.calculate_total_cost_smooth_scaling(
            row[node_distance_col],
            row[f'grid_connection_cost_per_km_{self.resource_type}'],
            row[f'tx_line_rebuild_cost_{self.resource_type}'],
            row[f'capex_{self.resource_type}'],
            reference_capacity  # Use fixed capacity
        )
        
        # O&M costs with reference capacity
        fom_annual = row[f'fom_{self.resource_type}'] * reference_capacity
        vom_annual = row[f'vom_{self.resource_type}'] * annual_energy
        
        # Calculate LCOE
        lcoe_millions = ((total_capex * CRF) + fom_annual + vom_annual) / annual_energy
        lcoe_dollars = lcoe_millions * 1E6
        
        return lcoe_dollars

    def calculate_score_per_mw(self,
                            row: pd.Series,
                            node_distance_col: str,
                            CF_column: str,
                            CRF: float) -> float:
        """
        Calculate LCOE per MW for capacity-independent comparison.
        """
        
        capacity_factor = row[CF_column]
        if capacity_factor == 0:
            return float('999999')
        
        # Calculate costs per MW
        tech_capex_per_mw = row[f'capex_{self.resource_type}']  # Already in M$/MW
        
        # Grid connection cost per MW (this is where the issue lies)
        grid_cost_total = (row[node_distance_col] * 
                        row[f'grid_connection_cost_per_km_{self.resource_type}'] + 
                        row[f'tx_line_rebuild_cost_{self.resource_type}'])
        
        # Assume minimum viable project size for grid connection allocation
        min_project_size = 10.0  # MW - adjust based on your analysis
        grid_cost_per_mw = grid_cost_total / min_project_size
        
        total_capex_per_mw = tech_capex_per_mw + grid_cost_per_mw
        
        # Annual energy per MW
        annual_energy_per_mw = 8760 * capacity_factor  # MWh/year per MW
        
        # O&M per MW
        fom_per_mw = row[f'fom_{self.resource_type}']  # Already per MW
        vom_per_mw = row[f'vom_{self.resource_type}'] * annual_energy_per_mw
        
        # LCOE per MW
        lcoe_millions = ((total_capex_per_mw * CRF) + fom_per_mw + vom_per_mw) / annual_energy_per_mw
        lcoe_dollars = lcoe_millions * 1E6
        
        return lcoe_dollars

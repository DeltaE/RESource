from pathlib import Path
import atlite
import numpy as np
import pandas as pd

from RES.boundaries import GADMBoundaries
import RES.utility as utils

import os
os.environ['CDSAPI_URL'] = 'https://cds.climate.copernicus.eu/api'

print_level_base=3

class ERA5Cutout(GADMBoundaries):
    def __post_init__(self):
        
        # Call the parent class __post_init__ to initialize inherited attributes
        super().__post_init__()
        
        # Set the targeted data specific attributes 
        self.cutout_config:dict = self.get_cutout_config()
        
        # Extract start and end years
        self.start_year = self.cutout_config["snapshots"]["start"][0][:4]
        self.end_year = self.cutout_config["snapshots"]["end"][0][:4]
        self.cutout_path:Path = self.get_cutout_path()

        
    def get_cutout_path(self)->Path:
        '''
        ### takes:
        cutout configuration dictionary. Specifically the snapshot information.
        
        ### does:
        creates an unique name based on the region and start/end year
        for a cutout.
        
        ### returns: 
        file path + unique name for the cutout described by selections in the
        cutout configuration.
        '''
        
        # Get the base directory and region name
        base_dir = Path(self.cutout_config['root'])
        
        # Construct the file name based on whether it's a single year or multi-year file
        if self.start_year == self.end_year:
            suffix = self.start_year
        else:
            suffix = "_".join([self.start_year, self.end_year])
        
        # Combine region and year(s) to form the file name
        file_name = f"{self.region_short_code}_{suffix}.nc"
        
        # Join the base directory and file name to form the full path
        file_path:Path = base_dir / file_name
        
        return file_path

        
    def get_era5_cutout(self) -> atlite.Cutout:
        """
        This method creates a cutout based on data for ERA5.

        Args:
            bounding_box (dict): A dictionary containing the bounding box with 'min_x', 'max_x', 'min_y', 'max_y'.
            region_code (str, optional): Optional string representing the code of the region for which the cutout is created.

        Returns:
            the 'cutout' object from atlite.

        Raises:
            ValueError: If the bounding box is not valid.
            ConnectionError: If there is an issue connecting to the data source.

        Note:
            After execution, all downloaded data is stored at cutout.path. By default, it is not loaded into memory but into Dask arrays to keep memory consumption low. The data is accessible via cutout.data, which is an xarray.Dataset.
        """
        utils.print_update(level=print_level_base+1,
                           message=f"{__name__}|  Processing ERA5's cutout...")
        
        MBR,region_boundary=self.get_bounding_box()
        utils.print_update(level=print_level_base+2,
                   message=f"{__name__}| ✓ MBR and regional boundary created. ")
        
        # Extract parameters from the configuration file
        dx, dy = self.cutout_config["dx"], self.cutout_config['dy']
        time_horizon = slice(self.cutout_config["snapshots"]['start'][0], self.cutout_config["snapshots"]['end'][0])
        min_x, max_x, min_y, max_y = MBR.values()

        # Create the cutout based on bounds found from above
        cutout = atlite.Cutout(
            path=self.cutout_path,
            module=self.cutout_config["module"],
            x=slice(min_x - dx, max_x + dx),  # Longitude
            y=slice(min_y - dy, max_y + dy),  # Latitude
            dx=dx,
            dy=dy,
            time=time_horizon
        )

        cutout.prepare(monthly_requests=True, 
                       concurrent_requests=False)  # Prepare the cutout data
        utils.print_info(info=""" Memory management remarks:
    * After execution, all downloaded data is stored at cutout.path. By default, it is not loaded into memory, but into dask arrays. This keeps the memory consumption extremely low.
    * The data is accessible in cutout.data, which is an xarray.Dataset. Querying the cutout gives us some basic information on which data is contained in it.
    * For more operations related to cutout, check the tool docs @ https://atlite.readthedocs.io/en/master/examples/create_cutout.html#
        """)
        # cutout,region_boundary
        
        # # Define a namedtuple
        # data_tuple = namedtuple('capacity_data', ['cutout','region_boundary'])
        
        # self.data=data_tuple(cutout,region_boundary)
        utils.print_update(level=print_level_base+1,
                   message=f"{__name__}| ✓ Cutout and regional boundary processed. ")
        return cutout,region_boundary
    
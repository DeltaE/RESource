import requests
import rasterio
from rasterio.mask import mask
from zipfile import ZipFile
from pathlib import Path
import matplotlib.pyplot as plt

from RES.boundaries import GADMBoundaries
from RES import utility as utils
print_level_base=4

# Define the GAEZRasterProcessor class
class GAEZRasterProcessor(GADMBoundaries):
    def __post_init__(self):
        """
        Initialize inherited attributes and OSM data-specific configuration.
        """
        super().__post_init__()

        self.gaez_config: dict = self.get_gaez_data_config()
        
        self.gaez_root = Path(self.gaez_config.get('root', 'data/downloaded_data/GAEZ'))
        self.gaez_root.mkdir(parents=True, exist_ok=True)

        self.zip_file = Path(self.gaez_config['zip_file'])

        self.Rasters_in_use_direct = Path(self.gaez_config['Rasters_in_use_direct'])
        self.Rasters_in_use_direct.mkdir(parents=True, exist_ok=True)

        self.raster_types = self.gaez_config['raster_types']


    def process_all_rasters(self,
                            show:bool=False):
        """Main pipeline to download, extract, clip, and plot rasters based on configuration."""
        if not (self.gaez_root / self.zip_file).exists():
            self.__download_resources_zip_file__()
        
        self.__extract_rasters__()
        self.region_boundary = self.get_region_boundary()
        
        utils.print_update(level=print_level_base,message=f"{__name__}| Clipping Rasters to regional boundaries.. ")
        # Loop over raster types and process each
        for raster_type in self.raster_types:
            self.__clip_to_boundary_n_plot__(raster_type, self.region_boundary.geometry,show)
        
        utils.print_update(level=print_level_base,message=f"{__name__}| ✔ All required rasters for GAEZ processed and plotted successfully.")

    def __download_resources_zip_file__(self):
        """Downloads the resources zip file from GAEZ if not already downloaded."""
        url = self.gaez_config.get('source', 'https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/LR.zip')
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(self.gaez_root / self.zip_file, 'wb') as f:
                f.write(response.content)
            utils.print_update(level=print_level_base,message=f"{__name__}| GAEZ Raster Resource '.zip' file downloaded and saved to: {self.gaez_root}")
        else:
            utils.print_update(level=print_level_base,message=f"{__name__}|  ❌ Failed to download the Resources zip file from GAEZ. Status code: {response.status_code}")

    def __extract_rasters__(self):
        """Extracts required raster files from the downloaded zip file."""
        with ZipFile(self.gaez_root / self.zip_file, 'r') as zip_ref:
            for raster_type in self.raster_types:
                raster_file = raster_type['raster']
                zip_direct = raster_type['zip_extract_direct']
                file_inside_zip = str(Path(zip_direct) / raster_file)  # Ensure it's a single string

                target_path = self.gaez_root / self.Rasters_in_use_direct / zip_direct / raster_file

                if not target_path.exists():
                    # Check for existence as a string in zip_ref
                    if file_inside_zip in zip_ref.namelist():
                        zip_ref.extract(file_inside_zip, path=self.gaez_root / self.Rasters_in_use_direct)
                        utils.print_update(level=print_level_base,message=f"{__name__}| Raster file '{raster_file}' extracted from {file_inside_zip}")
                    else:
                        utils.print_update(level=print_level_base,message=f"{__name__}| Raster file '{raster_file}' not found in the archive {file_inside_zip}")
                else:
                    utils.print_update(level=print_level_base,message=f"{__name__}| Raster file '{raster_file}' found in local directory, skipping download.")


    def __clip_to_boundary_n_plot__(self, raster_type, boundary_geom,show):
        """Clip the raster to region boundaries and generate a plot."""
        zip_direct = raster_type['zip_extract_direct']
        raster_file = raster_type['raster']
        plot_title = raster_type['name']
        color_map = raster_type['color_map']

        input_raster = self.gaez_root / self.Rasters_in_use_direct / zip_direct / raster_file
        output_dir = self.gaez_root / self.Rasters_in_use_direct / zip_direct 
        output_dir.mkdir(parents=True, exist_ok=True)

        clipped_raster_path = output_dir / f"{self.region_short_code}_{raster_file}"

        with rasterio.open(input_raster) as src:
            clipped_raster, clipped_transform = mask(src, boundary_geom, crop=True, indexes=src.indexes)
            clipped_meta = src.meta.copy()
            clipped_meta.update({
                'height': clipped_raster.shape[1],
                'width': clipped_raster.shape[2],
                'transform': clipped_transform
            })

            with rasterio.open(clipped_raster_path, 'w', **clipped_meta) as dst:
                dst.write(clipped_raster)
            
            # Call visualization method
            plot_save_to = Path('vis/misc') / raster_file.replace('.tif', f'_raster_{self.region_short_code}.png')
            raster_plot=self.plot_gaez_tif(clipped_raster_path, 
                               color_map, 
                               plot_title, 
                               plot_save_to,show)
            utils.print_update(level=print_level_base+1,message=f"{__name__}| Clipped Raster plot for {self.region_name} saved at: {plot_save_to}")
            return raster_plot

    def plot_gaez_tif(self, tif_path, color_map, plot_title, save_to, show=False):
        """Visualize and save the raster plot."""
        with rasterio.open(tif_path) as src:
            data = src.read(1, masked=True)
            extent = src.bounds
        save_to.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data, cmap=color_map, extent=[extent.left, extent.right, extent.bottom, extent.top])
        cbar = plt.colorbar(im, ax=ax, label="Layer Class", orientation="horizontal", fraction=0.05, pad=0.08)
        ax.set_title(plot_title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(visible=False)
        plt.tight_layout()
        plt.savefig(save_to)
        if show:
            plt.show()
        plt.close(fig)
        return fig
        

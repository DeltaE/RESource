from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import rasterio
import requests
from rasterio.mask import mask

from RES import utility as utils
from RES.AttributesParser import AttributesParser
from RES.boundaries import GADMBoundaries

print_level_base = 4

class CORINERasterProcessor(AttributesParser):
    """
    CORINE (Coordination of Information on the Environment) raster data processor for land cover analysis.
    
    This class handles the download, extraction, clipping, and visualization of CORINE land cover raster datasets
    used in environmental and renewable energy land constraint analysis. CORINE provides spatial data on land cover
    types across Europe, supporting land use planning and resource assessment.
    
    The processor integrates CORINE land cover data with regional boundaries to support siting decisions and capacity assessments.
    It automatically downloads the required raster datasets, extracts specific layers based on configuration, clips
    them to regional boundaries, and generates visualization outputs for analysis.
    """
    def __post_init__(self):
        super().__post_init__()
        self.required_args = {
            "config_file_path": self.config_file_path,
            "region_short_code": self.region_short_code,
            "resource_type": self.resource_type
        }
        self.gadmBoundary = GADMBoundaries(**self.required_args)
        self.corine_config: dict = super().get_corine_data_config()
        self.corine_root = Path(self.corine_config.get('root', 'data/downloaded_data/CORINE'))
        self.corine_root.mkdir(parents=True, exist_ok=True)
        self.zip_file = Path(self.corine_config['zip_file'])
        self.Rasters_in_use_direct = Path(self.corine_config['Rasters_in_use_direct'])
        self.Rasters_in_use_direct.mkdir(parents=True, exist_ok=True)
        self.raster_types = self.corine_config['raster_types']
        self.region_boundary = None

    def process_all_rasters(self, show: bool = False):
        if not (self.corine_root / self.zip_file).exists():
            self.__download_resources_zip_file__()
        raster_paths = {}
        self.__extract_rasters__()
        self.region_boundary = self.gadmBoundary.get_region_boundary()
        utils.print_update(level=print_level_base, message=f"{__name__}| Clipping CORINE Rasters to regional boundaries.. ")
        for raster_type in self.raster_types:
            raster_path = self.__clip_to_boundary_n_plot__(raster_type, self.region_boundary.geometry, show)
            raster_paths[raster_type['name']] = raster_path
        utils.print_update(level=print_level_base, message=f"{__name__}| ✔ All required rasters for CORINE processed and plotted successfully.")
        return raster_paths

    def __download_resources_zip_file__(self):
        url = self.corine_config.get('source', 'https://land.copernicus.eu/arcgis/rest/services/Corine/CLC2018/MapServer')
        response = requests.get(url)
        if response.status_code == 200:
            with open(self.corine_root / self.zip_file, 'wb') as f:
                f.write(response.content)
            utils.print_update(level=print_level_base, message=f"{__name__}| CORINE Raster Resource '.zip' file downloaded and saved to: {self.corine_root}")
        else:
            utils.print_update(level=print_level_base, message=f"{__name__}|  ❌ Failed to download the Resources zip file from CORINE. Status code: {response.status_code}")

    def __extract_rasters__(self):
        with ZipFile(self.corine_root / self.zip_file, 'r') as zip_ref:
            for raster_type in self.raster_types:
                raster_file = raster_type['raster']
                zip_direct = raster_type['zip_extract_direct']
                file_inside_zip = str(Path(zip_direct) / raster_file)
                target_path = self.corine_root / self.Rasters_in_use_direct / zip_direct / raster_file
                if not target_path.exists():
                    if file_inside_zip in zip_ref.namelist():
                        zip_ref.extract(file_inside_zip, path=self.corine_root / self.Rasters_in_use_direct)
                        utils.print_update(level=print_level_base, message=f"{__name__}| Raster file '{raster_file}' extracted from {file_inside_zip}")
                    else:
                        utils.print_update(level=print_level_base, message=f"{__name__}| Raster file '{raster_file}' not found in the archive {file_inside_zip}")
                else:
                    utils.print_update(level=print_level_base, message=f"{__name__}| Raster file '{raster_file}' found in local directory, skipping download.")

    def __clip_to_boundary_n_plot__(self, raster_type, boundary_geom, show):
        zip_direct = raster_type['zip_extract_direct']
        raster_file = raster_type['raster']
        plot_title = raster_type['name']
        color_map = raster_type['color_map']
        input_raster = self.corine_root / self.Rasters_in_use_direct / zip_direct / raster_file
        output_dir = self.corine_root / self.Rasters_in_use_direct / zip_direct
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
            plot_save_to = Path('vis/misc') / raster_file.replace('.tif', f'_raster_{self.region_short_code}.png')
            self.plot_corine_tif(clipped_raster_path, color_map, plot_title, plot_save_to, show)
            utils.print_update(level=print_level_base+1, message=f"{__name__}| Clipped CORINE Raster plot for {super().get_region_name()} saved at: {plot_save_to}")
            return clipped_raster_path

    def plot_corine_tif(self, tif_path, color_map, plot_title, save_to, show=False):
        with rasterio.open(tif_path) as src:
            data = src.read(1, masked=True)
            extent = src.bounds
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data, cmap=color_map, extent=[extent.left, extent.right, extent.bottom, extent.top])
        plt.colorbar(im, ax=ax, label="Land Cover Class", orientation="horizontal", fraction=0.05, pad=0.08)
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

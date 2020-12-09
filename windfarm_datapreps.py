##windfarm suitability analysis Group effort from Ovenden, Martin, Burton 

import numpy as np
import os
import rasterio 
from rasterio.plot import show, show_hist
import scipy
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd 
import rioxarray as rxr
from shapely.geometry import Point, LineString, Polygon, box
from rasterio.mask import mask
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from fiona.crs import from_epsg
from scipy.spatial import cKDTree
from rasterstats import zonal_stats
import fiona
import geocube

from Part1_FinalProject_mo import *


in_data_dir = r'.\data'


##Start data preparations##
#rasterize vector files using geocube
easement = vector(gpd.read_file(in_data_dir + './GDB_ME/Misc_Easements.gdb'), './easements_ras.tif')
oilgas = vector(gpd.read_file(in_data_dir + './gdb_osi/GDB_OSI.gdb'), './oil_gas_assets.tif')
coastal_lease = vector(gpd.read_file(in_data_dir + 'GDB_NonMineralPoly/Coastal_Leases_Poly.gdb'), './coastal_lease_ras.tif')
navdist = vector(gpd.read_file(in_data_dir + 'GDB_NavDist/NavDist.gdb'), './navdist_ras.tif')
shipping = vector(gpd.read_file(in_data_dir + 'shippinglanes/shippinglanes.shp'), '.shipping_ras.tif')
ppa = vector(gpd.read_file(in_data_dir + 'GDB_PPA/PPA.gdb'), './sensitive_habitat_ras.tif')

#reproject rasters
dem_projected = reproject_r('./corpuschristi_dem.tif','./reproject_corpuschristi_dem.tif')
wind_projected = reproject_r('./wtk_conus_100m_mean_masked.tif', './reproject_wtk_conus_100m_mean_masked.tif')
reproject_easment  = reproject_r('./easements_ras.tif', './reproject_easements_ras.tif') 
reproject_hab  = reproject_r('.\\ppa.tif', '.\\reproject_ppa.tif.tif') 
reproject_navdist = reproject_r('.\\navdist_ras.tif', '.\\reproject_navdist_ras.tif')
reproject_oil = reproject_r('.\\oil_gas_assets.tif', '.\\reproject_oil_gas_assets.tif')
reproject_shipping = reproject_r('.\\shipping_ras', '.\\reproject_shipping_ras.tif')
reproject_leases = reproject_r('.\\coastal_lease_ras', '.\\reproject_coastal_lease.tif')

#mask rasters to the dem_bounds
mask_easment  = mask(reproject_easment, '.\\Clip_reproject_easements_ras.tif')
mask_hab  = mask(reproject_hab, '.\\Cilp_reproject_habitat_ras.tif') 
mask_navdist = mask(reproject_navdist, '.\\Cilp_reproject_navdist_ras.tif')
mask_oil = mask(reproject_oil, '.\\Cilp_reproject_oil_gas_assets.tif')
mask_speed_wind = mask(wind_projected, '.\\Cilp_reproject_wind_suit.tif')
mask_lease = mask(reproject_leases, '.\\Cilp_reproject_coastal_lease.tif')
mask_shipping = mask(reproject_shipping, '.\\Clip_reproject_shipping_ras.tif')

import numpy as np
import os
import rasterio 
from rasterio.plot import show, show_hist
import glob
import scipy
#from lab5functions import *
import pandas as pd
import geopandas as gpd
#import netCDF4 as nc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array, osr, ogr
import rasterio 
import pandas as pd 
import geopandas as gpd
import numpy as np
import rioxarray as rxr
import os
from shapely.geometry import Point, LineString, Polygon, box
from rasterio.mask import mask
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from fiona.crs import from_epsg

from Part1_FinalProject_mo import *


in_data_dir = r'.\data'

#reproject dems
dem_projected = reproject_r(in_data_dir + './corpuschristi_dem.tif', in_data_dir +'./reproject_corpuschristi_dem.tif')
wind_projected = reproject_r(in_data_dir + './wtk_conus_100m_mean_masked.tif', in_data_dir +'./reproject_wtk_conus_100m_mean_masked.tif')

#rasterize shapefiles with geocube

#used the rasterized shapefiles, mask, reproject 
ship_lanes = reproject('./shippinglanes/shippinglanes.shp')


#rasterize shapefiles based on reprojected dem
shp_fn = (gpd.read_file(os.path.join(in_data_dir, 'shippinglanes/shippinglanes.shp')))
rst_fn = (in_data_dir + './reproject_corpuschristi_dem.tif')
out_fn = (in_data_dir + './rasterized.tif')

rst = rasterio.open(rst_fn)

meta = rst.meta.copy()
meta.update(compress='lzw')

with rasterio.open(out_fn, 'w+', **meta) as out:
    out_arr = out.read(1)
    out.write_band(1, out_arr)

reproject_easment  = reproject_r('./easements_ras.tif', './reproject_easements_ras.tif') 
reproject_hab  = reproject_r('.\\habitat_ras.tif', '.\\reproject_habitat_ras.tif') 
reproject_navdist = reproject_r('.\\navdist_ras.tif', '.\\reproject_navdist_ras.tif')
reproject_oil = reproject_r('.\\oil_gas_assets.tif', '.\\reproject_oil_gas_assets.tif')
reproject_speed_wind2 = reproject_r('.\\wind_suit.tif', '.\\reproject_wind_suit.tif')


mask_easment  = mask(reproject_easment, '.\\reproject_easements_ras.tif')
mask_hab  = mask(reproject_hab, '.\\Cilp_reproject_habitat_ras.tif') 
mask_navdist = mask(reproject_navdist, '.\\Cilp_reproject_navdist_ras.tif')
mask_oil = mask(reproject_oil, '.\\Cilp_reproject_oil_gas_assets.tif')
mask_speed_wind2 = mask(reproject_speed_wind2, '.\\Cilp_reproject_wind_suit.tif')

#reclassify 
eas_Raster = rasterio.open(in_data_dir + mask_easment , 'r')  
easments = eas_Raster.read(1) 
suit_eas = np.where( easments == 1, 1,0)
print('The number of sites ', suit_eas.sum())

avoid_Raster = rasterio.open(in_data_dir +  mask_avoid , 'r')  
area_avoid = aviod_Raster.read(1) 
suit_area = np.where(area_avoid == 1 , 1,0)
print('The number of sites ', suit_area.sum())

hab_Raster = rasterio.open(in_data_dir + mask_hab  , 'r')  
area_hab = hab_Raster.read(1) 
suit_hab = np.where(area_hab == 1, 1,0)
print('The number of sites ', suit_hab.sum())

navdist_Raster = rasterio.open(in_data_dir + mask_navdist  , 'r')  
navdist = navdist_Raster.read(1) 
suit_navdist= np.where(navdist == 1 , 1,0)
print('The number of sites ', suit_navdist.sum())

oil_Raster = rasterio.open(in_data_dir + mask_oil  , 'r')  
oil_rig_loc = oil_Raster.read(1) 
suit_no_oil = np.where(oil_rig_loc == 1, 1,0)
print('The number of sites ', suit_no_oil.sum())

speed_Wind_Raster = rasterio.open(in_data_dir + mask_speed_wind2  , 'r')  
Wind = speed_Wind_Raster.read(1) 
Suit_speed_wind = np.where( Wind == 1 , 1,0)
print('The number of sites ', Suit_speed_wind.sum())

Wind50_Raster = rasterio.open(in_data_dir + mask_Sites_wind50  , 'r')  
Wn50 = Wind50_Raster.read(1) 
Suit_Wn50 = np.where(  Wn50 == 1 , 1,0)
print('The number of sites ', Suit_Wn50.sum())

    
sum_area = suit_eas + suit_area + suit_hab + suit_navdist + suit_no_oil + Suit_speed_wind + Suit_Wn50 
suitable_areas = np.where(sum_area == 7, 1, 0)
print('Total number of suitable sites is ', suitable_areas.sum())
Suitarray = []
Suitarray.append(suitable_areas)
#export 

# reproject_avoid = reproject_r('.\\avoidance_area.tif', '.\\reproject_avoidance_area.tif')
#reclassify avoidance areas so that we have suitable areas NODATA = 1, 1 = NODATA
mask_avoid = mask(reproject_avoid, '.\\Cilp_reproject_avoidance_area.tif')


#export results to a final tif windsites_50
reproject_Sites_wind50 = reproject_r('.\\windsites_50m.tif', '.\\reproject_windsites_50m.tif')

#compute slope and aspect of each of the suites located, zonal stats

with rasterio.open(in_data_dir + './reproject_corpuschristi_dem.tif', 'r') as dem_file:
    dem = dem_file.read(1)
    dem_profile = dem_file.profile
    show(dem[1100:8350,1100:8350])
    dem_profile
    dem.shape
    #COMPUTE ZONAL STATS OF EACH OF THE SITES



with rasterio.open(in_data_dir + './rasterized.tif', 'r') as shiplanes:
    ship = shiplanes.read(1)
    ship_profile = shiplanes.profile
    show(ship)
    ship_profile
    ship.shape

#compute straightline distance to nearest windfarm station 
# #this calculation is not based on optimal route
xs = []
ys = []
with open(os.path.join(in_data_dir, 'transmission_stations.txt')) as coords:
    lines = coords.readlines()[1:]
    for l in lines:
        x,y = l.split(',')
        xs.append(float(x))
        ys.append(float(y))
    #np.vstack is for pixel data with height (first axis) width (second axis), concatenates along the first axis
    stations = np.vstack([xs, ys])
    stations = stations.T

with rasterio.open(os.path.join(in_data_dir, 'suitable_sites.tif')) as file:
    bounds = file.bounds
    topLeft = (bounds[0], bounds[3])
    lowRight = (bounds[2], bounds[1])
    cellSize = 1000
    x_coords = np.arange(topLeft[0] + cellSize/2, lowRight[0], cellSize) #gives range of x coordinates
    y_coords = np.arange(lowRight[1] + cellSize/2, topLeft[1], cellSize) #gives range of y coordinates 
    #meshgrid cretaes a rectangular grid out of two given one-dim arrays reprenting cartesian indexing       
    x,y = np.meshgrid(x_coords, y_coords)
    #np.c_ tranlates slice objects to concatenation along the second axes, flatten returns the array in one dimension
    coord = (np.c_[x.flatten(), y.flatten()])

#provides an index into a set of k-dimensional points which can be used to rapidly look up the nearest neighbors of any point.
tree = cKDTree(coord)

#performs the nearest neighbor operations, with k being the nearest neighbors to return 
dd, ii = tree.query(stations, k=5)


print('The maximum distance to the closest transmission substation among all of the suitable sites is ' 
      +  str(dd.max()) + ' meters')
print('The minimum distance to the closest transmission substation among all of the suitable sites is ' 
      +  str(dd.min()) + ' meters')
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

from project_functions import *


in_data_dir = r'.\data'

#start windfarm suitability analysis
#reclassify 
eas_Raster = rasterio.open(in_data_dir + './transmission_ras.tif', 'r')  
easments_arr = eas_Raster.read(1) 
eas_Raster.meta
suit_eas = np.where(easments_arr == 1, 0,1)
show(suit_eas)
print('The number of sites ', suit_eas.sum())

oil_Raster = rasterio.open(in_data_dir + './oil_gas_assets_ras.tif', 'r')  
oil_arr = oil_Raster.read(1) 
oil_Raster.meta
suit_no_oil = np.where(oil_arr == 1, 0,1)
show(suit_no_oil)
print('The number of sites ', suit_no_oil.sum())

leases_Raster = rasterio.open(in_data_dir + './leases_ras.tif'  , 'r')  
coastal_arr = leases_Raster.read(1) 
leases_Raster.meta
suit_leases= np.where(coastal_arr == 1, 0, 1)
show(suit_leases)
print('The number of sites ', suit_leases.sum())

navdist_Raster = rasterio.open(in_data_dir + './navdist_ras.tif'  , 'r')  
navdist_arr = navdist_Raster.read(1) 
navdist_Raster.meta
suit_navdist= np.where(navdist_arr == 1, 0, 1)
show(suit_navdist)
print('The number of sites ', suit_navdist.sum())

hab_Raster = rasterio.open(in_data_dir + './habitat_ras.tif'  , 'r')  
hab_arr = hab_Raster.read(1) 
hab_Raster.meta
suit_hab = np.where(hab_arr == 1, 0, 1)
show(suit_hab)
print('The number of sites ', suit_hab.sum())

dem_raster =  rasterio.open(in_data_dir + './depth_ras.tif', 'r')  
dem_arr = dem_raster.read(1) 
dem_raster.meta
dem_suit = np.where(dem_arr == 1, 1, 0)
show(dem_suit)
print('The number of sites ', dem_suit.sum())

speed_Wind_Raster = rasterio.open(in_data_dir + './wind_suit.tif', 'r')  
Wind_arr = speed_Wind_Raster.read(1) 
speed_Wind_Raster.meta
Suit_speed_wind = np.where(Wind_arr == 1, 1, 0)
show(Suit_speed_wind)
print('The number of sites ', Suit_speed_wind.sum())

beach_buff_Raster = rasterio.open(in_data_dir + './beach_buff.tif', 'r')  
beach_arr = beach_buff_Raster.read(1) 
beach_buff_Raster.meta
suit_beach = np.where(beach_arr == 0, 0, 1)
show(suit_beach)
print('The number of sites ', suit_beach.sum())

meta = speed_Wind_Raster.meta
meta.update({'dtype':'int32', 'nodata' : 0})

sum_area = suit_eas + suit_no_oil + suit_leases + suit_navdist + suit_hab + Suit_speed_wind + dem_suit
suit_arr = np.where(sum_area == 7, 1, 0)
show(suit_arr)
print('Total area of suitable sites is ', suit_arr .sum())

#export windfarm sites to raster
with rasterio.open(in_data_dir + './suit_windfarm.tif', 'w', **meta) as ds:
    ds.write_band(1, suit_arr)
    

#start computational efforts
with rasterio.open(in_data_dir + './reproject_corpuschristi_dem.tif', 'r') as dem_file:
    dem = dem_file.read(1)
    dem_meta = dem_file.profile
    #set nodata values to nan
    show(dem[100:350,100:350])
    #resample dem using nearest neighbor
    dem.shape
    cellSize = dem_file.meta['transform'][1]
    #COMPUTE ZONAL STATS OF EACH OF THE SITES   
    s,a = slopeAspect(dem, cellSize)
    #recalculates the aspect to 8 cardinal directions using the fuction 
    aspect = reclassAspect(a)
    #using a bin size of 10, recalculate the slope grid into 10 classes using the function
    #s = reclassByHisto(s, 10)
    #print(s,a)

#distance analysis
def findX(rasHeight, rasWidth):
    coordX = rasXmin + ((0.5+rasWidth)*cellSize)
    return coordX

def findY(rasHeight, rasWidth):
    coordY = rasYmin + ((0.5+rasHeight)*cellSize)
    return coordY  

with rasterio.open(in_data_dir + './suit_windfarm.tif', 'r') as ras:
    rasXmin = ras.bounds.left
    rasYmin = ras.bounds.bottom
    cellSize = ras.meta['transform'][1]

rasHeight = suit_arr.shape[0]
rasWidth = suit_arr.shape[1]

allXs = np.fromfunction(findX, (rasHeight,rasWidth))
centerXs = allXs[suit_arr==1]
centerXs = centerXs.flatten()

allYs = np.fromfunction(findY, (rasHeight,rasWidth))
centerYs = allYs[suit_arr==1]
centerYs = centerYs.flatten()

centroids = np.vstack([centerXs,centerYs])
centroids = centroids.T
print(centroids)

#iterate over value in raster attribute table to compute nearest neighbor  
transCoords = np.loadtxt(os.path.join(in_data_dir, 'windfarm_locations.txt'), delimiter=',', skiprows=1)
#provides an index into a set of k-dimensional points which can be used to rapidly look up the nearest neighbors of any point.
#performs the nearest neighbor operations, with k being the nearest neighbors to return 
#ii is the location of the neighbor
dist, indices = cKDTree(transCoords).query(centroids)

print('The shortest distance is:',np.min(dist),'and the maximum distance is:',np.max(dist))

 #work with rasters in isolated 
 #create a way to identify groups 
 #reclassify

#zonal stats of slope for each suitable site
def zonalStats(npArray, zoneArray,output_csv):
    vals = np.unique(zoneArray)
    df = pd.DataFrame(columns=['Zone', 'Count', 'Mean', 'Stddev', 'Min', 'Max'])
    df['Zone'] = vals[~np.isnan(vals)].astype(np.int)
    for zone in df['Zone']:
        inZone = npArray[zoneArray == zone]
        df.at[df['Zone'] == zone, 'Count'] = (zoneArray == zone).sum()
        df.at[df['Zone'] == zone, 'Mean'] = inZone.mean()
        df.at[df['Zone'] == zone, 'Stddev'] = inZone.std()
        df.at[df['Zone'] == zone, 'Min'] = inZone.min()
        df.at[df['Zone'] == zone, 'Max'] = inZone.max()
        df.to_csv(output_csv)
    return df

zonalStats(suit_arr, s, "slp.csv")
zonalStats(suit_arr, a, "aspect.csv")
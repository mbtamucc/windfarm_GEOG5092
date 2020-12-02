"Final Project Section 2"
import rasterio 
import fiona
import pandas as pd 
import geopandas as gpd
import numpy as np
import rioxarray as rxr
import os
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, box
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from fiona.crs import from_epsg
from Part1_FinalProject import reproject, reporject_r, mask

in_data_dir = r'.//data'

reproject_easment  = reproject_r('.\\easements_ras.tif', '.\\reproject_easements_ras.tif') 
reproject_avoid = reproject_r('.\\avoidance_area.tif', '.\\reproject_avoidance_area.tif')
reproject_hab  = reproject_r('.\\habitat_ras.tif', '.\\reproject_habitat_ras.tif') 
reproject_navdist = reproject_r('.\\navdist_ras.tif', '.\\reproject_navdist_ras.tif')
reproject_oil = reproject_r('.\\oil_gas_assets.tif', '.\\reproject_oil_gas_assets.tif')
reproject_speed_wind2 = reproject_r('.\\wind_suit.tif', '.\\reproject_wind_suit.tif')
reproject_Sites_wind50 = reproject_r('.\\windsites_50m.tif', '.\\reproject_windsites_50m.tif')


mask_easment  = mask(reproject_easment, '.\\reproject_easements_ras.tif')
mask_avoid = mask(reproject_avoid, '.\\Cilp_reproject_avoidance_area.tif')
mask_hab  = mask(reproject_hab, '.\\Cilp_reproject_habitat_ras.tif') 
mask_navdist = mask(reproject_navdist, '.\\Cilp_reproject_navdist_ras.tif')
mask_oil = mask(reproject_oil, '.\\Cilp_reproject_oil_gas_assets.tif')
mask_speed_wind2 = mask(reproject_speed_wind2, '.\\Cilp_reproject_wind_suit.tif')
mask_Sites_wind50 = mask(reproject_Sites_wind50, '.\\Cilp_reproject_windsites_50m.tif')


eas_Raster = rasterio.open(in_data_dir + mask_easment , 'r')  
easments = eas_Raster.read(1) 
suit_eas = np.where( easments == 1, 1,0)
print('The number of sites ', suit_eas.sum())

aviod_Raster = rasterio.open(in_data_dir +  mask_avoid , 'r')  
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


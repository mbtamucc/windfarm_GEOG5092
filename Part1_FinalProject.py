import rasterio 
import pandas as pd 
import geopandas as gpd
import numpy as np
import rioxarray as rxr
import os
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, box
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.mask import mask
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from fiona.crs import from_epsg
#import pycrs
#%matplotlib inline


in_data_dir = 'C:\\Users\\Aubre_M\\Documents\\Fall 2020\\GIS Programming_ Automation\\geog_4092\\Final Project'

def reprogject (layer):
    "This is the function that reproject the vector layers"
    master_crs = 'EPSG:26914'
    if layer.crs != master_crs:
        #print(list_of_lables[idx], 'is not on the master projection, reprojecting now..')
        layer = layer.to_crs(master_crs) 
        print (layer.crs, 'and', master_crs)
    
    return layer

def reproject_r(in_path, out_path):
    'This is the function to reproject the raster data '
    dst_crs = 'EPSG:26914'
    with rasterio.open(in_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rasterio.open(out_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
    return(out_path)

reproject_dem = reproject_r('.\\corpuschristi_dem.tif', '.\\reproject_corpuschristi_dem.tif')

with rasterio.open(in_data_dir + '.\\reproject_corpuschristi_dem.tif', 'r') as dem_file: 
    beach = dem_file.read(1) 
    
    suit_beach = np.where( beach == 0.0 , 1, 0)
    print('The number of sites where elevation is at sea level ', suit_beach.sum())


reproject_wind = reproject_r('.\\wtk_conus_100m_mean_masked.tif', '.\\reprojectDocuments_wtk_conus_100m_mean_masked.tif') 

with rasterio.open(in_data_dir + '.\\reprojectDocuments_wtk_conus_100m_mean_masked.tif', 'r') as dem_file:
    wind = dem_file.read(1) 
    suit_wind_speed = np.where( 7.0 < wind, 1, 0)
    print('The number of sites where wind speed is greater than 7 m ', suit_wind_speed.sum())
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
import fiona
import rasterio
import rasterio.mask

in_data_dir = r'.\data'

#this file contains 4 functions:
#reproject vector layers
#reproject raster layers 
#parse features from gdb for improved rasterio readability
#convert shapefiles to rasters

def reproject(layer):
    "This is the function that reproject the vector layers"
    master_crs = 'EPSG:26914'
    if layer.crs != master_crs:
        #print(list_of_lables[idx], 'is not on the master projection, reprojecting now..')
        layer = layer.to_crs(master_crs) 
        print (layer.crs, 'and', master_crs)
    
    return(layer)

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

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]




#rasterizing test using rasterio and geopandas
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd


#rasterize 3 work in progress
#use reproject shipping lanes
#use reporject dem as the base raster file
shp_fn = (gpd.read_file(os.path.join(in_data_dir, 'shippinglanes/shippinglanes.shp')))
rst_fn = (in_data_dir + './reproject_corpuschristi_dem.tif')
out_fn = (in_data_dir + './rasterized.tif')

rst = rasterio.open(rst_fn)

meta = rst.meta.copy()
meta.update(compress='lzw')

with rasterio.open(out_fn, 'w+', **meta) as out:
    out_arr = out.read(1)
    out.write_band(1, out_arr)


#masking function
import rasterio
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg

with fiona.open('./data/bounds/dem_bounds.shp', "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

with rasterio.open('./data/reproject_wtk_conus_100m_mean_masked.tif') as src:
    out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
    out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

with rasterio.open("RGB.byte.masked.tif", "w", **out_meta) as dest:
    dest.write(out_image)


def mask(raster_file_in, raster_file_out):
    with fiona.open('./data/bounds/dem_bounds.shp', "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open(raster_file_in) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta

        out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

    with rasterio.open(raster_file_out, "w", **out_meta) as dest:
        dest.write(out_image)
    return(raster_file_out)
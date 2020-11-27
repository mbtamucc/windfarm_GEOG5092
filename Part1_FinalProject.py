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

practice = reproject_r('./corpuschristi_dem.tif', './reproject_corpuschristi_dem.tif')
print(practice)

with rasterio.open(in_data_dir + '.\\wtk_conus_100m_mean_masked.tif', 'r') as dem_file:
    wind = dem_file.read(1) 
    #wind = reprogject(dem)
    #practice = reproject_raster(wind)
    #print(practic.crs)
    #crs_UTM13 = CRS.from_string('EPSG:26914')
    # Reproject the data using the crs object
    #layer_UTM13 = wind.rxr.reproject(crs_wgs84)
    #layer_UTM13.rxr.crs
#    suit_wind_speed = np.where( 7.0 < wind, 1, 0)
#    print('The number of sites where wind speed is greater than 7 m ', suit_wind_speed.sum())
    
#with rasterio.open(in_data_dir + '.\\corpuschristi_dem.tif', 'r') as dem_file:
#    beach = dem_file.read(1) 
    #beach = reprogject(dem)
#    suit_beach = np.where( beach == 0.0 , 1, 0)
#    print('The number of sites where elevation is at sea level ', suit_beach.sum())

data_dir = "L5_data"

# Input raster
fp = os.path.join(data_dir, "p188r018_7t20020529_z34__LV-FIN.tif")

# Output raster
out_tif = os.path.join(data_dir, "Helsinki_Masked.tif")

# Read the data
data = rasterio.open(fp)

# Visualize the NIR band
show((data, 4), cmap='terrain')


# WGS84 coordinates
minx, miny = 24.60, 60.00
maxx, maxy = 25.22, 60.35
bbox = box(minx, miny, maxx, maxy)

geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(4326))
print(geo)

# Project the Polygon into same CRS as the grid
geo = geo.to_crs(crs=data.crs.data)

# Print crs
geo.crs

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


coords = getFeatures(geo)
print(coords)


# Clip the raster with Polygon
out_img, out_transform = mask(dataset=data, shapes=coords, crop=True)


# Copy the metadata
out_meta = data.meta.copy()
print(out_meta)



# Parse EPSG code
epsg_code = int(data.crs.data['init'][5:])
print(epsg_code)

out_meta.update({"driver": "GTiff",
                 "height": out_img.shape[1],
                 "width": out_img.shape[2],
                 "transform": out_transform,
                 "crs": pycrs.parser.from_epsg_code(epsg_code).to_proj4()}
                         )


with rasterio.open(out_tif, "w", **out_meta) as dest:
        dest.write(out_img)
        

# Open the clipped raster file
clipped = rasterio.open(out_tif)

# Visualize
show((clipped, 5), cmap='terrain')
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

#os.getcwd()
in_data_dir = r'.\data'

#using repojrected dem 
with rasterio.open(in_data_dir + './reproject_corpuschristi_dem.tif', 'r') as dem_file:
    dem = dem_file.read(1)
    dem_profile = dem_file.profile
    show(dem[1100:8350,1100:8350])
    dem_profile
    dem.shape
    #calculates the slope and aspect using the DEM with a cell size of 30
    #as part of the analysis, we could determine the slope and aspect of each suitable parcel
    s,a = slopeAspect(dem, 30)
    #recalculates the aspect to 8 cardinal directions using the fuction 
    aspect = reclassAspect(a)
    #using a bin size of 10, recalculate the slope grid into 10 classes using the function
    s = reclassByHisto(s, 10)
    #row first, last row. first column, last column
    show(dem[1100:8350,1100:8350])
    dem_profile
    dem.shape
    print(s,a)
#set bounds to dem extents
#create a polygon frame and use geopandas clip
#site availablity section
#remove O&G wells, pipelines, coastal leases from study area

#reproject rasters into same CRS
dem_projected = reproject_r(in_data_dir + './corpuschristi_dem.tif', in_data_dir +'./reproject_corpuschristi_dem.tif')
wind_projected = reproject_r(in_data_dir + './wtk_conus_100m_mean_masked.tif', in_data_dir +'./reproject_wtk_conus_100m_mean_masked.tif')

#open with rasterio the shipping districts
with rasterio.open(in_data_dir + './rasters/navdist_ras.tif', 'r') as nav_dist:
    nav = nav_dist.read(1)
    nav_profile = nav_dist.profile
    show(nav)
    nav_profile
    nav.shape




shipping = gpd(os.path.join(in_data_dir, 'shippinglanes/shippinglanes.shp'))
input_shp = ogr.Open(shipping)
    #getting layer information of shapefile.
shp_layer = input_shp.GetLayer()

dataSource = ogr.Open(shipping)
daLayer = dataSource.GetLayer(0)
layerDefinition = daLayer.GetLayerDefn()
for i in range(layerDefinition.GetFieldCount()):
    print(layerDefinition.GetFieldDefn(i).GetName())


bufferdist = 500
shipping_buff = shipping.buffer(500)


#nearest neighbor
#compute distance to nearest wind farm

#convert shapefile to raster


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
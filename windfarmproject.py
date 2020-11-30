<<<<<<< HEAD
import rasterio
import pandas as pd
import geopandas as gpd
import scipy
import glob
from netCDF4 import Dataset
import os


os.getcwd()

with rasterio.open(in_data_dir + './corpuschristi_dem.tif', 'r') as dem_file:
    dem = dem_file.read(1)
    dem_profile = dem_file.profile
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
=======
import rasterio
import pandas as pd
import geopandas as gpd
import scipy
import glob
import os


os.getcwd()

with rasterio.open(in_data_dir + './corpuschristi_dem.tif', 'r') as dem_file:
    dem = dem_file.read(1)
    dem_profile = dem_file.profile
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
>>>>>>> 395c9cd0030662d4347ee091f555bfcab351699b

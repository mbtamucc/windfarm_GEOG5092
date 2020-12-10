import numpy as np
import os
from rasterio.plot import show, show_hist
import glob
import scipy
import pandas as pd
import geopandas as gpd
#import netCDF4 as nc
import numpy as np
import fiona
import matplotlib
import matplotlib.pyplot as plt
import rioxarray as rxr
import rasterio 
import pandas as pd 
import geopandas as gpd
import numpy as np
import rioxarray as rxr
import os
from shapely.geometry import Point, LineString, Polygon, box, shape
from rasterio.mask import mask
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from fiona.crs import from_epsg

in_data_dir = r'.\data'

#this file contains 4 functions:
#reproject vector layers
#reproject raster layers 
#parse features from gdb for improved rasterio readability
#convert shapefiles to rasters

def reproject (layer):
    "This is the function that reproject the vector layers"
    master_crs = 'EPSG:26914'
    if layer.crs != master_crs:
        #print(list_of_lables[idx], 'is not on the master projection, reprojecting now..')
        layer = layer.to_crs(master_crs) 
        print (layer.crs, 'and', master_crs)
    
    return layer

#geocube function to convert vector to raster
def vector (input,output):
    from geocube.api.core import make_geocube
    grid = make_geocube(
        vector_data=merged_gpd,
        resolution=(100, 100),
    )
    grid.present.rio.to_raster(output, compress="DEFLATE")
    return(output)
    
#reproject and resample
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

wind_clip = mask('./data/reproject_wtk_conus_100m_mean_masked.tif', './data/clip_reproject_wtk_conus_100m_mean_masked.tif') 

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

def slopeAspect(dem, cs):
    """Calculates slope and aspect using the 3rd-order finite difference method

    Parameters
    ----------
    dem : numpy array
        A numpy array of a DEM
    cs : float
        The cell size of the original DEM

    Returns
    -------
    numpy arrays
        Slope and Aspect arrays
    """

    from math import pi
    from scipy import ndimage
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    dzdx = ndimage.convolve(dem, kernel, mode='mirror') / (8 * cs)
    dzdy = ndimage.convolve(dem, kernel.T, mode='mirror') / (8 * cs)
    slp = np.arctan((dzdx ** 2 + dzdy ** 2) ** 0.5) * 180 / pi
    ang = np.arctan2(-dzdy, dzdx) * 180 / pi
    aspect = np.where(ang > 90, 450 - ang, 90 - ang)
    return slp, aspect


def reclassAspect(npArray):
    """Reclassify aspect array to 8 cardinal directions (N,NE,E,SE,S,SW,W,NW),
    encoded 1 to 8, respectively (same as ArcGIS aspect classes).

    Parameters
    ----------
    npArray : numpy array
        numpy array with aspect values 0 to 360

    Returns
    -------
    numpy array
        numpy array with cardinal directions
    """
    return np.where((npArray > 22.5) & (npArray <= 67.5), 2,
    np.where((npArray > 67.5) & (npArray <= 112.5), 3,
    np.where((npArray > 112.5) & (npArray <= 157.5), 4,
    np.where((npArray > 157.5) & (npArray <= 202.5), 5,
    np.where((npArray > 202.5) & (npArray <= 247.5), 6,
    np.where((npArray > 247.5) & (npArray <= 292.5), 7,
    np.where((npArray > 292.5) & (npArray <= 337.5), 8, 1)))))))


#work on resample


def reclassByHisto(npArray, bins):
    """Reclassify np array based on a histogram approach using a specified
    number of bins. Returns the reclassified numpy array and the classes from
    the histogram.

    Parameters
    ----------
    npArray : numpy array
        Array to be reclassified
    bins : int
        Number of bins

    Returns
    -------
    numpy array
        umpy array with reclassified values
    """
    histo = np.histogram(npArray, bins)[1]
    rClss = np.zeros_like(npArray)
    for i in range(bins):
        rClss = np.where((npArray >= histo[i]) & (npArray <= histo[i + 1]),
                         i + 1, rClss)
    return rClss


def clip_and_buffer_geodataframes(list_gdfs, 
                                  list_of_labels, 
                                  boundary, 
                                  master_crs, 
                                  buffer_dist=4400):
   #every time you write a funciton, be sure to explain that funciton. this one using numpy
    """Given a number of geodataframes, clips each one against the park boundary,
    perform a buffer operation using the provided distance, and dissolve all
    buffer into a single poly.

    Parameters
    ----------
    list_gdfs : list
        List of geodataframes.
    list_of_labels : list
        List of labels.
    boundary : Geopandas Geodataframe
        Park boundary
    master_crs : CRS
        CRS information to be use on all re-project operations.
        Final geodataframe will be on this projection.
    buffer_dist : int, optional
        Buffer distance in meters, by default 100

    Returns
    -------
    Geopandas Geodataframe
        Geodataframe containing the buffer.
    """
    buffer_geoms = []

    for idx, e in enumerate(list_gdfs):
        gdf_buffer = gpd.GeoDataFrame()

        if e.crs != master_crs:
            print(list_of_labels[idx], 'is not on the master projection, reprojecting now...')
            e = e.to_crs(master_crs)

        # Clip each feature to the park bnd, this reduces the number of features we are dealing with.
        print('Clipping', list_of_labels[idx])
        e = gpd.overlay(e, boundary, how='intersection')

        print('Buffering', list_of_labels[idx])
        gdf_buffer['geometry'] = e.buffer(buffer_dist)
        gdf_buffer['source'] = list_of_labels[idx]

        # Dissolving each feature after the buffer improved performance by a lot!
        print('Dissolving', list_of_labels[idx])
        gdf_buffer = gdf_buffer.dissolve(by='source')

        buffer_geoms.append(gdf_buffer)

    unioned_gdf = pd.concat(buffer_geoms)
    unioned_gdf.crs = master_crs

    return unioned_gdf

def polygonize(self, data=None, mask=None, connectivity=4, transform=None):
        """
        Yield (polygon, value) for each set of adjacent pixels of the same value.
        Wrapper around rasterio.features.shapes

        From rasterio documentation:

        Parameters
        ----------
        data : numpy ndarray
        mask : numpy ndarray
               Values of False or 0 will be excluded from feature generation.
        connectivity : 4 or 8 (int)
                       Use 4 or 8 pixel connectivity.
        transform : affine.Affine
                    Transformation from pixel coordinates of `image` to the
                    coordinate system of the input `shapes`.
        """
       
        if data is None:
            data = self.mask.astype(np.uint8)
        if mask is None:
            mask = self.mask
        if transform is None:
            transform = self.affine
        shapes = rasterio.features.shapes(data, mask=mask, connectivity=connectivity,
                                          transform=transform)
        return shapes 


#polgyonize 
def polygonize(raster_file, vector_file, driver, mask_value):
    
    with rasterio.drivers():
        
        with rasterio.open(raster_file) as src:
            image = src.read(1)
        
        if mask_value is not None:
            mask = image == mask_value
        else:
            mask = None
        
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) 
            in enumerate(
                shapes(image, mask=mask, transform=src.affine)))

        with fiona.open(
                vector_file, 'w', 
                driver=driver,
                crs=src.crs,
                schema={'properties': [('raster_val', 'int')],
                        'geometry': 'Polygon'}) as dst:
            dst.writerecords(results)
    
    return dst.name
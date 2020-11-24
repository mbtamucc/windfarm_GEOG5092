import os
import pandas as pd
import geopandas as gpd
import fiona
import random
from rasterio.plot import  show, show_hist
from rasterio.mask import mask
from shapely.geometry import Polygon
from matplotlib import pyplot
import rasterio
import glob
import os
import numpy as np

in_data_dir = '/Users/jonathanburton/Desktop/Fall2020/Geog5092/final_proj/data'

shipping = gpd.read_file(os.path.join(in_data_dir, 'shippinglanes/shippinglanes.shp'))
bufferdist = 500
shipping_buff = shipping.buffer(500)
shipping_buff.plot())

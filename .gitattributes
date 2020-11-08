# Auto detect text files and perform LF normalization
* text=auto

def reprogject (layer):
    master_crs = 'EPSG:26914'
    if layer.crs != master_crs:
        print(list_of_lables[idx], 'is not on the master projection, reprojecting now..')
        layer = layer.to_crs(master_crs) 
        print (layer.crs, 'and', master_crs)
    
    return layer
        
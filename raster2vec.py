import numpy as np
import scipy.spatial as spatial
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import GeometryCollection
from shapely.geometry import mapping
from osgeo import gdal
from osgeo import osr
import geojson
from shapely.ops import linemerge
from skimage.morphology import thin
from skimage import img_as_ubyte
import cv2 
import os 


def raster2vec(outputpath, inimage, fpnumber, src):
    
    if fpnumber.endswith('.png'):
        fpnumberr = fpnumber[:-4]  
    else: 
        fpnumberr = fpnumber       
    outresult = inimage.split(".")[0]+".geojson"
    
    if 'win' in str(inimage):
        os.chdir(outputpath + fpnumberr + "/")
        outresult = str(fpnumberr) + '_window.geojson'
    elif 'wall' in str(inimage):
        os.chdir(outputpath + fpnumberr + "/")
        outresult = str(fpnumberr) + '_wall.geojson'
    elif 'stair' in src:
        os.chdir(outputpath + fpnumberr + "/")
        outresult = str(fpnumberr) + '_stair.geojson'
    else:
        os.chdir(outputpath + fpnumberr + "/")
        outresult = str(fpnumberr) + '_lift.geojson'
    
    src_ds = gdal.Open(outputpath + fpnumberr + '/' + fpnumberr + src) #read image
    originX, pixelWidth, x_rotation, originY, y_rotation, pixelHeight = src_ds.GetGeoTransform() #Image GeoTransform
    
    # Image EPSG
    proj = osr.SpatialReference(wkt=src_ds.GetProjection())
    crs = "EPSG:{}".format(proj.GetAttrValue('AUTHORITY',1))
    
    img = src_ds.ReadAsArray() #Image as array
    imgbin = img == 255 #Create a boolean binary image
    imgbin = imgbin[1, :, :]
    img_thin = thin(imgbin) #The thinned image
    if img_thin.sum() == 0:
        return False
    img_thin = img_as_ubyte(img_thin) #Convert to 8-bit uint (0-255 range)
    img_thin = np.where(img_thin == 255) #this will return the indices of white pixels
    
    
    # Calculate the center point of each cell
    points = []
    for i,idY in enumerate(img_thin[0]):
        idX = img_thin[1][i]
        cX = idX
        cY = idY
        points.append((cX, cY))
   
    maxD = np.sqrt(2.0*pixelWidth*pixelWidth)
    tree = spatial.cKDTree(points)
    groups = tree.query_ball_tree(tree, maxD)
    
    out_lines = []
    for x, ptlist in enumerate(groups):
        for pt in ptlist:
            if pt>x:            
                out_lines.append(LineString([Point(points[x]), Point(points[pt])]))    
    
    # Merge the contiguous lines 
    merged_lines = linemerge(MultiLineString(out_lines))
    
    # Define the crs
    crs = {
        "type": "name",
        "properties": {
            "name": crs
        }
    }
    
    # Create a feature collection of linestrings
    feature_collection = []
    if isinstance(merged_lines,LineString):
        merged_lines = merged_lines.simplify(maxD, True)
        feature_collection.append(geojson.Feature(geometry=geojson.LineString(merged_lines.coords[:])))
    else:    
        for line in merged_lines:
            # Simplify the result to try to remove zigzag effect
            line = line.simplify(maxD, True)
            feature_collection.append(geojson.Feature(geometry=geojson.LineString(line.coords[:])))                                  
                   
    feature_collection = geojson.FeatureCollection(feature_collection,crs=crs)
    
    # Save the output to a geosjon file
    with open(outresult, 'w') as outfile:
        os.chdir(outputpath + fpnumberr + '/')
        geojson.dump(feature_collection, outfile)
    return feature_collection


# #===================================================================================

# mainpath = 'C:/Users/user/Desktop/module3_vec/module3/'   
# filepath = 'C:/Users/user/Desktop/module3_vec/module3/test_input/'
# outputpath = 'C:/Users/user/Desktop/module3_vec/module3/test_output/'
# vecoutput = 'C:/Users/user/Desktop/module3_vec/module3/test_output/1-1.png/line/10-1vec.shp'
# separated_path = 'C:/Users/user/Desktop/module3_vec/module3/separated_output/'
# fpnumberr = '1-1' 

# inimage = outputpath + str(fpnumberr) + '/' + str(fpnumberr) + '_wall_bin.bmp'
# a = raster2vec(outputpath, inimage, fpnumberr)
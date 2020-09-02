
# coding: utf-8

# # 라이브러리 Import

# In[1]:


import geopandas as gpd
from geopandas.geoseries import Point
from shapely.geometry import Point, LineString, shape, mapping
from shapely.ops import polygonize, split
import json
import fiona
import pandas
from osgeo import ogr
from fiona import collection
from pathlib import Path
import sys


# # 창문, 문, 벽체 레이어 결합하여 하나의 Polyline 레이어 생성

# In[2]:


def merge_layer(folder, merged):
    
    folder = Path(folder)
    
    # 입력한 폴더 내 .shp 파일 전체 merge
    shapefiles = folder.glob("*.shp")
    gdf = pandas.concat([
        gpd.read_file(shp)
        for shp in shapefiles
    ]).pipe(gpd.GeoDataFrame)
    
    # 결과 merged 레이어 생성
    gdf.to_file(merged)


# # 벡터라이징에서 생성된 Intersection으로 line 분할하여 segment 레이어 생성

# In[3]:


# wall line을 intersection point로 split
def split_wall(wall_line, wall_point, wall_splited): #xxxxx
        # point 레이어와 해당 point 레이어를 기준으로 split할 line 레이어 입력
        lines = gpd.GeoDataFrame.from_file(wall_line)
        points = gpd.GeoDataFrame.from_file(wall_point)
        
        # shapely library의 union(전체 객체를 multi part 객체로 union)과 split 함수 사용
        line = lines.geometry.unary_union
        point = points.geometry.unary_union
        split_result = split(line, point)
        
        # fiona library 활용하여 shapefile write
        schema = {'geometry': 'LineString','properties': {'id': 'int'}}
        with fiona.open(wall_splited, 'w', 'ESRI Shapefile', schema) as c:
            for poly_id,polyline in enumerate(split_result):
                c.write({
                    'geometry': mapping(polyline),
                    'properties': {'id': poly_id},
                })


# # 공간화(폴리곤화) 1 - 창문, 문, 벽체 레이어의 공간화

# In[4]:


def wall_polygonizing(wall_splited, space):

        # ogr library 활용 데이터 read
        file = ogr.Open(wall_splited)
        shape = file.GetLayer(0)
        feature = shape.GetFeature(0)
        lines = []

        # 입력 레이어의 feature를 json으로 읽은 후 각 segment의 양 끝점의 좌표를 list로 변환
        for feature in shape:
            first = feature.ExportToJson()
            data = json.loads(first)
            start_n = (data['geometry']['coordinates'][0][0],data['geometry']['coordinates'][0][1])
            end_n = (data['geometry']['coordinates'][1][0],data['geometry']['coordinates'][1][1])
            line = [(start_n, end_n)]
            lines = lines+line

        # shapely library의 polygonize 함수 사용
        poly = polygonize(lines)
        
        # fiona library 활용하여 shapefile write
        schema = {'geometry': 'Polygon','properties': {'id': 'int', 'space_f' : 'str','floor_num' : 'int','instl_type' : 'str', 'ru_point' : 'float','Auto_YN': 'int'}}
        with fiona.open(space, 'w', 'ESRI Shapefile', schema) as c:
            for poly_id,polygon in enumerate(poly):
                
                c.write({
                    'geometry': mapping(polygon),
                    'properties': {'id': poly_id, 'space_f' : '','floor_num' : '','instl_type' : '', 'ru_point' : '','Auto_YN': ''}, 
                })
                


# # 공간화(폴리곤화) 2 - 엘리베이터, 계단 레이어의 공간화

# In[5]:


def bi_polygonizing(wall_splited, space):

        # ogr library 활용 데이터 read
        file = ogr.Open(wall_splited)
        shape = file.GetLayer(0)
        feature = shape.GetFeature(0)
        lines = []

        # 입력 레이어의 feature를 json으로 읽은 후 각 segment의 양 끝점의 좌표를 list로 변환
        for feature in shape:
            first = feature.ExportToJson()
            data = json.loads(first)
            start_n = (data['geometry']['coordinates'][0][0],data['geometry']['coordinates'][0][1])
            end_n = (data['geometry']['coordinates'][1][0],data['geometry']['coordinates'][1][1])
            line = [(start_n, end_n)]
            lines = lines+line

        # shapely library의 polygonize 함수 사용
        poly = polygonize(lines)
        
        # fiona library 활용하여 shapefile write
        schema = {'geometry': 'Polygon','properties': {'id': 'int', 'instl_type' : 'int', 'space_f' : 'str','space_id' : 'int','lift_id' : 'int','stair_id' : 'int','lift_ent_id' : 'int','stair_ent_id' : 'int','width' : 'float','height' : 'float','floor_num' : 'int','start_floor' : 'int', 'end_floor' : 'int','mng_id': 'int','stair_num' : 'int','in_out' : 'int','Auto_YN': 'int'}}
        with fiona.open(space, 'w', 'ESRI Shapefile', schema) as c:
            for poly_id,polygon in enumerate(poly):
                
                c.write({
                    'geometry': mapping(polygon),
                    'properties': {'id': poly_id, 'instl_type' : '', 'space_f' : '','space_id' : '','lift_id' : '','stair_id' : '','lift_ent_id' : '','stair_ent_id' : '','width' : '','height' : '','floor_num' : '','start_floor' : '', 'end_floor' : '','mng_id': '','stair_num' : '','in_out' : '','Auto_YN': ''}, 
                })


# # 실명 텍스트 공간 레이어에 입력

# In[6]:


#일단 빼기


#폴리곤과 포인트 중첩   
def room_label(space,label,output):    
    # join할 space 레이어와 label 레이어 read
    label = gpd.GeoDataFrame.from_file(label)
    space = gpd.GeoDataFrame.from_file(space) 
    
    # geopandas의 spatial join 함수(.sjoin)를 통해 space 레이어와 label 레이어 join(label이 space에 포함될 경우 join) 
    pointInPoly = gpd.sjoin(space, label,  op='contains', how = 'left') 
    pointInPoly = pointInPoly[['id','space_f_right','geometry','floor_num','instl_type','Auto_YN']]
    pointInPoly = pointInPoly.rename(columns={'space_f_right':'space_f'})
    pointInPoly.to_file(output, driver='ESRI Shapefile')
    


# # 실행 코드1 (벽체, 문, 창문)

# In[9]:


# wall, door, window layer가 위치한 folder
folder = "E:/data/polygonize/test"
# merge file(merge 결과 파일 경로 및 이름)
merged = "E:/data/polygonize/test/merged_wall.shp"
# 벡터라이징 시 생성한 intersection point
wall_point = "E:/data/polygonize/test/point/903-3point.shp"
# splited wall line(segment 결과 파일 경로 및 이름)
wall_splited = "E:/data/polygonize/test/merged_wall_splited.shp"
# space polygon(공간화 결과 파일 경로 및 이름)
space = "E:/data/polygonize/test/903-3_space.shp"
# room label point layer(실명 추출한 결과 데이터)
label = "E:/data/polygonize/test/text/903-3label.shp"
# final space output
space_w_label = "E:/data/polygonize/test/903-3_space_w_label.shp"

merge_layer(folder, merged)
split_wall(merged, wall_point, wall_splited)
wall_polygonizing(wall_splited, space)
room_label(space, label, space_w_label)


# # 실행 코드2 (엘리베이터, 계단)

# In[12]:


# 엘리베이터 레이어
# bi file(lift, stair polyline파일 경로 및 이름)
bi= "E:/data/polygonize/test/bi/lift.shp"
# space polygon(공간화 결과 파일 경로 및 이름)
space = "E:/data/polygonize/test/bi/lift_space.shp"
# room label point layer(실명 추출한 결과 데이터)
label = "E:/data/polygonize/test/text/903-3label.shp"
# final space output
space_w_label = "E:/data/polygonize/test/bi/903-3_lift_w_label.shp"

bi_polygonizing(bi, space)
room_label(space, label, space_w_label)

# 계단 레이어
# bi file(lift, stair polyline파일 경로 및 이름)
bi2= "E:/data/polygonize/test/bi/stair.shp"
# space polygon(공간화 결과 파일 경로 및 이름)
space2 = "E:/data/polygonize/test/bi/stair_space.shp"
# room label point layer(실명 추출한 결과 데이터)
label2 = "E:/data/polygonize/test/text/903-3label.shp"
# final space output
space_w_label2 = "E:/data/polygonize/test/bi/903-3_stair_w_label.shp"

bi_polygonizing(bi2, space2)
room_label(space2, label2, space_w_label2)



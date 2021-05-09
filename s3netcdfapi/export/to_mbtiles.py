import mercantile
import json
import numpy as np
from mbtilesapi import Points2VT,send,VT2Tile
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
import numpy_groupies as npg
from geojson import Point, Feature, FeatureCollection, dumps

def getXY(lng,lat):
    r_major = 6378137.000
    x = r_major * np.radians(lng)
    scale = x/lng
    y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + lat * (np.pi/180.0)/2.0)) * scale
    return x, y

def getData(obj,data):
  # print(data)
  variable=data[obj['variable'][0]]
  values=np.squeeze(variable['data'])
  # print(variable['dimData']['node'].keys())
  lng=variable['dimData']['node']['subdata']['x']['data']
  lat=variable['dimData']['node']['subdata']['y']['data']
  
  ids=np.arange(len(lng))
  
  x,y=getXY(lng,lat)
  xy = np.column_stack((x,y))
  
  x=obj['mx']
  y=obj['my']
  z=obj['mz']
  # y = (1 << z) - 1 - y
  
  box=mercantile.xy_bounds(x, y, z)
  llbox=mercantile.bounds(x, y, z)
  
  ll = np.array([box.left, box.bottom])
  ur = np.array([box.right, box.top])
  
  inidx = np.all(np.logical_and(ll <= xy, xy <= ur), axis=1)
  outbox = xy[np.logical_not(inidx)]
  
  ids=ids[inidx]
  lng=lng[inidx]
  lat=lat[inidx]
  values=values[inidx]
  lnglat=np.column_stack((lng,lat))
  
  maxP=1E5
  if ids.size>maxP:
    
    
    dist=(llbox.east-llbox.west)/2048
    
    tree = cKDTree(lnglat)
    c=tree.query_ball_point(lnglat,dist,return_length=True)
    count=np.sum(c==1)
    
    ori_ids=ids[c==1]
    ori_ll=lnglat[c==1]
    clu_ll=lnglat[c!=1]
    
    ori_values=values[c==1]
    clu_values=values[c!=1]
    
    centroid, label = kmeans2(clu_ll, int(maxP-count), minit='points')
    noagg_values=clu_values[label]
    agg_values=npg.aggregate(label, noagg_values, func='sum')
    agg_ids=np.zeros(len(agg_values),dtype="i")-1
    
    ids=np.concatenate((ori_ids, agg_ids))
    lnglat=np.concatenate((ori_ll, centroid))
    values=np.concatenate((ori_values, agg_values))
  
  
  
  
  
  return ids,lnglat,values


def getGeojson(obj,ids,lnglat,values):
  
  feat_list=[]
  for i in range(len(ids)):
    f = Feature(geometry=Point((float(lnglat[i][0]), float(lnglat[i][1]))),
               properties = {"id":int(ids[i]),"value":float(values[i]),"clustered":int(bool(ids[i]==-1))})
    feat_list.append(f)
  collection = FeatureCollection(feat_list)
  
  filepath=obj['filepath']+".geojson"
  with open(filepath, 'w') as outfile:
    json.dump(collection,outfile,indent=0)
  return filepath

# def to_vt(obj,data):
#   return Points2VT(*getData(obj,data))
  
def to_mbtiles(obj,data):
  return getGeojson(obj,*getData(obj,data))
  # vt=Points2VT(*getData(obj,data))
  # return send(VT2Tile(vt))
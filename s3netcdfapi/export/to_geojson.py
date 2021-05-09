import json
import numpy as np
from .table import to_table

def to_geojson(obj,data):
  geojson = {'type':'FeatureCollection', 'features':[]}
  _x=['lon','x','lng','longitude','Longitude']
  _y=['lat','y','latitude','Latitude']
  
  if obj['isTable']:
    df,xyHeaders=to_table(obj,data,True)
    xname=next(key for key in xyHeaders.keys() if key in _x )
    yname=next(key for key in xyHeaders.keys() if key in _y )
    for _, row in df.iterrows():
      feature = {'type':'Feature','properties':json.loads(row.to_json()), 'geometry':
        {'type':'Point','coordinates':[row[xyHeaders[xname]],row[xyHeaders[yname]]]}}
      geojson['features'].append(feature)
  else:
    xname=_x[_x.index(data.keys())]
    yname=_y[_y.index(data.keys())]
    x=data[xname]
    y=data[yname]
    elem=data['elem']
    _p=np.stack((x[elem],y[elem])) # shape=(2,nelem,3)
    _p=np.einsum('ijk->jki', _p) # shape=(nelem,3,2)
    
    for coordinates in _p:
      coordinates=coordinates.flatten()
      feature = {'type':'Feature','properties':{}, 'geometry':
        {'type':'Polygon','coordinates':list(coordinates)}}
      geojson['features'].append(feature)
    
  filepath=obj['filepath']+".geojson"
  with open(filepath, 'w') as outfile:
    json.dump(geojson,outfile,indent=2)
  return filepath
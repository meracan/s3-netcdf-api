import json
import numpy as np
from .table import to_table

def to_geojson(obj,data):
  geojson = {'type':'FeatureCollection', 'features':[]}
  if obj['isTable']:
    df,xyHeaders=to_table(obj,data,True)
    for _, row in df.iterrows():
      feature = {'type':'Feature','properties':json.loads(row.to_json()), 'geometry':
        {'type':'Point','coordinates':[row[xyHeaders['x']],row[xyHeaders['y']]]}}
      geojson['features'].append(feature)
  else:
    x=data['x']
    y=data['y']
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
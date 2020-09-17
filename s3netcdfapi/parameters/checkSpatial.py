import numpy as np

def checkSpatial(netcdf2d,obj):
  """
  """
  if obj['longitude'] is not None:obj['x']=obj['longitude'] # Test1
  if obj['latitude'] is not None:obj['y']=obj['latitude']
  del obj['longitude']
  del obj['latitude']
  
  obj['xy']=None  
  
  if obj['inode'] is not None: # Test3
    if not isinstance(obj['inode'],list):obj['inode']=[obj['inode']]
    obj['x']=None
    obj['y']=None
  elif obj['x'] is not None or obj['y'] is not None:
    if obj['x'] is None or obj['y'] is None:raise Exception("x/longitude must be equal to y/latitude") 
    if obj['x'] is not None and not isinstance(obj['x'],list):obj['x']=[obj['x']] # Test1
    if obj['y'] is not None and not isinstance(obj['y'],list):obj['y']=[obj['y']]
    if len(obj['x']) !=len(obj['y']):raise Exception("x/longitude must be equal to y/latitude")
    obj['xy']=np.column_stack((obj['x'],obj['y']))
  
  return obj
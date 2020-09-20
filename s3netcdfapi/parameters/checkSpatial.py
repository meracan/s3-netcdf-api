import numpy as np
from matplotlib.tri import Triangulation
from scipy.spatial import cKDTree
from .utils import getIdx

def getMesh(netcdf2d,obj):
  if obj['meshx'] is None:
    dt=netcdf2d.query({'variable':'time'})
    obj['meshx']=netcdf2d.query(getIdx(obj,'meshx'))
  if obj['meshy'] is None:
    obj['meshy']=netcdf2d.query(getIdx(obj,'meshy'))
  if obj['elem'] is None:
    obj['elem']=netcdf2d.query(getIdx(obj,'elem'))
  return obj

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
    if obj['interpolation']['spatial']=='closest':obj=closest(netcdf2d,obj)
    elif obj['interpolation']['spatial']=='linear':obj=linear(netcdf2d,obj)
    else:raise Exception("closest and linear are exepted ({})".format(obj['interpolation']['spatial']))
  
  return obj

def linear(netcdf2d,obj):
  obj=getMesh(netcdf2d,obj)
  tri = Triangulation(obj['meshx'], obj['meshy'], obj['elem'].astype("int32"))
  trifinder = tri.get_trifinder()
  ielem=trifinder.__call__(obj['x'], obj['y'])
  idx=obj['elem'][ielem].astype("int32")
  inode=np.unique(idx)
  obj['_x']=obj['x'][inode]
  obj['_y']=obj['y'][inode]
  obj['_elem']=obj['elem'][ielem]
  obj['inode']=inode
  return obj

def closest(netcdf2d,obj):
  obj=getMesh(netcdf2d,obj)
  xy=np.column_stack((obj['meshx'],obj['meshy']))
  kdtree = cKDTree(xy)
  distance,inode=kdtree.query(obj['xy'],1)
  inode,xyIndex=np.unique(inode.ravel(),return_inverse=True)
  obj['inode']=inode
  obj['xyIndex']=xyIndex
  return obj
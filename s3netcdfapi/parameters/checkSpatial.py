import numpy as np
from matplotlib.tri import Triangulation
from scipy.spatial import cKDTree
from .utils import getIdx
# x      = User coordinate-x
# y      = User coordinate-y
# _meshx = All  coordinate-x from mesh
# _meshy = All  coordinate-y from mesh
# _elem  = All  elements from mesh
# meshx  = Selected coordinate-x from mesh
# meshy  = Selected coordinate-y from mesh
# elem   = Selected elements from mesh

def getMesh(netcdf2d,obj):
  if not '_meshx' in obj or obj['_meshx'] is None:obj['_meshx']=netcdf2d.query({"variable":"x"})
  if not '_meshy' in obj or obj['_meshy'] is None:obj['_meshy']=netcdf2d.query({"variable":"y"})
  if not '_elem' in obj or obj['_elem'] is None: obj['_elem']=netcdf2d.query({"variable":"elem"})
  return obj

def checkSpatial(netcdf2d,obj):
  """
  """
  if obj['longitude'] is not None:obj['x']=obj['longitude'] # Test1
  if obj['latitude'] is not None:obj['y']=obj['latitude']
  del obj['longitude']
  del obj['latitude']
  
    
  obj['user_xy']=False
  if obj['inode'] is not None: # Test3
    if isinstance(obj['inode'],(int)):obj['inode']=[obj['inode']]
    obj['x']=None
    obj['y']=None
    obj['xy']=None
  elif obj['x'] is not None or obj['y'] is not None:
    obj['user_xy']=True
    if obj['x'] is None or obj['y'] is None:raise Exception("x/longitude must be equal to y/latitude") 
    if obj['x'] is not None and not isinstance(obj['x'],list):obj['x']=[obj['x']] # Test1
    if obj['y'] is not None and not isinstance(obj['y'],list):obj['y']=[obj['y']]
    if len(obj['x']) !=len(obj['y']):raise Exception("x/longitude must be equal to y/latitude")
    obj['xy']=np.column_stack((obj['x'],obj['y']))
    if obj['inter.spatial']=='closest':obj=closest(netcdf2d,obj)
    elif obj['inter.spatial']=='linear':obj=linear(netcdf2d,obj)
    else:raise Exception("closest and linear are exepted ({})".format(obj['interpolation']['spatial']))
  
  return obj


def linear(netcdf2d,obj):
  """
  """
  obj=getMesh(netcdf2d,obj)
  tri = Triangulation(obj['_meshx'], obj['_meshy'], obj['_elem'].astype("int32"))
  trifinder = tri.get_trifinder()
  ielem=trifinder.__call__(obj['x'], obj['y'])
  idx=obj['_elem'][ielem].astype("int32")
  inode=np.unique(idx)
  obj['meshx']=obj['_meshx'][inode]
  obj['meshy']=obj['_meshy'][inode]
  obj['elem']=obj['_elem'][ielem]
  obj['inode']=inode
  return obj


def closest(netcdf2d,obj):
  """
  """  
  obj=getMesh(netcdf2d,obj)
  _meshxy=np.column_stack((obj['_meshx'],obj['_meshy']))
  kdtree = cKDTree(_meshxy)
  distance,inode=kdtree.query(obj['xy'],1)
  inode,xyIndex=np.unique(inode.ravel(),return_inverse=True)
  obj['inode']=inode
  obj['xyIndex']=xyIndex
  return obj
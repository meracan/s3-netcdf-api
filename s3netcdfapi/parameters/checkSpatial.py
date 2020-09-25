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
  mesh=netcdf2d.getMeshMeta()
  if not '_meshx' in obj or obj['_meshx'] is None:obj['_meshx']=netcdf2d.query({"variable":mesh['x']})
  if not '_meshy' in obj or obj['_meshy'] is None:obj['_meshy']=netcdf2d.query({"variable":mesh['y']})
  if not '_elem' in obj or obj['_elem'] is None: obj['_elem']=netcdf2d.query({"variable":mesh['elem']})
  return obj

def checkSpatial(netcdf2d,obj,dname,type='mesh'):
  """
  """
  if obj['longitude'] is not None:obj['x']=obj['longitude'] # Test1
  if obj['latitude'] is not None:obj['y']=obj['latitude']
  del obj['longitude']
  del obj['latitude']
  idname="i"+dname[1:]
    
  obj['user_xy']=False
  if obj[idname] is not None: # Test3
    if isinstance(obj[idname],(int)):obj[idname]=[obj[idname]]
    obj['x']=None
    obj['y']=None
    obj['xy']=None
  elif obj['x'] is not None or obj['y'] is not None:
    obj['user_xy']=True
    if obj['x'] is None or obj['y'] is None:raise Exception("x/longitude must be equal to y/latitude") 
    if obj['x'] is not None and not isinstance(obj['x'],list):obj['x']=[obj['x']] # Test1
    if obj['y'] is not None and not isinstance(obj['y'],list):obj['y']=[obj['y']]
    if len(obj['x']) !=len(obj['y']):raise Exception("x/longitude must be equal to y/latitude")
    obj['x']=np.array(obj['x'])
    obj['y']=np.array(obj['y'])
    obj['xy']=np.column_stack((obj['x'],obj['y']))
    if type=="mesh":
      if obj['inter.mesh']=='closest':obj=closest(netcdf2d,obj,idname)
      elif obj['inter.mesh']=='linear':obj=linear(netcdf2d,obj,idname)
      else:raise Exception("Unknown ({}) interpolation for inter.mesh".format(obj['inter.mesh']))
    elif type=="xy":
      if obj['inter.xy']=='closest':obj=closestXY(netcdf2d,obj,dname,idname)
      else:raise Exception("Unknown ({}) interpolation for inter.xy".format(obj['inter.xy']))
  return obj


def linear(netcdf2d,obj,idname):
  """
  """
  obj=getMesh(netcdf2d,obj)
  tri = Triangulation(obj['_meshx'], obj['_meshy'], obj['_elem'].astype("int32"))
  trifinder = tri.get_trifinder()
  ielem=trifinder.__call__(obj['x'], obj['y'])
  idx=obj['_elem'][ielem].astype("int32")
  inode,nodeIndex=np.unique(idx,return_inverse=True)
  obj['meshx']=obj['_meshx'][inode]
  obj['meshy']=obj['_meshy'][inode]
  obj['elem']=nodeIndex.reshape(idx.shape)
  obj[idname]=inode
  return obj


def closest(netcdf2d,obj,idname):
  """
  """  
  obj=getMesh(netcdf2d,obj)
  _meshxy=np.column_stack((obj['_meshx'],obj['_meshy']))
  kdtree = cKDTree(_meshxy)
  distance,inode=kdtree.query(obj['xy'],1)
  inode,nodeIndex=np.unique(inode.ravel(),return_inverse=True)
  obj[idname]=inode
  obj['nodeIndex']=nodeIndex
  return obj

def closestXY(netcdf2d,obj,dname,idname):
  """
  """
  vnames=netcdf2d.getVariablesByDimension(dname)
  x=next(x for x in vnames if x in obj['pointers']['xy']['x'])
  y=next(x for y in vnames if y in obj['pointers']['xy']['y'])
  
  obj['_x']=netcdf2d.query({"variable":x})
  obj['_y']=netcdf2d.query({"variable":y})

  sxy=np.column_stack((obj['_x'],obj['_y']))
  kdtree = cKDTree(sxy)
  distance,isnode=kdtree.query(obj['xy'],1)
  isnode,xyIndex=np.unique(isnode.ravel(),return_inverse=True)
  obj[idname]=isnode
  obj['xyIndex']=xyIndex
  return obj
from scipy.spatial import cKDTree
import numpy as np


def sInterpolate(netcdf2d,obj,variable):
  """
  """
  if obj['smethod']=='idw':return IDW(netcdf2d,obj,variable)
  elif obj['smethod']=='linear':return linear(netcdf2d,obj,variable)
  else: return closest(netcdf2d,obj,variable)

def IDW(netcdf2d,obj,variable,regularize_by=1e-9):
  """
  """  
  mesh=netcdf2d.getMesh()
  xy=np.column_stack((mesh['x'],mesh['y']))
  kdtree = cKDTree(xy)
  distances,ids=kdtree.query(obj['xy'],3)
  distances += regularize_by
  tmpData=netcdf2d.query({**obj,'variable':variable,'inode':ids.ravel()})
  weights = tmpData.reshape(ids.shape)
  return np.sum(weights/distances, axis=1) / np.sum(1./distances, axis=1)

def linear(netcdf2d,obj,variable):
  """
  """  
  None

def closest(netcdf2d,obj,variable):
  """
  """  
  None

def tInterpolate(netcdf2d,obj,variable):
  """
  """
  if obj['tmethod']=='closest':return tclosest(netcdf2d,obj,variable)
  else: return tlinear(netcdf2d,obj,variable)

def tlinear(netcdf2d,obj,variable):
  """
  """  
  None
def tclosest(netcdf2d,obj,variable):
  """
  """  
  None  

  

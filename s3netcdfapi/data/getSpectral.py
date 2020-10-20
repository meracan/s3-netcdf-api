import numpy as np
from matplotlib.tri import Triangulation
from scipy.spatial import cKDTree

# x      = User coordinate-x
# y      = User coordinate-y
# _meshx = All  coordinate-x from mesh
# _meshy = All  coordinate-y from mesh
# _elem  = All  elements from mesh
# meshx  = Selected coordinate-x from mesh
# meshy  = Selected coordinate-y from mesh
# elem   = Selected elements from mesh

def getSpectral(netcdf2d,obj,dname):
  """
  """
  
  if obj['station'] is not None:
    stationname=netcdf2d.query({"variable":netcdf2d.spectral['stationName']})
    stationids=np.where(obj['station']==stationname[:,np.newaxis])[0]
    if(len(stationids)==0):raise Exception("Station name(s) {} does not exist".format(obj['station']))
    _stationids=netcdf2d.query({"variable":netcdf2d.spectral['stationId']})
    isnode=np.where(stationids==_stationids[:,np.newaxis])[0]
    idname="i"+dname[1:]
    obj[idname]=isnode
  return obj


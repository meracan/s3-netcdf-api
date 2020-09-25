import numpy as np
from matplotlib.tri import Triangulation
from scipy.spatial import cKDTree

from .utils import getIdx

# sx      = User coordinate-x
# sy      = User coordinate-y
# sxy     = User coordinate-xy
# _sx     = Spectra point coordinate-x
# _sy     = Spectra point coordinate-y

def checkSpectral(netcdf2d,obj):
  """
  """
  if obj['slongitude'] is not None:obj['sx']=obj['slongitude'] # Test1
  if obj['slatitude'] is not None:obj['sy']=obj['slatitude']
  del obj['slongitude']
  del obj['slatitude']
  
    
  obj['user_sxy']=False
  if obj['isnode'] is not None: # Test3
    if isinstance(obj['isnode'],(int)):obj['isnode']=[obj['isnode']]
    obj['sx']=None
    obj['sy']=None
    obj['sxy']=None
  elif obj['sx'] is not None or obj['sy'] is not None:
    obj['user_sxy']=True
    if obj['sx'] is None or obj['y'] is None:raise Exception("x/longitude must be equal to y/latitude") 
    if obj['sx'] is not None and not isinstance(obj['sx'],list):obj['sx']=[obj['sx']] # Test1
    if obj['sy'] is not None and not isinstance(obj['sy'],list):obj['sy']=[obj['sy']]
    if len(obj['sx']) !=len(obj['sy']):raise Exception("sx/slongitude must be equal to sy/slatitude")
    obj['sx']=np.array(obj['sx'])
    obj['sy']=np.array(obj['sy'])
    obj['sxy']=np.column_stack((obj['sx'],obj['sy']))
    if obj['inter.spectral']=='closest':obj=closest(netcdf2d,obj)
    else:raise Exception("closest is exepted ({})".format(obj['interpolation']['spectral']))
  
  return obj



def closest(netcdf2d,obj):
  """
  """
  obj['_sx']=netcdf2d.query(getIdx(obj,'sx'))
  obj['_sy']=netcdf2d.query(getIdx(obj,'sy'))

  sxy=np.column_stack((obj['_sx'],obj['_sy']))
  kdtree = cKDTree(sxy)
  distance,isnode=kdtree.query(sxy,1)
  obj['isnode']=isnode
  return obj
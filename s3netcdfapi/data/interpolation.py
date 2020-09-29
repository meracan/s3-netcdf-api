from scipy.spatial import cKDTree
import numpy as np
from matplotlib.tri import Triangulation,LinearTriInterpolator
from scipy import interpolate
import sys
import time

def _checkBounds(_datetimes,datetimes):
  """
  """
  dt_min=np.min(datetimes)
  dt__min=np.min(_datetimes)
  dt_max=np.max(datetimes)
  dt__max=np.max(_datetimes)  
  if dt_min <dt__min:raise Exception("{} is below reference datetimes {}".format(dt_min,dt__min))
  if dt_max >dt__max:raise Exception("{} is above reference datetimes {}".format(dt_max,dt__max))

def timeSeries(_datetimes,datetimes,_data=None,bounds_error=True,kind='nearest'):
  """
  """
  if bounds_error:
    _checkBounds(_datetimes,datetimes)
  f = interpolate.interp1d(_datetimes.astype("f8"), _data,kind=kind,axis=0)
  return f(datetimes.astype("f8"))
        
def mesh(x,y,elem,data,_x,_y):
  """
  """
  tri = Triangulation(x, y, elem.astype("int32"))
  trifinder = tri.get_trifinder()
  
  if data.ndim==1:
    if len(data)!=len(x):raise Exception("x, y and data must be equal-length 1-D array")
    lti=LinearTriInterpolator(tri,data,trifinder)
    return lti(_x,_y)
  elif data.ndim==2:
    intdata=np.zeros((len(_x),data.shape[1]))
    for i in range(data.shape[1]):
      lti=LinearTriInterpolator(tri,data[:,i],trifinder)
      intdata[:,i]=lti(_x,_y)
    return intdata
  else:
    raise Exception("Not programmed")
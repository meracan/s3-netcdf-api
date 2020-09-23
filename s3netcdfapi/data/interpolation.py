from scipy.spatial import cKDTree
import numpy as np
from matplotlib.tri import Triangulation,LinearTriInterpolator

import sys

def _checkBounds(_datetimes,datetimes):
  """
  """
  dt_min=np.min(datetimes)
  dt__min=np.min(_datetimes)
  dt_max=np.max(datetimes)
  dt__max=np.max(_datetimes)  
  if dt_min <dt__min:raise Exception("{} is below reference datetimes {}".format(dt_min,dt__min))
  if dt_max >dt__max:raise Exception("{} is above reference datetimes {}".format(dt_max,dt__max))

def timeSeriesClosest(_datetimes,datetimes,_data=None,bounds_error=True):
  """
  """
  if bounds_error:
    _checkBounds(_datetimes,datetimes)
  
  i0=np.argsort(np.abs(_datetimes - datetimes[:, np.newaxis]))[:,0] # Closest index
  data = _data[i0]
  return data
  
  

def timeSeriesLinear(_datetimes,datetimes,_data=None,bounds_error=True):
  """
  Interpolate time-series
  
  Parameters
  ----------
  _datetimes:reference datetimes, np.datetime64
  datetimes:datetimes to interpolate,np.datetime64
  _data:(optional)reference data, np.ndarray
  
  Assumptions
  -----------
  Equal interval between time-step
  If bounds_error is False, automatic extrapolation based on nearest start-index or end-index
  _data= The first axis needs to be referecing datetimes
  
  Returns
  -------
  object(i0,w0,wi1,w1) if _data is None
  data:np.ndarray
  
  Examples
  --------
  _datetime = np.arange('2000-01-01', '2000-02-01', np.timedelta64(1, 'h'), dtype='datetime64[s]')
  datetime  = np.arange('2000-01-10T00:15', '2000-01-11', np.timedelta64(1, 'h'), dtype='datetime64[s]')
  -1D
    data=np.arange(len(_datetime))
  -2D
    data=np.arange(len(_datetime)*3).reshape(3,len(_datetime)).T
  inter.timeSeries(_datetime,datetime,data)
  
  """
  if bounds_error:
    _checkBounds(_datetimes,datetimes)
  
  _i0=np.argsort(np.abs(_datetimes - datetimes[:, np.newaxis]))[:,0] # Closest index
  _i1=np.argsort(np.abs(_datetimes - datetimes[:, np.newaxis]))[:,1] # Second closest
  
  i1=np.maximum(_i0,_i1) # id1 is the front index
  i0=np.minimum(_i0,_i1) # id0 is the back index
  
  t1=_datetimes[i1] # Front datetime
  t0=_datetimes[i0] # Back datetime
  
  w1 = (datetimes-t0) / (t1-t0) # Weight of front
  w0 = 1.0-w1  # Weight of back
  
  if _data is None:return {"i0":i0,"i1":i1,"w0":w0,"w1":w1}
  
  # If needed, expand dimensions to match _data
  for _ in range(_data.ndim-w1.ndim):
    w1=np.expand_dims(w1, axis=1)
    w0=np.expand_dims(w0, axis=1)

  data = _data[i1]*w1+_data[i0]*w0 # Interpolate
  return data

def barycentric(elem,x,y,p,_data=None):
  """
  Barycentric interpolation
  
  Parameters
  ----------
  _p:reference points,shape=(nelme,npe,nc)
    nelem:number of elements
    npe:number of points per elements
    nc:number of coordinates per point
  p:interpolate points, shape=(npoints,nc)
  _data:reference data, shape=(_npoints)
  """
  nelem=len(elem)
  npoint=len(p)
  _p=np.stack((x[elem],y[elem])) # shape=(2,nelem,3)
  _p=np.einsum('ijk->jki', _p) # shape=(nelem,3,2)
  a = _p[:, 0, :]
  b = _p[:, 1, :]
  c = _p[:, 2, :]
  
 
  v0 = b - a
  v1 = c - a
  v0 = v0[np.newaxis,:]
  v1 = v1[np.newaxis,:]
  
  d00 = (v0 * v0).sum(axis=2)
  d01 = (v0 * v1).sum(axis=2)
  d11 = (v1 * v1).sum(axis=2)
  denom = d00 * d11 - d01 * d01

  # Divide in parts to reduce memory
  step=int(np.ceil(1.0E7/nelem))
  nparts=int(np.ceil(npoint/step))
  
  isInside=np.zeros(npoint)
  iElem=np.zeros(npoint,dtype="int")
  weight=np.zeros((npoint,3))
  for _i in np.arange(nparts):
    idx=np.arange(_i*step,np.minimum((_i+1)*step,npoint))
    
    v2 = p[idx,np.newaxis] - a
    d20 = (v2 * v0).sum(axis=2)
    d21 = (v2 * v1).sum(axis=2)
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    
    l=np.logical_and(np.logical_and(u>=0.0,v>=0.0),w>=0.0)
    _isInside=np.any(l,axis=1)
    _iElem=np.argmax(l,axis=1) 
    weight_u=u[:,_iElem].diagonal()
    weight_v=v[:,_iElem].diagonal()
    weight_w=w[:,_iElem].diagonal()
    _weight=np.stack((weight_u,weight_v,weight_w)) # (npoints,(a,b,c))
    isInside[idx]=_isInside
    iElem[idx]=_iElem
    weight[idx]=_weight.T
  if _data is None:return isInside,iElem,weight
  return np.einsum('ij...,ij->i...',_data[elem[iElem]],weight)
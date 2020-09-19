import numpy as np
from .utils import getIdx


def checkTemporal(netcdf2d,obj):
  """ Check Temporal parameters
  """
  obj['dt']=None
  if obj['itime'] is not None:
    if not isinstance(obj['itime'],list):obj['itime']=[obj['itime']]
    obj['start']=None
    obj['end']=None
    obj['step']=None
  elif obj['start'] is not None or obj['end'] is not None:
    obj['datetime']=dt=netcdf2d.query(getIdx(obj,'time'))
    mindt=np.min(dt)
    maxdt=np.max(dt)
    
    if obj['start'] is None:obj['start']=mindt
    if obj['end'] is None:obj['end']=maxdt
    
    obj['start']=np.datetime64(obj['start']) #np.array(obj['start'],dtype='datetime64[h]')
    obj['end']=np.datetime64(obj['end']) #np.array(obj['end'],dtype='datetime64[h]')
    obj['step']=np.timedelta64(obj['step'], obj['stepUnit'])
    
    if obj['start']<mindt:raise Exception("{0} below limit of {1}".format(obj['start'],np.min(dt)))
    if obj['end']>maxdt:raise Exception("{0} below limit of {1}".format(obj['end'],np.max(dt)))
    
    obj['_datetime']=_datetime= np.arange(obj['start'], obj['end'],obj['step'], dtype='datetime64[h]')
    _i0=np.argsort(np.abs(dt - mindt))[0] # Closest index
    _i1=np.argsort(np.abs(dt - maxdt))[1] # Second closest
    
    i1=np.maximum(_i0,_i1) # id1 is the front index
    i0=np.minimum(_i0,_i1) # id0 is the back index
    obj['itime']=np.arange(i0,i1,dtype="int")
  
  return obj
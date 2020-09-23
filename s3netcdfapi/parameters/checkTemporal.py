import numpy as np
from .utils import getIdx


# _datetime      = All datetime
# datetime      = Selected datetime

def checkTemporal(netcdf2d,obj):
  """ Check Temporal parameters
  """
  obj['user_time']=False
  if obj['itime'] is not None:
    if isinstance(obj['itime'],int):obj['itime']=[obj['itime']]
    obj['start']=None
    obj['end']=None
    obj['step']=None
  elif obj['start'] is not None or obj['end'] is not None:
    obj['user_time']=True
    dt=obj['_datetime']=netcdf2d.query({"variable":"time"})
    mindt=np.min(dt)
    maxdt=np.max(dt)
    
    if obj['start'] is None:obj['start']=mindt
    if obj['end'] is None:obj['end']=maxdt
    
    obj['start']=np.datetime64(obj['start']) #np.array(obj['start'],dtype='datetime64[h]')
    obj['end']=np.datetime64(obj['end']) #np.array(obj['end'],dtype='datetime64[h]')
    obj['step']=np.timedelta64(obj['step'], obj['stepUnit'])
    
    if obj['start']<mindt:raise Exception("{0} below limit of {1}".format(obj['start'],np.min(dt)))
    if obj['end']>maxdt:raise Exception("{0} below limit of {1}".format(obj['end'],np.max(dt)))
    
    obj['datetime']= np.arange(obj['start'], obj['end'],obj['step'], dtype='datetime64[h]')
    
    # Get minimum index
    _s=np.argsort(np.abs(dt - mindt)) 
    s=np.minimum(_s[0],_s[1]) 
    
    # Get maximum index
    _e=np.argsort(np.abs(dt - maxdt))
    e=np.maximum(_e[0],_e[1]) 
    
    obj['itime']=np.arange(s,e,dtype="int")
  
  return obj
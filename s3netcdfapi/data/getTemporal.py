import numpy as np
# _datetime      = All datetime
# datetime      = Selected datetime


def getTemporal(netcdf2d,obj,dname):
  """ Check Temporal parameters
  """
  idname="i"+dname[1:]
  obj['user_time']=False
 
  if obj[idname] is not None:
    if isinstance(obj[idname],int):obj[idname]=[obj[idname]]
    obj['start']=None
    obj['end']=None
    obj['step']=None
    
  elif obj['start'] is not None or obj['end'] is not None:
    obj['user_time']=True
    # vnames=netcdf2d.getVariablesByDimension(dname)
    
    
    dt=obj['_time']=netcdf2d.query({"variable":netcdf2d.temporal['time']})
    mindt=np.min(dt)
    maxdt=np.max(dt)
    
    if obj['start'] is None:obj['start']=mindt
    if obj['end'] is None:obj['end']=maxdt
    
    obj['start']=np.datetime64(obj['start']) #np.array(obj['start'],dtype='datetime64[h]')
    obj['end']=np.datetime64(obj['end']) #np.array(obj['end'],dtype='datetime64[h]')
    obj['step']=np.timedelta64(obj['step'], obj['stepUnit'])
    
    if obj['start']<mindt:raise Exception("{0} below limit of {1}".format(obj['start'],np.min(dt)))
    if obj['end']>maxdt:raise Exception("{0} below limit of {1}".format(obj['end'],np.max(dt)))
    
    obj['time']= np.arange(obj['start'], obj['end']+obj['step'],obj['step'], dtype='datetime64[s]')

    # Get minimum index
    _s=np.argsort(np.abs(dt - obj['start'])) 
    s=np.minimum(_s[0],_s[1]) 
    
    # Get maximum index
    _e=np.argsort(np.abs(dt - obj['end']))
    e=np.maximum(_e[0],_e[1]) 
    
    
    
    obj[idname]=np.arange(s,e+1,dtype="int")
    obj['_time']=obj['_time'][obj[idname]]
  
  return obj
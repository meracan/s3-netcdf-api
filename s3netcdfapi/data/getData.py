import numpy as np
import collections
from . import interpolation as inter
from .utils import cleanObject,swapAxes,swapAxe


def getData(netcdf2d,obj):
  data={}
  for variable in obj['variable']:
    data[variable]=_getData(netcdf2d,obj,variable)
  return data


def _getData(netcdf2d,obj,variable,_odimensions=None):
  """
  
  Get data and interpolate data from S3-NetCDF
  
  Parameters
  ----------
  _odimensions:list or tuple, order of the dimension, example (nsnode,ntime,nfreq,ndir)
    If the _odimensions is not specified, the order is random    
  
  """
  # _dimensions=netcdf2d.getDimensionsByVariable(variable)
  _dimensions=netcdf2d.getMetaByVariable(variable)['dimensions']
  if _odimensions is None:_odimensions=_dimensions
  
  # print(cleanObject({**obj,'variable':variable},_dimensions))
  # Get data from s3-NetCDF
  _obj=cleanObject({**obj,'variable':variable},_dimensions)
  data,dimensions=netcdf2d.query(_obj,return_dimensions=True)
  
  
  # Data Interpolation
  for dim in _dimensions:
    data,dimensions=swapAxe(data,dimensions,dim)
    # if dim=="ntime" and obj['user_time']: # Does it need to be interpolated?
    if dim in obj['pointers']['temporal']['dimensions'] and obj['user_time']: # Does it need to be interpolated?
      data=inter.timeSeries(obj["_time"],obj["time"],data,kind=obj['inter.temporal'])
    elif dim in obj['pointers']['mesh']['dimensions'] and obj['user_xy']:
      if obj['inter.mesh']=="nearest":
        data=data[obj['nodeIndex']]
      elif obj['inter.mesh']=="linear":
        insideData=data[:len(obj["meshx"])]
        outsideData=data[len(obj["meshx"]):]
        data=inter.mesh(obj["meshx"],obj["meshy"],obj["elem"],insideData,obj['x'],obj['y'])
        if obj['extra.mesh']=="nearest":
          data[obj['iOutsideElem']]=outsideData[obj['nodeIndex']]
          
      else:raise Exception("Method {} does not exist".format(obj['inter.mesh']))
    elif dim in obj['pointers']['xy']['dimensions']  and obj['user_xy']: 
      if obj['inter.xy']=="nearest":data=data[obj['xyIndex']]
      else:raise Exception("")
  
  # Swap axes as required
  data,dimensions=swapAxes(data,dimensions,_odimensions)

  
  # Prepare return object
  newObj={"name":variable,"data":data,"meta":netcdf2d.getMetaByVariable(variable),"dimData":None,"dimensions":dimensions}
  if len(dimensions)==1 or obj['dataOnly']:return newObj
  

  # Get dimension data
  newObj['dimData']=getDimData(netcdf2d,obj,dimensions)
 
  return newObj



def getDimData(netcdf2d,obj,dnames):
  if obj['dataOnly'] or len(dnames)==1:return None
  data = collections.OrderedDict()
  invPointer=getInversePointer(obj['pointers'])
  
  for dname in dnames:
    name=dname[1:]
    # _name="mesh" if name=="node" else name
 
    if name in invPointer:types=invPointer[name]
    elif dname=="nelem" or dname=='npe':continue  
    else:types=[];pointer=None
    
    if 'mesh' in types or "xy" in types:
      if 'mesh' in types:pointer=obj['pointers']["mesh"]
      else:pointer=obj['pointers']["xy"]
      x=netcdf2d.getVariableByDimension(dname,pointer,"x")
      y=netcdf2d.getVariableByDimension(dname,pointer,"y")
      # if 'user_xy' in obj and not obj['user_xy']:
      if not 'x' in obj or obj['x'] is None: 
        obj['x']=netcdf2d.query(cleanObject({**obj,'variable':x},['i{}'.format(name)]))
      if not 'y' in obj or obj['y'] is None:
        obj['y']=netcdf2d.query(cleanObject({**obj,'variable':y},['i{}'.format(name)]))
        
      data[name]={"name":name,"data":None,"subdata":{
        "x":{"name":"x","data":obj['x'],"meta":netcdf2d.getMetaByVariable(x)},
        "y":{"name":"y","data":obj['y'],"meta":netcdf2d.getMetaByVariable(y)}
      }}
    else:
      nvar=name
      # if not 'user_{}'.format(name) in obj or not obj['user_'+name]:
      if not nvar in obj or obj[nvar] is None: 
        if name in types:
          pointer=obj['pointers'][name]
          nvar=netcdf2d.getVariableByDimension(dname,pointer,name)
        obj[nvar]=netcdf2d.query(cleanObject({**obj,'variable':nvar},['i{}'.format(nvar)]))
        
        
      data[nvar]={"name":nvar,"data":obj[nvar],"meta":netcdf2d.getMetaByVariable(nvar)}
  
  # print(data)
  return data
  
def getInversePointer(pointers):
  ts={}
  for t in pointers:
    dnames=pointers[t]
    for dname in dnames:
      if dname in ts:ts[dname].append(t)
      else:ts[dname]=[t]
  return ts

# def getVariableByDimension(netcdf2d,dname,pointer,pointername):
#   """
#   """
#   vnames=netcdf2d.getVariablesByDimension(dname)
#   vname=next(x for x in vnames if x in pointer[pointername])
#   return vname
  
import numpy as np
import collections
from .interpolation import timeSeriesClosest,timeSeriesLinear,barycentric

from .utils import cleanObject,swapAxes,swapAxe


def get(netcdf2d,obj):
  data={}
  for variable in obj['variable']:
    data[variable]=getData(netcdf2d,obj,variable)
  return data


def getData(netcdf2d,obj,variable,_odimensions=None):
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
  
  # Get data from s3-NetCDF
  _obj=cleanObject({**obj,'variable':variable},_dimensions)
  data,dimensions=netcdf2d.query(_obj,return_dimensions=True)
  
  
  # Data Interpolation
  for dim in _dimensions:
    data,dimensions=swapAxe(data,dimensions,dim)
    # if dim=="ntime" and obj['user_time']: # Does it need to be interpolated?
    if dim in obj['pointers']['temporal']['dimensions'] and obj['user_time']: # Does it need to be interpolated?
      if obj['inter.temporal']['temporal']=="closest":data=timeSeriesClosest(obj["_time"],obj["time"],data)
      elif obj['inter.temporal']=="linear":data=timeSeriesLinear(obj["_time"],obj["time"],data)
      else:raise Exception("")
    # elif dim=="nnode" and obj['user_xy']: # Does it need to be interpolated?
    elif dim in obj['pointers']['mesh']['dimensions'] and obj['user_xy']: # Does it need to be interpolated?
      if obj['inter.mesh']=="closest":data=data[obj['nodeIndex']]
      elif obj['inter.mesh']=="linear":data=barycentric(obj["elem"],obj["meshx"],obj["meshy"],obj['xy'],data)
      else:raise Exception("")
    elif dim in obj['pointers']['xy']['dimensions']  and obj['user_xy']: # Does it need to be interpolated?
      if obj['inter.xy']=="closest":data=data[obj['xyIndex']]
      else:raise Exception("")
  
  # Swap axes as required
  data,dimensions=swapAxes(data,dimensions,_odimensions)

  
  # Prepare return object
  newObj={"name":variable,"data":data,"meta":netcdf2d.getMetaByVariable(variable),"dimData":None,"dimensions":dimensions}
  if len(dimensions)==1 or obj['dataOnly']:return newObj
  
  # print(dimensions,data.shape)
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
      x=getVariableByDimension(netcdf2d,dname,pointer,"x")
      y=getVariableByDimension(netcdf2d,dname,pointer,"y")
      if 'user_xy' in obj:
        obj['x']=netcdf2d.query(cleanObject({**obj,'variable':x},['i{}'.format(name)]))
        obj['y']=netcdf2d.query(cleanObject({**obj,'variable':y},['i{}'.format(name)]))
      data[name]={"data":None,"subdata":{
        "x":{"data":obj['x'],"meta":netcdf2d.getMetaByVariable(x)},
        "y":{"data":obj['y'],"meta":netcdf2d.getMetaByVariable(y)}
      }}
    else:
      if not 'user_{}'.format(name) in obj or not obj['user_'+name]:
        if name in types:
          pointer=obj['pointers'][name]
          nvar=getVariableByDimension(netcdf2d,dname,pointer,name)
        else:nvar=name
        obj[nvar]=netcdf2d.query(cleanObject({**obj,'variable':nvar},['i{}'.format(nvar)]))
      data[nvar]={"data":obj[nvar],"meta":netcdf2d.getMetaByVariable(nvar)}
  
  return data
  
def getInversePointer(pointers):
  ts={}
  for t in pointers:
    dnames=pointers[t]
    for dname in dnames:
      if dname in ts:ts[dname].append(t)
      else:ts[dname]=[t]
  return ts

def getVariableByDimension(netcdf2d,dname,pointer,pointername):
  """
  """
  vnames=netcdf2d.getVariablesByDimension(dname)
  vname=next(x for x in vnames if x in pointer[pointername])
  return vname
  

# def getDimData(netcdf2d,obj,dimensions):
#   """ Dim data is the dimension data of the array. For u velocity, shape is (ntime,nnode).
#   Dim data is the array of time and array of node (and array of x and y)
#   """
#   if obj['dataOnly'] or len(dimensions)==1:return None
  
#   # dimensions = ["{}".format(dimension[1:]) for dimension in dimensions]
#   # data=[]
#   data = collections.OrderedDict()
#   for dname in dimensions:
#     # if dim in spatial:
#     # elif nelem,npe
#     # else
    
#     if dname in obj['pointers']['mesh']['dimensions']:
#       if not obj['user_xy']:
#         vnames=netcdf2d.getVariablesByDimension(dname)
#         x=next(x for x in vnames if x in obj['pointers']['mesh']['x'])
#         y=next(x for y in vnames if y in obj['pointers']['mesh']['y'])
#         obj['x']=netcdf2d.query(cleanObject({**obj,'variable':x},['inode']))
#         obj['y']=netcdf2d.query(cleanObject({**obj,'variable':y},['inode']))

#       data[dname[1:]]={"data":None,"subdata":{
#         "x":{"data":obj['x'],"meta":netcdf2d.getMetaByVariable(x)},
#         "y":{"data":obj['y'],"meta":netcdf2d.getMetaByVariable(y)}}
#       }
#     elif dname in obj['pointers']['xy']['dimensions']:
#       if not obj['user_xy']:
#         vnames=netcdf2d.getVariablesByDimension(dname)
#         x=next(x for x in vnames if x in obj['pointers']['xy']['x'])
#         y=next(x for y in vnames if y in obj['pointers']['xy']['y'])        
#         obj['x']=netcdf2d.query(cleanObject({**obj,'variable':x},['ixy']))
#         obj['y']=netcdf2d.query(cleanObject({**obj,'variable':y},['ixy']))
      
#       data[dname[1:]]={"data":None,"subdata":{
#         "sx":{"data":obj['x'],"meta":netcdf2d.getMetaByVariable(x)},
#         "sy":{"data":obj['y'],"meta":netcdf2d.getMetaByVariable(y)}
#       }}
#     elif dname=="time":
#       if not obj['user_time']:
#         obj['datetime']=netcdf2d.query(cleanObject({**obj,'variable':'time'},['itime']))
#         # TODO : add itime=arange(ntime)
#       # data.append({"name":dim,"data":obj['datetime'],"meta":netcdf2d.getMetaByVariable('time')})
#       data["time"]={"data":obj['datetime'],"meta":netcdf2d.getMetaByVariable('time')}
#     elif dname=="npe":continue
#     elif dname=="nelem":continue
#     else:
#       _data=netcdf2d.query(cleanObject({**obj,'variable':dim},['i{}'.format(dim)]))
#       # data.append({"name":dim,"data":_data,"meta":netcdf2d.getMetaByVariable(dim)})
#       data[dim]={"data":_data,"meta":netcdf2d.getMetaByVariable(dim)}
    
#   return data
  
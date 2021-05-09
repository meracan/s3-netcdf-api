import numpy as np
import copy
import collections
from . import interpolation as inter
from .utils import cleanObject,swapAxes,swapAxe


def getData(netcdf2d,obj):
  """
  """
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
  _dimensions=netcdf2d.variables[variable]['dimensions']
  

  
  # Get data from s3-NetCDF
  _obj=cleanObject({**obj,'variable':variable},_dimensions)
  
  data,dimensions,indices=netcdf2d.query(_obj,return_dimensions=True,return_indices=True)
  _odimensions=copy.copy(dimensions)


  # Data Interpolation
  if any(x in _dimensions for x in netcdf2d.alls):
    for dim in _dimensions:
      data,dimensions=swapAxe(data,dimensions,dim)
      if dim in netcdf2d.temporals and obj['user_time']: # Does it need to be interpolated?
        data=inter.timeSeries(obj["_time"],obj["time"],data,kind=obj['inter.temporal'])
      elif dim in netcdf2d.spectrals and obj['user_xy']:
        if obj['inter.xy']=="nearest":
          data=data[obj['xyIndex']]
        else:raise Exception("")
      elif dim in netcdf2d.spatials and obj['user_xy']:
          if obj['inter.mesh']=="nearest":
            data=data[obj['nodeIndex']]
          elif obj['inter.mesh']=="linear":
            insideData=data[:len(obj["meshx"])]
            outsideData=data[len(obj["meshx"]):]
            data=inter.mesh(obj["meshx"],obj["meshy"],obj["elem"],insideData,obj['x'],obj['y'])
            if obj['extra.mesh']=="nearest":
              data[obj['iOutsideElem']]=outsideData[obj['nodeIndex']]
          else:raise Exception("Method {} does not exist".format(obj['inter.mesh']))
        
    
    # Swap axes as required
    data,dimensions=swapAxes(data,dimensions,_odimensions)
    
  
  # Prepare return object
  
  # newObj={"name":variable,"data":data,"meta":netcdf2d.variables[variable],"dimData":None,"dimensions":dimensions,"indices":indices}
  newObj={"name":variable,"data":data,"meta":netcdf2d.variables[variable],"dimData":None,"dimensions":dimensions}
  if "nchar" in dimensions or len(dimensions)==1 or obj['dataOnly']:return newObj

  # Get dimension data
  if(obj['extra']):
    newObj['dimData']=getDimData(netcdf2d,obj,dimensions)
 
  return newObj



def getDimData(netcdf2d,obj,dnames):
  """
  """
  if obj['dataOnly'] or len(dnames)==1:return None
  data = collections.OrderedDict()
  
  for dname in dnames:
    name=dname[1:]
    iname="i"+name
    if dname=="nelem":
      data["elem"]={"name":"elem","data":np.arange(netcdf2d.obj['dimensions']['nelem'],dtype="I"),"meta":""}
    elif dname=='npe':
      data["pe"]={"name":"npe","data":np.arange(3,dtype="I"),"meta":""}
    elif dname in netcdf2d.spatials:
      data[name]={"name":name,"data":None,"subdata":_getSpatial(netcdf2d,obj,dname,iname,netcdf2d.spatial['x'],netcdf2d.spatial['y'])}
    elif dname in netcdf2d.spectrals:
      data[name]={"name":name,"data":None,"subdata":_getSpectral(netcdf2d,obj,dname,iname)}
    
    else:
      if not name in obj or obj[name] is None:
        obj[name]=netcdf2d.query(cleanObject({**obj,'variable':name},[iname]))
      data[name]={"name":name,"data":obj[name],"meta":netcdf2d.variables[name]}
  
  return data
  

def _getSpatial(netcdf2d,obj,dname,iname,xname,yname):
  x=netcdf2d.query(cleanObject({**obj,'variable':xname},[iname])) if obj['x'] is None else obj['x']
  y=netcdf2d.query(cleanObject({**obj,'variable':yname},[iname])) if obj['y'] is None else obj['y']
  return {
      "x":{"name":"x","data":x,"meta":netcdf2d.variables[xname]},
      "y":{"name":"y","data":y,"meta":netcdf2d.variables[yname]},
      }


def _getSpectral(netcdf2d,obj,dname,iname):
 
  xy=_getSpatial(netcdf2d,obj,dname,iname,netcdf2d.spectral['x'],netcdf2d.spectral['y'])
  sid ="stationid";
  sname ="name";
  obj['stationid']=netcdf2d.query(cleanObject({**obj,'variable':sid},[iname]))
  if obj['x'] is not None:obj['stationid']=obj['stationid'][obj['xyIndex']]
  obj['stationname']=netcdf2d.query({'variable':sname})
  obj['stationname']=obj['stationname'][obj['stationid']]
  nameobj={
    "stationid":{"name":"stationid","data":obj['stationid'],"meta":netcdf2d.variables[sid]},
    "stationname":{"name":"stationname","data":obj['stationname'],"meta":netcdf2d.variables[sname]}
  } 
  return {**xy,**nameobj}
  
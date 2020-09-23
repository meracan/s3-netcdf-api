import numpy as np
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
    if dim=="ntime" and obj['user_time']: # Does it need to be interpolated?
      if obj['interpolation']['temporal']=="closest":data=timeSeriesClosest(obj["_datetime"],obj["datetime"],data)
      elif obj['inter.temporal']=="linear":data=timeSeriesLinear(obj["_datetime"],obj["datetime"],data)
      else:raise Exception("")
    elif dim=="nnode" and obj['user_xy']: # Does it need to be interpolated?
      if obj['inter.spatial']=="closest":data=data[obj['xyIndex']]
      elif obj['inter.spatial']=="linear":data=barycentric(obj["elem"],obj["meshx"],obj["meshy"],obj['xy'],data)
      else:raise Exception("")
    elif dim=="nsnode" and obj['user_sxy']: # Does it need to be interpolated?
      if obj['inter.spectral']=="closest":data=data[obj['sxyIndex']]
      else:raise Exception("")
  
  # Swap axes as required
  data,dimensions=swapAxes(data,dimensions,_odimensions)

  print(dimensions)
  # Prepare return object
  newObj={"name":variable,"data":data,"meta":netcdf2d.getMetaByVariable(variable),"dimData":None,"dimensions":dimensions}
  if len(dimensions)==1 or obj['dataOnly']:return newObj
  
  # print(dimensions,data.shape)
  # Get dimension data
  newObj['dimData']=getDimData(netcdf2d,obj,dimensions)
  return newObj


def getDimData(netcdf2d,obj,dimensions):
  """ Dim data is the dimension data of the array. For u velocity, shape is (ntime,nnode).
  Dim data is the array of time and array of node (and array of x and y)
  """
  if obj['dataOnly'] or len(dimensions)==1:return None
  
  dimensions = ["{}".format(dimension[1:]) for dimension in dimensions]
  data=[]
  for dim in dimensions:
    if dim=="node":
      if not obj['user_xy']:
        obj['x']=netcdf2d.query(cleanObject({**obj,'variable':'x'},['inode']))
        obj['y']=netcdf2d.query(cleanObject({**obj,'variable':'y'},['inode']))
        # TODO : add inode=arange(nnode)
          
      data.append([
        {"name":"x","data":obj['x'],"meta":netcdf2d.getMetaByVariable('x')},
        {"name":"y","data":obj['y'],"meta":netcdf2d.getMetaByVariable('y')}
        ])
      
    elif dim=="snode":
      if not obj['user_sxy']:
        obj['sx']=netcdf2d.query(cleanObject({**obj,'variable':'sx'},['isnode']))
        obj['sy']=netcdf2d.query(cleanObject({**obj,'variable':'sy'},['isnode']))
        # TODO : add isnode=arange(nsnode)
      data.append([
        {"name":"sx","data":obj['sx'],"meta":netcdf2d.getMetaByVariable('sx')},
        {"name":"sy","data":obj['sy'],"meta":netcdf2d.getMetaByVariable('sy')}
        ])
      
    elif dim=="time":
      if not obj['user_time']:
        obj['datetime']=netcdf2d.query(cleanObject({**obj,'variable':'time'},['itime']))
        # TODO : add itime=arange(ntime)
      data.append({"name":dim,"data":obj['datetime'],"meta":netcdf2d.getMetaByVariable('time')})
    elif dim=="pe":continue
    elif dim=="elem":continue
    else:
      _data=netcdf2d.query(cleanObject({**obj,'variable':dim},['i{}'.format(dim)]))
      data.append({"name":dim,"data":_data,"meta":netcdf2d.getMetaByVariable(dim)})
    
  return data
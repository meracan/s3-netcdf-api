import numpy as np
from .interpolation import timeSeriesClosest,timeSeriesLinear,barycentric

from .utils import cleanObject,swapAxes,swapAxe


def getData(netcdf2d,obj,variable,_odimensions=None):
  """
  
  Get data and interpolate data from S3-NetCDF
  
  Parameters
  ----------
  _odimensions:list or tuple, order of the dimension, example (nsnode,ntime,nfreq,ndir)
    If the _odimensions is not specified, the order is random    
  
  
  
  """
  _dimensions=netcdf2d.getDimensionsByVariable(variable)
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

  # Prepare return object
  newObj={"data":data,"header":getHeader(netcdf2d,variable),"dimData":None}
  if len(dimensions)==1 or obj['dataOnly']:return newObj
  
  # Get dimension data
  newObj['dimData']=getDimData(netcdf2d,obj,dimensions)
  return newObj
  
    
def getHeader(netcdf2d,vname):
  """ Get variable header by getting standard_name and units
  """
  meta=netcdf2d.getMetaByVariable(vname)
  header=meta['standard_name']
  if header!="Datetime" and meta['units']!="":header="{},{}".format(header,meta['units'])
  return header


def getDimData(netcdf2d,obj,dimensions):
  """
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
        {"data":obj['x'],"header":getHeader(netcdf2d,'x')},
        {"data":obj['y'],"header":getHeader(netcdf2d,'y')}
        ])
      
    elif dim=="snode":
      if not obj['user_sxy']:
        obj['sx']=netcdf2d.query(cleanObject({**obj,'variable':'sx'},['isnode']))
        obj['sy']=netcdf2d.query(cleanObject({**obj,'variable':'sy'},['isnode']))
        # TODO : add isnode=arange(nsnode)
      data.append([
        {"data":obj['sx'],"header":getHeader(netcdf2d,'sx')},
        {"data":obj['sy'],"header":getHeader(netcdf2d,'sy')}
        ])
      
    elif dim=="time":
      if not obj['user_time']:
        obj['datetime']=netcdf2d.query(cleanObject({**obj,'variable':'time'},['itime']))
        # TODO : add itime=arange(ntime)
      data.append({"data":obj['datetime'],"header":getHeader(netcdf2d,'time')})
    else:
      _data=netcdf2d.query(cleanObject({**obj,'variable':dim},['i{}'.format(dim)]))
      data.append({"data":_data,"header":getHeader(netcdf2d,dim)})
    
  return data
  
  
  
  

# def _getDimensionValues(netcdf2d,shape,obj,dimensions):
#   """
#   """
#   if obj['dataOnly'] or len(dimensions)==1:return None
  
#   dimensions = ["{}".format(dimension[1:]) for dimension in dimensions]
#   dimIndexValue=[]
#   maxLength=len(dimensions)
#   headers=[]
  
#   for dim in dimensions:
#     if dim=="node":
#       values=np.array(["{},{}".format(_x,_y)for _x,_y in zip(obj['x'],obj['y'])])
#       headers.append(getHeader(netcdf2d,'x'))
#       headers.append(getHeader(netcdf2d,'y'))
#     elif dim=="snode":
#       values=np.array(["{},{}".format(_x,_y)for _x,_y in zip(obj['sx'],obj['sy'])])
#       headers.append(getHeader(netcdf2d,'sx'))
#       headers.append(getHeader(netcdf2d,'sy'))
#     elif dim=="time":
#       values=np.datetime_as_string(obj['datetime'], timezone='UTC')
#       headers.append(getHeader(netcdf2d,'time'))
#     else:
#       values=obj[dim]
#       headers.append(getHeader(netcdf2d,dim))
      
#     if isinstance(values,list):values=np.array(values)
#     values=values.astype('str')
#     _max=max([len(x) for x in values])
#     maxLength+=_max
#     values=values.astype('|S{}'.format(_max))
#     dimIndexValue.append(values)
    
  
#   a=np.chararray((np.prod(shape)), itemsize=maxLength).reshape(shape)
#   a[:]=""
  
#   if len(shape)!=len(dimIndexValue):raise Exception("Shape of dimension index values does not match the data")
#   for i,(ishape,indexValue) in enumerate(zip(shape,dimIndexValue)):
#     if ishape!=len(indexValue):raise Exception("Error here {},{}".format(ishape,indexValue))    
    
#     t=[slice(None)for j in range(i)]
#     for k in range(ishape):
#         _t=tuple(list(t)+[k])
#         if isinstance(a[_t],str):a[_t]=a[_t].encode()+indexValue[k]+b","
#         else:a[_t]=a[_t]+indexValue[k]+b","
        

#   a = np.char.strip(a.astype('<U{}'.format(maxLength)), ',')

#   return a,headers  
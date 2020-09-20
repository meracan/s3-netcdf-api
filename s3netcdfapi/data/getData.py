import numpy as np
import interpolation as inter


def cleanObject(obj,dimensions):
  """
  Get only the variables required to netcdf2d
  
  Parameters
  ----------
  obj:object
  dimensions={nnode,ntime}
  
  Notes
  -----
  idimensions={inode,itime}
    
  """
  if not 'variable' in obj:raise Exception("Add {} to the default parameter object".format('variable'))
  
  _obj={}
  idimensions = ["{}{}".format('i',dimension[1:]) for dimension in dimensions]
  for name in idimensions:
    if name in obj:
      _obj[name]=obj[name]
  _obj['variable']=obj['variable']
  return _obj


def swapAxe(data,dimensionNames,name,i=0):
  """
  Swap axe from data
  
  Parameters
  ----------
  data:np.ndarray
  dimensionNames:list, name of dimensions.
  name:str,name of dimension to be swap
  i:int, index to be swap
  """
  if data.dim!=len(dimensionNames):raise Exception("List size should equal to data dimensions")
  index=dimensionNames.index(name)
  data=np.swapaxes(data, i, index) 
  dimensionNames[i], dimensionNames[index] = dimensionNames[index], dimensionNames[i]
  return data,dimensionNames


def swapAxes(data,dimensionNames,names):
  """
  Swap axes based on the specified dimension list
  Parameters
  ----------
  data:np.ndarray
  dimensionNames:list, name of dimensions.
  names:str,name of dimension to be swap
  """
  for i,name in enumerate(names):
    data,dimensionNames=swapAxe(data,dimensionNames,name,i)
  return data,dimensionNames 


def getData(netcdf2d,obj,variable):
  """
  Parameters
  ----------
  
  """
  _dimensions=netcdf2d.getDimensionsByVariable(variable)
  data,dimensions=netcdf2d.query(cleanObject({**obj,'variable':variable},_dimensions),True)
  
  
  for dim in _dimensions:
    data,dimensions=swapAxe(data,dimensions,dim)
    if dim=="ntime":
      if obj['datetime'] is None:
        obj['datetime']=netcdf2d.query({"variable":"time","itime":obj.get('time')})
      elif obj['interpolation']['temporal']=="closest":data=inter.timeSeriesClosest(obj["datetime"],obj["datetime"],data)
      elif obj['interpolation']['temporal']=="linear":data=inter.timeSeriesLinear(obj["datetime"],obj["datetime"],data)
      else:raise Exception("")
      
      
      
      # obj['datetime'] = obj['datetime']+
    elif dim=="nnode":
      if obj['xy'] is None:
        obj['x']=netcdf2d.query({'variable':'x','inode':obj['inode']})
        obj['y']=netcdf2d.query({'variable':'y','inode':obj['inode']})
      elif obj['interpolation']['spatial']=="closest":data=data[obj['xyIndex']]
      elif obj['interpolation']['spatial']=="linear":data=inter.barycentric(obj["_elem"],obj["_x"],obj["_y"],obj['xy'],data)
      else:raise Exception("")
    elif dim=="nsnode":
      if obj['sxy'] is None:
        obj['sx']=netcdf2d.query({'variable':'sx','inode':obj['isnode']})
        obj['sy']=netcdf2d.query({'variable':'sy','inode':obj['isnode']})
      elif obj['interpolation']['spectral']=="closest":data=data[obj['sxyIndex']]
      else:raise Exception("")
    elif dim=="nfreq":
      obj['freq']=netcdf2d.query({'variable':'freq'})
    elif dim=="ndir":
      obj['dir']=netcdf2d.query({'variable':'dir'})
  
  # Swap to original axes
  data,dimensions=swapAxes(data,dimensions,_dimensions)
  
  return {
    "data":data,
    "header":getHeader(obj,dimensions),
    "dimValues":getDimensionValues(data.shape,obj,dimensions),
    "dimHeaders":getDimensionHeaders(obj,dimensions)}

def getHeader(obj,variable):
  """
  """
  meta=obj['meta']['variable']
  
  header=""
  if obj['standard_name']:header="{}, {}".format(header,meta[variable]['standard_name'])
  if obj['units']:header="{}, {}".format(header,meta[variable]['units'])
  
  return header

def getDimensionHeaders(obj,dimensions):
  """
  """
  return [getHeader(obj,dim) for dim in dimensions]

def getDimensionValues(shape,obj,dimensions):
  """
  """
  dimensions = ["{}".format(dimension[1:]) for dimension in dimensions]
  dimIndexValue=[obj[dim] for dim in dimensions]
  
  # TODO: for node and snode, combine x,y or sx,sy
  # TODO: Compute itemSize
  # TODO: Compute values to string
  
  a=np.chararray((np.prod(shape)), itemsize=32).reshape(shape)
  a[:]=""
  if len(shape)!=len(dimIndexValue):raise Exception("Shape of dimension index values does not match the data")
  for i,(ishape,indexValue) in enumerate(zip(shape,dimIndexValue)):
    if ishape!=len(indexValue):raise Exception("Error here")    
    t=[slice(None)for j in range(i)]
    for k in range(ishape):
        _t=tuple(list(t)+[k])
        a[_t]=a[_t]+indexValue[k]+","
  a = np.char.strip(a, ',')

  return a  
  
  
# def _header():
#   a=np.chararray((27), itemsize=16).reshape((3,3,3))
#   a[:]=""
#   # print(a+b)
#   A=['a','b','c']
#   B=['d','e','f']
#   C=['g','h','i']
#   D=['g','h','i']
#   shape=a.shape
#   headers=(A,B,C,D)
#   if len(shape)!=len(headers):raise Exception("a")
#   for i,(ishape,header) in enumerate(zip(shape,headers)):
#     if ishape!=len(header):raise Exception("b")    
#     t=[slice(None)for j in range(i)]
#     for k in range(ishape):
#         _t=tuple(list(t)+[k])
#         a[_t]=a[_t]+header[k]+","
#   a = np.char.strip(a, ',')

  
  
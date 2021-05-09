import numpy as np


def cleanObject(obj,dimensions):
  """
  Get only the variables required to netcdf2d
  
  Parameters
  ----------
  obj:object
  dimensions=list,['nnode','ntime']
  
  Notes
  -----
  idimensions=['inode','itime']
    
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
  
  if data.ndim!=len(dimensionNames):print(data,data.ndim,dimensionNames)
  if data.ndim!=len(dimensionNames):raise Exception("List size should equal to data dimensions")
  index=dimensionNames.index(name)
  data=np.swapaxes(data, i, index) 
  dimensionNames[i], dimensionNames[index] = dimensionNames[index], dimensionNames[i]
  return data,dimensionNames


def swapAxes(data,dimensionNames,names,return_dimensions=True):
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
  if return_dimensions:return data,dimensionNames 
  return data

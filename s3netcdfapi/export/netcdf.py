
from s3netcdf.netcdf2d_func import createNetCDF

def to_netcdf(obj,data):
  
  dimensions={}
  variables={}
  for vname in data:
    variable=data[vname]
    variables=prepareData(variables,variable)
    
    for i,dim in enumerate(variable['dimensions']):
      dimensions[dim]=variable['data'].shape[i]
    
    _dimData = variable['dimData']
    
    for v in _dimData:
      if not isinstance(v,list):v=[v]
      for _v in v:
        variables=prepareData(variables,_v)
        
      
  createNetCDF(obj['output']+".nc",metadata={"title":""},dimensions=dimensions,variables=variables)

def prepareData(obj,variable):
  meta=variable['meta']
  meta['data']=variable['data']
  obj[variable['name']]=meta
  return obj
  
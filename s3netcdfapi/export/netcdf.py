
from s3netcdf.netcdf2d_func import createNetCDF

def to_netcdf(obj,data):
  
  dimensions={}
  variables={}
  for vname in data:
    variable=data[vname]
    variables=prepareData(variables,variable)
    
    for i,dim in enumerate(variable['dimensions']):
      dimensions[dim]=variable['data'].shape[i]
    
     
    dimData=variable['dimData']
    for dName in variable['dimData']:
      _dimData=dimData[dName]
      if _dimData['data'] is None:
        subData=_dimData['subdata']
        for subdim in subData:
          subdata=subData[subdim]
          subdata['name']=subdim
          variables=prepareData(variables,subdata)
      else:
        _dimData['name']=dName
        variables=prepareData(variables,_dimData)

      
        
        
  filepath=obj['filepath']+".nc"    
  createNetCDF(filepath,metadata={"title":""},dimensions=dimensions,variables=variables)
  return filepath

def prepareData(obj,variable):
  meta=variable['meta']
  meta['data']=variable['data']
  obj[variable['name']]=meta
  return obj

# def prepareData(obj,variable):
#   meta=variable['meta']
#   meta['data']=variable['data']
#   obj[variable['name']]=meta
#   if 'dimData' in variable:
#     for v in variable['dimData']:
#       if not isinstance(v,list):v=[v]
#       for _v in v:
#         obj=prepareData(obj,_v)
  
#   return obj  
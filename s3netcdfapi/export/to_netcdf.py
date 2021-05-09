
# from s3netcdf.s3netcdf_func import createNetCDF
from netcdf import NetCDF

def to_netcdf(obj,data,netcdf3=False):
  
  dimensions={}
  variables={}
  for vname in data:
    variable=data[vname]
    variables=prepareData(variables,variable)
    for i,dim in enumerate(variable['dimensions']):
      dimensions[dim]=variable['data'].shape[i]

    dimData=variable['dimData']
    if dimData is not None:
      for dName in dimData:
        _dimData=dimData[dName]
        if _dimData['data'] is None:
          subData=_dimData['subdata']
          for subdim in subData:
            subdata=subData[subdim]
             
            subdata['name']=subdim
            variables=prepareData(variables,subdata)
            dimensions=getOtherDimensions(dimensions,subdata)
            
        else:
          _dimData['name']=dName
          variables=prepareData(variables,_dimData)

      
  filepath=obj['filepath']+".nc"    
  NetCDF.create(filepath,metadata={"title":""},dimensions=dimensions,variables=variables,netcdf3=netcdf3)
  return filepath

def prepareData(variables,variable):
  newvariable=variable['meta']
  newvariable['data']=variable['data']
  if 'dimensions' in variable:
    newvariable['dimensions']=variable['dimensions']
  variables[variable['name']]=newvariable
 
  
  return variables

def getOtherDimensions(dimensions,variable):
  for dim in variable['meta']['dimensions']:
    if not dim in dimensions:
      if dim=="nchar":
        dimensions[dim]=16
      else:
        dimensions[dim]=len(variable['data'])
  return dimensions

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
import os
import json
import boto3

from s3netcdf import NetCDF2D
from export import export
from parameters import getParameters
from response import response
from credentials import getCredentials
from getData import getData

def handler(event, context):
  try:
    parameters =  event.get("queryStringParameters",{})
    credentials = getCredentials(event)
    return query(parameters,credentials)
  except Exception as err:
    print(err)
    return {
     "statusCode": 500,
     "body": json.dumps({'Error': str(err),'Reference': str(context.aws_request_id)}),
     "headers": {"Access-Control-Allow-Origin": "*"},
     }

def checkNetCDFExist(credentials,id,prefix,bucket):
  s3 = boto3.client('s3',**credentials)
  _prefix= "" if prefix is None else prefix+"/"
  key="{0}{1}/{1}.nca".format(_prefix,id)
  s3.head_object(Bucket=bucket, Key=key)# Check if object exist, if not returns Exception

def query(parameters,credentials):
  id = parameters.pop("id",os.environ["AWS_DEFAULTMODEL"])
  prefix = parameters.pop("prefix",os.environ.get("AWS_PREFIX",None))
  bucket=parameters.pop("bucket",os.environ['AWS_BUCKETNAME'])
  checkNetCDFExist(credentials,id,prefix,bucket)
  
  netcdf2d=NetCDF2D({"name":id,"prefix":prefix,"bucket":bucket,"localOnly":False,"cacheLocation":"/tmp"})
  
  # Export Metadata Only
  variable=parameters.get('variable',None)
  mesh=parameters.get('mesh',None)
  if variable is None and mesh is None:
    meta=netcdf2d.meta()
    return response("application/json",False,json.dumps(meta)) 
  
  # Check parameters
  obj=getParameters(netcdf2d,parameters)
  
  # Get data
  data={}
  for variable in obj['variable']:
    data[variable]=getData(netcdf2d,obj,variable)
  
  # Export Data
  return response(**export(obj,data))



  
  # data={}
  # if var=="mesh" or format=='slf':
  #   # Get mesh data
  #   # TODO: these group and variable name are hardcoded...might include the key words in the nca metadata
  #   data['elem']=netcdf2d.query({"group":"elem","variable":"elem"})
  #   data['x']=netcdf2d.query({"group":"nodes","variable":"lon"})
  #   data['y']=netcdf2d.query({"group":"nodes","variable":"lat"})


  # if var!="mesh":
  #   # Get data using name of variable
  #   if format in ["geojson", "csv"]:
  #     data['parameter'] = var

  #     if var == "spectra":
  #       data['lons'] = netcdf2d.query({"group": "stations", "variable": "slon"})
  #       data['lats'] = netcdf2d.query({"group": "stations", "variable": "slat"})
  #       data['freq'] = netcdf2d.query({"group": "freq", "variable": "freq"})
  #       data['dir'] = netcdf2d.query({"group": "dir", "variable": "dir"})
  #       data['station'] = parameters.get('station', None)
  #       data['freq_indices'] = parameters.get('freq', None)
  #       data['dir_indices'] = parameters.get('dir', None)
  #     else:
  #       data['lons'] = netcdf2d.query({"group": "nodes", "variable": "lon"})
  #       data['lats'] = netcdf2d.query({"group": "nodes", "variable": "lat"})
  #       data['bath'] = netcdf2d.query({"group": "nodes", "variable": "bed"})

  #     data['n_indices'] = parameters.get('node', None)
  #     data['t_indices'] = parameters.get('time', None)
  #     data['times'] = netcdf2d.query({"group": "time", "variable": "time"})

  #   if format == "jsontest2":
  #     nodes = parameters.get('node', None)
  #     data['nodes'] = nodes


  #   data[var]=netcdf2d.query(parameters)

    
  # return save(format,data)




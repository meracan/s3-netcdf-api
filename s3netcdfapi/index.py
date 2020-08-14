import os
import json
import boto3

from s3netcdf import NetCDF2D
from save import save,response

def handler(event, context):
  try:
    parameters = event['queryStringParameters']
    if parameters is None:parameters={}
    
    credentials = checkCredentials(event)
    return query(parameters,credentials)
    
  except Exception as err:
    print(err)
    return {
     "statusCode": 500,
     "body": json.dumps({'Error': str(err),'Reference': str(context.aws_request_id)}),
     "headers": {"Access-Control-Allow-Origin": "*"},
     }

def checkCredentials(event):
  try:
    role=event['requestContext']['authorizer']['claims']['cognito:roles']
    
    sts = boto3.client('sts')
    obj = sts.assume_role(RoleArn=role,RoleSessionName="APIrole")
    
    credentials={
      "aws_access_key_id":obj['Credentials']['AccessKeyId'],
      "aws_secret_access_key":obj['Credentials']['SecretAccessKey'],
      "aws_session_token":obj['Credentials']['SessionToken']
    }
    return credentials
  except Exception as err:
    return {}
  
def query(parameters,credentials):
  
  id = parameters.pop("id",os.environ["AWS_DEFAULTMODEL"])
  Bucket=parameters.pop("bucket",os.environ['AWS_BUCKETNAME'])
  
  s3 = boto3.client('s3',**credentials)
  key="{0}/{0}.nca".format(id)
  
  s3.head_object(Bucket=Bucket, Key=key)
  
  netcdf2d=NetCDF2D({"name":id,"bucket":Bucket,"localOnly":False,"cacheLocation":"/tmp"})
  
  meta=netcdf2d.meta()
  
  var=parameters.get('variable',None)
  if var is None:return response("application/json",json.dumps(meta)) 
  
  format= parameters.pop('format',"json")
  data={}
  if var=="mesh" or format=='slf':
    # Get mesh data
    # TODO: these group and variable name are hardcoded...might include the key words in the nca metadata
    data['elem']=netcdf2d.query({"group":"elem","variable":"elem"})
    data['x']=netcdf2d.query({"group":"nodes","variable":"lon"})
    data['y']=netcdf2d.query({"group":"nodes","variable":"lat"})


  if var!="mesh":
    # Get data using name of variable
    if format in ["geojson", "csv"]:
      data['parameter'] = var

      if var == "spectra":
        data['lons'] = netcdf2d.query({"group": "stations", "variable": "slon"})
        data['lats'] = netcdf2d.query({"group": "stations", "variable": "slat"})
        data['freq'] = netcdf2d.query({"group": "freq", "variable": "freq"})
        data['dir'] = netcdf2d.query({"group": "dir", "variable": "dir"})
        data['station'] = parameters.get('station', None)
        data['freq_indices'] = parameters.get('freq', None)
        data['dir_indices'] = parameters.get('dir', None)
      else:
        data['lons'] = netcdf2d.query({"group": "nodes", "variable": "lon"})
        data['lats'] = netcdf2d.query({"group": "nodes", "variable": "lat"})
        data['bath'] = netcdf2d.query({"group": "nodes", "variable": "bed"})

      data['n_indices'] = parameters.get('node', None)
      data['t_indices'] = parameters.get('time', None)
      data['times'] = netcdf2d.query({"group": "time", "variable": "time"})

    if format == "jsontest2":
      nodes = parameters.get('node', None)
      data['nodes'] = nodes


    data[var]=netcdf2d.query(parameters)

    
  return save(format,data)




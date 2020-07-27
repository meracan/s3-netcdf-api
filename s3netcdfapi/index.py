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
    data['x']=netcdf2d.query({"group":"node","variable":"lon"})
    data['y']=netcdf2d.query({"group":"node","variable":"lat"})

  if var!="mesh":
    # Get data using name of variable
    if format == "geojson":
      # needs start and end nodes and times?
      xdata = netcdf2d.query({"group": "node", "variable": "lon"})
      ydata = netcdf2d.query({"group": "node", "variable": "lat"})
      zdata = netcdf2d.query({"group": "node", "variable": "bed"})
      tdata = netcdf2d.query({"group": "time", "variable": "time"})

      # 'node' and 'time' is a string or tuple?
      nodes = parameters.get('node', None)
      times = parameters.get('time', None)

      """
      # depends on what type the 'node' and 'time' parameters are
      data['x'] = xdata[nodes[0]:nodes[1]]
      data['y'] = ydata[nodes[0]:nodes[1]]
      data['z'] = zdata[nodes[0]:nodes[1]]
      data['times'] = tdata[times[0]:times[1]]
      data['parameter'] = var
      """

    if format == "csv":

      xdata = netcdf2d.query({"group": "node", "variable": "lon"})
      ydata = netcdf2d.query({"group": "node", "variable": "lat"})

      nodes = parameters.get('node')
      times = parameters.get('time')

      """
      if nodes is not None and len(nodes) == 1: nodes =
      if times is not None and len(times) == 1: times =

      data['times'] =
      data['parameter'] = var
      data['lons'] =
      data['lats'] =
      """

    data[var]=netcdf2d.query(parameters)

    
  return save(format,data)




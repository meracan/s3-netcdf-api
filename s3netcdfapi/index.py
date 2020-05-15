import os
import json
import boto3

from s3netcdf import NetCDF2D
from save import save,response

def handler(event, context):
  try:
    parameters = event['parameters']
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
  s3.head_object(Bucket=Bucket, Key=id)

  netcdf2d=NetCDF2D({"name":id,"bucket":Bucket,"localOnly":False})
  meta=netcdf2d.meta()
  var=parameters.get('variable',None)
  if var is None:return response("text/json",json.dumps(meta)) 
  
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
    data[var]=netcdf2d.query(parameters)
    
  return save(format,data)




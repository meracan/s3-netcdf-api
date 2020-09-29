import json
from .s3netcdf2dapi import S3NetCDFAPI
from .credentials import getCredentials

def handler(event, context):
  try:
    print(event)
    if event is None:event={}
    parameters =  event.get("queryStringParameters",{})
    if parameters is None:parameters={}
    print(parameters)
    credentials = getCredentials(event)
    netcdf2d=S3NetCDFAPI.init(parameters,credentials)
    return netcdf2d.run(parameters)
  except Exception as err:
    print(err)
    return {
     "statusCode": 500,
     "body": json.dumps({'Error': str(err),'Reference': str(context.aws_request_id)}),
     "headers": {"Access-Control-Allow-Origin": "*"},
     }
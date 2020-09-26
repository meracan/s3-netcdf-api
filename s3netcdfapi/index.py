import json
from .s3netcdf2dapi import S3NetCDFAPI
from .credentials import getCredentials

def handler(event, context):
  try:
    parameters =  event.get("queryStringParameters",{})
    credentials = getCredentials(event)
    netcdf2d=S3NetCDFAPI.create(parameters,credentials)
    return netcdf2d.run(parameters)
  except Exception as err:
    print(err)
    return {
     "statusCode": 500,
     "body": json.dumps({'Error': str(err),'Reference': str(context.aws_request_id)}),
     "headers": {"Access-Control-Allow-Origin": "*"},
     }
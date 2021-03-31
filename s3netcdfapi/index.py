import json
import os
from .s3netcdf2dapi import S3NetCDFAPI
from .credentials import getCredentials

def handler(event, context):
  # print(event);
  try:
    if event is None:event={}
    
    parameters =  event.get("queryStringParameters",{})
    pathParameters=event.get("pathParameters",{})
    if parameters is None:parameters={}
    if pathParameters is None:pathParameters={}
    id=pathParameters.get("id",os.environ.get("AWS_DEFAULTMODEL",None))
    
    credentials = getCredentials(event)
    
    headers=event.get("headers",{})
    if headers is None:headers={}
    host=headers.get("Host","")
    requestContext=event.get("requestContext",{})
    stage=requestContext.get("stage","")
    url=host+"/"+stage+"/"+id
    
    
    parameters['id']=id
    parameters['url']=url
    netcdf2d=S3NetCDFAPI.init(parameters,credentials)
    return netcdf2d.run(parameters)
  except Exception as err:
    print(err)
    return {
     "statusCode": 500,
     "body": json.dumps({'Error': str(err),'Reference': str(context.aws_request_id)}),
     "headers": {"Access-Control-Allow-Origin": "*"},
     }
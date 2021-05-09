import json
import os
from .s3netcdfapi import S3NetCDFAPI,checkNetCDFExist
from .credentials import getCredentials
from awstools import DynamoDB 


def getURL(event,id):
  headers=event.get("headers",{})
  if headers is None:headers={}
  host=headers.get("Host","")
  requestContext=event.get("requestContext",{})
  stage=requestContext.get("stage","")
  url=host+"/"+stage+"/"+id
  return url


def handler(event, context):
  # AWS_TABLENAME
  try:
    if event is None:event={}
    parameters =  event.get("queryStringParameters",None)
    if parameters is None:parameters={}
    pathParameters=event.get("pathParameters",None)
    if pathParameters is None:pathParameters={}
    id=pathParameters.get("id",None)
    if id is None:raise Exception("Api needs a model id")
    
    credentials = getCredentials(event)
    url=getURL(event,id)
    debug = os.environ.get('AWS_DEBUG',"False")
    cache = os.environ.get('AWS_CACHE','/tmp')
    
    os.makedirs(cache,exist_ok=True)
    os.makedirs(os.path.join(cache,"tmp"),exist_ok=True)
    
    if debug == "False":
      dyno      = DynamoDB()
      item      = dyno.get(id=id)
      bucket    = item.get("bucket")
      if bucket is None:raise Exception("Model id={} does not exists".format(id)) 
      s3prefix  = item.get("s3-predix")
      verbose   = False
      localOnly = False
      checkNetCDFExist(bucket,s3prefix,id,credentials)
      
      # TODO: Get region of bucket, check region of lambda, call different different lambda
    else:
      bucket    = None
      s3prefix  = None
      verbose   = True
      localOnly = True
    
    obj={
      "name":id,
      "bucket":bucket,
      "s3prefix":s3prefix,
      "verbose":verbose,
      "localOnly":localOnly,
      "cacheLocation":cache,
      "credentials":credentials,
    }
    parameters['id']=id
    parameters['url']=url
    
    with S3NetCDFAPI(obj) as api:
      return api.run(parameters)
  
  except Exception as err:
    print(err)
    return {
     "statusCode": 500,
     "body": json.dumps({'Error': str(err),'Reference': str(context.aws_request_id)}),
     "headers": {"Access-Control-Allow-Origin": "*"},
     }
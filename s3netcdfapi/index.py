import os
import sys
import json
import boto3
import uuid
from collections import OrderedDict 

from s3netcdf import NetCDF2D
from .s3api import S3API
from .parameters import getParameters
from .response import response,responseSignedURL
from .credentials import getCredentials
from .data import get
from .export import export
sys.path.append(os.path.join(os.path.dirname(__file__)))
isDebug=os.environ.get('AWS_DEBUG',"True")

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

def query(parameters,credentials={}):
  id = parameters.pop("id",os.environ.get("AWS_DEFAULTMODEL",None))
  if id is None:raise Exception("Api needs a model id")
  if isDebug=="True":
    bucket=os.environ.get('AWS_BUCKETNAME',"uvic-bcwave")
    prefix = os.environ.get("AWS_PREFIX",None)
    cache=os.environ.get('AWS_CACHE',r"../s3/tmp")
    netcdf2d=NetCDF2D({"name":id,"s3prefix":prefix,"bucket":bucket,"localOnly":True,"cacheLocation":r"../s3","credentials":credentials})
  else:
    bucket=os.environ.get('AWS_BUCKETNAME',None)
    prefix = os.environ.get("AWS_PREFIX",None)
    cache=os.environ.get('AWS_CACHE','/tmp')
    checkNetCDFExist(credentials,id,prefix,bucket)
    netcdf2d=NetCDF2D({"name":id,"prefix":prefix,"bucket":bucket,"localOnly":False,"cacheLocation":cache})
  
  if not os.path.exists(cache):os.makedirs(cache)
  
  s3=S3API({"name":id,"s3prefix":"tmp","bucket":bucket,"cacheLocation":cache,"credentials":credentials})
  
  # Export Metadata Only
  variable=parameters.get('variable',None)
  if variable is None:
    meta=netcdf2d.meta()
    return response("application/json",False,json.dumps(meta)) 
  
  # TODO:Get cache file
  # dict1 = OrderedDict(sorted(parameters.items())) 
  # print(dict1)
  
  # Check parameters
  obj=getParameters(netcdf2d,parameters)
  
  # Get data
  data=get(netcdf2d,obj)
  
  # Export data to file
  obj['output']=str(uuid.uuid4())
  obj['filepath']=os.path.join(cache,obj['output'])
  filepath=export(obj,data)
  
  # Upload to S3
  s3.upload(filepath)
  url=s3.generate_presigned_url(filepath)
  
  return responseSignedURL(url)
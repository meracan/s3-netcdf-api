import base64
import os
# def response(ContentType,Body):
#   """
#   Response to AWS API GATEWAY
  
#   Parameters
#   ----------
#   ContentType:
#   isBase64Encoded:
#   Body:
#   """
#   return {
#       'statusCode': 200,
#       'headers': {
#         "Content-Type": ContentType,
#         'Cache-Control':"max-age=31536000",
#         "Access-Control-Allow-Origin": "*",
#         'Access-Control-Allow-Headers': 'Content-Type',
#         'Access-Control-Expose-Headers': 'Content-Type',
#       },
#       'body':Body
#     } 

def response(data,contentType,compressed=False):
  isBase64Encoded=False
  contentEncoding=""
  if compressed:
    isBase64Encoded=True
    contentEncoding="gzip"
    filePath=data
    with open(filePath, "rb") as file:
      data=file.read()
      data=base64.b64encode(data).decode('utf-8')
    os.remove(filePath)
    
  return {
    'statusCode': 200,
    'isBase64Encoded': isBase64Encoded,
    'body': data,
    'headers': {
      'Access-Control-Allow-Headers': 'Content-Type,Content-Encoding',
      'Access-Control-Allow-Methods': 'OPTIONS,GET',
      'Access-Control-Allow-Origin': '*',
      'Cache-Control':"max-age=31536000",
      "Content-Encoding": contentEncoding,
      "Content-Type": contentType,
    },
  } 

def responseSignedURL(signedUrl,origin=None):
  """
  Response to AWS API GATEWAY
  
  Parameters
  ----------
  ContentType:
  isBase64Encoded:
  Body:
  """
  return {
  	"statusCode": 303,
  	"headers": {
  		"Location": signedUrl,
  		# "Access-Control-Allow-Origin": origin,
  		"Access-Control-Allow-Origin": "*",
  		'Cache-Control':"max-age=86400",
  		'Access-Control-Allow-Headers': 'Content-Type',
  		'Access-Control-Expose-Headers': 'Content-Type',
  		# "Access-Control-Allow-Credentials": "true",
  		# "Vary": "Origin",
  	}
  }    
 
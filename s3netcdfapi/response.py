
def response(ContentType,Body):
  """
  Response to AWS API GATEWAY
  
  Parameters
  ----------
  ContentType:
  isBase64Encoded:
  Body:
  """
  return {
      'statusCode': 200,
      'headers': {
        "Content-Type": ContentType,
        "Access-Control-Allow-Origin": "*",
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Expose-Headers': 'Content-Type',
      },
      'body':Body
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
  		'Access-Control-Allow-Headers': 'Content-Type',
  		'Access-Control-Expose-Headers': 'Content-Type',
  		# "Access-Control-Allow-Credentials": "true",
  		# "Vary": "Origin",
  	}
  }    
 
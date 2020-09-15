
def response(ContentType,isBase64Encoded,Body):
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
      'headers': {"content-type": ContentType},
      'isBase64Encoded': isBase64Encoded,
      'body':Body
    } 
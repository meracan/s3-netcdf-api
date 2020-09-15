import boto3

def getCredentials(event):
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
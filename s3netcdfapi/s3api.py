import os
from s3netcdf.s3client import S3Client

class S3API(S3Client):
  def __init__(self, obj):
    super().__init__(obj)
  
  def generate_presigned_url(self,filepath):
    s3path = self._gets3path(filepath)
    expiration=3600
    URL=self.s3.generate_presigned_url("get_object",Params={'Bucket': self.bucket,'Key': s3path}, ExpiresIn=expiration)
    return URL
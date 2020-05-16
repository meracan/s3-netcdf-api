aws cloudformation deploy \
  --template-file api.yaml \
  --stack-name $1 \
  --parameter-overrides BucketName=$AWS_BUCKETNAME \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
aws cloudformation describe-stacks --stack-name $1 > output.json

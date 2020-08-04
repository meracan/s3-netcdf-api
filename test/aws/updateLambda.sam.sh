FOLDER=~/Desktop/Jan2020/PRIMED
FUNCTIONARN=arn:aws:lambda:us-west-2:440480703237:function:tests3netcdfapi-MyFunction-8445GG2UE18J
ZIPFILE=s3netcdfapi.zip

aws s3 cp s3://$AWS_BUCKETNAME/lambda/function/s3netcdfapi.base.zip ./$ZIPFILE

rm -R lambda
mkdir lambda
cd lambda
cp -r ~/PycharmProjects/MARACAN/s3-netcdf-api/s3netcdfapi/*.py .



rm -R ./*dist-info

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
zip -r ../$ZIPFILE *

aws s3 cp ../$ZIPFILE s3://$AWS_BUCKETNAME/lambda/function/$ZIPFILE

cd ..
aws lambda update-function-code --function-name  $FUNCTIONARN --zip-file fileb://$ZIPFILE
rm -R lambda
rm $ZIPFILE



#aws s3 cp ./s3netcdfapi.zip s3://$AWS_BUCKETNAME/lambda/function/s3netcdfapi.zip
#aws lambda update-function-code --function-name  arn:aws:lambda:us-west-2:440480703237:function:tests3netcdfapi-MyFunction-8445GG2UE18J --zip-file fileb://s3netcdfapi.zip
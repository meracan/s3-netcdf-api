FOLDER=/home/ec2-user/environment
ZIPFILE=s3netcdfapi.zip
FUNCTIONARN=arn:aws:lambda:us-west-2:440480703237:function:tests3netcdfapi-MyFunction-8445GG2UE18J

rm -R lambda
mkdir lambda
cd lambda
# cp -r $FOLDER/s3-netcdf-api/s3netcdfapi/*.py .
mkdir app
cd app
cp -r $FOLDER/s3-netcdf-api/s3netcdfapi/* .
cd ..

pip install --no-deps -t . netcdf4
pip install --no-deps -t . cftime

pip install --no-deps -t . pandas
pip install --no-deps -t . pytz

pip install --no-deps -t . matplotlib
pip install --no-deps -t . pyparsing
pip install --no-deps -t . cycler
pip install --no-deps -t . kiwisolver
pip install --no-deps -t . pillow


pip install --no-deps -t . $FOLDER/binpy
pip install --no-deps -t . $FOLDER/s3-netcdf
pip install --no-deps -t . $FOLDER/slf-py

rm -R ./*dist-info

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
zip -r ../$ZIPFILE *

aws s3 cp ../$ZIPFILE s3://$AWS_BUCKETNAME/lambda/function/s3netcdfapi.base.zip

cd ..
aws lambda update-function-code --function-name  $FUNCTIONARN --zip-file fileb://$ZIPFILE --region us-west-2
rm -R lambda
rm $ZIPFILE

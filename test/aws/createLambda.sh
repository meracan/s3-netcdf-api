FOLDER=/home/ec2-user/environment
ZIPFILE=s3netcdfapi.zip

rm -R lambda
mkdir lambda
cd lambda
cp -r $FOLDER/s3-netcdf-api/s3netcdfapi/*.py .

pip install --no-deps -t . netcdf4
pip install --no-deps -t . cftime
pip install --no-deps -t . $FOLDER/binary-py
pip install --no-deps -t . $FOLDER/s3-netcdf
 
rm -R ./*dist-info

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
zip -r ../$ZIPFILE *
aws s3 cp ../$ZIPFILE s3://$AWS_BUCKETNAME/lambda/function/$ZIPFILE
cd ..
rm -R lambda
rm $ZIPFILE
echo s3://$AWS_BUCKETNAME/lambda/function/$ZIPFILE
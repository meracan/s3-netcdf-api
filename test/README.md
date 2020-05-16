# Testing

```bash 
conda install pytest
pytest
```



## Testing with AWS
These are steps to create AWS services such as Api Gateway and Lambda using CloudFormation.

### Pre-Installation
If conda evnironment `s3netcdfapi` is not created, please follow the steps [here](../README.md) and activate the environment `conda activate s3netcdfapi`.


#### Set AWS_BUCKETNAME
Set environment variable `AWS_BUCKETNAME`
```bash
python env.template.py uvic-bcwave
```

#### Upload Lambda function to S3
Please check paths in `createLambda.sh`. Some of libraries needs to be downloaded from github.
```bash
bash createLambda.sh
```
Check and review `s3://uvic-bcwave/lambda/function/s3netcdfapi.zip` in `api.yaml` file.

### Create AWS services
Create AWS services using the `STACKNAME` and `BUCKETNAME` environment variables. This will generate an output.json with all the services id.
```bash
# bash createAWS.sh {STACKNAME} {BUCKETNAME}
bash createAWS.sh tests3netcdfapi
```

Optional, change region
```bash
export AWS_DEFAULT_REGION=us-west-2
bash createAWS.sh tests3netcdfapi

```

### Post create
Save environment variables to conda's environment
```bash
python extract.py
conda deactivate
conda activate s3netcdfapi
```
Here's the list of environment variables:
- `AWS_BUCKETNAME`
- `AWS_API`

### (For development) - Review environment variables
To modify conda's environment vairables:
```bash
conda activate meracan
vi $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
vi $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
```
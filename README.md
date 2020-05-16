# s3-netcdf-api
Lambda api that handles http request to get data from s3-netcdf.

## Installation
```bash
conda create -n s3netcdfapi python=3.8
conda activate s3netcdfapi
conda install -c conda-forge boto3 numpy scipy netcdf4
pip install -e ../s3-netcdf
pip install -e ../s3-netcdf-api
pip install -e ../binary-py
```

## AWS Setup
```
git clone fff
```

#### TODO
- Complete save.py
- Some of the export format needs the mesh (e.g slf,geojson)
- For mesh: the group and variable name are hardcoded...might include the key words in the nca metadata
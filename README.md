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

## Parameters
```json
{
    "variable":{"default":None,"type":(str,list),"values":[]},
    
    "inode":{"default":None,"type":(int,list,slice),"minmax":[],"comment":"index of the model node"},
    "isnode":{"default":None,"type":(int,list,slice),"minmax":[],"comment":"index of the model node"},
    "itime":{"default":None,"type":(int,list,slice),"minmax":[]},
    
    
    "longitude":{"default":None,"type":(float,list),"extent":[]},
    "latitude":{"default":None,"type":(float,list)},"extent":[],
    "x":{"default":None,"type":(float,list),"extent":[]},
    "y":{"default":None,"type":(float,list),"extent":[]},
    
    
    "start":{"default":None,"type":"str","extent":""},
    "end":{"default":None,"type":"str","extent":""},
    "step":{"default":1,"type":"int"},
    "stepUnit":{"default":"h","type":"str"},
    
    "inter.mesh":{"default":"nearest","type":"str","values":["nearest","linear"]},
    "inter.temporal":{"default":"nearest","type":"str","values":["nearest","linear"]},
    "inter.xy":{"default":"nearest","type":str,"values":["nearest"]},
    
    "export":{"default":"json","type":str,"values":[]},
    "sep":{"default":",","type":str,"values":[]},
}

```

## AWS Setup
```
git clone fff
```

##



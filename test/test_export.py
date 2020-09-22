
from s3netcdf import NetCDF2D
from s3netcdfapi.query.get import getData,getHeader,getDimensionValues
from s3netcdfapi.export.table import table

input={
  "name":"input1",
  "cacheLocation":"../s3",
  "localOnly":True,
  "verbose":True,
  "maxPartitions":40
}

netcdf2d=NetCDF2D(input)



def test_table():
  # Test 1
  obj1={"inode":[0,1]}
  data={
    "x":getData(netcdf2d,obj1,"x"),
    "y":getData(netcdf2d,obj1,"y"),
    }
  table(obj1,data)
  
  # Test 2
  obj2={"dataOnly":False,"itime":[0]}
  data={
    "u":getData(netcdf2d,obj2,"u")
    }
  table(obj2,data)

def test_binary():
    export.binary()
    
def test_csv():
    export.csv()

def test_geojson():
    export.geojson()

def test_json():
    export.json()

def test_jsontest():
    export.jsontest()

def test_mat():
    export.mat()

def test_netcdf():
    export.netcdf()

def test_shapefile():
    export.shapefile()

def test_slf():
    export.slf()

def test_tri():
    export.tri()

if __name__ == "__main__":
  test_table()
#   test_binary()
#   test_csv()
#   test_geojson()
#   test_json()
#   test_jsontest()
#   test_mat()
#   test_netcdf()
#   test_shapefile()
#   test_slf()
#   test_tri()
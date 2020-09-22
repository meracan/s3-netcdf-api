
from s3netcdf import NetCDF2D
from s3netcdfapi.parameters.parameters import getParameters
from s3netcdfapi.query.get import getData,getHeader
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
  # # Test 1
  # obj1=getParameters(netcdf2d,{"variable":"x,y","inode":[0,1]})
  # data={
  #   "x":getData(netcdf2d,obj1,"x"),
  #   "y":getData(netcdf2d,obj1,"y"),
  #   }
  # print(table(obj1,data))
  
  # # Test 2
  # obj2=getParameters(netcdf2d,{"variable":"u,v","inode":[0,1,2],"itime":[0,1]})
  
  # data={
  #   "u":getData(netcdf2d,obj2,"u"),
  #   "v":getData(netcdf2d,obj2,"v"),
  #   }
  # print(table(obj2,data))
  
  # Test 3
  obj3=getParameters(netcdf2d,{"variable":"spectra","isnode":[0],"itime":[0,1]})
  
  data={
    "spectra":getData(netcdf2d,obj3,"spectra"),
    }
  print(table(obj3,data))

# def test_binary():
#     export.binary()
    
# def test_csv():
#     export.csv()

# def test_geojson():
#     export.geojson()

# def test_json():
#     export.json()

# def test_jsontest():
#     export.jsontest()

# def test_mat():
#     export.mat()

# def test_netcdf():
#     export.netcdf()

# def test_shapefile():
#     export.shapefile()

# def test_slf():
#     export.slf()

# def test_tri():
#     export.tri()

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
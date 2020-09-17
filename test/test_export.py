import s3netcdfapi.export as export

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
  test_binary()
  test_csv()
  test_geojson()
  test_json()
  test_jsontest()
  test_mat()
  test_netcdf()
  test_shapefile()
  test_slf()
  test_tri()
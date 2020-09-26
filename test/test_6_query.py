from s3netcdfapi import S3NetCDFAPI


def test_query():
    netcdf2d=S3NetCDFAPI.create({"id":"input1"},{})
    # print(netcdf2d.run({"id":"input1","export":"csv","variable":"u,v","x":-160.0,"y":40.0,"itime":[0,1]}))
    print(netcdf2d.run({}))
    
if __name__ == "__main__":
  test_query()
  
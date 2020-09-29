from s3netcdfapi import S3NetCDFAPI


def test_query():
    
    # print(netcdf2d.run({"id":"input1","export":"csv","variable":"u,v","x":-160.0,"y":40.0,"itime":[0,1]}))
    # print(netcdf2d.run({}))
    
    netcdf2d=S3NetCDFAPI.init({"id":"SWANv5","bucket":"uvic-bcwave","localOnly":False},{})
    
    # Test different combination
    
    # 1 variable
    # print(netcdf2d.run({"export":"csv","variable":"hs","inode":0,"itime":[0,1]}))
    # print(netcdf2d.run({"export":"csv","variable":"hs","inode":[0,1],"itime":[0,1]}))
    # print(netcdf2d.run({"export":"csv","variable":"hs","itime":0}))
    # print(netcdf2d.run({"export":"csv","variable":"hs","inode":0}))
    # print(netcdf2d.run({"export":"csv","variable":"hs","x":-136.7264224,"y":57.39504017,"itime":[0,1,2,3,4,5]}))
    # print(netcdf2d.run({"export":"csv","variable":"hs","x":[-136.7264224,-135.0],"y":[57.39504017,57.],"itime":[0,1,2,3,4,5]}))
    # print(netcdf2d.run({"export":"csv","variable":"hs","inode":0,"start":"2014-01-01","end":"2014-01-10"}))
    
    # Error testing
    # print(netcdf2d.run({"export":"csv","variable":"hs","inode":0,"start":"2003-01-01","end":"2019-01-10"}))
    # print(netcdf2d.run({"export":"csv","variable":"hs","inode":0,"start":"2016-01-01","end":"2019-01-10"}))
    # print(netcdf2d.run({"export":"csv","variable":"hs","x":-160.0,"y":0,"itime":0,"inter.mesh":"linear"}))
    
    
    print(netcdf2d.run({}))
    
    
    
if __name__ == "__main__":
  test_query()
  
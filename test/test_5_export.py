import os
import numpy as np
import pandas as pd
import json
import base64
# from netCDF4 import Dataset,chartostring
from netcdf import NetCDF
from s3netcdfapi import S3NetCDFAPI
# import binpy
import scipy.io as sio
from mbtilesapi import getTile,getVT,readVT,send,VT2Tile,Points2VT,getVTfromBinary

from s3netcdfapi.data import getData
import s3netcdfapi.export as export
input={
  "name":"s3netcdfapi_test",
  "cacheLocation":"../s3",
  "localOnly":True,
  "verbose":True,
  "maxPartitions":40,
  "autoRemove":False,
}




def test_table():
  with S3NetCDFAPI(input) as netcdf:
    # Test X,Y
    obj=netcdf.prepareInput({"variable":"x,y","inode":[0,1]})
    df=export.to_table(obj,getData(netcdf,obj))
    np.testing.assert_array_almost_equal(df['Longitude'].values,[-160.0,-159.899994])
    
    obj=netcdf.prepareInput({"variable":"x,y"})
    df=export.to_table(obj,getData(netcdf,obj))
    np.testing.assert_array_almost_equal(df['Longitude'].values,netcdf['node','x'])
    np.testing.assert_array_almost_equal(df['Latitude'].values,netcdf['node','y'])
    
    # Test Elem
    obj=netcdf.prepareInput({"variable":"elem"})
    df=export.to_table(obj,getData(netcdf,obj))
    
    # Test 2
    obj=netcdf.prepareInput({"variable":"u,v","inode":[0,1,2],"itime":[0,1]})
    df2=export.to_table(obj,getData(netcdf,obj))
    np.testing.assert_array_equal(df2['Latitude'].values,[40,40,40,40,40,40])
    np.testing.assert_array_equal(df2['U Velocity,m/s'].values,[0,1,2,10302,10303,10304])
    
    obj=netcdf.prepareInput({"variable":"u,v","itime":[0]})
    df=export.to_table(obj,getData(netcdf,obj))
    np.testing.assert_array_equal(df['Latitude'].values,netcdf['node','y'])
    np.testing.assert_array_equal(df['U Velocity,m/s'].values,np.squeeze(netcdf['s','u',0]))
    
    # Test 3
    obj=netcdf.prepareInput({"variable":"spectra","isnode":[0],"itime":[0]})
    df3=export.to_table(obj,getData(netcdf,obj))
    np.testing.assert_array_almost_equal(df3['Direction,radian'].values,np.tile(np.arange(36)/36.0,(33,1)).flatten())
    np.testing.assert_array_almost_equal(df3['Frequency,Hz'].values,np.tile(np.arange(33)/33.0,(36,1)).T.flatten())
    np.testing.assert_array_equal(df3['VaDens,m2/Hz/degr'].values,np.arange(33*36)/1000000.0)
    
    
    # Test - Datetime
    obj=netcdf.prepareInput({"variable":"u","start":"2000-01-01","inode":0})
    df=export.to_table(obj,getData(netcdf,obj))
    np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,np.squeeze(netcdf['t','u',0]))
    
    obj=netcdf.prepareInput({"variable":"u","start":"2000-01-01T01","inode":0})
    df=export.to_table(obj,getData(netcdf,obj))
    np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,np.squeeze(netcdf['t','u',0,1:]))
    
    obj=netcdf.prepareInput({"variable":"u","start":"2000-01-01T01","end":"2000-01-01T02","inode":0})
    df=export.to_table(obj,getData(netcdf,obj))
    np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,np.squeeze(netcdf['t','u',0,[1,2]]))
    
    obj=netcdf.prepareInput({"variable":"u","start":"2000-01-01T01:30","end":"2000-01-01T02:30","inode":0})
    df=export.to_table(obj,getData(netcdf,obj))
    np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,np.squeeze(netcdf['t','u',0,[1,2]]))  
    
    obj=netcdf.prepareInput({"variable":"u","start":"2000-01-01T01:30","end":"2000-01-01T02:30","inter.temporal":'linear',"inode":0})
    df=export.to_table(obj,getData(netcdf,obj))
    np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[15453,25755])
    
    obj=netcdf.prepareInput({"variable":"u","start":"2000-01-01T01:30","end":"2000-01-01T02:30","inter.temporal":'linear',"x":-160.0,"y":40.0})
    df=export.to_table(obj,getData(netcdf,obj))
    np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[15453,25755])
    
    obj=netcdf.prepareInput({"variable":"u","start":"2000-01-01T01:30","end":"2000-01-01T02:30","inter.temporal":'linear',"inode":1})
    df=export.to_table(obj,getData(netcdf,obj))
    np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[15454,25756])
    
    obj=netcdf.prepareInput({"variable":"u","start":"2000-01-01T01:30","end":"2000-01-01T02:30","inter.temporal":'linear',"inter.mesh":'linear',"x":-159.95,"y":40.0})
    df=export.to_table(obj,getData(netcdf,obj))
    np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[15453.5,25755.5],4)  
    
    # Test - Spatial  
    obj=netcdf.prepareInput({"variable":"u","itime":0,"x":-159.95,"y":40.0})
    df=export.to_table(obj,getData(netcdf,obj))
    np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[0.0])
    
    obj=netcdf.prepareInput({"variable":"u","itime":0,"x":[-159.95,-159.90],"y":[40.0,40.0]})
    df=export.to_table(obj,getData(netcdf,obj))
    np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[0.0,1.])
    
    obj=netcdf.prepareInput({"variable":"u","itime":0,"inter.mesh":'linear',"x":[-159.95,-159.90],"y":[40.0,40.0]})
    df=export.to_table(obj,getData(netcdf,obj))
    np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[0.5,1.],4)
    
    obj=netcdf.prepareInput({"variable":"time"})
    df=export.to_table(obj,getData(netcdf,obj))
    np.testing.assert_array_almost_equal(df['Datetime'].values.astype("datetime64[h]").astype("float64"),netcdf['time','time'].astype("datetime64[h]").astype("float64"))
    
  
def test_csv():
  with S3NetCDFAPI(input) as netcdf:
    # Test 1
    obj=netcdf.prepareInput({"variable":"u,v","inode":[0,1,2],"itime":[0,1]})
    export.to_csv(obj,getData(netcdf,obj))
    
    df=pd.read_csv(obj['filepath']+".csv")
    np.testing.assert_array_equal(df['Longitude'].values,[-160.0,-159.9,-159.8,-160.0,-159.9,-159.8])
    np.testing.assert_array_equal(df['Latitude'].values,[40.0,40.0,40.0,40.0,40.0,40.0])
    np.testing.assert_array_equal(df['U Velocity,m/s'].values,[0.,1.,2.,10302.,10303.,10304.])

def test_json():
  with S3NetCDFAPI(input) as netcdf:
    
    # Test 1
    obj=netcdf.prepareInput({"variable":"u,v","inode":[0,1,2],"itime":[0,1]})
    export.to_json(obj,getData(netcdf,obj))
    
    df=pd.read_json(obj['filepath']+".json")
    np.testing.assert_array_almost_equal(df['Longitude'].values,[-160.0,-159.9,-159.8,-160.0,-159.9,-159.8],5)
    np.testing.assert_array_almost_equal(df['Latitude'].values,[40.0,40.0,40.0,40.0,40.0,40.0],5)
    np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[0.,1.,2.,10302.,10303.,10304.],5)  

def test_geojson():
  with S3NetCDFAPI(input) as netcdf:
    # Test 1
    obj=netcdf.prepareInput({"variable":"u,v","inode":[0,1,2],"itime":[0,1]})
    export.to_geojson(obj,getData(netcdf,obj))
    with open(obj['filepath']+".geojson") as f:
      geojson = json.load(f)
      data=[feature['properties'] for feature in geojson['features']]
      df=pd.DataFrame(data)
      np.testing.assert_array_almost_equal(df['Longitude'].values,[-160.0,-159.9,-159.8,-160.0,-159.9,-159.8],5)
      np.testing.assert_array_almost_equal(df['Latitude'].values,[40.0,40.0,40.0,40.0,40.0,40.0],5)
      np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[0.,1.,2.,10302.,10303.,10304.],5)     

def test_netcdf():
  with S3NetCDFAPI(input) as netcdf:
    # Test 1
    obj=netcdf.prepareInput({"variable":"u,v","inode":[0,1,2],"itime":[0,1]})
    export.to_netcdf(obj,getData(netcdf,obj))
    
    with NetCDF(obj["filepath"]+".nc", "r") as src_file:
      np.testing.assert_array_equal(src_file['time'][:],np.array(['2000-01-01T00','2000-01-01T01'],dtype="datetime64[ms]"))
      np.testing.assert_array_almost_equal(src_file['x'][:],[-160.0,-159.9,-159.8],5)
      np.testing.assert_array_almost_equal(src_file['y'][:],[40.0,40.0,40.0])
      np.testing.assert_array_almost_equal(src_file['u'][:],[[0.,1.,2.],[10302.,10303.,10304.]])
      
    obj=netcdf.prepareInput({"variable":"spectra","isnode":[0],"itime":[0,1]})
    export.to_netcdf(obj,getData(netcdf,obj))
    
    with NetCDF(obj["filepath"]+".nc", "r") as src_file:
      np.testing.assert_array_equal(src_file['time'][:].astype("datetime64[s]"),np.array(['2000-01-01T00','2000-01-01T01'],dtype="datetime64[h]"))
      np.testing.assert_array_equal(src_file['x'][:],[-160.0])
      np.testing.assert_array_equal(src_file['y'][:],[40.0])
      np.testing.assert_array_equal(src_file['freq'][:],netcdf["freq",'freq'])
      np.testing.assert_array_equal(src_file['dir'][:],netcdf["dir",'dir'])
      np.testing.assert_array_equal(src_file['spectra'][:],netcdf["spc",'spectra',0,:2])
      np.testing.assert_array_equal(src_file['stationname'][:],netcdf["station",'name',0])
      np.testing.assert_array_equal(src_file['stationid'][:],netcdf["snode",'stationid',0])

    obj=netcdf.prepareInput({"variable":"spectra","isnode":[0],"itime":[0,1]})
    export.to_netcdf(obj,getData(netcdf,obj),netcdf3=True)
    with NetCDF(obj["filepath"]+".nc", "r") as src_file:
      assert src_file.file_format=="NETCDF3_CLASSIC" 
      np.testing.assert_array_equal(src_file['time'][:].astype("datetime64[s]"),np.array(['2000-01-01T00','2000-01-01T01'],dtype="datetime64[h]"))
      np.testing.assert_array_equal(src_file['x'][:],[-160.0])
      np.testing.assert_array_equal(src_file['y'][:],[40.0])
      np.testing.assert_array_equal(src_file['freq'][:],netcdf["freq",'freq'])
      np.testing.assert_array_equal(src_file['dir'][:],netcdf["dir",'dir'])
      np.testing.assert_array_equal(src_file['spectra'][:],netcdf["spc",'spectra',0,:2])
      np.testing.assert_array_equal(src_file['stationname'][:],netcdf["station",'name',0])
      np.testing.assert_array_equal(src_file['stationid'][:],netcdf["snode",'stationid',0])
    
    obj=netcdf.prepareInput({"variable":"u","inode":[0],"extra":"false"})
    export.to_netcdf(obj,getData(netcdf,obj),netcdf3=True)
    with NetCDF(obj["filepath"]+".nc", "r") as src_file:
      assert src_file.file_format=="NETCDF3_CLASSIC" 
      

def test_mat():
  with S3NetCDFAPI(input) as netcdf:
    # Test 1
    obj=netcdf.prepareInput({"variable":"u,v","inode":[0,1,2],"itime":[0,1]})
    
    export.to_mat(obj,getData(netcdf,obj))
    
    mat=sio.loadmat(obj["filepath"]+".mat")
    np.testing.assert_array_equal(np.squeeze(mat['time'].astype("datetime64[ms]")),np.array(['2000-01-01T00','2000-01-01T01'],dtype="datetime64[h]"))
    np.testing.assert_array_almost_equal(np.squeeze(mat['x']),[-160.0,-159.9,-159.8],5)
    np.testing.assert_array_almost_equal(np.squeeze(mat['y']),[40.0,40.0,40.0])
    np.testing.assert_array_almost_equal(mat['u'],[[0.,1.,2.],[10302.,10303.,10304.]])
      
    obj=netcdf.prepareInput({"variable":"spectra","isnode":[0],"itime":[0,1]})
    export.to_mat(obj,getData(netcdf,obj))
    
    mat=sio.loadmat(obj["filepath"]+".mat")
    np.testing.assert_array_equal(np.squeeze(mat['time'].astype("datetime64[ms]")),np.array(['2000-01-01T00','2000-01-01T01'],dtype="datetime64[h]"))
    np.testing.assert_array_equal(np.squeeze(mat['x']),[-160.0])
    np.testing.assert_array_equal(np.squeeze(mat['y']),[40.0])
    np.testing.assert_array_equal(np.squeeze(mat['freq']),netcdf["freq",'freq'])
    np.testing.assert_array_equal(np.squeeze(mat['dir']),netcdf["dir",'dir'])
    np.testing.assert_array_equal(mat['spectra'],netcdf["spc",'spectra',0,:2])
    # np.testing.assert_array_equal(np.squeeze(mat['stationname']),netcdf["station",'name',0]) #TODO: Matlab add empty space
    np.testing.assert_array_equal(np.squeeze(mat['stationid']),netcdf["snode",'stationid',0])


def test_mbtiles():
  with S3NetCDFAPI(input) as netcdf:
    # Test 1
    obj=netcdf.prepareInput({"export":"mbtiles","variable":"u","itime":0,"x":0,"y":5,"z":4})
    export.to_mbtiles(obj,getData(netcdf,obj))
    with open(obj['filepath']+".geojson") as f:
      geojson = json.load(f)
      # print(geojson)

# def test_slf():
#   with S3NetCDFAPI(input) as netcdf:
#     obj=netcdf.prepareInput({"export":"slf","variable":"u,v","inode":[0,1,2],"itime":[0,1]})
#     export.to_slf(obj,getData(netcdf,obj))


# def test_binary():
#   with S3NetCDFAPI(input) as netcdf:
#     obj=netcdf.prepareInput({"export":"bin","variable":"mesh"})
#     export.to_binary(netcdf,obj,getData(netcdf,obj))
#     with open(obj["filepath"]+".bin","rb") as f:results=binpy.read(f)
#     np.testing.assert_array_almost_equal(results['elem'],netcdf['elem','elem'])
#     np.testing.assert_array_almost_equal(results['x'],netcdf['node','x'])
#     np.testing.assert_array_almost_equal(results['y'],netcdf['node','y'])
    
#     obj=netcdf.prepareInput({"export":"bin","variable":"time"})
#     export.to_binary(netcdf,obj,getData(netcdf,obj))
#     with open(obj["filepath"]+".bin","rb") as f:results=binpy.read(f)
#     np.testing.assert_array_equal(results['time'],netcdf['time','time'])
    
    
#     obj=netcdf.prepareInput({"export":"bin","variable":"freq"})
#     export.to_binary(netcdf,obj,getData(netcdf,obj))
#     with open(obj["filepath"]+".bin","rb") as f:results=binpy.read(f)
#     np.testing.assert_array_equal(results['freq'],netcdf['freq','freq'])
    
#     obj=netcdf.prepareInput({"export":"bin","variable":"u","itime":0})
#     export.to_binary(netcdf,obj,getData(netcdf,obj),0,10301)
#     with open(obj["filepath"]+".bin","rb") as f:results=binpy.read(f)
#     image=results['u_s_0'].reshape(netcdf.res()*netcdf.res(),2)
#     nnode=netcdf._meta['dimensions']['nnode']
#     np.testing.assert_array_almost_equal(np.round(export.decode(image,0,10301)[:nnode]),np.squeeze(netcdf['s','u',0]))
    
#     obj=netcdf.prepareInput({"export":"bin","variable":"u","inode":0})
#     export.to_binary(netcdf,obj,getData(netcdf,obj))
#     with open(obj["filepath"]+".bin","rb") as f:results=binpy.read(f)
#     np.testing.assert_array_almost_equal(results['u_t_0'],np.squeeze(netcdf['t','u',0]))  
    
#     obj=netcdf.prepareInput({"export":"bin","variable":"u","x":-159.0,"y":40.0})
#     export.to_binary(netcdf,obj,getData(netcdf,obj))
#     with open(obj["filepath"]+".bin","rb") as f:results=binpy.read(f)
#     np.testing.assert_array_almost_equal(results['u_t_-159.0_40.0'],np.squeeze(netcdf['t','u',10]))  
    
#     obj=netcdf.prepareInput({"export":"bin","variable":"spectra","isnode":0,"itime":0})
#     export.to_binary(netcdf,obj,getData(netcdf,obj))
#     with open(obj["filepath"]+".bin","rb") as f:results=binpy.read(f)
#     np.testing.assert_array_almost_equal(results['spectra_0_0'],np.squeeze(netcdf['spc','spectra',0,0]))  
    
#     obj=netcdf.prepareInput({"export":"bin","variable":"spectra","itime":0,"x":-159.0,"y":40.0})
#     export.to_binary(netcdf,obj,getData(netcdf,obj))
#     with open(obj["filepath"]+".bin","rb") as f:results=binpy.read(f)
#     np.testing.assert_array_almost_equal(results['spectra_-159.0_40.0_0'],np.squeeze(netcdf['spc','spectra',5,0]))  
    
#     obj=netcdf.prepareInput({"export":"bin","variable":"spectra","start":"2000-01-01T02","end":"2000-01-01T02","x":-159.0,"y":40.0})
#     export.to_binary(netcdf,obj,getData(netcdf,obj))
#     with open(obj["filepath"]+".bin","rb") as f:results=binpy.read(f)
#     np.testing.assert_array_almost_equal(results['spectra_-159.0_40.0_2000-01-01T02:00:00'],np.squeeze(netcdf['spc','spectra',5,2])) 
 
  
# def test_mat():
#     export.mat()


# def test_shapefile():
#     export.shapefile()


# def test_tri():
#     export.tri()


if __name__ == "__main__":
  test_table()
  test_csv()
  test_json()
  test_geojson()
  test_netcdf()
  test_mbtiles()
  test_mat()
  # test_binary()
  
  # test_slf()
  # test_mat()
  # test_shapefile()
  # test_tri()
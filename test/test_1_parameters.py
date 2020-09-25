import pytest
import numpy as np
import copy
from s3netcdf import NetCDF2D
from s3netcdfapi.parameters import getParameters,getGroups,setGroups,parseParameters,checkSpatial,checkTemporal,checkExport,check




input={
  "name":"input1",
  "cacheLocation":"../s3",
  "localOnly":True,
  "verbose":True
}
pointers={"pointers":{
    "mesh":{"dimensions":["nnode"],"x":["x","lng",'longitude','lon'],"y":["y","lat","latitude"]},
    "temporal":{"dimensions":["ntime"],"time":["time"]},
    "xy":{"dimensions":["nsnode"],"x":["x","sx"],"y":["y","sy"]},
    }}
    
default={
    'output':{"default":"output","type":str},
    'export':{"default":"json","type":str},
    "dataOnly":{"default":False,"type":bool},

    'variable':{"default":None,"type":(str,list)},
    
    'inode':{"default":None,"type":(int,list,slice)},
    'isnode':{"default":None,"type":(int,list,slice)},
    'longitude':{"default":None,"type":(float,list)},
    'latitude':{"default":None,"type":(float,list)},
    'x':{"default":None,"type":(float,list)},
    'y':{"default":None,"type":(float,list)},
    
    'itime':{"default":None,"type":(int,list,slice)},
    'start':{"default":None,"type":str},
    'end':{"default":None,"type":str},
    'step':{"default":1,"type":int},
    'stepUnit':{"default":'h',"type":str},
    
    'inter.mesh':{"default":'closest',"type":str},
    'inter.temporal':{"default":'closest',"type":str},
    'inter.xy':{"default":'closest',"type":str},
    
    'sep':{"default":',',"type":str},
}

netcdf2d=NetCDF2D(input)


def test_getGroups():
  assert getGroups(netcdf2d,{'variable':[]})==[]
  assert getGroups(netcdf2d,{'variable':['u']})==[['s','t']]
  assert getGroups(netcdf2d,{'variable':['u','v']})==[['s','t']]
  assert getGroups(netcdf2d,{'variable':['x','y']})==[['node']]
  assert getGroups(netcdf2d,{'variable':['sx','sy']})==[['snode']]
  assert getGroups(netcdf2d,{'variable':['u','x']})==[['s','t'],['node']]
  assert getGroups(netcdf2d,{'variable':['u','v','x']})==[['s','t'],['node']]


def test_parseParameters():
  # Test1
  obj=copy.deepcopy(default)
  parameters={'variable':'[u,v]','itime':'0:10','step':'2'}
  assert parseParameters(obj,parameters)['variable']==['u','v']
  assert parseParameters(obj,parameters)['itime']==slice(0,10,None)
  assert parseParameters(obj,parameters)['step']==2

  # Test2
  obj=copy.deepcopy(default)
  parameters={'variable':'u,v]','itime':':10'}
  assert parseParameters(obj,parameters)['variable']==['u','v']
  assert parseParameters(obj,parameters)['itime']==slice(None,10,None)
  assert parseParameters(obj,parameters)['step']==1
  
  # Test3
  obj=copy.deepcopy(default)
  parameters3={'variable':'u,v'}
  assert parseParameters(obj,parameters)['variable']==['u','v']
  
 
  # Test4
  obj=copy.deepcopy(default)
  parameters={'variable':'u,v','longitude':'0.1,0.2'}
  assert parseParameters(obj,parameters)['variable']==['u','v']
  assert parseParameters(obj,parameters)['longitude']==[0.1,0.2]


  # # Test5
  obj=copy.deepcopy(default)
  parameters={'variable':'u,v','longitude':[0.1,0.2]}
  assert parseParameters(obj,parameters)['variable']==['u','v']
  assert parseParameters(obj,parameters)['longitude']==[0.1,0.2]
  
  # Test6
  obj=copy.deepcopy(default)
  parameters={'variable':'u,v','longitude':[0.1,'0.2']}
  assert parseParameters(obj,parameters)['variable']==['u','v']
  assert parseParameters(obj,parameters)['longitude']==[0.1,0.2]
  
  # Test7
  obj7=copy.deepcopy(default)
  parameters7={'variable':'u,v','longitude':[0.1,'0..2']}
  with pytest.raises(Exception):assert parseParameters(obj7,parameters7)
  
  # Test8
  obj8=copy.deepcopy(default)
  parameters8={'export':'u,v'}
  with pytest.raises(Exception):assert parseParameters(obj8,parameters8)

def test_check():
  #Test 1
  obj=copy.deepcopy(default)
  obj=setGroups(netcdf2d,{**pointers,**parseParameters(default,{'longitude':[0.1],'latitude':[0.2],"variable":"x"})})
  np.testing.assert_array_equal(check(netcdf2d,obj)['xy'],[[0.1, 0.2]])
  
  # Test 2
  obj=copy.deepcopy(default)
  obj=setGroups(netcdf2d,{**pointers,**parseParameters(default,{'longitude':[0.1,0.1],'latitude':[0.2,0.2],"variable":"x"})})
  obj=check(netcdf2d,obj)
  np.testing.assert_array_equal(obj['xy'],[[0.1, 0.2],[0.1, 0.2]])
  np.testing.assert_array_equal(obj['nodeIndex'],[0,0])
  
  # Test 3
  obj=copy.deepcopy(default)
  obj=setGroups(netcdf2d,{**pointers,**parseParameters(obj,{'longitude':[-159.85],'latitude':[40.0],'inter.mesh':"linear","variable":"x"})})
  obj=check(netcdf2d,obj)
  np.testing.assert_array_almost_equal(obj['meshx'],[-159.9,-159.8,-159.9],5)
  np.testing.assert_array_almost_equal(obj['meshy'],[40.0,40.0,40.1],5)
  np.testing.assert_array_equal(obj['elem'],[[0,2,1]])
  
  obj=copy.deepcopy(default)
  obj=setGroups(netcdf2d,{**pointers,**parseParameters(obj,{'longitude':[-159.85,-159.75],'latitude':[40.0,40.0],'inter.mesh':"linear","variable":"x"})})
  obj=check(netcdf2d,obj)
  np.testing.assert_array_almost_equal(obj['meshx'],[-159.9,-159.8,-159.7,-159.9,-159.8],5)
  np.testing.assert_array_almost_equal(obj['meshy'],[40.0,40.0,40.0,40.1,40.1],5)
  np.testing.assert_array_equal(obj['elem'],[[0,3,1],[1,4,2]])
  
  # Test 4
  obj=copy.deepcopy(default)
  obj=setGroups(netcdf2d,{**pointers,**parseParameters(obj,{'longitude':[-159.85,-159.75],'latitude':[40.0,40.0],'inode':'0:10',"variable":"x"})})
  obj=check(netcdf2d,obj)
  assert obj['inode']==slice(0, 10, None)

  # Test 5
  obj=copy.deepcopy(default)
  obj=setGroups(netcdf2d,{**pointers,**parseParameters(obj,{'itime':0,'start':'2000-01-01',"variable":"time"})})
  obj=check(netcdf2d,obj)
  assert obj['itime']==[0]
  assert obj['start']==None

  # Test 6
  obj=copy.deepcopy(default)
  obj=setGroups(netcdf2d,{**pointers,**parseParameters(obj,{'start':'2000-02-11T13',"variable":"time"})})
  obj=check(netcdf2d,obj)
  np.testing.assert_array_equal(obj['start'],np.array(np.datetime64('2000-02-11T13')))
  np.testing.assert_array_equal(obj['end'],np.array(np.datetime64('2000-02-11T15')))


def test_checkExport():
  # Test 1
  obj1=copy.deepcopy(default)
  parameters1={'export':'csv','variable':'u,x'}
  with pytest.raises(Exception):assert checkExport(netcdf2d,parseParameters(obj1,parameters1))  

def test_getParameters():
  # Test 1
  parameters={'variable':'u','longitude':[0.1,0.1],'latitude':[0.2,0.2],'itime':0}
  obj=getParameters(netcdf2d,parameters)
  np.testing.assert_array_equal(obj['nodeIndex'],[0,0])

if __name__ == "__main__":
  test_getGroups()
  test_parseParameters()
  test_check()
  test_checkExport()
  test_getParameters()

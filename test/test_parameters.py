from s3netcdf import NetCDF2D
from s3netcdfapi.parameters import getParameters,getGroups,setGroups,parseParameters,checkSpatial,checkTemporal,checkExport
import copy
import pytest
import numpy as np

input={
  "name":"input1",
  "cacheLocation":"../s3",
  "localOnly":True,
  "verbose":True
}

default={
  'export':{"default":"json","type":str},
  "dataOnly":{"default":False,"type":bool},
  'variable':{"default":None,"type":(str,list)},
  
  'inode':{"default":None,"type":(int,list,slice)},
  'longitude':{"default":None,"type":(float,list)},
  'latitude':{"default":None,"type":(float,list)},
  'x':{"default":None,"type":(float,list)},
  'y':{"default":None,"type":(float,list)},
  
  'isnode':{"default":None,"type":(int,list,slice)},
  'slongitude':{"default":None,"type":(float,list)},
  'slatitude':{"default":None,"type":(float,list)},    
  'sx':{"default":None,"type":(float,list)},
  'sy':{"default":None,"type":(float,list)},    
  
  'itime':{"default":None,"type":(int,list,slice)},
  'start':{"default":None,"type":str},
  'end':{"default":None,"type":str},
  'step':{"default":1,"type":int},
  'stepUnit':{"default":'h',"type":str},

  
    
  # 'pointer':{"default":{
  #   "meshx":{'variable':'x'},
  #   "meshy":{'variable':'y'},
  #   "elem":{'variable':'elem'},
  #   "time":{'variable':'time'},
  #   "sx":{'variable':'sx'},
  #   "sy":{'variable':'sy'},
  #   "dimensions":{"itime":'itime',"inode":'inode',"isnode":'isnode'},
  #   },"type":(object)},
        
  
  # 'meshx':{"default":None,"type":(float,list)},
  # 'meshy':{"default":None,"type":(float,list)},
  # 'elem':{"default":None,"type":(float,list)},
  
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
  obj1=copy.deepcopy(default)
  parameters1={'variable':'[u,v]','itime':'0:10','step':'2'}
  
  assert parseParameters(obj1,parameters1)=={'export': 'json', 'dataOnly': False, 'variable': ['u', 'v'], 'inode': None, 'longitude': None, 'latitude': None, 'x': None, 'y': None, 'isnode': None, 'slongitude': None, 'slatitude': None, 'sx': None, 'sy': None, 'itime': slice(0, 10, None), 'start': None, 'end': None, 'step': 2, 'stepUnit': 'h'}
  
  # Test2
  obj2=copy.deepcopy(default)
  parameters2={'variable':'u,v]','itime':':10'}
  assert parseParameters(obj2,parameters2)=={'export': 'json', 'dataOnly': False, 'variable': ['u', 'v'], 'inode': None, 'longitude': None, 'latitude': None, 'x': None, 'y': None, 'isnode': None, 'slongitude': None, 'slatitude': None, 'sx': None, 'sy': None, 'itime': slice(None, 10, None), 'start': None, 'end': None, 'step': 1, 'stepUnit': 'h'}
  
  # Test3
  obj3=copy.deepcopy(default)
  parameters3={'variable':'u,v'}
  
  assert parseParameters(obj3,parameters3)=={'export': 'json', 'dataOnly': False, 'variable': ['u', 'v'], 'inode': None, 'longitude': None, 'latitude': None, 'x': None, 'y': None, 'isnode': None, 'slongitude': None, 'slatitude': None, 'sx': None, 'sy': None, 'itime': None, 'start': None, 'end': None, 'step': 1, 'stepUnit': 'h'}
 
  # Test4
  obj4=copy.deepcopy(default)
  parameters4={'variable':'u,v','longitude':'0.1,0.2'}
  
  assert parseParameters(obj4,parameters4)=={'export': 'json', 'dataOnly': False, 'variable': ['u', 'v'], 'inode': None, 'longitude': [0.1, 0.2], 'latitude': None, 'x': None, 'y': None, 'isnode': None, 'slongitude': None, 'slatitude': None, 'sx': None, 'sy': None, 'itime': None, 'start': None, 'end': None, 'step': 1, 'stepUnit': 'h'}

  # # Test5
  obj5=copy.deepcopy(default)
  parameters5={'variable':'u,v','longitude':[0.1,0.2]}
  assert parseParameters(obj5,parameters5)=={'export': 'json', 'dataOnly': False, 'variable': ['u', 'v'], 'inode': None, 'longitude': [0.1, 0.2], 'latitude': None, 'x': None, 'y': None, 'isnode': None, 'slongitude': None, 'slatitude': None, 'sx': None, 'sy': None, 'itime': None, 'start': None, 'end': None, 'step': 1, 'stepUnit': 'h'}
  
  # Test6
  obj6=copy.deepcopy(default)
  parameters6={'variable':'u,v','longitude':[0.1,'0.2']}
  assert parseParameters(obj6,parameters6)=={'export': 'json', 'dataOnly': False, 'variable': ['u', 'v'], 'inode': None, 'longitude': [0.1, 0.2], 'latitude': None, 'x': None, 'y': None, 'isnode': None, 'slongitude': None, 'slatitude': None, 'sx': None, 'sy': None, 'itime': None, 'start': None, 'end': None, 'step': 1, 'stepUnit': 'h'}
  
  # Test7
  obj7=copy.deepcopy(default)
  parameters7={'variable':'u,v','longitude':[0.1,'0..2']}
  with pytest.raises(Exception):assert parseParameters(obj7,parameters7)
  
  # Test8
  obj8=copy.deepcopy(default)
  parameters8={'export':'u,v'}
  with pytest.raises(Exception):assert parseParameters(obj8,parameters8)


def test_checkSpatial():
  default['inter.spatial']={"default":'closest',"type":str}
  default['inter.temporal']={"default":'closest',"type":str}
  default['inter.spectral']={"default":'closest',"type":str}
  
  # Test 1
  obj1=copy.deepcopy(default)
  parameters1={'longitude':0.1,'latitude':0.2}
  r1=checkSpatial(netcdf2d,parseParameters(obj1,parameters1))
  np.testing.assert_array_equal(r1['xy'],[[0.1, 0.2]])
  
  # Test 2
  obj2=copy.deepcopy(default)
  parameters2={'longitude':[0.1,0.1],'latitude':[0.2,0.2]}
  r2=checkSpatial(netcdf2d,parseParameters(obj2,parameters2))
  np.testing.assert_array_equal(r2['xy'],[[0.1, 0.2],[0.1, 0.2]])
  np.testing.assert_array_equal(r2['xyIndex'],[0,0])
  
  # Test 3
  obj3=copy.deepcopy(default)
  parameters3={'longitude':[0.1,0.1],'latitude':[0.2,0.2],'inter.spatial':"linear"}
  r3=checkSpatial(netcdf2d,parseParameters(obj3,parameters3))
  np.testing.assert_array_equal(r3['xy'],[[0.1, 0.2],[0.1, 0.2]])
  np.testing.assert_array_almost_equal(r3['meshx'],[-150.,-150.1000061,-150.])
  np.testing.assert_array_almost_equal(r3['meshy'],[50.,50.09999847,50.09999847])
  
  # Test 4
  obj4=copy.deepcopy(default)
  parameters4={'longitude':[0.1,0.1],'latitude':[0.2,0.2],'inode':'0:10'}
  r4=checkSpatial(netcdf2d,parseParameters(obj4,parameters4))
  assert r4['inode']==slice(0, 10, None)

def test_checkTemporal():
  # Test 1
  obj1=copy.deepcopy(default)
  parameters1={'itime':0,'start':'2000-01-01'}
  r1=checkTemporal(netcdf2d,parseParameters(obj1,parameters1))
  assert r1=={'export': 'json', 'dataOnly': False, 'variable': None, 'inode': None, 'longitude': None, 'latitude': None, 'x': None, 'y': None, 'isnode': None, 'slongitude': None, 'slatitude': None, 'sx': None, 'sy': None, 'itime': [0], 'start': None, 'end': None, 'step': None, 'stepUnit': 'h', 'user_time': False}

  # Test 2
  obj2=copy.deepcopy(default)
  parameters2={'start':'2000-02-11T13'}
  r2=checkTemporal(netcdf2d,parseParameters(obj2,parameters2))
  np.testing.assert_array_equal(r2['start'],np.array(np.datetime64('2000-02-11T13')))
  np.testing.assert_array_equal(r2['end'],np.array(np.datetime64('2000-02-11T15')))


def test_checkExport():

  # Test 1
  obj1=copy.deepcopy(default)
  # parameters1={'mesh':True,'export':'csv'}
  # with pytest.raises(Exception):assert checkExport(netcdf2d,parseParameters(obj1,parameters1))
  
  # Test 2
  # obj2=copy.deepcopy(default)
  # parameters2={'mesh':True,'export':'geojson','variable':'u'}
  # with pytest.raises(Exception):assert checkExport(netcdf2d,parseParameters(obj2,parameters2))  
  
  # Test 3
  obj3=copy.deepcopy(default)
  parameters3={'export':'csv','variable':'u,x'}
  with pytest.raises(Exception):assert checkExport(netcdf2d,parseParameters(obj3,parameters3))  

def test_getParameters():

  # Test 1
  parameters={'variable':'u','longitude':[0.1,0.1],'latitude':[0.2,0.2],'itime':0}
  obj=getParameters(netcdf2d,parameters)
  np.testing.assert_array_equal(obj['xyIndex'],[0,0])
  
  parameters={"export":"geojson",'variable':'mesh'}
  obj=getParameters(netcdf2d,parameters)
  # print(obj)
  # np.testing.assert_array_equal(obj['xyIndex'],[0,0])
  


if __name__ == "__main__":
  test_getGroups()
  test_parseParameters()
  test_checkSpatial()
  test_checkTemporal()
  test_checkExport()
  test_getParameters()

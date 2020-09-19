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
  'mesh':{"default":False,"type":bool},
  'variable':{"default":None,"type":(str,list)},
  'inode':{"default":None,"type":(int,slice,list)},
  'longitude':{"default":None,"type":(float,list)},
  'latitude':{"default":None,"type":(float,list)},
  'x':{"default":None,"type":(float,list)},
  'y':{"default":None,"type":(float,list)},
  'itime':{"default":None,"type":(int,slice,list)},
  'start':{"default":None,"type":str},
  'end':{"default":None,"type":str},
  'step':{"default":1,"type":int},
  'stepUnit':{"default":'h',"type":str},
  'smethod':{"default":'closest',"type":str},
  'tmethod':{"default":'closest',"type":str},
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
  parameters1={'variable':'[u,v]','itime':'0:10','mesh':'True','step':'2'}
  assert parseParameters(obj1,parameters1)=={'export':'json','mesh': True, 'variable': ['u', 'v'], 'inode': None, 'longitude': None, 'latitude': None, 'x': None, 'y': None, 'itime': slice(0, 10, None), 'start': None, 'end': None, 'step': 2, 'stepUnit': 'h', 'smethod': 'closest', 'tmethod': 'closest'}
  
  # Test2
  obj2=copy.deepcopy(default)
  parameters2={'variable':'u,v]','itime':':10','mesh':'False'}
  assert parseParameters(obj2,parameters2)=={'export': 'json', 'mesh': False, 'variable': ['u', 'v'], 'inode': None, 'longitude': None, 'latitude': None, 'x': None, 'y': None, 'itime': slice(None, 10, None), 'start': None, 'end': None, 'step': 1, 'stepUnit': 'h', 'smethod': 'closest', 'tmethod': 'closest'}
  
  # Test3
  obj3=copy.deepcopy(default)
  parameters3={'variable':'u,v'}
  assert parseParameters(obj3,parameters3)=={'export': 'json', 'mesh': False, 'variable': ['u', 'v'], 'inode': None, 'longitude': None, 'latitude': None, 'x': None, 'y': None, 'itime': None, 'start': None, 'end': None, 'step': 1, 'stepUnit': 'h', 'smethod': 'closest', 'tmethod': 'closest'}
 
  # Test4
  obj4=copy.deepcopy(default)
  parameters4={'variable':'u,v','longitude':'0.1,0.2'}
  assert parseParameters(obj4,parameters4)=={'export': 'json', 'mesh': False, 'variable': ['u', 'v'], 'inode': None, 'longitude': [0.1,0.2], 'latitude': None, 'x': None, 'y': None, 'itime': None, 'start': None, 'end': None, 'step': 1, 'stepUnit': 'h', 'smethod': 'closest', 'tmethod': 'closest'}

  # Test5
  obj5=copy.deepcopy(default)
  parameters5={'variable':'u,v','longitude':[0.1,0.2]}
  assert parseParameters(obj5,parameters5)=={'export': 'json', 'mesh': False, 'variable': ['u', 'v'], 'inode': None, 'longitude': [0.1,0.2], 'latitude': None, 'x': None, 'y': None, 'itime': None, 'start': None, 'end': None, 'step': 1, 'stepUnit': 'h', 'smethod': 'closest', 'tmethod': 'closest'}
  
  # Test6
  obj6=copy.deepcopy(default)
  parameters6={'variable':'u,v','longitude':[0.1,'0.2']}
  assert parseParameters(obj6,parameters6)=={'export': 'json', 'mesh': False, 'variable': ['u', 'v'], 'inode': None, 'longitude': [0.1,0.2], 'latitude': None, 'x': None, 'y': None, 'itime': None, 'start': None, 'end': None, 'step': 1, 'stepUnit': 'h', 'smethod': 'closest', 'tmethod': 'closest'}
  
  # Test7
  obj7=copy.deepcopy(default)
  parameters7={'variable':'u,v','longitude':[0.1,'0..2']}
  with pytest.raises(Exception):assert parseParameters(obj7,parameters7)
  
  # Test8
  obj8=copy.deepcopy(default)
  parameters8={'export':'u,v'}
  with pytest.raises(Exception):assert parseParameters(obj8,parameters8)


def test_checkSpatial():
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
  
  # Test 3
  obj3=copy.deepcopy(default)
  parameters3={'longitude':[0.1,0.1],'latitude':[0.2,0.2],'inode':'0:10'}
  r3=checkSpatial(netcdf2d,parseParameters(obj3,parameters3))
  assert r3=={'export': 'json', 'mesh': False, 'variable': None, 'inode': [slice(0, 10, None)], 'x': None, 'y': None, 'itime': None, 'start': None, 'end': None, 'step': 1, 'stepUnit': 'h', 'smethod': 'closest', 'tmethod': 'closest', 'xy': None}

def test_checkTemporal():
  # Test 1
  obj1=copy.deepcopy(default)
  parameters1={'itime':0,'start':'2000-01-01'}
  r1=checkTemporal(netcdf2d,parseParameters(obj1,parameters1))
  assert r1=={'export': 'json', 'mesh': False, 'variable': None, 'inode': None, 'longitude': None, 'latitude': None, 'x': None, 'y': None, 'itime': [0], 'start': None, 'end': None, 'step': None, 'stepUnit': 'h', 'smethod': 'closest', 'tmethod': 'closest', 'dt': None}

  # Test 2
  obj2=copy.deepcopy(default)
  parameters2={'start':'2000-02-11T13'}
  r2=checkTemporal(netcdf2d,parseParameters(obj2,parameters2))
  np.testing.assert_array_equal(r2['start'],np.array(np.datetime64('2000-02-11T13')))
  np.testing.assert_array_equal(r2['end'],np.array(np.datetime64('2000-02-11T15')))


def test_checkExport():

  # Test 1
  obj1=copy.deepcopy(default)
  parameters1={'mesh':True,'export':'csv'}
  with pytest.raises(Exception):assert checkExport(netcdf2d,parseParameters(obj1,parameters1))
  
  # Test 2
  obj2=copy.deepcopy(default)
  parameters2={'mesh':True,'export':'geojson','variable':'u'}
  with pytest.raises(Exception):assert checkExport(netcdf2d,parseParameters(obj2,parameters2))  
  
  # Test 3
  obj3=copy.deepcopy(default)
  parameters3={'export':'csv','variable':'u,x'}
  with pytest.raises(Exception):assert checkExport(netcdf2d,parseParameters(obj3,parameters3))  

def test_getParameters():

  # Test 1
  parameters1={'variable':'u','longitude':[0.1,0.1],'latitude':[0.2,0.2],'itime':0}
  obj=getParameters(netcdf2d,parameters1)
  # del r1['xy']
  # del r1['pointer']
  # assert r1=={'export': 'json', 'mesh': False, 'variable': [], 'inode': None, 'meshx': None, 'meshy': None, 'elem': None, 'x': [0.1, 0.1], 'y': [0.2, 0.2], 'itime': [0], 'start': None, 'end': None, 'step': None, 'stepUnit': 'h', 'smethod': 'closest', 'tmethod': 'closest', 'groups': [], 'ngroups': 0, 'isTable': False, 'dt': None}
  # print(r1)
  
  # Get data
  data={}
  for variable in obj['variable']:
    dimensions=netcdf2d.getDimensionsByVariable(variable)
    tmp=[]
    for dimension in dimensions:
      if not dimension in obj['methods']:raise Exception("No method was specified for {}".format(dimension))
      # if obj['methods'][dimension]=='spatial': getSpatialIdx
      # if obj['methods'][dimension]=='temporal': getTemporalIdx
    # if len(tmp)>1 and spatial>temporal, change spatial to all
    # elif len(tmp)>1 and temporal>spatial, 
    # else if temoral> timesries, spatial>
    # Get Data
    
        
    # conditions=[obj['methods'][dimension] for dimension in dimensions]
    # if 'spectral' in conditions:
    #   None # get x,y
    # elif 'spatial' in conditions and 'temporal'in conditions:
    #   None
    # elif 'spatial' in conditions:
    #   None
    # elif 'temporal' in conditions:
    #   None  
    # else:
    #   None
    #   # if obj['methods'][dimension]=='spatial': getSpatialIdx
    #   # if obj['methods'][dimension]=='temporal': getSpatialIdx
    #   print(obj['methods'][dimension])



if __name__ == "__main__":
  # test_getGroups()
  # test_parseParameters()
  # test_checkSpatial()
  # test_checkTemporal()
  # test_checkExport()
  test_getParameters()

import pytest
import numpy as np
import copy
from s3netcdfapi import S3NetCDFAPI
from s3netcdfapi.utils import parseParameters

input={
  "name":"input1",
  "cacheLocation":"../s3",
  "apiCacheLocation":"../s3/tmp",
  "localOnly":True,
  "verbose":True
}

netcdf2d=S3NetCDFAPI(input)

def test_getDefaultParameters():
  # print(netcdf2d.getDefaultParameters())
  None

def test_getGroups():
  
  assert netcdf2d.getGroups({'variable':[]})==[]
  assert netcdf2d.getGroups({'variable':['u']})==[['s','t']]
  assert netcdf2d.getGroups({'variable':['u','v']})==[['s','t']]
  assert netcdf2d.getGroups({'variable':['x','y']})==[['node']]
  assert netcdf2d.getGroups({'variable':['sx','sy']})==[['snode']]
  assert netcdf2d.getGroups({'variable':['u','x']})==[['s','t'],['node']]
  assert netcdf2d.getGroups({'variable':['u','v','x']})==[['s','t'],['node']]


def test_parseParameters():
  # Test1
  obj=netcdf2d.getDefaultParameters()
  
  parameters={'variable':'[u,v]','itime':'0:10','step':'2'}
  assert parseParameters(obj,parameters)['variable']==['u','v']
  assert parseParameters(obj,parameters)['itime']==slice(0,10,None)
  assert parseParameters(obj,parameters)['step']==2

  # Test2
  obj=netcdf2d.getDefaultParameters()
  parameters={'variable':'u,v]','itime':':10'}
  assert parseParameters(obj,parameters)['variable']==['u','v']
  assert parseParameters(obj,parameters)['itime']==slice(None,10,None)
  assert parseParameters(obj,parameters)['step']==1
  
  # Test3
  obj=netcdf2d.getDefaultParameters()
  parameters3={'variable':'u,v'}
  assert parseParameters(obj,parameters)['variable']==['u','v']
  
 
  # Test4
  obj=netcdf2d.getDefaultParameters()
  parameters={'variable':'u,v','longitude':'0.1,0.2'}
  assert parseParameters(obj,parameters)['variable']==['u','v']
  assert parseParameters(obj,parameters)['longitude']==[0.1,0.2]


  # # Test5
  obj=netcdf2d.getDefaultParameters()
  parameters={'variable':'u,v','longitude':[0.1,0.2]}
  assert parseParameters(obj,parameters)['variable']==['u','v']
  assert parseParameters(obj,parameters)['longitude']==[0.1,0.2]
  
  # Test6
  obj=netcdf2d.getDefaultParameters()
  parameters={'variable':'u,v','longitude':[0.1,'0.2']}
  assert parseParameters(obj,parameters)['variable']==['u','v']
  assert parseParameters(obj,parameters)['longitude']==[0.1,0.2]
  
  # Test7
  obj7=netcdf2d.getDefaultParameters()
  parameters7={'variable':'u,v','longitude':[0.1,'0..2']}
  with pytest.raises(Exception):assert parseParameters(obj7,parameters7)
  
  # Test8
  obj8=netcdf2d.getDefaultParameters()
  parameters8={'export':'u,v'}
  with pytest.raises(Exception):assert parseParameters(obj8,parameters8)

def test_checkExport():
  # Test 1
  obj1=netcdf2d.getDefaultParameters()
  parameters1={'export':'csv','variable':'u,x'}
  with pytest.raises(Exception):assert netcdf2d.checkExport(parseParameters(obj1,parameters1)) 

def test_prepareInput():
  #Test 1
  obj=netcdf2d.prepareInput({'longitude':[0.1],'latitude':[0.2],"variable":"x"})
  np.testing.assert_array_equal(obj['xy'],[[0.1, 0.2]])
  
  # Test 2
  obj=netcdf2d.prepareInput({'longitude':[0.1,0.1],'latitude':[0.2,0.2],"variable":"x"})
  np.testing.assert_array_equal(obj['xy'],[[0.1, 0.2],[0.1, 0.2]])
  np.testing.assert_array_equal(obj['nodeIndex'],[0,0])
  
  # Test 3

  obj=netcdf2d.prepareInput({'longitude':[-159.85],'latitude':[40.0],'inter.mesh':"linear","variable":"x"})
  np.testing.assert_array_almost_equal(obj['meshx'],[-159.9,-159.8,-159.9],5)
  np.testing.assert_array_almost_equal(obj['meshy'],[40.0,40.0,40.1],5)
  np.testing.assert_array_equal(obj['elem'],[[0,2,1]])
  

  obj=netcdf2d.prepareInput({'longitude':[-159.85,-159.75],'latitude':[40.0,40.0],'inter.mesh':"linear","variable":"x"})
  np.testing.assert_array_almost_equal(obj['meshx'],[-159.9,-159.8,-159.7,-159.9,-159.8],5)
  np.testing.assert_array_almost_equal(obj['meshy'],[40.0,40.0,40.0,40.1,40.1],5)
  np.testing.assert_array_equal(obj['elem'],[[0,3,1],[1,4,2]])
  
  # Test 4
  obj=netcdf2d.prepareInput({'longitude':[-159.85,-159.75],'latitude':[40.0,40.0],'inode':'0:10',"variable":"x"})
  assert obj['inode']==slice(0, 10, None)

  # Test 5
  obj=netcdf2d.prepareInput({'itime':0,'start':'2000-01-01',"variable":"time"})
  assert obj['itime']==[0]
  assert obj['start']==None

  # Test 6
  obj=netcdf2d.prepareInput({'start':'2000-02-11T13',"variable":"time"})
  np.testing.assert_array_equal(obj['start'],np.array(np.datetime64('2000-02-11T13')))
  np.testing.assert_array_equal(obj['end'],np.array(np.datetime64('2000-02-11T15')))

# def test_cacheName():
#   parameters={'longitude':[0.1],'latitude':[0.2],"variable":"x"}
#   parameters={'longitude':[0.1,0.1],'latitude':[0.2,0.2],"variable":"x,y"}
#   parameters={'longitude':[0.1,0.1],'latitude':[0.2,0.2],"variable":"x,y,hs","start":"2000-01-01","end":"2000-01-02"}
#   netcdf2d.getCacheName(parameters)
#   None

if __name__ == "__main__":
  test_getDefaultParameters()
  test_getGroups()
  test_parseParameters()
  test_checkExport()
  test_prepareInput()
  # test_cacheName()
import pytest
import numpy as np
import copy
from s3netcdfapi import S3NetCDFAPI
from s3netcdfapi.utils import parseParameters

input={
  "name":"s3netcdfapi_test",
  "cacheLocation":"../s3",
  "localOnly":True,
  "verbose":True
}


def test_getGroups():
  with S3NetCDFAPI(input) as netcdf:
    assert netcdf.getGroups({'variable':[]})==[]
    np.testing.assert_array_equal(np.sort(netcdf.getGroups({'variable':['u']})),np.array([['s','t']]))
    np.testing.assert_array_equal(np.sort(netcdf.getGroups({'variable':['u','v']})),np.array([['s','t']]))
    np.testing.assert_array_equal(np.sort(netcdf.getGroups({'variable':['x','y']})),np.array([['node']]))
    np.testing.assert_array_equal(np.sort(netcdf.getGroups({'variable':['sx','sy']})),np.array([['snode']]))
    
    a=sorted(list(map(lambda x:sorted(x),netcdf.getGroups({'variable':['u','x']}))))
    b=sorted(list(map(lambda x:sorted(x),netcdf.getGroups({'variable':['u','v','x']}))))
    np.testing.assert_array_equal(a,np.array([['node'],['s','t']]))
    np.testing.assert_array_equal(b,np.array([['node'],['s','t']]))

  
def test_parseParameters():
  with S3NetCDFAPI(input) as netcdf:
    obj=netcdf.defaultParameters
    
    # Test1
    parameters={'variable':'[u,v]','itime':'0:10','step':'2'}
    assert parseParameters(obj,parameters)['variable']==['u','v']
    assert parseParameters(obj,parameters)['itime']==slice(0,10,None)
    assert parseParameters(obj,parameters)['step']==2
  
    # Test2
    
    parameters={'variable':'u,v]','itime':':10'}
    assert parseParameters(obj,parameters)['variable']==['u','v']
    assert parseParameters(obj,parameters)['itime']==slice(None,10,None)
    assert parseParameters(obj,parameters)['step']==1
    
    # Test3
    parameters3={'variable':'u,v'}
    assert parseParameters(obj,parameters3)['variable']==['u','v']
    
   
    # Test4
    parameters={'variable':'u,v','x':'0.1,0.2'}
    assert parseParameters(obj,parameters)['variable']==['u','v']
    assert parseParameters(obj,parameters)['x']==[0.1,0.2]
  
  
    # Test5
    parameters={'variable':'u,v','x':[0.1,0.2]}
    assert parseParameters(obj,parameters)['variable']==['u','v']
    assert parseParameters(obj,parameters)['x']==[0.1,0.2]
    
    # Test6
    parameters={'variable':'u,v','x':[0.1,'0.2']}
    assert parseParameters(obj,parameters)['variable']==['u','v']
    assert parseParameters(obj,parameters)['x']==[0.1,0.2]
    
    # Test7
    obj7=netcdf.defaultParameters
    parameters7={'variable':'u,v','x':[0.1,'0..2']}
    with pytest.raises(Exception):assert parseParameters(obj7,parameters7)
    
    # Test8
    obj8=netcdf.defaultParameters
    parameters8={'export':'u,v'}
    with pytest.raises(Exception):assert parseParameters(obj8,parameters8)

def test_checkExport():
  with S3NetCDFAPI(input) as netcdf:
    # Test 1
    obj1=netcdf.defaultParameters
    parameters1={'export':'csv','variable':'u,x'}
    with pytest.raises(Exception):assert netcdf.checkExport(parseParameters(obj1,parameters1)) 

def test_prepareInput():
  with S3NetCDFAPI(input) as netcdf:
    #Test 1
    obj=netcdf.prepareInput({'longitude':[0.1],'latitude':[0.2],"variable":"x"})
    np.testing.assert_array_equal(obj['xy'],[[0.1, 0.2]])
    
    # Test 2
    obj=netcdf.prepareInput({'longitude':[0.1,0.1],'latitude':[0.2,0.2],"variable":"x"})
    np.testing.assert_array_equal(obj['xy'],[[0.1, 0.2],[0.1, 0.2]])
    np.testing.assert_array_equal(obj['nodeIndex'],[0,0])
    
    # Test 3
    obj=netcdf.prepareInput({'longitude':[-159.85],'latitude':[40.0],'inter.mesh':"linear","variable":"x"})
    np.testing.assert_array_almost_equal(obj['meshx'],[-159.9,-159.8,-159.9],5)
    np.testing.assert_array_almost_equal(obj['meshy'],[40.0,40.0,40.1],5)
    np.testing.assert_array_equal(obj['elem'],[[0,2,1]])
    

    obj=netcdf.prepareInput({'longitude':[-159.85,-159.75],'latitude':[40.0,40.0],'inter.mesh':"linear","variable":"x"})
    np.testing.assert_array_almost_equal(obj['meshx'],[-159.9,-159.8,-159.7,-159.9,-159.8],5)
    np.testing.assert_array_almost_equal(obj['meshy'],[40.0,40.0,40.0,40.1,40.1],5)
    np.testing.assert_array_equal(obj['elem'],[[0,3,1],[1,4,2]])
    
    # Test 4
    obj=netcdf.prepareInput({'longitude':[-159.85,-159.75],'latitude':[40.0,40.0],'inode':'0:10',"variable":"x"})
    assert obj['inode']==slice(0, 10, None)
  
    # Test 5
    obj=netcdf.prepareInput({'itime':0,'start':'2000-01-01',"variable":"time"})
    assert obj['itime']==[0]
    assert obj['start']==None
  
    # Test 6
    obj=netcdf.prepareInput({'start':'2000-02-11T13',"variable":"time"})
    np.testing.assert_array_equal(obj['start'],np.array(np.datetime64('2000-02-11T13')))
    np.testing.assert_array_equal(obj['end'],np.array(np.datetime64('2000-02-11T15')))
    np.testing.assert_array_equal(obj['extra'],True)
  

if __name__ == "__main__":
  test_getGroups()
  test_parseParameters()
  test_checkExport()
  test_prepareInput()
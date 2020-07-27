from struct import unpack,pack
import numpy as np
import io
dtypeToFormat={
  "int16":"h",
  "int32":"i",
  "int64":"q",
  "uint16":"H",
  "uint32":"I",
  "uint64":"Q",
  "float32":"f",
  "float64":"d",
}

formatToDtype={
  "h":"int16",
  "i":"int32",
  "q":"int64",
  "H":"uint16",
  "I":"uint32",
  "Q":"uint64",
  "f":"float32",
  "d":"float64",
}



def read(input, return_header=False):
  obj={}
  with io.BytesIO(input) as f:
    endian = ">"
    placeholder=f.read(2)
    one,=unpack(endian+'H',placeholder)
    if one!=1:
      endian="<"
      one,=unpack(endian+'H',placeholder)
      if one!=1:
        raise Exception("Not a binary file from binarypy or binaryjs")
    
    # Header 
    title,nvar=unpack(endian+'16sH',f.read(16+2))
    title=title.decode('ascii').rstrip('\x00')
    
    
    # variables={}
    for _ in range(nvar):
      position,name,format,ndim=unpack(endian+'I16s1sH',f.read(4+16+1+2))
      name=name.decode('ascii').rstrip('\x00')
      format=format.decode('ascii').rstrip('\x00')
      shape=[]
      for _ in range(ndim):
        dim,=unpack(endian+'I',f.read(4))
        shape.append(dim)
      obj[name]={"shape":shape,"size":np.prod(shape),"format":format,"position":position}
    
    
    if return_header:return obj
    
    # Read Data
    for name in obj:
      variable=obj[name]
      f.seek(variable['position'],0)
      itemsize=np.dtype(formatToDtype[variable['format']]).itemsize
      buf=f.read(itemsize*variable['size'])
      obj[name]=np.frombuffer(buf,dtype=formatToDtype[variable['format']],count=variable['size']).reshape(variable['shape'])
    
    obj["title"]=title
    return obj      
    


# def write(output,obj):
def write(variables):  
  """
  {
    title:"",
    variableA:np.ndarray,
    variableB:np.ndarray,
  }
  """
  with io.BytesIO() as f:
  # with open(output,'wb') as f:
    
    endian = ">"
    
    # Header 
    title=variables.pop("title","")[:16].encode()
    
    nvar = len(variables)
    
    f.write(pack(endian+'H16sH',1,title,nvar))
    
    headerposition={}
    for name in variables:
      data=variables[name]
      if not isinstance(data,np.ndarray):raise Exception("Needs to be an ndarray")
      ndim=data.ndim
      shape=data.shape
      format=dtypeToFormat[data.dtype.name].encode()
      headerposition[name]=f.tell()
      f.write(pack(endian+'I',0))
      f.write(pack(endian+'16s1sH',name[:16].encode(),format,ndim))
      [f.write(pack(endian+'I',shape[i])) for i in range(ndim)]
    
    position={}
    # Write Data
    for name in variables:
      data=variables[name]
      position[name]=f.tell()
      f.write(data.tobytes())
    
    # Save variable position in the header
    for name in variables:
      f.seek(headerposition[name],0)
      f.write(pack(endian+'I',position[name]))
    return f.getvalue()
import json
import binarypy
import base64

def saveJSON(data):
  for var in data:
    data[var]=list(data[var])
  return json.dumps(data)

def saveGeoJSON(data):
  None

def saveCSV(data):
  None

def saveBinary(data):
  data=base64.b64encode(binarypy.write(data))
  return data
  # return binarypy.write(data)

def saveNetCDF(data):
  None

def saveMat(data):
  None  

def saveTri(data):
  None    

def saveSLF(data):
  None

def saveShapefile(data):
  None
  
def save(format,data):
  if format=="json":return response("application/json",False,saveJSON(data))
  if format=="geojson":return response("",False,saveGeoJSON(data))
  if format=="csv":return response("",False,saveCSV(data))
  if format=="bin":return response("application/octet-stream",True,saveBinary(data))
  if format=="nc":return response("application/octet-stream",True,saveNetCDF(data))
  if format=="mat":return response("application/octet-stream",True,saveMat(data))
  if format=="tri":return response("application/octet-stream",True,saveTri(data))
  if format=="slf":return response("application/octet-stream",True,saveSLF(data))
  if format=="shp":return response("application/octet-stream",True,saveShapefile(data))

def response(ContentType,isBase64Encoded,Body):
  return {
      'statusCode': 200,
      'headers': {"content-type": ContentType},
      'isBase64Encoded': isBase64Encoded,
      'body':Body
      # 'body': Body,
      # "headers": {"Access-Control-Allow-Origin": "*"},
    } 
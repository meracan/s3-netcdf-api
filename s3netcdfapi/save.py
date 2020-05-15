import binarypy


def saveJSON(data):
  None

def saveGeoJSON(data):
  None

def saveCSV(data):
  None

def saveBinary(data):
  None

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
  if format=="json":return response("",saveJSON(data))
  if format=="geojson":return response("",saveGeoJSON(data))
  if format=="csv":return response("",saveCSV(data))
  if format=="bin":return response("",saveBinary(data))
  if format=="nc":return response("",saveNetCDF(data))
  if format=="mat":return response("",saveMat(data))
  if format=="tri":return response("",saveTri(data))
  if format=="slf":return response("",saveSLF(data))
  if format=="shp":return response("",saveShapefile(data))

def response(ContentType,Body):
  return {
      'statusCode': 200,
      'Content-Type':ContentType,
      'body': Body,
      # "headers": {"Access-Control-Allow-Origin": "*"},
    } 
import binarypy
import base64

def binary(data):
  bpy_wdata = binarypy.write(data)
  data = base64.b64encode(bpy_wdata)
  return data
  # return binarypy.write(data)
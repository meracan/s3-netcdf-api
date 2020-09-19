
def getIdx(obj,name):
  pointer=obj['pointer'][name]
  group=pointer['group']
  variable=pointer['variable']
  return (group,variable)
  

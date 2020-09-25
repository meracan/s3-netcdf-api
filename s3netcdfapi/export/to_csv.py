
from .table import to_table

def to_csv(obj,data):
  filepath=obj['filepath']+".csv"
  to_table(obj,data).to_csv(filepath,sep=obj['sep'],index=False)
  return filepath
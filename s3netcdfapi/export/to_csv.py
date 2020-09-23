
from .table import to_table

def to_csv(obj,data):
  to_table(obj,data).to_csv(obj['output']+".csv",sep=obj['sep'],index=False)
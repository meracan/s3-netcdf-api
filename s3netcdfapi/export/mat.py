from io import BytesIO as bio
from scipy.io import loadmat, savemat

def mat(data):
  _ = bio()
  savemat(_, {})
  mat = loadmat(_)
  for var in data:
    # mat[var] = data[var]
    mat['spectra']=data['spectra']
    mat['freq']=data['freq'] 
    mat['dir']=data['dir'] 
  return mat
import json
import os
import sys
template=sys.argv[1]
activate=os.path.join(os.environ.get("CONDA_PREFIX"),"etc/conda/activate.d")
pathActivate=os.path.join(activate,"env_vars.sh")
os.makedirs(activate, exist_ok=True)
with open(pathActivate,"a+") as file:
  file.write("export AWS_BUCKETNAME={}\n".format(template))


deactivate=os.path.join(os.environ.get("CONDA_PREFIX"),"etc/conda/deactivate.d")
pathDeactivate=os.path.join(deactivate,"env_vars.sh")
os.makedirs(deactivate, exist_ok=True)
with open(pathDeactivate,"a+") as file:
  file.write("unset AWS_BUCKETNAME\n")
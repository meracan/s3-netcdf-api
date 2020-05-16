import json
import os
with open('output.json') as file:
  data = json.load(file)
obj={}

for item in data['Stacks'][0]['Outputs']:
  obj[item['OutputKey']]=item['OutputValue']  

activate=os.path.join(os.environ.get("CONDA_PREFIX"),"etc/conda/activate.d")
pathActivate=os.path.join(activate,"env_vars.sh")
os.makedirs(activate, exist_ok=True)
with open(pathActivate,"a+") as file:
  file.write("export AWS_API={}\n".format(obj['AWSAPI']))


deactivate=os.path.join(os.environ.get("CONDA_PREFIX"),"etc/conda/deactivate.d")
pathDeactivate=os.path.join(deactivate,"env_vars.sh")
os.makedirs(deactivate, exist_ok=True)
with open(pathDeactivate,"a+") as file:
  file.write("unset AWS_API\n")


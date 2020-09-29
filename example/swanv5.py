import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("https://w4wj62194j.execute-api.us-west-2.amazonaws.com/Prod?variable=hs&inode=0&export=csv")
fig = go.Figure([go.Scatter(x=df['Datetime'], y=df['hs,m'])])
fig.write_image("hs_0.png")





from datetime import date, timedelta
from dash import dcc, dash_table, html, Dash
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

df= pd.read_excel('C://Abdelouaheb//CSA//Platform_Preparation//World Bank Data//Population Growth.xlsx')
df_countries=df.melt(id_vars=["Country Name", "Indicator Name"],
        var_name="Year", 
        value_name="Value")

value =df_countries['Indicator Name'].unique()


filter= dcc.RadioItems(
    options=[{"label":x, "value":x} for x in value ],
    value='Population growth (annual %)',
    labelStyle={'display': 'inline-block'},
    id= "test"
)

app = Dash(__name__)

graph= html.Div([
    dcc.Graph(id='graph-with-slider'),

])

app.layout= html.Div([filter,graph])

@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('test', 'value'))

def update_figure(value):
    filtered_df = df_countries[df_countries['Indicator Name']== value]

    df_countrydate = filtered_df.groupby(['Year', 'Country Name']).sum().reset_index()

    # Creating the visualization
    fig = px.choropleth(df_countrydate,
                        locations="Country Name",
                        locationmode="country names",
                        color="Value",
                        hover_name="Country Name",
                        animation_frame="Year"
                        )

    fig.update_layout(
        title_text='Average Grwoth rate',
        title_x=0.5,
        geo=dict(
            showframe=False,
            showcoastlines=False,
        ))

    fig.update_layout(
    title_text = 'Growth population rate',
    title_x = 0.5,
    geo=dict(
    showframe = False,
    showcoastlines = False,
    projection_type = 'equirectangular'
)
)


    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
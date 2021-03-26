import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go
import tickerData as td

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

# ---------- Import and clean data (importing csv into pandas)
# df = pd.read_csv("intro_bees.csv")
ticker = 'MSFT'
df = td.getTickerPriceData(ticker,period='90d',interval='1d')
signals_df = td.makeTickerDfSignals(df,interval='1d')
signals_df.drop(columns='Close',inplace=True)
comb_df = pd.concat([df,signals_df],join='inner',axis=1)

#df = df.groupby(['State', 'ANSI', 'Affected by', 'Year', 'state_code'])[['Pct of Colonies Impacted']].mean()
#df.reset_index(inplace=True)
print(comb_df[:5])

def update_graph(data_df,ticker=None):

    dff = data_df.copy()
    #dff = dff[dff["Year"] == option_slctd]
    #dff = dff[dff["Affected by"] == "Varroa_mites"]
    '''
    # Plotly Express
    fig = px.choropleth(
        data_frame=dff,
        locationmode='USA-states',
        locations='state_code',
        scope="usa",
        color='Pct of Colonies Impacted',
        hover_data=['State', 'Pct of Colonies Impacted'],
        color_continuous_scale=px.colors.sequential.YlOrRd,
        labels={'Pct of Colonies Impacted': '% of Bee Colonies'},
        template='plotly_dark'
    )
    '''
    # Plotly Graph Objects (GO)
    fig = go.Figure(
        data=[go.Candlestick(x=dff.index,
        open=dff['Open'],
        high=dff['High'],
        low=dff['Low'],
        close=dff['Close'],
        )]
    )

    fig.update_layout(
        title_text="%s Stock Prices"%ticker,
        title_xanchor="center",
        title_font=dict(size=24),
        title_x=0.5,
    )

    return fig

fig = update_graph(comb_df,ticker)
# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("Our Ticker Data with Dash", style={'text-align': 'center'}),

    html.Div(id='none',children=[]),

    dcc.Graph(id='my_ticker_chart', figure=fig)

])


# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
#@app.callback(
     #[Output(component_id='my_ticker_chart', component_property='figure')],
    #[Input(component_id='none', component_property='children')]
#)


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)

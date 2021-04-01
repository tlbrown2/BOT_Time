import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go
import tickerData as td
import executeTrade as et
import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dtable
from dash.dependencies import Input, Output


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Establishing Web App Environment
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

# ---------- Import and clean data (importing csv into pandas)

# Getting the SPY (S&P 500) ticker list
table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
spy_df = table[0]
spy_list = spy_df.Symbol.to_list()

# User input for ticker symbol, short & long windows, & time sample period
ticker = input ('Enter the ticker symbol that you want to use: ')
short_window = int(input ('Enter the short window period (days): '))
long_window = int(input ('Enter the long window period (days): '))
period = int(input ('Enter the ticker sample period (days): '))

# Read in initial ticker price data
df = td.getTickerPriceData(ticker,period='%id'%period,interval='1d')

# Adding 1st level of trade signals to the initial dataframe
signals_df = td.makeTickerDfSignals(df,interval='1d',short_window=short_window,long_window=long_window)
signals_df.drop(columns='Close',inplace=True)


# Joining the Price & 1st level of signals dataframe
comb_df = pd.concat([df,signals_df],join='inner',axis=1)

# Loading the models and back-testing the data w/ signals; returns dataframe
all_df,recommendation, predicted_price, strike_price_call, strike_price_put = td.execute_backtest(comb_df,initial_capital=10000.00,shares=500)

# Load Ticker Options Chain from the Webull Trading Application
if ticker != 'SPY':
    options_df = et.get_webull_options(ticker)

print(all_df[:5])


# Web App function to update the data
def update_graph(data_df,ticker=None,future_price=0):

    dff = data_df.copy()
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
    #Trade Signals
    # Plotly Graph Objects (GO)
    candlestick = go.Figure(
        data=[go.Candlestick(x=dff.index,
        open=dff['Open'],
        high=dff['High'],
        low=dff['Low'],
        close=dff['Close'],
        )]
    )

    candlestick.update_layout(
        title_text="%s Stock Prices"%ticker,
        title_xanchor="center",
        title_font=dict(size=24),
        title_x=0.5,
    )
    candlestick.update_yaxes(title_text='Close Prices', tickprefix='$', color='blue')
    bollinger = go.Figure()
    bollinger.add_trace(go.Scatter(x=dff.index,y=dff['Close'],name=dff['Close'].name,line=dict(color='orange',width=5)))
    bollinger.add_trace(go.Scatter(x=dff.index, y=dff['Upper_Band'], name=dff['Upper_Band'].name,line=dict(dash='dash',color='green')))
    bollinger.add_trace(go.Scatter(x=dff.index, y=dff['Lower_Band'], name=dff['Lower_Band'].name,line=dict(dash='dash',color='black')))
    bollinger.update_yaxes(title_text='Close Prices', tickprefix='$', color='blue')

    rsi = go.Figure()
    rsi.add_trace(go.Scatter(x=dff.index,y=dff['RSI'],name=dff['RSI'].name,line=dict(color='orange',width=5)))
    rsi.add_trace(go.Scatter(x=dff.index, y=dff['RSI_Upper_Lim'], name=dff['RSI_Upper_Lim'].name, line=dict(color='red')))
    rsi.add_trace(go.Scatter(x=dff.index, y=dff['RSI_Lower_Lim'], name=dff['RSI_Lower_Lim'].name, line=dict(color='red')))
    rsi.update_yaxes(title_text='RSI', color='orange')

    xover = go.Figure()
    xover.add_trace(go.Scatter(x=dff.index, y=dff['SMA%s'%long_window], name='Long', line=dict(color='blue')))
    xover.add_trace(go.Scatter(x=dff.index, y=dff['SMA%s'%short_window], name='Short', line=dict(color='brown')))
    #xover.add_trace(go.Scatter(x=dff.index, y=dff[dff['Entry/Exit']== 1.0]['Close'], name='Entry', line=dict(color='green',dash='dot',width=8)))
    #xover.add_trace(go.Scatter(x=dff.index, y=dff[dff['Entry/Exit'] == -1.0]['Close'], name='Exit', line=dict(color='red',dash='dot',width=8)))
    xover.add_trace(go.Scatter(x=dff.index, y=dff['Close'], name='Price', line=dict(color='orange')))
    xover.update_yaxes(title_text='Close Prices', tickprefix='$', color='blue')

    #prediction = td.rm3.plot_graph(dff)
    prediction = go.Figure()
    prediction.add_trace(go.Scatter(x=dff.index, y=dff['true_adjclose_15'], name='Actual Price', line=dict(color='blue',width=6)))
    prediction.add_trace(go.Scatter(x=dff.index, y=dff['adjclose_15'], name='Predicted Price', line=dict(color='red',width=6)))
    prediction.update_yaxes(title_text='Close Prices',tickprefix='$',color='blue')


    #prediction price
    predicted_price_val = go.Figure()
    predicted_price_val.add_trace(go.Indicator(
    mode = "number+delta",
    value = float(predicted_price),
    number= {'prefix' : '$'},
    delta = {'position': 'bottom','reference': dff['Close'].iloc[-1]},
    domain = {'x': [0,1], 'y': [0,1]}))
    predicted_price_val.update_layout(paper_bgcolor = 'lightblue')

    # Recommended Call Option Price
    strike_call = go.Figure()
    strike_call.add_trace(go.Indicator(
        mode="number+delta",
        value=float(strike_price_call),
        number={'prefix': '$'},
        delta={'position': 'bottom', 'reference': dff['Close'].iloc[-1]},
        domain={'x': [0, 1], 'y': [0, 1]}))
    strike_call.update_layout(paper_bgcolor='green')

    # Recommended Put Option Price
    strike_put = go.Figure()
    strike_put.add_trace(go.Indicator(
        mode="number+delta",
        value=float(strike_price_put),
        number={'prefix': '$'},
        delta={'position': 'bottom', 'reference': dff['Close'].iloc[-1]},
        domain={'x': [0, 1], 'y': [0, 1]}))
    strike_put.update_layout(paper_bgcolor='red')


    if ticker != 'SPY':
        options_df = et.get_webull_options(ticker)
        options = go.Figure(data=[go.Table(header=dict(values=list(options_df.columns),align='left'),)])
    else:
        options = go.Figure()
        options.add_trace(go.Scatter(x=dff.index,y=1))

    return candlestick,bollinger,rsi,xover,prediction,predicted_price_val,strike_call,strike_put,options

candlestick,bollinger,rsi,xover,prediction,predicted_price_val,strike_call,strike_put,options = update_graph(all_df,ticker,predicted_price)

# App layout
app.layout = html.Div(style={'backgroundColor': colors['background']},children = [
    html.Div([html.H1("ITS BOT-TIME with Dash", style={'text-align': 'center','color':'white'}),

    html.Div(id='output',children=[]),

    dcc.Graph(id='my_ticker_chart', figure=candlestick)

    ]),

    html.Div([
        html.H1(children='%s Bollinger Bands'%ticker,style={'text-align': 'center','color':'white'}),

        html.Div(children=[]),

        dcc.Graph(id='Bollinger',figure=bollinger),

    ]),
    html.Div([
        html.H1(children='%s RSI'%ticker,style={'text-align': 'center','color':'white'}),

        html.Div(children=[]),

        dcc.Graph(id='RSI',figure=rsi),

    ]),
    html.Div([
        html.H1(children='%s Long %i Short %i Crossover'%(ticker,long_window,short_window),style={'text-align': 'center','color':'white'}),

        html.Div(children=[]),

        dcc.Graph(id='XOVER',figure=xover),

    ]),
    html.Div([
        html.H1(children='%s Prediction'%ticker,style={'text-align': 'center','color':'white'}),

        html.Div(children=[]),

        dcc.Graph(id='PREDICT',figure=prediction),

    ]),

    html.Div([
        html.H1(children='%s Prediction Price'%ticker,style={'text-align': 'center','color':'white'}),

        html.Div(children=[]),

        dcc.Graph(id='Predicted Price',figure=predicted_price_val),

    ]),
    html.Div([
        html.H1(children='%s'%recommendation,style={'text-align': 'center','color':'yellow'}),

        html.Div(children=[]),

        #dcc.Graph(id='RECOMMEND',figure=prediction),

    ]),
    html.Div([
        html.H1(children='%s Recommended Call Strike Price'%ticker,style={'text-align': 'center','color':'white'}),

        html.Div(children=[]),

        dcc.Graph(id='Strike Call Price',figure=strike_call),

    ]),
    html.Div([
        html.H1(children='%s Recommended Put Strike Price'%ticker,style={'text-align': 'center','color':'white'}),

        html.Div(children=[]),

        dcc.Graph(id='Strike Put Price',figure=strike_put),

    ]),
    html.Div([
        dtable.DataTable(id='table',columns=[{'name': i, 'id':i} for i in options_df.columns],
                         data = options_df.to_dict('records'),
                         ),
    ])
])


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)

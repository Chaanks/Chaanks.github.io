# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import numpy as np
import pandas as pd


def RobbinsMonro_const(p, step_size=10, nrounds=1000):
    theta_ = np.zeros((nrounds,))
    theta = 0

    for n in range(nrounds):
        csi = np.random.choice(range(1, len(p)+1), p=p)
        theta = theta+1/(step_size)*(csi-theta);
        theta_[n] = theta;
    
    return theta_

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('exemple.csv')

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    dcc.Store(id='memory-output'),

    html.Label('Slider'),
    dcc.Slider(
        id='nrounds',
        min=0,
        max=5000,
        step=50,
        value=1000,
    ),
    html.Div(id='slider-output-container'),

    html.Div(dcc.Input(id='input-box', type='text', value='10')),
    html.Button('Submit', id='button'),
    html.Div(id='output-container-button',
             children='Enter a value and press submit'),

    dcc.Graph(id="line-chart")
])

@app.callback(
    [dash.dependencies.Output('slider-output-container', 'children'), dash.dependencies.Output('memory-output', 'data')],
    [dash.dependencies.Input('nrounds', 'value')])
def update_slider_output(value):
    data = value
    return 'You have selected "{}"'.format(value), data


@app.callback(
    dash.dependencies.Output("line-chart", "figure"),
    [dash.dependencies.Input('memory-output', 'data'), dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')])
def update_submit_output(nrounds, n_clicks, value):
    p = np.array([1/12, 1/6, 1/3, 5/12])
    m = sum(range(1, len(p)+1) * p)
     
    step = list(map(int, value.split(';')))
    print(step)

    data = {'time': range(nrounds)}
    for i in range(len(step)):
        data[str(step[i])] = RobbinsMonro_const(p, step[i], nrounds)
    df = pd.DataFrame(data)

    fig = px.line(df, x='time', y=list(map(str, step)))
    fig.add_hline(y=m, line_dash='dot')

    return fig

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go

import sympy as sp
import numpy as np
from vanilla_gradient import functions, vanilla_gradient_descent, plot_gradient_3D

x, y = sp.symbols(['x', 'y'])

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(children=[
    html.Div(className='header', style={'display': 'block', 'margin': '20px'}, children=[
        html.H3(
            'Vanilla Gradient Descent',
            style={'text-align': 'center'}
        ),
        dcc.Markdown(
            """ 
            **Date:** 25.03.2021            
            """,
            style={'text-align': 'center'}
        )
    ]),
    html.Div(className='offset-by-six.columns', style={},
             children=[
        html.Label(
            'Choose starting Points for Gradient'
        ),
        html.Div(children=[
            dcc.Input(
                id='starting_point_X',
                value=1,
                type='number',
                placeholder='X',
                style={'width': '200px', 'height': 'auto'}
            ),
            dcc.Input(
                id='starting_point_Y',
                value=2,
                type='number',
                placeholder='Y',
                style={'width': 'max', 'height': 'auto'}
            )
        ]),
        html.Div(children=[
            html.Label(
                'Choose a Function'
            ),
            dcc.Dropdown(
                id='dropdown_function',
                options=[
                    {'label': str(functions[1]), 'value': 1},
                    {'label': str(functions[2]), 'value': 2},
                    {'label': str(functions[3]), 'value': 3},
                    {'label': str(functions[4]), 'value': 4},
                    {'label': str(functions[5]), 'value': 5}
                ],
                value=2,
                style={'width': 'max', 'height': 'auto'}
            )
        ]),
        html.Div(children=[
            html.Label(
                f'Choose the Amount of Steps from {1} to {100}'
            ),
            dcc.Slider(
                id='steps',
                min=1,
                max=100,
                step=1,
            )
        ])
    ]),
    html.Div(className='six columns', style={'text-align': 'center'}, children=[
        dcc.Graph(
            id='figure',
            style={'heigth': 'auto', 'width': 'auto'}
        )
    ])
])

@app.callback(
    dash.dependencies.Output('figure', 'figure'), [
        dash.dependencies.Input('starting_point_X', 'value'),
        dash.dependencies.Input('starting_point_Y', 'value'),
        dash.dependencies.Input('dropdown_function', 'value'),
        dash.dependencies.Input('steps', 'value')
    ],
)
def perform_gradient_descent(start_x, start_y, function_nr, steps):
    points = vanilla_gradient_descent(f=sp.lambdify([x, y], functions[function_nr]),
                                      gradx=sp.lambdify([x, y], sp.diff(functions[function_nr], x)),
                                      grady=sp.lambdify([x, y], sp.diff(functions[function_nr], y)),
                                      P=(start_x, start_y),
                                      learning_rate=.1
                                      )

    function = sp.lambdify([x, y], functions[function_nr])
    points_x = np.array(points[0][:steps])
    points_y = np.array(points[1][:steps])

    limits = (-10, 10)
    x_ = np.linspace(limits[0], limits[1], 30)
    y_ = np.linspace(limits[0], limits[1], 30)

    X, Y = np.meshgrid(x_, y_)

    Z = function(X, Y)

    fig = go.Figure()
    fig.add_trace(go.Surface(x=X,
                             y=Y,
                             z=Z,
                             colorscale='ice',
                             opacity=.5,
                             showscale=False
                             )
                  )
    fig.add_trace(go.Scatter3d(x=points_x,
                               y=points_y,
                               z=function(points_x, points_y)
                               )
                  )
    fig.update_layout(
        title='',
        showlegend=False,
        height=700,
        width=700,
        template=None,
        uirevision='steps'
    )
    fig.update_xaxes(
        title='X'
    )
    fig.update_yaxes(
        title='Y'
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

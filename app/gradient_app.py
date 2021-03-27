import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import sympy as sp
import numpy as np
from vanilla_gradient import functions, vanilla_gradient_descent

x, y = sp.symbols(['x', 'y'])


external_stylesheets = [dbc.themes.FLATLY]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

##### Forms #####
controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Starting Points:"),
                html.Br(),
                dcc.Input(
                    id='starting_point_X',
                    value=0, min=-2, max=2,
                    type='number',
                    placeholder='X',
                ),
                dcc.Input(
                    id='starting_point_Y',
                    value=0, min=-2, max=2,
                    type='number',
                    placeholder='Y',
                )
            ]
        ),
        html.Br(),
        dbc.FormGroup(
            [
                dbc.Label("Choose Function:"),
                dcc.Dropdown(
                    id='dropdown_function',
                    options=[
                        {'label': str(functions[1]), 'value': 1},
                        {'label': str(functions[2]), 'value': 2},
                        {'label': str(functions[3]), 'value': 3},
                        {'label': str(functions[4]), 'value': 4},
                        {'label': str(functions[5]), 'value': 5}
                    ],
                    value=2
                )
            ]
        ),
        html.Br(),
        dbc.FormGroup(
            [
                dbc.Label('Control Learning Rate:'),
                dcc.Slider(id='learning_rate', min=0.01, max=5, step=.1, value=.01),
                html.Span(id='learning_rate_label')
            ]
        ),
    ],
    body=True,
)
controls2 = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Gradient Steps:"),
                html.Br(),
                dbc.ButtonGroup(
                    [
                        dbc.Button("Step Backward", id='step_backward',
                                   outline=True, n_clicks=0, color='dark',
                                   size='lg'),
                        dbc.Button("Step Forward", id='step_forward',
                                   outline=True, n_clicks=1, color='dark',
                                   size='lg'),
                        dbc.Button('Reset', id='reset_steps',
                                   outline=True, n_clicks=0, color='dark',
                                   size='lg'),
                    ]
                )
            ]
        )
    ],
    body=True,
)

### Dashboard Layout
app.layout = dbc.Container(
    [
        html.H1("Visualization of Vanilla Gradient Descent", style={'text-align': 'center'}),
        dcc.Markdown("""**Author:** Simon Staehli / **Date of Creation:** 26.03.2021""",
                     style={'text-align': 'center'}),
        dbc.Row(
            [
                dbc.Col([controls, html.Br(),  controls2], md=4),
                dbc.Col(dcc.Graph(id='figure'), md=6),
            ],
            align="center",
        ),
    ],
    fluid=True,
)

# Print chosen Steps as Label
@app.callback(
    dash.dependencies.Output('learning_rate_label', 'children'),
    [
        dash.dependencies.Input('learning_rate', 'value')
    ]
)
def update_label_steps(learning_rate):
    return '--> Learning Rate: ' + str(learning_rate)

# Reset Forward Step of Gradient
@app.callback(
    dash.dependencies.Output('step_forward', 'n_clicks'),
    [
        dash.dependencies.Input('reset_steps', 'n_clicks')
    ]
)
def reset_counts(reset):
    return 0

## Reset Backward counter
@app.callback(
    dash.dependencies.Output('step_backward', 'n_clicks'),
    [
        dash.dependencies.Input('reset_steps', 'n_clicks')
    ]
)
def reset_counts(reset):
    return 0

## Update Figure
@app.callback(
    dash.dependencies.Output('figure', 'figure'),
    [
        dash.dependencies.Input('starting_point_X', 'value'),
        dash.dependencies.Input('starting_point_Y', 'value'),
        dash.dependencies.Input('dropdown_function', 'value'),
        dash.dependencies.Input('learning_rate', 'value'),
        dash.dependencies.Input('step_forward', 'n_clicks'),
        dash.dependencies.Input('step_backward', 'n_clicks'),
    ],
)
def perform_gradient_descent(start_x, start_y, function_nr, learning_rate, clicks_forward, clicks_backward):
    points = vanilla_gradient_descent(f=sp.lambdify([x, y], functions[function_nr]),
                                      gradx=sp.lambdify([x, y], sp.diff(functions[function_nr], x)),
                                      grady=sp.lambdify([x, y], sp.diff(functions[function_nr], y)),
                                      P=(start_x, start_y),
                                      learning_rate=learning_rate,
                                      max_steps=1000
                                      )

    if clicks_forward - clicks_backward <= 0:
        steps = 1
    else:
        steps = clicks_forward - clicks_backward

    function = sp.lambdify([x, y], functions[function_nr])
    points_x = np.array(points[0][:steps])
    points_y = np.array(points[1][:steps])

    limits = (-2, 2)

    x_ = np.linspace(limits[0], limits[1], 50)
    y_ = np.linspace(limits[0], limits[1], 50)

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
        height=800,
        width=800,
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
    app.run_server(host='127.0.0.1', debug=True)
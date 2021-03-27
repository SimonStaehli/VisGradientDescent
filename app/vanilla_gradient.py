import numpy as np
import sympy as sp

import plotly.graph_objects as go


x, y = sp.symbols(['x', 'y'])

functions = {
    1: .1 * sp.sin(x**2) + .1*sp.cos(y**2),
    2: y**2 + x**2,
    3: x*sp.exp(-(x**2+y**2)),
    4: .5*sp.sin(x) + .5*y**2,
    5: (x**2+y**2)*sp.exp(-1*((x**2+y**2)/2))
}


def plot_gradient_3D(function, points_x, points_y, limits=(-10, 10)):
    """
    Plots a 3D Function with visualized Gradient Descent on local Minumum.
    """
    x = np.linspace(limits[0], limits[1], 30)
    y = np.linspace(limits[0], limits[1], 30)
    X, Y = np.meshgrid(x, y)
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
        height=1000, width=1500,
        template=None
    )
    fig.update_xaxes(
        title='X'
    )
    fig.update_yaxes(
        title='Y'
    )

    return fig

def vanilla_gradient_descent(f, gradx, grady, grad_require_xy=True, P=(0, 0), learning_rate=.01, epsilon=.001,
                             max_steps=100):
    """
    Performs stepwise Vanilla Gradient Descent for a given 3-Dimensional Function.

    returns: Points for each step to the local minimum
    """
    P = np.array([float(i) for i in P])
    points_x, points_y, points_z = [P[0]], [P[1]], [f(P[0], P[1])]
    gradient_x, gradient_y = 1, 1

    steps_count = 0
    while (steps_count < max_steps) and (np.linalg.norm(np.array([gradient_x, gradient_y])) > epsilon):

        if grad_require_xy:
            gradient_x = gradx(P[0], P[1])
            gradient_y = grady(P[0], P[1])
        else:
            gradient_x = gradx(P[0])
            gradient_y = grady(P[1])

        P[0] = P[0] - learning_rate * gradient_x
        P[1] = P[1] - learning_rate * gradient_y

        points_x.append(P[0])
        points_y.append(P[1])
        points_z.append(f(P[0], P[1]))

        steps_count += 1

    return (points_x, points_y, points_z)

import plotly.offline as py
import plotly.graph_objs as go
import numpy as np


def plot(img: np.ndarray, title: str) -> None:
    """
    Generate graphs. Export them as html file into `./output/plot` folder
    """
    data = [go.Heatmap(z=img)]

    layout = go.Layout(
        title=title,
        height=800,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        xaxis=dict(
            title='Time',
            ticklen=5,
            gridwidth=2,
        ),
        yaxis=dict(
            title='Frequency',
            type='log',
            ticklen=5,
            gridwidth=2,
        )
    )

    py.plot(go.Figure(data, layout),
            filename='./output/plot/' + title + '.html')

import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
from os import path
from typing import Tuple, List


def export_graph(img: np.ndarray, title: str, out_dir: str = './output/graphs/') -> None:
    """
    Generate graphs. Export them as html file into out_dir
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

    py.plot(go.Figure(data, layout), filename=path.join(out_dir, title + '.html'))


def format_as_2d_array(peak_points: List[List[List[int]]], shape: Tuple) -> np.ndarray:
    """
    Returns 2d numpy array which can be passed into export_graph function
    :param peak_points:
    :param shape: shape of output 2d array
    :return: peak indices which indicates background as -1 and peak lines as index starts from 0
    """
    peak_indices = np.zeros(shape) - 1
    for chunk_idx in range(len(peak_points)):
        peak_indices[tuple(map(tuple, np.array(peak_points[chunk_idx]).T))] = chunk_idx

    return peak_indices

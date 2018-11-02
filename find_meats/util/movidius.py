import sys
from typing import Union
from pathlib import Path
from mvnc import mvncapi as mvnc
import numpy as np

def load_graph_on_mvnc(model_path: Union[str, Path], device: int=0):
    '''
    load mvnc format graph onto movidius device.

    :param model_path: mvnc format model path wihch is converted from tensorflow format.
    :param device: movidius devide number.
    :return: mvnc graph which reads model parameters.
    '''

    if isinstance(model_path, str):
        model_path = Path(model_path)
    assert model_path.exists()

    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No devices found')
        sys.exit(1)

    device = mvnc.Device(devices[device])
    device.OpenDevice()

    with model_path.open('rb') as f:
        graphfile = f.read()

    graph = device.AllocateGraph(graphfile)

    return graph

import numpy as np 
from dataclasses import dataclass


@dataclass
class Input:
    matrix: np.ndarray
    b: np.ndarray
    correct: np.ndarray


@dataclass
class AlgParams:
    w: float
    stop_epsilon: float 
    max_iter: int


def read_data(filename) -> Input:
    arr = np.genfromtxt(filename)
    size = int(arr[0])

    matrix = arr[1: size*size + 1].reshape((size, size))
    bc = arr[size*size + 1:]
    l = int(len(bc) / 2)
    b = bc[: l]
    correct = bc[l:]

    return Input(matrix, b, correct)




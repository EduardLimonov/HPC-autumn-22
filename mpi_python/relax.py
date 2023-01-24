from __future__ import annotations
import numpy as np
from utils import *
from mpi_python import comm, rank, n_nodes


def calc_relax(input: Input, params: AlgParams):
    global comm, rank, n_nodes

    if rank == 0:
        current_x = np.zeros(input.matrix.shape[0])
    else:
        current_x = np.zeros((1,))

    iter = 0

    while True:
        last_epsilon = update_x(current_x, input, params)
        iter += 1

        can_stop = comm.bcast(check_stop(last_epsilon, params.stop_epsilon, iter, params.max_iter), root=0)
        if can_stop:
            break
        
    return current_x


def update_x(x, input: Input, params: AlgParams) -> float:
    ll = comm.bcast(len(x), root=0)

    epsilons = np.zeros_like(x)
    for i in range(ll):
        if rank == 0:
            xi = x[i]
            x[i] = 0
        mul_res = multiply(input.matrix[min(len(input.matrix) - 1, i)], x)
        if rank == 0:
            new_xi = (input.b[i] - mul_res) * params.w / input.matrix[i, i] + (1 - params.w) * xi
            epsilons[i] = new_xi - xi
            x[i] = new_xi

    return np.linalg.norm(epsilons)


def check_stop(last_epsilon, stop_epsilon, iter, max_iter):
    return last_epsilon <= stop_epsilon or iter >= max_iter


def multiply(m_i: np.ndarray, x: np.ndarray):
    global rank, n_nodes, comm
    sum_len = comm.bcast(len(x), root=0)
    my_len = int(sum_len / n_nodes)

    my_matrix = np.empty(my_len, dtype='d')
    my_x = np.empty(my_len, dtype='d')

    comm.Scatter(m_i, my_matrix, root=0, )
    comm.Scatter(x, my_x, root=0, )

    res = (my_matrix * my_x).sum()
    sums = np.empty(n_nodes, dtype='d')
    comm.Gather(res, sums, root=0)

    return sums.sum()

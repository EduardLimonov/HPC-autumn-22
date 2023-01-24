from __future__ import annotations
from mpi4py import MPI
import numpy as np
import time
from relax import *
from utils import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_nodes = comm.Get_size()


def run_test(test_name, nrepeats=1):
    if rank == 0:
        inp = read_data(test_name)
    else:
        inp = Input(np.zeros((1, 1)), np.zeros((1, )), np.zeros((1, )))

    params = AlgParams(w=1.1, stop_epsilon=0.00001, max_iter=10000)

    sum_time = 0
    for i in range(nrepeats):
        start = time.time()
        res = calc_relax(inp, params)
        end = time.time()
        sum_time += end - start

    if rank == 0:
        print(test_name, sum_time / nrepeats, flush=True)
        print('error: %f' % np.linalg.norm(res - inp.correct))
        #comm.barrier()


def main():
    test_paths = [
        #'../resource/test0',
        #'../resource/test1',
        '../resource/test2',
        '../resource/test3',
        '../resource/test4',
        '../resource/test5',
        '../resource/test6',
        
    ]
    for path in test_paths:
        run_test(path, 10)


if __name__ == '__main__':
    main()

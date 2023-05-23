import pyreadr
import numpy as np
def read_data(data_name='jp'):
    result = pyreadr.read_r('data/mortality_sp.RData')
    result = result['data.log.sm']
    data_np = result.to_numpy().transpose((1, 2, 0)).transpose(0, 2, 1)  # shape (47, 96, 43)
    num_dim, num_tasks, num_points = 47, 33, 96

    return data_np, num_dim, num_tasks, num_points
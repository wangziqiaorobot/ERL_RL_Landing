#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from array import array
import os
# from IPython import embed
# import rslgym.algorithm.modules as rslgym_module


"""
saves weights of actuator model mlp
"""

if __name__ == '__main__':
    importPath = "/home/ziqiao/RL/ERL_RL_Landing/experiments/learning/results/save-landing-ppo-kin-ld-06.03.2022_12.15.09/06.03.2022_12.15.16/"

    if torch.cuda.is_available():
        dev = "cuda:0"
        print("GPU found")
    else:
        dev = "cpu"

    snapshot = torch.load(importPath + 'best_mode.zip', map_location=torch.device(dev))

    actor_net = snapshot['actor_state_dict']['architecture']


    weights = [i for k, i in actor_net.items() if (k.endswith('.weight'))]
    biases = [i for k, i in actor_net.items() if (k.endswith('.bias'))]


    paramsConcat = np.array([])
    for w, b in zip(weights, biases):
        w = w.cpu().numpy().transpose()
        b = b.cpu().numpy().transpose()

        paramsConcat = np.concatenate((paramsConcat, w.flatten(order='C')))
        paramsConcat = np.concatenate((paramsConcat, b.flatten(order='C')))

    dims = array('L', [paramsConcat.shape[0], 1])
    params = array('f', paramsConcat)

    out = open(importPath + '/weights.bin', 'wb')
    dims.tofile(out)
    params.tofile(out)
    out.close()
    print("saved to folder %s " % importPath)

    action_size = 4
    observation_size = 23

    """
    load policy the way we do it in the test scripts and double check result
    """

    actor_net = rslgym_module.MLP([256,256,256,128],
                                  nn.Tanh,
                                  observation_size,
                                  action_size,
                                  0.0)

    actor = rslgym_module.Actor(actor_net,
                                rslgym_module.MultivariateGaussianDiagonalCovariance(4, 1.0),observation_size,action_size,
                                'cpu')

    snapshot = torch.load(importPath + 'snapshot1800.pt')
    actor.load_state_dict(snapshot['actor_state_dict'])

    ob = np.zeros(shape=(1, observation_size), dtype=np.float32)

    for i in range(0, (observation_size-1)):
        ob[0, i] = observation_size-1 - i
    print(ob)

    act = actor.noiseless_action(ob).cpu().detach().numpy()

    print(act)

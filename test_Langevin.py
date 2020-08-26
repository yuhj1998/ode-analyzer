#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File 	 : test_Langevin.py
@Time 	 : 2020/07/21
@Author :  Haijn Yu <hyu@lsec.cc.ac.cn>
@Desc	 : Apply ode_analyzer to Langevin dynamics
'''
# %% 1. import library and set parameters
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import ode_analyzer as oa


class LangevinNet(oa.OdeModel):
    """ A deterministic Langevin dyanmics """

    def __init__(self, gamma, kappa,
                 gid=0,   # 0=constant resistance, 1=nonlinear resistance
                 kid=0    # 0=Hookean model, 1=Pendulum model
                 ):
        super().__init__()
        self.nVar = 2
        self.gamma = gamma
        self.kappa = kappa
        self.gid = gid
        self.kid = kid

    def forward(self, inputs):
        """ the inputs is a tensor of size=batch_size x 2 """
        inputs = inputs.view(-1, 2)
        if self.gid == 0:
            gamma = self.gamma
        else:              # nonlinear resistance
            gamma = self.gamma * torch.sum(inputs**2, dim=1)
        if self.kid == 0:  # Hookean
            Vx = self.kappa * inputs[:, 0]
        else:              # Pendulum
            c = np.pi * 0.5
            Vx = self.kappa/(c) * torch.sin(c*inputs[:, 0])
        output = inputs[:, [1, 0]]
        output[:, 1] = - gamma*inputs[:, 1] - Vx
        return output


def plot_traj_structure(ode, T, fixed_pts, lcs=None,
                 dt=0.001, nOut=100,
                 region=[-1., 1., -1., 1.0], savefile='structure'):
    ''' @ode : the ODE system
    '''
    nS = 12
    n = ode.nVar
    p1 = torch.zeros(nOut, nS, n)
    x = torch.linspace(region[0], region[1], 5)
    y = torch.linspace(region[2], region[3], 5)
    p1[0, :, 0] = torch.cat((x[1:4], x[1:4],
                             torch.tensor([region[0]]*3+[region[1]]*3).float()
                             ), dim=0)
    p1[0, :, 1] = torch.cat((torch.tensor([region[2]]*3+[region[3]]*3).float(),
                             y[1:4], y[1:4]), dim=0)
    with torch.no_grad():
        print('Calculating evaluation data ...', end=' ')
        for i in range(nOut-1):
            nt = int(T/nOut/dt)
            p1[i+1, :, :] = ode.ode_rk3(p1[i, :, :], dt, nt)
        print('Done.')

    if lcs is not None:
        nLC = lcs.shape[0]
        pLC = torch.zeros(nOut, nLC, n)
        pLC[0, :, 0:n] = torch.tensor(lcs[:, 0:n]).float()
        nt = 10
        with torch.no_grad():
            print('Calculating limit cycle data ...', end=' ')
            for ip in range(nLC):
                Ti = lcs[ip, n]
                for i in range(nOut-1):
                    dt = Ti/(nOut-1)/nt
                    pLC[i+1, ip, :] = ode.ode_rk3(pLC[i, ip, :], dt, nt)
            print('done.')
    else:
        nLC = 0

    f = plt.figure(figsize=[9, 9])
    ax = f.add_subplot(111)
    for ip in np.arange(nS):
        plt.plot(p1[:, ip, 0], p1[:, ip, 1], linewidth=1,
                 alpha=0.7, zorder=1)
        for ip in np.arange(nLC):
            plt.plot(pLC[:, ip, 0], pLC[:, ip, 1], color='yellow',
                     linewidth=1, alpha=0.6, zorder=4)
        ax.scatter(fixed_pts[:, 0], fixed_pts[:, 1], color='red',
                   marker='+', alpha=0.9, edgecolors=None,
                   zorder=5)
    ax.set_xlim(region[0:2])
    ax.set_ylim(region[2:4])
    plt.xlabel('x')
    plt.ylabel('v')

    plt.tight_layout()
    plt.savefig(savefile+'.pdf', bbox_inches='tight', dpi=288)
    print(f'The result is saved to figure {savefile}.pdf!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Langevin dynamics')
    parser.add_argument('-gamma', type=float, default=3.0, metavar='g',
                        help='input diffusion constant gamma (default: 3.0)')
    parser.add_argument('-kappa', type=float, default=4, metavar='k',
                        help='input elastic constant kappa (default: 4.0)')
    parser.add_argument('-gid', type=int, default=1, metavar='gid',
                        help='input diffusion type (0=const, 1=nonlinear)')
    parser.add_argument('-kid', type=int, default=1, metavar='kid',
                        help='input potential type (0=Hookean, 1=Pendulum)')
    parser.add_argument('-lr', type=float, default=0.0064, metavar='LR',
                        help='step size (default: 0.0064)')
    parser.add_argument('--niter', type=int, default=200, metavar='epoch',
                        help='number of iterations to calculate a limit cycle')
    args = parser.parse_args()
    print(args)

    gamma = args.gamma
    kappa = args.kappa
    gid = args.gid
    kid = args.kid

    niter = args.niter
    lr = args.lr

    LgNet = LangevinNet(gamma, kappa, gid=gid, kid=kid)
    oa1 = oa.OdeAnalyzer(LgNet)

    # %% find fixed point
    x10 = np.array([0., 0.])
    x10 = torch.tensor(x10).float()
    print(f'Initial guess of the fixed point is x1={x10}')
    x1 = oa1.find_fixed_pt(x10)
    x1 = x1[0]
    print(f'The fixed point found is x1={x1}')

    # check the solution
    x1t = torch.tensor(x1).float()
    f1 = oa1.ode(x1t)
    print(f'Check the fixed points found  f(x1)={f1}')
    fixed_pts = x1.reshape((1, LgNet.nVar))

    # %% find the limit cycle
    nLC = 1
    x0 = np.zeros((nLC, LgNet.nVar))
    T0 = np.zeros((nLC, 1))
    if abs(gamma - 3) < 1e-3:
        x0[0] = np.array([0.25, 0.061], dtype=float)
        T0[0] = 3.138
    else:
        x0[0] = np.array([0.284, 0.0305], dtype=float)
        T0[0] = 3.1406
    lcs = np.zeros((nLC, LgNet.nVar+1), dtype=float)
    x, T = oa1.find_limit_cycle(x0, T0, niter=niter, lr=lr)
    lcs[0, 0:LgNet.nVar] = x
    lcs[0, LgNet.nVar] = T

    # %% plot some trajectories, together with fixed point and limit cycle
    plot_traj_structure(LgNet, T=25, fixed_pts=fixed_pts, lcs=lcs,
                 dt=0.001, nOut=200, savefile='Langevin_structure')

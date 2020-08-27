#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File 	 : test_Lorenz.py
@Time 	 : 2020/07/21
@Author :  Haijn Yu <hyu@lsec.cc.ac.cn>
@Desc	 : apply ode_analyzer.py to Lorenz system
'''

import numpy as np
import torch
import ode_analyzer as oa
import argparse
import matplotlib.pyplot as plt


class LorenzNet(oa.OdeModel):
    """ A neural network to simulate Lorenz dyanmics """

    def __init__(self, r, sigma=10, b=8.0/3):
        super().__init__()
        self.nVar = 3
        self.sigma = sigma
        self.b = b
        self.r = r

    def forward(self, inputs):
        """ the inputs is a tensor of size=batch_size x 2 """
        inputs = inputs.view(-1, self.nVar)
        x, y, z = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        output = torch.zeros_like(inputs)
        s = self.sigma
        r = self.r
        b = self.b
        output[:, 0] = s * (y-x)
        output[:, 1] = x * (r-z) - y
        output[:, 2] = x*y - b*z
        return output


def plot_traj_structure(ode: oa.OdeModel,
                        T, fixed_pts, lcs=None,
                        dt=0.001, nOut=100,
                        region=[-10.0, 10, -10, 10, 0, 1],
                        savefile='run_cmp'):
    nS = 4
    n = ode.nVar
    if T/nOut > 0.025:
        nOut = int(T/0.025)
    p1 = torch.zeros(nOut, nS, n)
    p1[0, :, 0] = torch.tensor([region[0], -1.0, 1.0, region[1]]).float()
    p1[0, :, 1] = torch.tensor([region[2], -1.0, 1.0, region[3]]).float()
    p1[0, :, 2] = torch.tensor(region[4]).float()
    with torch.no_grad():
        print('Calculating evaluation data ...', end=' ')
        for i in range(nOut-1):
            nt = int(T/nOut/dt)
            p1[i+1, ...] = ode.ode_rk3(p1[i, ...], dt, nt)
        print('done.')

    if lcs is not None:
        nLC = lcs.shape[0]
        pLC = torch.zeros(nOut, nLC, n)
        pLC[0, :, 0:n] = torch.tensor(lcs[:, 0:n]).float()
        nt = 10
        with torch.no_grad():
            print('Calculating limit cycle data ...', end=' ')
            for ip in range(nLC):
                Tlc = lcs[ip, n]
                for i in range(nOut-1):
                    dt = Tlc/(nOut-1)/nt
                    pLC[i+1, ip, :] = ode.ode_rk3(pLC[i, ip, :], dt, nt)
            print('done.')
    else:
        nLC = 0

    f = plt.figure(figsize=[12, 4], dpi=144)
    ax = f.add_subplot(131)
    ax.scatter(fixed_pts[:, 0], fixed_pts[:, 1], color='red', 
                label='fixed points',
               marker='+', alpha=0.9, edgecolors=None, zorder=5)
    for ip in np.arange(nLC):
        plt.plot(pLC[:, ip, 0], pLC[:, ip, 1], color='yellow', 
                 label='limit cycles',
                 linewidth=1, alpha=0.6, zorder=4)
    for ip in np.arange(nS):
        plt.plot(p1[:, ip, 0], p1[:, ip, 1], 
                 markersize=1, alpha=0.8, zorder=1)
    # plt.legend(['fixed points', 'limit cycles'])
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')

    ax = f.add_subplot(132)
    for ip in np.arange(nS):
        plt.plot(p1[:, ip, 0], p1[:, ip, 2], 
                 markersize=1, alpha=0.8, zorder=1)
    for ip in np.arange(nLC):
        plt.plot(pLC[:, ip, 0], pLC[:, ip, 2], color='yellow',
                    linewidth=1, alpha=0.6, zorder=4)
    ax.scatter(fixed_pts[:, 0], fixed_pts[:, 2], color='red',
                marker='+', alpha=0.9, edgecolors=None,
                zorder=5)
    plt.xlabel('X')
    plt.ylabel('Z')

    ax = f.add_subplot(133)
    for ip in np.arange(nS):
        plt.plot(p1[:, ip, 1], p1[:, ip, 2],
                 markersize=1, alpha=0.8, zorder=1)
    for ip in np.arange(nLC):
        plt.plot(pLC[:, ip, 1], pLC[:, ip, 2], color='yellow',
                    linewidth=1, alpha=0.6, zorder=4)
    ax.scatter(fixed_pts[:, 1], fixed_pts[:, 2], color='red',
                marker='+', alpha=0.9, edgecolors=None,
                zorder=5)
    plt.xlabel('Y')
    plt.ylabel('Z')
    
    plt.savefig(savefile+'.pdf', bbox_inches='tight', dpi=288)


def calc_Lyapunov_exp1(lz_net, T0=5, nOut=5000, dt=0.01):
    print(f'T0={T0}, nOut={nOut}, dt={dt}')
    T = int(dt*nOut)
    nOut_tot = int((T+T0)/dt)
    Path = lz_net.gen_sample_paths([-25., 25, -25, 25, -5, 30],
                         nS=1, T=T+T0, dt=0.001, nOut=nOut_tot)
    data = Path[:, 0]
    data = data[nOut_tot-nOut:nOut_tot]
    print(f'dt={dt}, T={T},  len(data)=', data.shape)
    x = np.arange(nOut) * dt
    Tmean = oa.plot_fft(x, data)
    K = int(2/dt)
    P = nOut//15
    print('Tmean=', Tmean)
    Lindex, yy = oa.estimate_Lyapunov_exp1(data, dt, P=P, J=11, m=5, K=K)


def analyze_Lorenz_structure(ode_net, r, b, lr, niter):
    oa1 = oa.OdeAnalyzer(ode_net)
    # %% find the two fixed points
    q = np.sqrt(b*(r-1.0))
    x10 = np.array([q, q, r-1.0])
    x20 = np.array([-q, -q, r-1.0])
    x10 = torch.tensor(x10).float()
    x20 = torch.tensor(x20).float()
    x1 = oa1.find_fixed_pt(x10)
    x2 = oa1.find_fixed_pt(x20)
    print('Initial guess of two fixed points are\n',
          f' x1={x10}\n',
          f' x2={x20}')
    print('The two fixed points found are\n',
          f' x1={x1}\n',
          f' x2={x2}')
    x1 = x1[0]
    x2 = x2[0]
    # check the solution
    x1t = torch.tensor(x1).float()
    f1 = ode_net(x1t)
    x2t = torch.tensor(x2).float()
    f2 = ode_net(x2t)
    print('Check the two fixed points found\n',
          f' f(x1)={f1}\n',
          f' f(x2)={f2}')
    fixed_pts = np.vstack((x1, x2))

    if r == 16:
        nLC = 2
        x0 = np.zeros((nLC, ode_net.nVar))
        T0 = np.zeros((nLC, 1))
        x0[0] = np.array([1.1, 1.8, 3.2], dtype=float)
        T0[0] = 1.3
        x0[1] = np.array([-1.1, -1.8, 3.2], dtype=float)
        T0[1] = 1.3
    elif r == 22:
        nLC = 2
        x0 = np.zeros((nLC, ode_net.nVar))
        T0 = np.zeros((nLC, 1))
        x0[0] = np.array([10.3, 13.4, 20.1], dtype=float)
        T0[0] = 0.76
        x0[1] = np.array([-10.3, -13.4, 20.1], dtype=float)
        T0[1] = 0.76
    else: 
        nLC = 0
    lcs = np.zeros((nLC, nPC+1), dtype=float)

    for i in np.arange(nLC):
        lcs[i, 0:nPC], lcs[i, nPC] = oa1.find_limit_cycle(x0[i], T0[i],
                                                      niter=niter, lr=lr)
        print('The start point and poerid of limit cycle is\n',
              f' xlc={lcs[i,0:nPC]}\n',
              f' Tlc={lcs[i, nPC]}')
    return fixed_pts, lcs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lorenz example')
    parser.add_argument('-r', type=float, default=22,
                        help='scaled Rayleigh number')
    parser.add_argument('-lr', type=float, default=0.032, metavar='LR',
                        help='learning rate (default: 0.032)')
    parser.add_argument('--niter', type=int, default=40, metavar='epoch',
                        help='niter (default:200)')
    args = parser.parse_args()
    print(args)

    sigma = 10
    b = 8.0/3
    nPC = 3
    r = args.r
    niter = args.niter
    lr = args.lr

    # %% test Lyapunov index of Lorenz system
    lz_net = LorenzNet(r=r, sigma=sigma, b=b)
    print(f'Calc Lyapunov index of Lorenz(r={r}, sigma={sigma}, b={b})')
    calc_Lyapunov_exp1(lz_net, nOut=5000, dt=0.01)

    # %% Calculate the fixed points and limit cycles of Lorenz system
    fixed_pts, lcs = analyze_Lorenz_structure(lz_net, r, b, lr, niter)
    plot_traj_structure(lz_net, T=30, fixed_pts=fixed_pts, lcs=lcs,
                        dt=0.001, nOut=200,
                        savefile='Lorenz_structure')

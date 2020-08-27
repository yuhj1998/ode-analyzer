#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File 	 : ode_analyzer.py
@Time 	 : 2020/05/1
@Author  : Haijn Yu <hyu@lsec.cc.ac.cn>
@Desc	 : Define Runge-Kutta integrators and analyzer 
            for ODE systems using pyTorch
'''

# %% 1. import library and set parameters
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.optimize as pyopt


class OdeModel(nn.Module):
    """ An abstract ODE model using pyTorch framework """

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        """ abstract forward procedure, must implemented by sub-class 
        to do the detailed calculation: the evaluation of rhs """
        pass

    def ode_rk1(self, h_in, dt, nt=1):
        """ Solve the ODE system with Euler's method """
        hn = h_in
        for i in np.arange(nt):
            h0 = hn
            rhs1 = self(h0)
            hn = h0 + dt * rhs1
        return hn

    def ode_rk2(self, h_in, dt, nt=1):
        """ March the ODE system with RK2 (Heun's method) """
        hn = h_in
        for i in np.arange(nt):
            h0 = hn
            rhs1 = self(h0)
            h1 = h0 + dt * rhs1
            rhs2 = self(h1)
            hn = h0 + dt/2 * (rhs1 + rhs2)
        return hn

    def ode_rk3(self, h_in, dt, nt=1):
        """ Solve the ODE system with RK3 """
        hn = h_in
        for i in np.arange(nt):
            h0 = hn
            rhs1 = self(h0)
            h1 = h0 + dt * rhs1
            rhs2 = self(h1)
            h2 = 3/4*h0 + 1/4*h1 + 1/4*dt*rhs2
            rhs3 = self(h2)
            hn = 1/3*h0 + 2/3*h2 + 2/3*dt*rhs3
        return hn

    def ode_rk4(self, h_in, dt, nt=1):
        """ Solve the ODE system with RK4 """
        hn = h_in
        for i in np.arange(nt):
            h0 = hn
            rhs1 = self(h0)
            h1 = h0 + dt/2.0 * rhs1
            rhs2 = self(h1)
            h2 = h0 + dt/2.0 * rhs2
            rhs3 = self(h2)
            h3 = h0 + dt * rhs3
            rhs4 = self(h3)
            hn = h0 + (rhs1 + 2*rhs2 + 2*rhs3 + rhs4)*dt/6.
        return hn

    def ode_run(self, hinit, dt, T=1, Tout=1):
        Terr = abs(T - int(T/Tout)*Tout)
        assert Terr <= 1e-6, f'T={T} should be multiple of Tout={Tout}'
        Toerr = abs(Tout - int(Tout/dt)*dt)
        assert Toerr < 1e-6,  f'Time step error: Tout={Tout}, dt={dt}'
        nPath = hinit.shape[0]
        nPC = hinit.shape[1]
        nOut = int(T/Tout) + 1
        with torch.no_grad():
            h_ode = np.zeros([nPath, nOut, nPC])
            h_ode[:, 0, :] = hinit[:, :]
            for it in np.arange(nOut-1):
                h0 = torch.FloatTensor(h_ode[:, it, :])
                hf = self.ode_rk3(h0, dt, int(Tout/dt))
                h_ode[:, it+1, :] = hf.data
        return h_ode

    def gen_sample_paths(self, region, nS=1, T=10, dt=0.001, nOut=100):
        n = self.nVar
        init_val = torch.zeros(nS, n)
        for i in np.arange(n):
            init_val[:, i] = torch.Tensor(nS).uniform_(region[2*i], region[2*i+1])
        paths = self.ode_run(init_val, dt, T, Tout=T/nOut)
        paths = paths.reshape([-1, n])
        return paths
    

class OdeAnalyzer(nn.Module):
    """ A tools to analyze the ODE system: 
        find fixed points, limit cycles, etc.
    """

    def __init__(self, odenet):
        super().__init__()
        self.ode = odenet

    def find_fixed_pt(self, x0):
        def ode_fun(x):
            ''' need convert x to tensor '''
            shape = x.shape
            x0 = torch.tensor(x).float().view(-1, self.ode.nVar)
            f = self.ode(x0)
            return f.detach().numpy().reshape(shape)

        xfix = pyopt.fsolve(ode_fun, x0, full_output=1)
        return xfix

    def find_limit_cycle(self, x0, T0, niter=100, lr=0.0128):
        x = torch.nn.Parameter(torch.tensor(x0, requires_grad=True).float())
        T = torch.nn.Parameter(torch.tensor(T0, requires_grad=True).float())
        optimizer = optim.Adam([{'params': [x, T]}], lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         'min',
                                                         factor=0.5,
                                                         patience=10)
        nt = 200
        T_beta = 100
        Tth = torch.tensor(T0, requires_grad=False).float()
        print(f'Initial guess of the limit cycle start at {x.data}', 
              f' with period {T.data}.')
        for e in range(niter):
            xmiddle = self.ode.ode_rk3(x, T/2./nt, nt)
            xt = self.ode.ode_rk3(xmiddle, T/2./nt, nt)
            loss = (torch.sum((xt-x)**2) + T_beta * F.relu(Tth/3-T)
                    + T_beta * F.relu(0.0001-torch.sum(xmiddle-x)**2))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([x, T], 0.5)
            optimizer.step()

            scheduler.step(loss)
            last_lr = optimizer.param_groups[0]["lr"]

            if e % 10 == 0 or e == niter-1:
                print(f'iter:{e:4d}/{niter}', end=' ')
                print(f'loss: {loss.item():.3e}', end=' ')
                print(f'x: {x.data}', end=' ')
                print(f'T: {T.data}', end=' ')
                print(f'lr: {last_lr}', flush=True)
        return x.detach().data, T.detach().data


if __name__ == '__main__':
    print('This is a library that defines',
          ' Runge-Kutta integrators and Analyzers for ODE systems.')

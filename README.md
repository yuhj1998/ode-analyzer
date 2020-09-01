# ODE Analyzer

A PyTorch tool to calculate the fixed points and limit cycles of ODE systems
without using Jacobian matrices.
A handy function to estimate the largest Lyapunov index based on only the trajectory data of an ODE
system is included.

## Requirements

    python3
    pytorch
    numpy
    scipy
    matploblib

## Usage

`ode_analyzer.py` defines the abstract class of an ODE and the analyzing tools.

1. Define your ODE systems by implemnenting an `OdeModel` subclass
2. Find the fixed points and limit cycles using `OdeAnayzer` defined in `ode_analyzer.py`.

Note that you need provide initial values for finding fixed points and limit cycles, which 
might be obtained by looking at some simulations of the ODE system. A set of Runge-Kutta metohds
is included in `ode_analyzer.py` for simulating ODE systems.

## Examples

#### Application to a determinisitic Langevin system 

$$
   x_t = v, \quad  v_t = - \gamma g(v) - U'(x)
$$

where,
$g(v)$ could be constant 1 or function $v^2$, which specified by parameter `gid`;
$U(x)$ is the potential energy function, for the Hookean potential it is $\frac{\kappa}{2} x^2$,
or pendulum potential which is a cosine function. This is controlled by parameter `kid` and
`kappa`.

Use

    python test_Langevin.py    
to run the code with default parameters.

Use

    python test_Langevin.py --help
to see how to call with different parameters.  

#### Application to the Lorenz '63 system

Lorenz system is a minimal 3-mode ODE system that exhibits chaotic solution.
We include a method in `ode_analyzer.py` to estimate the largest Lyapunov index 
based on only solution data. The largest Lyapunov index (>0) can be regarded as an indicator of chaos.

Use

    python test_Lorenz.py    
to run the code with default parameters.

Use

    python test_Lorenz.py --help
to see how to call with different parameters.  

Note that to generate a chaotic solution, parameter `r` need be larger than 24.06. 
The value of `r` used by Lorenz (in his '63 paper) is 28.

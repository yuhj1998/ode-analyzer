# ode-analyzer

A PyTorch tool to calculate the fixed points and limit cycles of ODE systems
and also estimate the Lyapunov index based on the trajectory data of the ODE
system.

## Requirements

    python3
    pytorch
    numpy
    scipy
    matploblib

## Usage

`ode_analyzer.py` defines the abstract class of an ODE and analyzing tools.

1. Define yourself ODE systems by implemnenting an `OdeModel` subclass
2. Find the fixed points and limit cycles using `OdeAnayzer` defined in `ode_analyzer.py`.

Note that you need provide initial values for finding fixed points and limit cycles.

## Examples

#### Application to a determinisitic Langevin system 

$$
   x_t = v, \quad  v_t = - \gamma g(v) - U'(x)
$$

where,
$g(v)$ could be constant 1 or function $v^2$, which specified by parameter `gid`;
$U(x)$ is the potential energy function, for the Hookean potential it is $\frac{\kappa}{2} x^2$,
or Pendulum potential which is a cosine function. This is controlled by parameter `kid` and
`kappa`.

Use

    python test_Langevin.py    
to run the code with default parameters.

Use

    python test_Langevin.py --help
to see how to call with different parameters.  

#### Application to the Lorenz system

Lorenz system is a minimal 3-mode ODE system that exhibits chaotic solution.
We include a method in `ode_analyzer.py` to estimate the largest Lyapunov index 
of given solution data, which can be regarded as an indicator of chaos.

Use

    python test_Lorenz.py    
to run the code with default parameters.

Use

    python test_Lorenz.py --help
to see how to call with different parameters.  

Note that to generate a chaotic solution parameter `r` need be larger than 24.06. 
The value of `r` used by Lorenz is 28.

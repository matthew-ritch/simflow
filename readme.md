We model fluid flow using [constant-density incompressible navier-stokes (convective)](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations#Incompressible_flow)
$$ \frac{\partial u}{\partial t} + (u ⋅ ∇)u - (\frac{\mu}{\rho_{0}}) * ∇^{2}u = -∇(\frac{P}{\rho_{0}}) + g $$

simflow.evolve_flow_2d does this for 2D flow fields


simflow.evolve_flow_3d does this for 3D flow fields

see use_2d_simflow.py and use_3d_simflow.py for examples of how to use the evolve_flow functions

TODO:

- Add non-constant density
- Use GPU for these ndimage convolutions
- Interface for defining boundary conditions


Requirements:

- numpy
- scipy (for ndimage)
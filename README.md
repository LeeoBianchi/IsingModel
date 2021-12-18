# IsingModel

**A parallelized (multi-threading) version of Wolff's algorithm for the analyisis of the Ising Model.**

- Wolff.py implements a universal Wolff algorithm for the simulation of a both 1 and 2-dimensional Ising lattice.

- sim1D.py provides a simulation of a 1D Ising Chain in order to compute the correlation function C(r) for different r values. The txt output is being read and plotted by the Main script.

- sim2D.py parallelizes Wolff's algorithm to compute the average magnetization per site at different temperatures, it provides a sampling to be plotted through the Main script.

- sim2D_Gamma.py parallelizes Wolff's algorithm to compute the Gamma function at different temperatures (around the critical region) for different lattice sizes, it provides a serie of samplings that are plotted in the Main script for showing the critical temperature's displacement for finite-size lattices.

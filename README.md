# Noether-normalisation
Examples of explicit normalisation constructions using Noether's normalisation lemma. 

# Includes
 1. Jupyter notebook with all code. 
 2. `surface.png` is the image used in the final report. 
 3. `rotation.mp4` is a video rotating the surface.
 4. All code in separate folders

# Requirements
 - sympy (symbolic computations)
 - numpy (datastructure for points to plot)
 - matplotlib (_inefficient_ plotting)
 - mpl_toolkits (general matplotlib toolkit)
 - vispy (_efficient_ plotting)
 - skimage (measure)
 - imageio_ffmpeg (saving to video)


# The surface $f = x^2 z^2 - xz^3 + y^2 z^2 - yz^3 - 1$

In `testSurface.py`, we test an equation by taking 
```math
v \in \left\{ x , y , z \right\}
```
and setting $u = \alpha - v^e$, $w = \beta - v^k$ for 
$$
\left\{ \alpha, \beta \right\} = \left\{ x , y , z \right\} \setminus \{ v \}.
$$
The code then tests for all $e,k \in \left\{ 0, 1 \right\}$ whether the leading coefficient of $v$ is independent of $u$ and $w$. 
The specific surface $f = x^2 z^2 - xz^3 + y^2 z^2 - yz^3 - 1$ does not satisfy this, hence in the construction for the normalisation we need to take $e \geq 2$ or $k \geq 2$.

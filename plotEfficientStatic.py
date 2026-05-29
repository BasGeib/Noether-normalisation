# Use vispy to create more responsive interactive plots

import numpy as np
import vispy
from vispy import app, scene
from vispy.scene.visuals import Mesh
from vispy.scene import TurntableCamera
from skimage import measure


def f(x, y, z):
    return (x + z**2)*(y + z)*(x + z**2 - z) + 1

def f2(x, y, z):
    return x*(x - z)*y + 1

# Change bounds to show more of the plot
def generate_volume(fn, bounds=(-4, 4), resolution=200):
    xmin, xmax = bounds
    grid = np.linspace(xmin, xmax, resolution)
    X, Y, Z = np.meshgrid(grid, grid, grid, indexing='ij')
    V = fn(X, Y, Z)
    return V, grid

def compute_isosurface(V, grid):
    # marching cubes
    verts, faces, normals, values = measure.marching_cubes(V, level=0)
    scale = (grid[-1] - grid[0]) / (len(grid) - 1)
    verts = verts * scale + grid[0]
    return verts, faces



def plot_implicit_vispy(fn):
    V, grid = generate_volume(fn)
    verts, faces = compute_isosurface(V, grid)
    canvas = scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()

    z_vals = verts[:, 2]
    z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min())

    colors = np.zeros((len(verts), 4))
    colors[:, 0] = z_norm # red
    colors[:, 1] = 0.03 # green
    colors[:, 2] = 2 - z_norm # blue
    colors[:, 3] = 1.0 # alpha
    print(len(verts))
    mesh = scene.visuals.Mesh(
    vertices=verts,
    faces=faces,
    vertex_colors=colors,
    shading='smooth'
    )
    view.add(mesh)
    view.camera = TurntableCamera(fov=60, distance=10)
    axis = scene.visuals.XYZAxis(parent=view.scene)
    app.run()

plot_implicit_vispy(f2)

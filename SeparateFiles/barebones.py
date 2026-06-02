import numpy as np
from skimage import measure
from PIL import Image
import pyvista as pv


def f3(x, y, z):
    return x**2 * z**2 - x * z**3 + y**2 * z**2 - y * z**3 - 1

def f2(x,y,z):
    return x*y*(x-z)+1

def generate_volume(fn, bounds=(-4, 4), resolution=700):
    xmin, xmax = bounds
    grid = np.linspace(xmin, xmax, resolution, dtype=np.float32)
    X, Y, Z = np.meshgrid(grid, grid, grid, indexing="ij")
    V = fn(X, Y, Z).astype(np.float32)
    return V, grid


def compute_isosurface(V, grid):
    from scipy.ndimage import gaussian_filter
    V = gaussian_filter(V, sigma=1.0)
    verts, faces, normals, values = measure.marching_cubes(V, level=0)
    print("Number of vertices:", len(verts))
    scale = (grid[-1] - grid[0]) / (len(grid) - 1)
    verts = verts * scale + grid[0]
    return verts, faces

def show_mesh_with_save(verts, faces):
	faces_pv = np.hstack(
		[np.full((faces.shape[0], 1), 3), faces]
	).astype(np.int64).ravel()

	mesh = pv.PolyData(verts, faces_pv)

	z_vals = verts[:, 2]
	z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-12)
	colors = np.zeros((len(verts), 4), dtype=np.float32)
	colors[:, 0] = np.clip(2 * z_norm, 0, 1) # red
	colors[:, 1] = 0.0 # green
	colors[:, 2] = np.clip(1 - z_norm, 0, 1) # blue
	colors[:, 3] = 1.0 # alpha

	colors_uint8 = (255 * colors).astype(np.uint8)

	mesh.point_data["rgba"] = colors_uint8

	plotter = pv.Plotter()
	plotter.set_background("white")
	plotter.add_mesh(
	mesh,
	scalars="rgba",
	rgba=True,
	smooth_shading=True
	)


	def save_image():
		import datetime as dt
		date = dt.datetime.now()
		date = date.strftime("%A, %d %B %Y %I %M%p")
		filename = "pyvista_"+date+".png"
		plotter.screenshot(filename)
		print(f"Saved {filename}")
	def save_movie():
		import datetime as dt
		date = dt.datetime.now()
		date = date.strftime("%A, %d %B %Y %I %M%p")
		filename = "pyvista_"+date+".mp4"
		path = plotter.generate_orbital_path(n_points=200, shift=mesh.length)
		plotter.open_movie(filename)
		print(plotter.camera_position)
		plotter.orbit_on_path(path, write_frames=True, progress_bar=True, focus = plotter.camera_position[0])
		print(f"Saved {filename}")
		plotter.close()
	plotter.add_key_event("s", save_image)
	plotter.add_key_event("m", save_movie)
	plotter.show()



def test(fn):
	V, grid = generate_volume(fn)
	verts, faces = compute_isosurface(V, grid)
	show_mesh_with_save(verts, faces)


if __name__ == "__main__":

    test(f2)

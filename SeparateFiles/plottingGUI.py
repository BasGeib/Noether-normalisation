import os
os.environ["QT_OPENGL"] = "software"


import traceback
import numpy as np
#from scipy.ndimage import gaussian_filter
#from skimage.measure import marching_cubes

import pyvista as pv
from pyvistaqt import QtInteractor

from PyQt5 import QtCore, QtWidgets


SAFE_NAMES = {
    "np": np,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "where": np.where,
    "pi": np.pi,
    "e": np.e,
}


class UserCancelledError(RuntimeError):
    """Raised when the user cancels a long-running operation."""


def make_function_from_string(expr: str):
    expr = expr.strip()
    if not expr:
        raise ValueError("Please enter a function of x, y, z.")

    code = compile(expr, "<user_function>", "eval")
    for name in code.co_names:
        if name not in SAFE_NAMES and name not in {"x", "y", "z"}:
            raise ValueError(f"Unsupported name in expression: {name}")

    def fn(x, y, z):
        local_vars = {"x": x, "y": y, "z": z}
        return eval(code, {"__builtins__": {}}, {**SAFE_NAMES, **local_vars})

    return fn


def _maybe_cancel(cancel_cb):
    if cancel_cb and cancel_cb():
        raise UserCancelledError("Operation cancelled by user.")


def generate_volume(fn, bounds=(-4.0, 4.0), resolution=200, progress_cb=None, cancel_cb=None):
    import pyvista as pv
    import numpy as np

    xmin, xmax = bounds

    _maybe_cancel(cancel_cb)
    if progress_cb:
        progress_cb(5, "Creating grid...")
    grid = np.linspace(xmin, xmax, resolution, dtype=np.float32)

    _maybe_cancel(cancel_cb)
    if progress_cb:
        progress_cb(20, "Creating 3D meshgrid...")
    X, Y, Z = np.meshgrid(grid, grid, grid, indexing="ij")

    _maybe_cancel(cancel_cb)
    if progress_cb:
        progress_cb(45, "Evaluating function...")
    V = fn(X, Y, Z).astype(np.float32)

    _maybe_cancel(cancel_cb)
    if progress_cb:
        progress_cb(65, "Smoothing scalar field...")

    
    V_padded = np.pad(V, 1, mode='edge')

    V = (
    V_padded[1:-1, 1:-1, 1:-1] +
    V_padded[2:, 1:-1, 1:-1] + V_padded[:-2, 1:-1, 1:-1] +
    V_padded[1:-1, 2:, 1:-1] + V_padded[1:-1, :-2, 1:-1] +
    V_padded[1:-1, 1:-1, 2:] + V_padded[1:-1, 1:-1, :-2]
    ) / 7


    _maybe_cancel(cancel_cb)
    if progress_cb:
        progress_cb(85, "Extracting isosurface...")

    dx = (xmax - xmin) / (resolution - 1)

    #grid_data = pv.UniformGrid()
    grid_data = pv.ImageData()

    grid_data.dimensions = V.shape
    grid_data.spacing = (dx, dx, dx)
    grid_data.origin = (xmin, xmin, xmin)

    grid_data.point_data["values"] = V.flatten(order="F")

    mesh = grid_data.contour([0])

    verts = mesh.points

    faces = mesh.faces.reshape(-1, 4)[:, 1:]

    _maybe_cancel(cancel_cb)
    if progress_cb:
        progress_cb(100, "Done")

    return verts, faces


def build_mesh(verts, faces):
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64).ravel()
    mesh = pv.PolyData(verts, faces_pv)

    z_vals = verts[:, 2]
    z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-12)
    colors = np.zeros((len(verts), 4), dtype=np.float32)
    colors[:, 0] = np.clip(2 * z_norm, 0, 1)  # red
    colors[:, 1] = 0.0  # green
    colors[:, 2] = np.clip(1 - z_norm, 0, 1)  # blue
    colors[:, 3] = 1.0  # alpha
    mesh.point_data["rgba"] = (255 * colors).astype(np.uint8)
    return mesh


class MeshComputeThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)
    result = QtCore.pyqtSignal(object, object)
    error = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal(str)

    def __init__(self, expr, bounds, resolution, parent=None):
        super().__init__(parent)
        self.expr = expr
        self.bounds = bounds
        self.resolution = resolution
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            fn = make_function_from_string(self.expr)
            verts, faces = generate_volume(
                fn,
                bounds=self.bounds,
                resolution=self.resolution,
                progress_cb=lambda v, msg: self.progress.emit(v, msg),
                cancel_cb=lambda: self._cancel_requested,
            )
            if self._cancel_requested:
                raise UserCancelledError("Mesh computation cancelled.")
            self.result.emit(verts, faces)
        except UserCancelledError as exc:
            self.cancelled.emit(str(exc))
        except Exception:
            self.error.emit(traceback.format_exc())


class MovieRenderThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)
    done = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal(str)

    #def __init__(self, expr, bounds, resolution, camera_position, output_path, parent=None):
    def __init__(self, expr, bounds, resolution, camera_position, output_path, show_axes=True, parent=None):
        super().__init__(parent)
        self.expr = expr
        self.bounds = bounds
        self.resolution = resolution
        self.camera_position = camera_position
        self.output_path = output_path
        
        self.show_axes = bool(show_axes)

        self._cancel_requested = False
        self._plotter = None

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            fn = make_function_from_string(self.expr)

            self.progress.emit(5, "Computing high-resolution mesh...")
            verts, faces = generate_volume(
                fn,
                bounds=self.bounds,
                resolution=self.resolution,
                progress_cb=lambda v, msg: self.progress.emit(min(v, 70), msg),
                cancel_cb=lambda: self._cancel_requested,
            )
            if self._cancel_requested:
                raise UserCancelledError("Movie rendering cancelled.")
            mesh = build_mesh(verts, faces)

            self.progress.emit(75, "Preparing movie renderer...")
            plotter = pv.Plotter(off_screen=True)
            self._plotter = plotter
            plotter.set_background("white")
            plotter.enable_anti_aliasing()

            
            plotter.add_mesh(mesh, scalars="rgba", rgba=True, smooth_shading=True)
            if self.show_axes:
                plotter.show_axes()
            plotter.camera_position = self.camera_position




            n_points = 300
            orbital_path = plotter.generate_orbital_path(factor=3.0, n_points=n_points, shift=mesh.length)
            path_points = np.asarray(orbital_path.points)
            if path_points.size == 0:
                raise RuntimeError("Orbital path generation returned no points.")

            focal_point = plotter.camera.focal_point
            viewup = plotter.camera.up
            plotter.open_movie(self.output_path)

            self.progress.emit(80, "Rendering movie frames...")
            for i, position in enumerate(path_points):
                if self._cancel_requested:
                    raise UserCancelledError("Movie rendering cancelled.")
                plotter.camera_position = [tuple(position), tuple(focal_point), tuple(viewup)]
                plotter.write_frame()
                progress = 80 + int(20 * (i + 1) / len(path_points))
                self.progress.emit(progress, f"Rendering frame {i+1}/{len(path_points)}...")

            plotter.close()
            self._plotter = None
            self.done.emit(self.output_path)
        except UserCancelledError as exc:
            try:
                if self._plotter is not None:
                    self._plotter.close()
            except Exception:
                pass
            self._plotter = None
            self.cancelled.emit(str(exc))
        except Exception:
            try:
                if self._plotter is not None:
                    self._plotter.close()
            except Exception:
                pass
            self._plotter = None
            self.error.emit(traceback.format_exc())


class EmbeddedPlotWidget(QtWidgets.QFrame):
    enterPressed = QtCore.pyqtSignal()

    def __init__(self, parent=None, show_axes=False):
        super().__init__(parent)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        self.layout().addWidget(self.plotter.interactor)

        self.plotter.set_background("white")
        self.axes_visible = bool(show_axes)
        if self.axes_visible:
            self.plotter.show_axes()

        self.plotter.enable_anti_aliasing()
        self.mesh = None
        self.plotter.interactor.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self.plotter.interactor and event.type() == QtCore.QEvent.KeyPress:
            if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                self.enterPressed.emit()
                return True
        return super().eventFilter(obj, event)

    def set_mesh(self, mesh):
        self.mesh = mesh
        self.plotter.clear()
        self.plotter.add_mesh(mesh, scalars="rgba", rgba=True, smooth_shading=True)
        self.plotter.reset_camera()
        self.set_axes_visible(self.axes_visible)
        self.plotter.render()


    def set_axes_visible(self, visible: bool):
        self.axes_visible = bool(visible)
        if self.axes_visible:
            self.plotter.show_axes()
        else:
            self.plotter.hide_axes()
        self.plotter.render()

    def save_screenshot(self, path):
        self.plotter.screenshot(path)

    def get_camera_position(self):
        return self.plotter.camera_position

    def set_camera_position(self, camera_position):
        self.plotter.camera_position = camera_position


class PreviewDialog(QtWidgets.QDialog):
    def __init__(self, expr, bounds, final_resolution, parent=None, show_axes=True):
        super().__init__(parent)
        self.expr = expr
        self.bounds = bounds
        self.final_resolution = final_resolution
        self.compute_thread = None
        self.movie_thread = None
        self.preview_mesh = None
        self.show_axes = show_axes

        self.setWindowTitle("Movie Preview")
        self.resize(1000, 700)

        layout = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "Preview is rendered at resolution 100. Rotate/zoom the surface.\n"
            "Press Enter inside the preview window or click 'Render Final Movie' to export the final movie\n"
            "using the main window resolution and the camera orientation chosen here."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        
        
        self.viewer = EmbeddedPlotWidget(self, show_axes=self.show_axes)
        layout.addWidget(self.viewer, stretch=1)

        controls = QtWidgets.QHBoxLayout()

        self.axes_checkbox = QtWidgets.QCheckBox("Show axes")
        self.axes_checkbox.setChecked(self.show_axes)
        self.axes_checkbox.toggled.connect(self.on_axes_toggled)
        controls.addWidget(self.axes_checkbox)

        
        self.render_movie_btn = QtWidgets.QPushButton("Render Final Movie")
        self.render_movie_btn.clicked.connect(self.render_final_movie)
        controls.addWidget(self.render_movie_btn)

        self.cancel_btn = QtWidgets.QPushButton("Cancel Render")
        self.cancel_btn.clicked.connect(self.cancel_movie_render)
        self.cancel_btn.setEnabled(False)
        controls.addWidget(self.cancel_btn)

        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        controls.addWidget(self.close_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.progress_label = QtWidgets.QLabel("Generating preview mesh...")
        layout.addWidget(self.progress_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.viewer.enterPressed.connect(self.render_final_movie)
        self.start_preview_compute()

    def start_preview_compute(self):
        self.render_movie_btn.setEnabled(False)
        self.compute_thread = MeshComputeThread(self.expr, self.bounds, 100, self)
        self.compute_thread.progress.connect(self.on_progress)
        self.compute_thread.result.connect(self.on_preview_ready)
        self.compute_thread.error.connect(self.on_error)
        self.compute_thread.cancelled.connect(self.on_cancelled)
        self.compute_thread.start()
        
    def on_axes_toggled(self, checked):
        self.show_axes = bool(checked)
        self.viewer.set_axes_visible(self.show_axes)

    def on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def on_preview_ready(self, verts, faces):

        self.preview_mesh = build_mesh(verts, faces)
        self.viewer.set_mesh(self.preview_mesh)
        self.viewer.set_axes_visible(self.axes_checkbox.isChecked())

        self.progress_label.setText("Preview ready. Adjust the camera, then press Enter or click Render Final Movie.")
        self.progress_bar.setValue(100)
        self.render_movie_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.viewer.setFocus()

    def on_error(self, tb_text):
        self.render_movie_btn.setEnabled(self.preview_mesh is not None)
        self.cancel_btn.setEnabled(False)
        QtWidgets.QMessageBox.critical(self, "Error", tb_text)

    def on_cancelled(self, message):
        self.progress_label.setText(message)
        self.cancel_btn.setEnabled(False)
        self.render_movie_btn.setEnabled(self.preview_mesh is not None)

    def render_final_movie(self):
        if self.preview_mesh is None or (self.movie_thread is not None and self.movie_thread.isRunning()):
            return

        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Movie",
            "",
            "MP4 video (*.mp4);;All files (*.*)",
        )
        if not output_path:
            return

        camera_position = self.viewer.get_camera_position()
        self.render_movie_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.movie_thread = MovieRenderThread(
            self.expr,
            self.bounds,
            self.final_resolution,
            camera_position,
            output_path,
            show_axes=self.show_axes,
            parent=self)
        self.movie_thread.progress.connect(self.on_progress)
        self.movie_thread.done.connect(self.on_movie_done)
        self.movie_thread.error.connect(self.on_error)
        self.movie_thread.cancelled.connect(self.on_cancelled)
        self.movie_thread.start()

    def cancel_movie_render(self):
        if self.movie_thread is not None and self.movie_thread.isRunning():
            self.progress_label.setText("Cancelling movie render...")
            self.cancel_btn.setEnabled(False)
            self.movie_thread.request_cancel()

    def on_movie_done(self, output_path):
        self.progress_label.setText("Movie saved.")
        self.progress_bar.setValue(100)
        self.render_movie_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        QtWidgets.QMessageBox.information(self, "Saved", f"Movie saved to:\n{output_path}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Implicit Surface Viewer (Qt Embedded)")
        self.resize(1200, 850)

        self.current_mesh = None
        self.compute_thread = None
        self.movie_thread = None
        self.current_expression = "x*y*(x-z)+1"
        
        self.axes_visible = False


        self._build_ui()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        form = QtWidgets.QGridLayout()
        root.addLayout(form)

        title = QtWidgets.QLabel("Implicit function f(x, y, z) = 0")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        form.addWidget(title, 0, 0, 1, 4)

        self.function_edit = QtWidgets.QLineEdit(self.current_expression)
        form.addWidget(self.function_edit, 1, 0, 1, 4)

        hint = QtWidgets.QLabel(
            "Examples: x*y*(x-z)+1    |    x**2*z**2 - x*z**3 + y**2*z**2 - y*z**3 - 1"
        )
        hint.setStyleSheet("color: #555555;")
        form.addWidget(hint, 2, 0, 1, 4)

        form.addWidget(QtWidgets.QLabel("Resolution"), 3, 0)
        self.resolution_spin = QtWidgets.QSpinBox()
        self.resolution_spin.setRange(20, 1000)
        self.resolution_spin.setValue(200)
        form.addWidget(self.resolution_spin, 4, 0)

        form.addWidget(QtWidgets.QLabel("Bounds min"), 3, 1)
        self.bounds_min = QtWidgets.QDoubleSpinBox()
        self.bounds_min.setRange(-1e6, 1e6)
        self.bounds_min.setDecimals(3)
        self.bounds_min.setValue(-4.0)
        form.addWidget(self.bounds_min, 4, 1)

        form.addWidget(QtWidgets.QLabel("Bounds max"), 3, 2)
        self.bounds_max = QtWidgets.QDoubleSpinBox()
        self.bounds_max.setRange(-1e6, 1e6)
        self.bounds_max.setDecimals(3)
        self.bounds_max.setValue(4.0)
        form.addWidget(self.bounds_max, 4, 2)

        buttons = QtWidgets.QHBoxLayout()
        root.addLayout(buttons)

        self.plot_btn = QtWidgets.QPushButton("Plot")
        self.plot_btn.clicked.connect(self.plot_surface)
        buttons.addWidget(self.plot_btn)

        self.preview_btn = QtWidgets.QPushButton("Movie Preview")
        self.preview_btn.clicked.connect(self.open_movie_preview)
        buttons.addWidget(self.preview_btn)

        self.save_image_btn = QtWidgets.QPushButton("Save Image")
        self.save_image_btn.clicked.connect(self.save_image)
        self.save_image_btn.setEnabled(False)
        buttons.addWidget(self.save_image_btn)

        self.save_movie_btn = QtWidgets.QPushButton("Save Movie")
        self.save_movie_btn.clicked.connect(self.save_movie_from_main_view)
        self.save_movie_btn.setEnabled(False)
        buttons.addWidget(self.save_movie_btn)
        
        
        self.axes_checkbox = QtWidgets.QCheckBox("Show axes")
        self.axes_checkbox.setChecked(True)
        self.axes_checkbox.toggled.connect(self.on_axes_toggled)
        buttons.addWidget(self.axes_checkbox)


        self.cancel_btn = QtWidgets.QPushButton("Cancel Render")
        self.cancel_btn.clicked.connect(self.cancel_current_render)
        self.cancel_btn.setEnabled(False)
        buttons.addWidget(self.cancel_btn)

        buttons.addStretch(1)

        self.status_label = QtWidgets.QLabel(
            "Plot embeds the 3D viewer directly in this window. Use Movie Preview for a fast preview at resolution 100."
        )
        root.addWidget(self.status_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        root.addWidget(self.progress_bar)

        #self.viewer = EmbeddedPlotWidget(self)
        
        self.viewer = EmbeddedPlotWidget(self, show_axes=self.axes_visible)

        root.addWidget(self.viewer, stretch=1)

    def on_axes_toggled(self, checked):
        self.axes_visible = bool(checked)
        self.viewer.set_axes_visible(self.axes_visible)

    def get_bounds(self):
        bmin = self.bounds_min.value()
        bmax = self.bounds_max.value()
        if bmin >= bmax:
            raise ValueError("Bounds min must be smaller than bounds max.")
        return (bmin, bmax)

    def set_busy(self, busy: bool):
        self.plot_btn.setEnabled(not busy)
        self.preview_btn.setEnabled(not busy)

    def on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def on_compute_error(self, tb_text):
        self.set_busy(False)
        self.cancel_btn.setEnabled(False)
        self.save_movie_btn.setEnabled(self.current_mesh is not None)
        QtWidgets.QMessageBox.critical(self, "Error", tb_text)

    def on_cancelled(self, message):
        self.set_busy(False)
        self.cancel_btn.setEnabled(False)
        self.save_movie_btn.setEnabled(self.current_mesh is not None)
        self.status_label.setText(message)

    def plot_surface(self):
        try:
            expr = self.function_edit.text().strip()
            resolution = int(self.resolution_spin.value())
            bounds = self.get_bounds()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
            return

        self.current_expression = expr
        self.current_mesh = None
        self.save_image_btn.setEnabled(False)
        self.save_movie_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.set_busy(True)
        self.on_progress(0, "Starting mesh computation...")

        self.compute_thread = MeshComputeThread(expr, bounds, resolution, self)
        self.compute_thread.progress.connect(self.on_progress)
        self.compute_thread.result.connect(self.on_mesh_ready)
        self.compute_thread.error.connect(self.on_compute_error)
        self.compute_thread.cancelled.connect(self.on_cancelled)
        self.compute_thread.start()

    def on_mesh_ready(self, verts, faces):
        #self.current_mesh = build_mesh(verts, faces)
        #self.viewer.set_mesh(self.current_mesh)
        
        self.current_mesh = build_mesh(verts, faces)
        self.viewer.set_mesh(self.current_mesh)
        self.viewer.set_axes_visible(self.axes_visible)

        self.save_image_btn.setEnabled(True)
        self.save_movie_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.set_busy(False)
        self.on_progress(100, f"Plot ready. Vertices: {len(verts):,}")

    def save_image(self):
        if self.current_mesh is None:
            return

        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;All files (*.*)",
        )
        if not output_path:
            return

        self.viewer.save_screenshot(output_path)
        QtWidgets.QMessageBox.information(self, "Saved", f"Image saved to:\n{output_path}")

    def save_movie_from_main_view(self):
        if self.current_mesh is None:
            return

        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Movie",
            "",
            "MP4 video (*.mp4);;All files (*.*)",
        )
        if not output_path:
            return

        expr = self.function_edit.text().strip()
        bounds = self.get_bounds()
        resolution = int(self.resolution_spin.value())
        camera_position = self.viewer.get_camera_position()

        self.set_busy(True)
        self.cancel_btn.setEnabled(True)
        self.save_movie_btn.setEnabled(False)
        self.on_progress(0, "Preparing movie export...")

        self.movie_thread = MovieRenderThread(expr, bounds, resolution, camera_position, output_path, show_axes=self.axes_visible, parent=self)

        self.movie_thread.progress.connect(self.on_progress)
        self.movie_thread.done.connect(self.on_movie_done)
        self.movie_thread.error.connect(self.on_compute_error)
        self.movie_thread.cancelled.connect(self.on_cancelled)
        self.movie_thread.start()

    def on_movie_done(self, output_path):
        self.set_busy(False)
        self.cancel_btn.setEnabled(False)
        self.save_movie_btn.setEnabled(True)
        self.on_progress(100, "Movie saved.")
        QtWidgets.QMessageBox.information(self, "Saved", f"Movie saved to:\n{output_path}")

    def cancel_current_render(self):
        if self.movie_thread is not None and self.movie_thread.isRunning():
            self.status_label.setText("Cancelling movie render...")
            self.cancel_btn.setEnabled(False)
            self.movie_thread.request_cancel()
            return
        if self.compute_thread is not None and self.compute_thread.isRunning():
            self.status_label.setText("Cancelling mesh computation...")
            self.cancel_btn.setEnabled(False)
            self.compute_thread.request_cancel()

    def open_movie_preview(self):
        try:
            expr = self.function_edit.text().strip()
            resolution = int(self.resolution_spin.value())
            bounds = self.get_bounds()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
            return

        dlg = PreviewDialog(expr, bounds, resolution, parent=self, show_axes=self.axes_visible)
        dlg.exec_()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()

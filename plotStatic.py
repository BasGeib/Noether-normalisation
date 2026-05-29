from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def plot_implicit(fn, bbox=(-5,5), cmap=cm.viridis):
    xmin, xmax = bbox
    ymin, ymax = bbox
    zmin, zmax = bbox

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    A = np.linspace(xmin, xmax, 200)
    B = np.linspace(xmin, xmax, 40)
    A1, A2 = np.meshgrid(A, A)
    norm = plt.Normalize(vmin=xmin, vmax=xmax)

    for z in B:
        X, Y = A1, A2
        Z = fn(X, Y, z)

        ax.contour(
            X, Y, Z + z,
            levels=[z],
            zdir='z',
            colors=[cmap(norm(z))],
            alpha=0.7
        )
    for y in B:
        X, Z = A1, A2
        Y = fn(X, y, Z)

        ax.contour(
            X, Y + y, Z,
            levels=[y],
            zdir='y',
            colors=[cmap(norm(y))],
            alpha=0.5
        )
    for x in B:
        Y, Z = A1, A2
        X = fn(x, Y, Z)

        ax.contour(
            X + x, Y, Z,
            levels=[x],
            zdir='x',
            colors=[cmap(norm(x))],
            alpha=0.5
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def f(x,y,z):
    return (x+z**2)*(y+z)*(x+z**2-z)+1

def f2(x,y,z):
    return x*(x-z)*y+1
plot_implicit(f2)

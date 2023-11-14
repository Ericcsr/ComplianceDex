import matplotlib
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

from matplotlib import cm
from matplotlib.ticker import LinearLocator
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data", type=str, default="gpis.npz")
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--axis", type=str, default="z")
parser.add_argument("--isf_limit", type=float, default=1.0)

args = parser.parse_args()

data = np.load(args.data)
test_mean = data["mean"]
test_var = data["var"]
num_steps = test_mean.shape[0]


if args.axis == "x": 
    X, Y = np.meshgrid(np.linspace(data["lb"][1],data["ub"][1],num_steps),
                    np.linspace(data["lb"][2],data["ub"][2],num_steps), indexing="xy")
elif args.axis=="y":
    X, Y = np.meshgrid(np.linspace(data["lb"][0],data["ub"][0],num_steps),
                    np.linspace(data["lb"][2],data["ub"][2],num_steps), indexing="xy")
else:
    X, Y = np.meshgrid(np.linspace(data["lb"][0],data["ub"][0],num_steps),
                    np.linspace(data["lb"][1],data["ub"][1],num_steps), indexing="xy")

for i in range(0, num_steps, args.stride):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    if args.axis == "x":
        Z = test_mean[i]
        color_dimension = test_var[i]
    elif args.axis == "y":
        Z = test_mean[:,i]
        color_dimension = test_var[:,i]
    else:
        Z = test_mean[:,:,i]
        color_dimension = test_var[:,:,i]
    
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)

    # Plot the surface.
    print(fcolors)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,facecolors=fcolors,
                        linewidth=0, antialiased=True)
    surf2 = ax.plot_surface(X, Y, np.zeros_like(Z))

    # Customize the z axis.
    ax.set_zlim(-args.isf_limit, args.isf_limit)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
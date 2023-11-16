import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
import matplotlib.colors as colors

# Define the function to plot
def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

# Generate data for the x, y, and z coordinates
x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
cmap = colors.ListedColormap(['black'])

# Create a 3D figure and a contour plot side by side
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

# Add labels and title to both subplots
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('3D Plot of f(x, y) = sin(sqrt(x^2 + y^2))')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Level Set Contour Plot of f(x, y) = sin(sqrt(x^2 + y^2))')

# Plot the surface on the left subplot
ax1.plot_surface(X, Y, Z, cmap='jet')


i = 1 # intialize i
level_set_speed = .075 # how quickly the level sets expand
plane_speed = .05 # how quickly the plane moves up

Z_temp = Z.copy()

for a in np.arange(-1,1.05,plane_speed): # controls the movement of the plane
    Z_temp = Z.copy()
    i += level_set_speed

    #Plot the plane moving up the surface on the left
    ax1.cla()
    plane = np.zeros_like(X)
    plane = np.zeros_like(X) + a
    
    Z_temp[Z<a] = np.nan
    
    ax1.plot_surface(X, Y, Z_temp, cmap='jet')
    ax1.plot_wireframe(X, Y, plane, color='black')
    ax1.set_zlim(np.nanmin(Z), np.nanmax(Z))

    # Plot the contour on the right subplot
    contour_levels = np.arange(np.nanmin(Z), np.nanmin(Z)+i, i/2)
    ax2.contourf(X, Y, Z, levels=contour_levels, cmap=cmap, extend='min')\
    
    #snapshot
    plt.pause(.1)

ax1.plot_surface(X, Y, Z, cmap='jet')
    

# Show the plot
plt.show()
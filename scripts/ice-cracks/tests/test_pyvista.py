import pyvista as pv

# Start virtual framebuffer if necessary
try:
    pv.start_xvfb()
except:
    print("Virtual framebuffer could not be started, proceeding without it.")

# Create a simple sphere to test PyVista
sphere = pv.Sphere()

# Plot the sphere
plotter = pv.Plotter()
plotter.add_mesh(sphere, color="blue")
plotter.add_scalar_bar(title="Test Scalar Bar")
plotter.show()

from decimal import *
from dolfin import *
from mshr import *
from math import pi, sin, cos, sqrt, fabs
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.tri as tri
import matplotlib.mlab as mlab




mesh = UnitSquareMesh(15, 15)
gdim = mesh.geometry().dim()
V = FunctionSpace(mesh, 'CG', 1)
f = Expression('sin(x[0])*exp(-5*x[0])*cos(x[1])', degree=1)
function = project(f,V)
dofmap = V.dofmap()

dofs = dofmap.dofs()
# Get coordinates as len(dofs) x gdim array
dofs_x = V.tabulate_dof_coordinates().reshape((-1, gdim))

for dof, dof_x in zip(dofs, dofs_x):
    print( dof, ':', dof_x, '---', function.vector().get_local()[dof])


def min_location(u):

    V = u.function_space()

    if V.mesh().topology().dim() != 2:
       raise ValueError("Only minimum of scalar function in 2D can be computed.")

    dofs_x = V.tabulate_dof_coordinates().reshape((-1, gdim))

    function_array = u.vector().get_local()
    minimum = min(u.vector().get_local())

    min_index = np.where(function_array == minimum)
    min_loc = dofs_x[min_index]

    return min_loc

min_loc = min_location(function)

print( 'Function minimum:', min_loc)




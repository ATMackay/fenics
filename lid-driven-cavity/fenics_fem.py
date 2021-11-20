"""Base Code for the Finite Element solution of the Lid Driven Cavity Flow"""



from decimal import *
from dolfin import *
from mshr import *
from math import pi, sin, cos, sqrt, fabs, tanh
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.tri as tri
import matplotlib.mlab as mlab
import time, sys 
import scipy.interpolate as sci

## Constants
pi=3.14159265359

# Progress Bar
def update_progress(job_title, progress):
    length = 50 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 4))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()

# MATPLOTLIB CONTOUR FUNCTIONS
def mesh2triang(mesh):
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells()) # Mesh Diagram

def mplot(obj):                     # Function Plot
    plt.gca().set_aspect('equal')
    if isinstance(obj, Function):
        mesh = obj.function_space().mesh()
        if (mesh.geometry().dim() != 2):
            raise(AttributeError)
        if obj.vector().size() == mesh.num_cells():
            C = obj.vector().array()
            plt.tripcolor(mesh2triang(mesh), C)
        else:
            C = obj.compute_vertex_values(mesh)
            plt.tripcolor(mesh2triang(mesh), C, shading='gouraud')
    elif isinstance(obj, Mesh):
        if (obj.geometry().dim() != 2):
            raise(AttributeError)
        plt.triplot(mesh2triang(obj), color='k')

def expskewcavity(x,y,N):
    """exponential Skew Mapping"""
    xi = 0.5*(1+np.tanh(2*N*(x-0.5)))
    ups= 0.5*(1+np.tanh(2*N*(y-0.5)))
    return(xi,ups)

# Skew Mapping
def skewcavity(x,y):
    xi = 0.5*(1-np.cos(x*pi))**1
    ups =0.5*(1-np.cos(y*pi))**1
    return(xi,ups)

def xskewcavity(x,y):
    xi = 0.5*(1-np.cos(x*pi))**1
    ups = y
    return(xi,ups)


def DGP_Mesh(mm, B, L):
    # Define Geometry
    x0 = 0
    y0 = 0
    x1 = B
    y1 = L

    # Mesh refinement comparison Loop
    nx=mm*B
    ny=mm*L

    #c = min(x1-x0,y1-y0)
    base_mesh= RectangleMesh(Point(x0,y0), Point(x1, y1), nx, ny) # Rectangular Mesh


    # Create Unstructured mesh
    #u_rec=Rectangle(Point(0.0,0.0),Point(1.0,1.0))
    #mesh0=generate_mesh(u_rec, mm)

    mesh1 = base_mesh

    """
    # MESH CONSTRUCTION CODE
    nv= base_mesh.num_vertices()
    nc= base_mesh.num_cells()
    coorX = base_mesh.coordinates()[:,0]
    coorY = base_mesh.coordinates()[:,1]
    cells0 = base_mesh.cells()[:,0]
    cells1 = base_mesh.cells()[:,1]
    cells2 = base_mesh.cells()[:,2]



    # OLD MESH COORDINATES -> NEW MESH COORDINATES
    r=list()
    l=list()
    x = list()
    for i in range(nv):
        r.append(xskewcavity(coorX[i], coorY[i])[0])
        l.append(xskewcavity(coorX[i], coorY[i])[1])

    r=np.asarray(r)
    l=np.asarray(l)
    #x=np.asarray(x)
    # MESH GENERATION (Using Mesheditor)
    mesh1 = Mesh()
    editor = MeshEditor()
    editor.open(mesh1, "triangle", 2,2)
    editor.init_vertices(nv)
    editor.init_cells(nc)
    for i in range(nv):
        editor.add_vertex(i, r[i], l[i])
    for i in range(nc):
        editor.add_cell(i, cells0[i], cells1[i], cells2[i])


    editor.close()
    """
    return mesh1

def DGP_structured_mesh(mm, x_0, y_0, x_1, y_1, B, L):
    nx=mm*B
    ny=mm*L
    base_mesh= RectangleMesh(Point(x_0,y_0), Point(x_1, y_1), nx, ny)

    nv= base_mesh.num_vertices()
    nc= base_mesh.num_cells()
    coorX = base_mesh.coordinates()[:,0]
    coorY = base_mesh.coordinates()[:,1]
    cells0 = base_mesh.cells()[:,0]
    cells1 = base_mesh.cells()[:,1]
    cells2 = base_mesh.cells()[:,2]

    # OLD MESH COORDINATES -> NEW MESH COORDINATES
    r=list()
    l=list()
    for i in range(nv):
      r.append(xskewcavity(coorX[i], coorY[i])[0])
      l.append(xskewcavity(coorX[i], coorY[i])[1])

      r=np.asarray(r)
      l=np.asarray(l)

    # MESH GENERATION (Using Mesheditor)
    mesh1 = Mesh()
    editor = MeshEditor()
    editor.open(mesh1, "triangle", 2,2)
    editor.init_vertices(nv)
    editor.init_cells(nc)
    for i in range(nv):
        editor.add_vertex(i, r[i], l[i])
    for i in range(nc):
        editor.add_cell(i, cells0[i], cells1[i], cells2[i])
    editor.close()
    
    return mesh1

def refine_boundary(mesh, times):
    for i in range(times):
          g = (max(x_1,y_1)-max(x_0,y_0))*0.025/(i+1)
          cell_domains = CellFunction("bool", mesh)
          cell_domains.set_all(False)
          for cell in cells(mesh):
              x = cell.midpoint()
              if  (x[0] < x_0+g or x[1] < y_0+g) or (x[0] > x_1-g or x[1] > y_1-g): 
                  cell_domains[cell]=True

          mesh = refine(mesh, cell_domains, redistribute=True)
    return mesh

def refine_top(mesh, times):
    for i in range(times):
          g = (max(x_1,y_1)-max(x_0,y_0))*0.025/(i+1)
          cell_domains = CellFunction("bool", mesh)
          cell_domains.set_all(False)
          for cell in cells(mesh):
              x = cell.midpoint()
              if  x[1] > y_1-g:
                  cell_domains[cell]=True
          mesh_refine = refine(mesh, cell_domains, redistribute=True)
    return mesh_refine

def ramp_function(t):
    f = 1.0 + tanh(8*(t-0.5))
    return f


# Reshapes 3-valued vector elements as 2x2 matrices
def reshape_elements(D0_vec, D12_vec, Ds_vec, D1_vec, tau0_vec, tau12_vec, tau1_vec):
    D0 = as_matrix([[D0_vec[0], D0_vec[1]],
                    [D0_vec[1], D0_vec[2]]])        #DEVSS STABILISATION
    D12 = as_matrix([[D12_vec[0], D12_vec[1]],
                    [D12_vec[1], D12_vec[2]]])
    Ds = as_matrix([[Ds_vec[0], Ds_vec[1]],
                    [Ds_vec[1], Ds_vec[2]]])
    D1 = as_matrix([[D1_vec[0], D1_vec[1]],
                    [D1_vec[1], D1_vec[2]]]) 
    tau0 = as_matrix([[tau0_vec[0], tau0_vec[1]],
                    [tau0_vec[1], tau0_vec[2]]])        # Stress 
    tau12 = as_matrix([[tau12_vec[0], tau12_vec[1]],
                    [tau12_vec[1], tau12_vec[2]]]) 
    tau1 = as_matrix([[tau1_vec[0], tau1_vec[1]],
                    [tau1_vec[1], tau1_vec[2]]])
    return D0, D12, Ds, D1, tau0, tau12, tau1

def solution_functions(Q, W, V, Z):
    #Solution Functions
    rho0 = Function(Q)
    rho12 = Function(Q)
    rho1 = Function(Q)
    p0 = Function(Q)       # Pressure Field t=t^n
    p1 = Function(Q)       # Pressure Field t=t^n+1
    T0 = Function(Q)       # Temperature Field t=t^n
    T1 = Function(Q)       # Temperature Field t=t^n+1
    tau0_vec = Function(Z)     # Stress Field (Vector) t=t^n
    tau12_vec = Function(Z)    # Stress Field (Vector) t=t^n+1/2
    tau1_vec = Function(Z)     # Stress Field (Vector) t=t^n+1
    w0 = Function(W)
    w12 = Function(W)
    ws = Function(W)
    w1 = Function(W)
    (u0, D0_vec) = w0.split()
    (u12, D12_vec) = w12.split()
    (us, Ds_vec) = ws.split()
    (u1, D1_vec) = w1.split()
    return rho0, rho12, rho1, p0, p1, T0, T1, u0, u12, us, u1, D0_vec, D12_vec, Ds_vec, D1_vec, w0, w12, ws, w1, tau0_vec, tau12_vec, tau1_vec

def trial_functions(Q, Z, W):
    # Trial Functions
    rho=TrialFunction(Q)
    p = TrialFunction(Q)
    T = TrialFunction(Q)
    tau_vec = TrialFunction(Z)
    (u, D_vec) = TrialFunctions(W)
    D =  as_matrix([[D_vec[0], D_vec[1]],
                    [D_vec[1], D_vec[2]]])
    tau = as_matrix([[tau_vec[0], tau_vec[1]],
                    [tau_vec[1], tau_vec[2]]]) 
    return rho, p, T, tau_vec, u, D_vec, D, tau

def function_spaces(mesh, order):
    # Discretization  parameters

    V_s = VectorElement("CG", mesh.ufl_cell(), order)       # Velocity Elements
    V_d = VectorElement("DG", mesh.ufl_cell(), order-1)
    V_se = VectorElement("Bubble", mesh.ufl_cell(),  order+1)
    
    Z_c = VectorElement("CG", mesh.ufl_cell(),  order, 3)     # Stress Elements
    Z_s = VectorElement("DG", mesh.ufl_cell(),  order-1, 3)
    Z_se = VectorElement("Bubble", mesh.ufl_cell(),  order+1, 3)
    Z_d = VectorElement("DG", mesh.ufl_cell(),  order-2, 3)

    Q_s = FiniteElement("CG", mesh.ufl_cell(), order-1)   # Pressure/Density Elements
    Q_p = FiniteElement("Bubble", mesh.ufl_cell(), order+1, 3)

    # Function spaces
    W = FunctionSpace(mesh,V_s*Z_d)             # F.E. Spaces 
    V = FunctionSpace(mesh,V_s)
    Vd = FunctionSpace(mesh,V_d)
    Z = FunctionSpace(mesh,Z_s)
    Zd = FunctionSpace(mesh,Z_d)
    Zc = FunctionSpace(mesh,Z_c)
    Q = FunctionSpace(mesh,Q_s)
    Qt = FunctionSpace(mesh, "DG", order-2)
    Qr = FunctionSpace(mesh,Q_s)
    return W, V, Vd, Z, Zd, Zc, Q, Qt, Qr

def  tgrad (w):
    """ Returns  transpose  gradient """
    w_grad = grad(w)

    tran_w_grad = w_grad.T

    return  tran_w_grad

def Dincomp (w):
    """ Returns 2* the  rate of  strain  tensor """
    return (grad(w) + tgrad(w))/2
def Dcomp (w):
    """ Returns 2* the  rate of  strain  tensor """
    return ((grad(w) + tgrad(w))-(2.0/3)*div(w)*Identity(len(w)))/2

def sigmain(u, p, Tau, We, betav):
    return 2*betav*Dincomp(u) - p*Identity(len(u)) + ((1-betav)/We)*(Tau-Identity(len(u)))

def sigma(u, p, Tau, We, betav):
    return 2*betav*Dcomp(u) - p*Identity(len(u)) + ((1-betav)/We)*(Tau-Identity(len(u)))

def Fdef(u, Tau):
    return dot(u,grad(Tau)) - dot(grad(u),Tau) - dot(Tau,tgrad(u))

def Fdefcon(u, Tau):
    return dot(u,grad(Tau)) - dot(grad(u),Tau) - dot(Tau,tgrad(u)) + div(u)*Tau 

def normalize_solution(u):
    "Normalize u: return u divided by max(u)"
    u_array = u.vector().array()
    u_max = np.max(np.abs(u_array))
    u_array /= u_max
    u.vector()[:] = u_array
    #u.vector().set_local(u_array)  # alternative
    return u

# Adaptive Mesh Refinement 
def adaptive_refinement(mesh, kapp, ratio):
    kapp_array = kapp.vector().get_local()
    kapp_level = np.percentile(kapp_array, (1-ratio)*100)

    cell_domains = MeshFunction("bool", mesh, 2)
    cell_domains.set_all(False)
    for cell in cells(mesh):
        x = cell.midpoint()
        if  kapp([x[0], x[1]]) > kapp_level:
            cell_domains[cell]=True

    mesh = refine(mesh, cell_domains, redistribute=True)
    return mesh

def stream_function(u):
    '''Compute stream function of given 2-d velocity vector.'''
    V = u.function_space().sub(0).collapse()

    if V.mesh().topology().dim() != 2:
        raise ValueError("Only stream function in 2D can be computed.")

    psi = TrialFunction(V)
    phi = TestFunction(V)

    a = inner(grad(psi), grad(phi))*dx
    L = inner(u[1].dx(0) - u[0].dx(1), phi)*dx
    bc = DirichletBC(V, Constant(0.), DomainBoundary())

    A, b = assemble_system(a, L, bc)
    psi = Function(V)
    solve(A, psi.vector(), b)

    return psi

def comp_stream_function(rho, u):
    '''Compute stream function of given 2-d velocity vector.'''
    V = u.function_space().sub(0).collapse()

    if V.mesh().topology().dim() != 2:
        raise ValueError("Only stream function in 2D can be computed.")

    psi = TrialFunction(V)
    phi = TestFunction(V)

    a = inner(grad(psi), grad(phi))*dx
    L = inner(rho*u[1].dx(0) - rho*u[0].dx(1), phi)*dx
    bc = DirichletBC(V, Constant(0.), DomainBoundary())

    A, b = assemble_system(a, L, bc)
    psi = Function(V)
    solve(A, psi.vector(), b)

    return psi

def shear_rate(u):
    '''Compute shear rate of given 2-d velocity vector.'''
    V = u.function_space().sub(0).collapse()

    if V.mesh().topology().dim() != 2:
        raise ValueError("Only stream function in 2D can be computed.")

    psi = TrialFunction(V)
    phi = TestFunction(V)

    a = inner(grad(psi), grad(phi))*dx
    gamma = grad(u) + tgrad(u) 
    g = inner(0.5*gamma, gamma)
    L = inner(g, phi)*dx
    bc = DirichletBC(V, Constant(0.), DomainBoundary())

    A, b = assemble_system(a, L, bc)
    psi = Function(V)
    solve(A, psi.vector(), b)

    return psi

def min_location(u, mesh):
    V = u.function_space()

    if V.mesh().topology().dim() != 2:
       raise ValueError("Only minimum of scalar function in 2D can be computed.")

    gdim = mesh.geometry().dim()
    dofs_x = V.tabulate_dof_coordinates().reshape((-1, gdim))

    function_array = u.vector().get_local()
    minimum = min(u.vector().get_local())

    min_index = np.where(function_array == minimum)
    min_loc = dofs_x[min_index]

    return min_loc

def max_location(u, mesh):
    V = u.function_space()
    if V.mesh().topology().dim() != 2:
       raise ValueError("Only minimum of scalar function in 2D can be computed.")

    gdim = mesh.geometry().dim()
    dofs_x = V.tabulate_dof_coordinates().reshape((-1, gdim))

    function_array = u.vector().get_local()
    maximum = max(u.vector().get_local())

    max_index = np.where(function_array == maximum)
    max_loc = dofs_x[max_index]

    return max_loc


def l2norm_solution(u):
    u_array = u.vector().array()
    u_l2 = norm(u, 'L2')
    u_array /= u_l2
    u.vector()[:] = u_array
    return u
    

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def absolute(u):
    u_array = np.absolute(u.vector().get_local())
    u.vector()[:] = u_array
    return u
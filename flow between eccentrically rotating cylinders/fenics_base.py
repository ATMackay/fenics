"""
Flow Between Eccentrically Rotating Cylinders - Alex Mackay 2018
This Python module contains helper functions for finite element discretisation.
...

"""
from decimal import *
from dolfin import *
from mshr import * 
from math import pi, sin, cos, sqrt, fabs, tanh
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.tri as tri
import scipy.interpolate as sci
#import matplotlib.mlab as mlab



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
        plt.axis('off')
    elif isinstance(obj, Mesh):
        if (obj.geometry().dim() != 2):
            raise(AttributeError)
        plt.triplot(mesh2triang(obj), color='k', linewidth = 0.25)
        plt.axis('off')

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;


# ADAPTIVE MESH REFINEMENT (METHOD 2) "MESH"

# Mesh Refine Code

def refine_boundaries(mesh,times):
    for i in range(times):
          g=1.0/(3*(i+1))
          #print(g)
          cell_domains = MeshFunction("bool", mesh, 2)
          cell_domains.set_all(False)
          for cell in cells(mesh):
              x = cell.midpoint()
              if  (x[0]-x_1)**2+(x[1]-y_1)**2 < ((1-g)*r_a**2+g*r_b**2) or x[0] < -0.7*r_a: #or (x[0]-x_2)**2+(x[1]-y_2)**2 > (g*r_a**2+(1-g)*r_b**2): 
                  cell_domains[cell]=True
          #plot(cell_domains, interactive=True)
          mesh = refine(mesh, cell_domains, redistribute=True)
    return mesh

def refine_narrow(mesh,times):
    for i in range(times):
          g=1.0/(3*(i+1))
          #print(g)
          cell_domains = MeshFunction("bool", mesh, 2)
          cell_domains.set_all(False)
          for cell in cells(mesh):
              x = cell.midpoint()
              if  (x[0]-x_1) < -0.9*r_a: #or (x[0]-x_2)**2+(x[1]-y_2)**2 > (g*r_a**2+(1-g)*r_b**2): 
                  cell_domains[cell]=True
          #plot(cell_domains, interactive=True)
          mesh = refine(mesh, cell_domains, redistribute=True)
    return mesh

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

# Some Useful Functions
def  tgrad (w):
    """ Returns  transpose  gradient """
    w_grad = grad(w)

    tran_w_grad = w_grad.T
    #tran_w_grad = w_grad.transpose()

    return  tran_w_grad


def Dincomp (w):
    """ Returns 2* the  rate of  strain  tensor """
    return (grad(w) + tgrad(w))/2
def Dcomp (w):
    """ Returns 2* the  rate of  strain  tensor """
    return ((grad(w) + tgrad(w))-(2.0/3)*div(w)*Identity(len(w)))/2.
def DinG (D):
    """ Returns 2* the  rate of  strain  tensor """
    return (D + D.T)/2

def sigma(u, p, Tau, betav, We):
    return 2*betav*Dcomp(u) - p*Identity(len(u)) + Tau

def sigmacon(u, p, Tau, betav, We):
    return 2*betav*Dcomp(u) - p*Identity(len(u)) + ((1.0-betav)/We)*(Tau-Identity(len(u)))

def fene_sigma(u, p, Tau, b, lambda_d, betav, We):
    return 2.0*betav*Dincomp(u) - p*Identity(len(u)) + ((1.-betav)/We)*( phi_def(u, lambda_d)*( fene_func(Tau, b)*Tau-Identity(len(u)) ) )

def fene_sigmacom(u, p, Tau, b,lambda_d, betav, We):
    return 2.0*betav*Dcomp(u) - p*Identity(len(u)) + ((1.-betav)/We)*( phi_def(u, lambda_d)*( fene_func(Tau, b)*Tau-Identity(len(u)) ) )

def Fdef(u, Tau):
    return dot(u,grad(Tau)) - dot(grad(u),Tau) - dot(Tau,tgrad(u)) + div(u)*Tau 

def FdefG(u, G, Tau): # DEVSS-G
    return dot(u,grad(Tau)) - dot(G,Tau) - dot(Tau,G.T) 

def normalize_solution(u):
    "Normalize u: return u divided by max(u)"
    u_array = u.vector().get_local()
    u_max = np.max(np.abs(u_array))
    u_array /= u_max
    u.vector()[:] = u_array
    #u.vector().set_local(u_array)  # alternative
    return u

def magnitude(u):
    return np.power((u[0]*u[0]+u[1]*u[1]), 0.5)

def euc_norm(tau):
    return np.power((tau[0]*tau[0] + tau[1]*tau[1] + tau[2]*tau[2]), 0.5)

def absolute(u):
    u_array = np.absolute(u.vector().get_local())
    u.vector()[:] = u_array
    return u

def phi(u, p, T, A, B, K_0, N, betap):
    
    K = 1. + A*p - B*(T/(1.+T))
    scalar_strain = np.power(2*inner(Dincomp(u), Dincomp(u)),0.5)
    
    Kthin = (K_0*K*scalar_strain)**N 
    ftheta = project(1./(1.+Kthin), Q)

    variable = (betap + (1.-betap)*ftheta)*K

    return variable

def phi_s(T, A_0):
    
    T_0 = 300.0
    T_h = 350.0
    k_b = 8.3144598 # Boltzmann Gas Constant
    
    A = A_0/(k_b*(T_h-T_0))
    C = T_h/(T_h-T_0)

    variable = 1. - A*(1./(C+T))

    return variable

def phi_ewm(tau,T, k, B):
    return (1.-B*(T/(1.+T)))*((0.5*Tr(tau)+0.00001)**k)

def phi_ewm_inv(tau,T, k, B):
    return 1.0/((1.-B*(T/(1.+T)))*((0.5*Tr(tau)+0.00001)**k))

def Tr(tau):
    return abs(tau[0,0] + tau[1,1])

def Trace(tau):
    return tau[0,0] + tau[1,1]




def fene_func(tau, b):
    f_tau = 1.0/(1.-(Tr(tau)-2.)/(b*b))
    return f_tau

def orthog_proj(u, V, V_d):
    
    u_d = project(u, V_d)
    u_orth_proj = project(u - u_d, V)
    
    return u_orth_proj

def frob_norm(tau):

    mult = tau*(tau.T)
    frob = Trace(mult)

    return np.power(frob, 0.5)

def ernesto_k(u, h, c_a, c_b):
 
    k = (c_a*h*magnitude(u) + c_b*h*h*frob_norm(grad(u)))

    return k

# Invariants of Tensor 
def I_3(A):
    return A[0,0]*A[1,1]-A[1,0]*A[0,1]


def I_2(A):
    return 0.5*(A[0,0]*A[0,0] + A[1,1]*A[1,1] + 2*A[1,0]*A[0,1]) 

def phi_def(u, lambda_d):
    D_u = as_matrix([[Dincomp(u)[0,0], Dincomp(u)[0,1]],
                     [Dincomp(u)[1,0], Dincomp(u)[1,1]]])
    phi = 1. + (lambda_d*3*I_3(D_u)/(I_2(D_u)+0.0000001))**2  # Dolfin epsillon used to avoid division by zero
    return phi

def psi_def(phi):
    return 0.5*(phi - 1.)
    


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
    u_array = u.vector().get_local()
    u_l2 = norm(u, 'L2')
    u_array /= u_l2
    u.vector()[:] = u_array
    return u
    

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]



def ramp_function(t):
    f = 1.0 + tanh(8*(t-0.5))
    return f


# Rehapes 3-valued vector elements as 2x2 matrices
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


    #Z_e = Z_c + Z_se
    #Z_e = EnrichedElement(Z_c,Z_se)                 # Enriched Elements
    #Z_e = MixedElement(Z_c,Z_se)
    #V_e = EnrichedElement(V_s,V_se) 
    #Q_rich = EnrichedElement(Q_s,Q_p)

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
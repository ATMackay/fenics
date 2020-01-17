
from decimal import *
from dolfin import *
#from fenics import *
#from ufl import *
from mshr import *
from math import pi, sin, cos, sqrt, fabs
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.tri as tri
import matplotlib.mlab as mlab

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



def JBP_mesh(mm):
    c0 = Circle(Point(0.0,0.0), r_a, 256) 
    c1 = Circle(Point(x_1,y_1), r_a, 256)
    c2 = Circle(Point(x_2,y_2), r_b, 256)

    ex = x_2 - x_1
    ey = y_2 - y_1
    ec = np.sqrt(ex**2+ey**2)
    c = r_b - r_a
    ecc = (ec)/(c)

    if c <= 0.0:
       print("ERROR! Journal radius greater than bearing radius")
       quit()

    # Create mesh
    cyl0 = c2 - c0
    cyl = c2 - c1

    mesh = generate_mesh(cyl, mm)

    return mesh


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

    #tran_w_grad = as_matrix(w_grad[i,j], (j,i))
    tran_w_grad = w_grad.T
    #tran_w_grad = w_grad.transpose()

    return  tran_w_grad


def Dincomp (w):
    """ Returns 2* the  rate of  strain  tensor """
    return (grad(w) + tgrad(w))/2
def Dcomp (w):
    """ Returns 2* the  rate of  strain  tensor """
    return ((grad(w) + tgrad(w))-(2.0/3)*div(w)*Identity(len(u)))/2.
def DinG (D):
    """ Returns 2* the  rate of  strain  tensor """
    return (D + D.T)/2

def sigma(u, p, Tau):
    return 2*betav*Dcomp(u) - p*Identity(len(u)) + Tau

def sigmacon(u, p, Tau):
    return 2*betav*Dcomp(u) - p*Identity(len(u)) + ((1.0-betav)/We)*(Tau-Identity(len(u)))

def fene_sigma(u, p, Tau, b, lambda_d):
    return 2.0*betav*Dincomp(u) - p*Identity(len(u)) + ((1.-betav)/We)*( phi_def(u, lambda_d)*( fene_func(Tau, b)*Tau-Identity(len(u)) ) )

def fene_sigmacom(u, p, Tau, b,lambda_d):
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

def phi(u, p, T, A, B, K_0, N):
    
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


def max_location(u):

    V = u.function_space()

    if V.mesh().topology().dim() != 2:
       raise ValueError("Only minimum of scalar function in 2D can be computed.")

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




# HOLOLOW CYLINDER MESH

# Parameters
r_a = 1.0 #Journal Radius
r_b = 2.0#1.25 #Bearing Radius
x_1 = -0.80 #-0.2
y_1 = 0.0
x_2 = 0.0
y_2 = 0.0
ex = x_2-x_1
ey = y_2-y_1
ec = np.sqrt(ex**2+ey**2)
c = r_b-r_a
ecc = (ec)/(c)

c3 = Circle(Point(x_1,y_1), 0.99*r_a, 256)  # Empty hole in mesh

mm = 35
mesh = JBP_mesh(mm)

#mesh =refine_narrow(mesh,1)




gdim = mesh.geometry().dim() # Mesh Geometry


meshc= generate_mesh(c3, 15)

mplot(mesh)
plt.savefig("JBP_mesh_"+str(mm)+".png")
plt.clf() 
plt.close()

#quit()


#Jounral Boundary                                                                              
class Omega0(SubDomain):
      def inside(self, x, on_boundary):
          return True if (x[0]-x_1)**2+(x[1]-y_1)**2 < (0.9*r_a**2+0.1*r_b**2) and on_boundary  else False  # and 
omega0= Omega0()

# Bearing Boundary
class Omega1(SubDomain):
      def inside(self, x, on_boundary):
          return True if (x[0]-x_2)**2 + (x[1]-y_2)**2 > (0.1*r_a**2+0.9*r_b**2) and on_boundary else False  #
omega1= Omega1()

# Subdomian for the pressure boundary condition at (r_a,0)
class POmega(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[0] < 0.5*(r_a+r_b) and x[0] > 0 and x[1] < r_a*0.02 and x[1] > -r_a*0.05 and on_boundary else False 
POmega=POmega()


# Create mesh functions over the cell facets (Verify Boundary Classes)
#print(mesh.topology().dim())
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
sub_domains.set_all(0)
omega0.mark(sub_domains, 2)
omega1.mark(sub_domains, 3)
#POmega.mark(sub_domains, 4)

#file = File("subdomains.pvd")
#file << sub_domains
#quit()

#plot(sub_domains, interactive=False, scalarbar = False)
#quit()

#Define Boundary Parts
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
#boundary_parts = FacetFunction("size_t", mesh)
omega0.mark(boundary_parts,0)
omega1.mark(boundary_parts,1)
ds = Measure("ds")[boundary_parts]


# Discretization  parameters
family = "CG"; dfamily = "DG"; rich = "Bubble"
shape = "triangle"; order = 2

#mesh.ufl_cell()


# Define spaces

V_s = VectorElement(family, mesh.ufl_cell(), order)       # Elements
Z_c = VectorElement(family, mesh.ufl_cell(),  order, 3)
Z_s = VectorElement(dfamily, mesh.ufl_cell(),  order-1, 3)
Z_d = VectorElement(dfamily, mesh.ufl_cell(),  order-2, 3)
Z_se = VectorElement(rich, mesh.ufl_cell(),  order+1, 3)
Q_s = FiniteElement(family, mesh.ufl_cell(), order-1)
Q_p = FiniteElement(rich, mesh.ufl_cell(), order+1, 3)
#Z_e = EnrichedElement(Z_c,Z_se)                 # Enriched Elements
#Q_rich = EnrichedElement(Q_s,Q_p)


W = FunctionSpace(mesh,V_s*Z_d)             # F.E. Spaces 
V = FunctionSpace(mesh,V_s)

Z = FunctionSpace(mesh,Z_s)
Zd = FunctionSpace(mesh,Z_d)
#Ze = FunctionSpace(mesh,Z_e)
Zc = FunctionSpace(mesh,Z_c)
Q = FunctionSpace(mesh,Q_s)
Qt = FunctionSpace(mesh,Q_s)
#Qt = FunctionSpace(mesh, "DG", order-2)
Qr = FunctionSpace(mesh,Q_s)


# Define trial and test functions 
rho=TrialFunction(Q)
p = TrialFunction(Q)
T = TrialFunction(Q)
q = TestFunction(Q)
r = TestFunction(Q)
uu0 = Function(V)

p0=Function(Q)       # Pressure Field t=t^n
p1=Function(Q)       # Pressure Field t=t^n+1
rho0=Function(Q)
rho1=Function(Q)
T0=Function(Q)       # Temperature Field t=t^n
T1=Function(Q)       # Temperature Field t=t^n+1


(v, R_vec) = TestFunctions(W)
(u, D_vec) = TrialFunctions(W)

tau_vec = TrialFunction(Zc)
Rt_vec = TestFunction(Zc)


tau0_vec=Function(Zc)     # Stress Field (Vector) t=t^n
tau12_vec=Function(Zc)    # Stress Field (Vector) t=t^n+1/2
tau1_vec=Function(Zc)     # Stress Field (Vector) t=t^n+1

w0= Function(W)
w12= Function(W)
ws= Function(W)
w1= Function(W)

(u0, D0_vec) = w0.split()
(u12, D12_vec) = w12.split()
(u1, D1_vec) = w1.split()
(us, Ds_vec) = ws.split()


I = Expression((('1.0','0.0'), ('0.0','1.0')), element = Zc.ufl_element())




# Project Vector Trial Functions of Stress onto SYMMETRIC Tensor Space

D =  as_matrix([[D_vec[0], D_vec[1]],
                [D_vec[1], D_vec[2]]])

tau = as_matrix([[tau_vec[0], tau_vec[1]],
                 [tau_vec[1], tau_vec[2]]])  

# Project Vector Test Functions of Stress onto SYMMETRIC Tensor Space

Rt = as_matrix([[Rt_vec[0], Rt_vec[1]],
                 [Rt_vec[1], Rt_vec[2]]])        # DEVSS Space

R = as_matrix([[R_vec[0], R_vec[1]],
                 [R_vec[1], R_vec[2]]])

# Project Vector Functions of Stress onto SYMMETRIC Tensor Space

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

# Default nondim parameters

T_0 = 300
T_h = 350
conv=1                                      # Non-inertial Flow Parameter (Re=0)
We = 0.25
betav = 0.5                 # Viscosity Fraction
betap = 0.1                 # Shear Thinning Parameter
Re = 25
Ma = 0.0005
c0 = 1.0/Ma



# Define boundary conditions
w = Expression(('(0.5*(1.0+tanh(8*(t-0.5))))*(x[1]-y1)/r_a' , '-(0.5*(1.0+tanh(8*(t-0.5))))*(x[0]-x1)/r_a' ), degree=2, r_a=r_a, x1=x_1, y1=y_1 , t=0.0)


# Define unit Normal/tangent Vector at inner and outer Boundary (Method 2)
n0 =  Expression(('(x[0]-x1)/r_a' , '(x[1]-y1)/r_a' ), degree=2, r_a=r_a, x1=x_1, y1=y_1)
n1 =  Expression(('(x[0]-x2)/r_b' , '(x[1]-y2)/r_b' ), degree=2, r_b=r_b, x2=x_2, y2=y_2)
t0 =  Expression(('(x[1]-y1)/r_a' , '-(x[0]-x1)/r_a' ), degree=2, r_a=r_a, x1=x_1, y1=y_1)
t1 =  Expression(('(x[1]-y2)/r_b' , '-(x[0]-x2)/r_b' ), degree=2, r_b=r_b, x2=x_2, y2=y_2)

n = FacetNormal(mesh)
tang = as_vector([n[1], -n[0]])
h = CellDiameter(mesh)

 # Dirichlet Boundary Conditions  (LID DRIVEN CAVITY)

spin =  DirichletBC(W.sub(0), w, omega0)  #The inner cylinder will be rotated with constant angular velocity w_a
noslip  = DirichletBC(W.sub(0), (0.0, 0.0), omega1) #The outer cylinder remains fixed with zero velocity 
temp0 =  DirichletBC(Q, T_h, omega0)    #Temperature on Omega0 

#Collect Boundary Conditions
bcu = [noslip, spin]
bcp = []
bcT = [temp0]
bctau = []

Np= len(p0.vector().get_local())
Nv= len(w0.vector().get_local())  
Nvel = len(uu0.vector().get_local()) 
Ntau= len(tau0_vec.vector().get_local())
dof= 3*Nv+2*Ntau+Np
print('############# Discrete Space Characteristics ############')
print('Degree of Elements', order)
print('Size of Pressure Space = %d ' % Np)
print('Size of Velocity Space = %d ' % Nvel)    
print('Size of Velocity/DEVSS Space = %d ' % Nv)
print('Size of Stress Space = %d ' % Ntau)
print('Degrees of Freedom = %d ' % dof)
print('Number of Cells:', mesh.num_cells())
print('Number of Vertices:', mesh.num_vertices())
print('Minimum Cell Diamter:', mesh.hmin())
print('Maximum Cell Diamter:', mesh.hmax())


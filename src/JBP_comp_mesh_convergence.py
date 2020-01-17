
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

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

# HOLOLOW CYLINDER MESH

# Parameters
r_a = 1.0 #Journal Radius
r_b = 1.25 #Bearing Radius
x_1 = -0.20
y_1 = 0.0
x_2 = 0.0
y_2 = 0.0
ex = x_2-x_1
ey = y_2-y_1
ec = np.sqrt(ex**2+ey**2)
c = r_b-r_a
ecc = (ec)/(c)

c3 = Circle(Point(x_1,y_1), 0.99*r_a, 256)  # Empty hole in mesh

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


#quit()

# ADAPTIVE MESH REFINEMENT (METHOD 2) "MESH"

# Mesh Refine Code

def refine_boundaries(mesh,times):
    for i in range(times):
          g=1.0/(3*(i+1))
          #print(g)
          cell_domains = CellFunction("bool", mesh)
          cell_domains.set_all(False)
          for cell in cells(mesh):
              x = cell.midpoint()
              if  (x[0]-x_1)**2+(x[1]-y_1)**2 < ((1-g)*r_a**2+g*r_b**2) or x[0] < -0.7*r_a or (x[0]-x_2)**2+(x[1]-y_2)**2 > (g*r_a**2+(1-g)*r_b**2): 
                  cell_domains[cell]=True
          #plot(cell_domains, interactive=True)
          mesh = refine(mesh, cell_domains, redistribute=True)
    return mesh



# Some Useful Functions
def  tgrad (w):
    """ Returns  transpose  gradient """
    return  transpose(grad(w))
def Dincomp (w):
    """ Returns 2* the  rate of  strain  tensor """
    return (grad(w) + tgrad(w))/2
def Dcomp (w):
    """ Returns 2* the  rate of  strain  tensor """
    return ((grad(w) + tgrad(w))-(2.0/3)*div(w)*Identity(len(u)))/2.
def DinG (D):
    """ Returns 2* the  rate of  strain  tensor """
    return (D + transpose(D))/2

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
    return dot(u,grad(Tau)) - dot(G,Tau) - dot(Tau,transpose(G)) 

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

    mult = tau*transpose(tau)
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







dt = 0.001  #Time Stepping  
T_f = 4.0
Tf = T_f
loopend = 3
j = 0
jj = 0
jjj = 0
tol = 10E-6
defpar = 1.0


U = 1.0
w_j = 1.0
Di = 0.005                         # Diffusion Number
Vh = 0.005
Bi = 0.2
rho_0 = 1.0

conv = 1                                      # Non-inertial Flow Parameter (Re=0)
We = 0.5
Re = 50
Ma = 0.005
betav = 0.5
betap = 0.1

c0 = 1.0/Ma


al = 0.1                # Nonisothermal Parameter between 0 and 1

A_0 = 1200 # Solvent viscosity thinning
k_ewm = -0.7 # Shear thinning (EWM)
B = 0.1 # Polymeric viscosity thinning (EWM)
K_0 = 0.01

T_0 = 300
T_h = 350

alph1 = 1.0
c1 = 0.1
c2 = 0.01
c3 = 0.1
th = 1.0               # DEVSS
           # DEVSS


# FEM Solution Convergence/Energy Plot
x1=list()
x2=list()
x3=list()
x4=list()
x5=list()
y=list()
z=list()
zz=list()
zzz=list()
zl=list()
ek1=list()
ek2=list()
ek3=list()
ek4=list()
ee1=list()
ee2=list()
ee3=list()
ee4=list()
ek5=list()
ee5=list()
y1 = list()
zx1 = list()
z1 = list()
y2 = list()
zx2 = list()
z2 = list()
y3 = list()
zx3 = list()
z3 = list()
y4 = list()
zx4 = list()
z4 = list()
y5 = list()
zx5 = list()
z5 = list()
while j < loopend:
    j+=1

    t = 0.0
    if j==1:
       mm = 45
    elif j==2:
       mm = 35
    elif j==3:
       mm = 38

    mesh = JBP_mesh(mm)

 


    # ADAPTIVE MESH REFINEMENT (METHOD 2) "MESH"

    # Mesh Refine Code


    if j==1 or j==2:
       mesh = refine_boundaries(mesh, 1)
    if j==3:
       mesh = refine_boundaries(mesh, 1)        

    dt = 2*min(mesh.hmin()**2, 0.001)

    gdim = mesh.geometry().dim() 

    mplot(mesh)
    plt.savefig("JBP_mesh_"+str(mm)+".png")
    plt.clf()
    plt.close()

    #quit()


    #Jounral Boundary                                                                              
    class Omega0(SubDomain):
          def inside(self, x, on_boundary):
              return True if (x[0]-x_1)**2+(x[1]-y_1)**2 < (0.96*r_a**2+0.04*r_b**2) and on_boundary  else False  # and 
    omega0= Omega0()

    # Bearing Boundary
    class Omega1(SubDomain):
          def inside(self, x, on_boundary):
              return True if (x[0]-x_2)**2 + (x[1]-y_2)**2 > (0.04*r_a**2+0.96*r_b**2) and on_boundary else False  #
    omega1= Omega1()

    # Subdomian for the pressure boundary condition at (r_a,0)
    class POmega(SubDomain):
          def inside(self, x, on_boundary):
              return True if x[0] < 0.5*(r_a+r_b) and x[0] > 0 and x[1] < r_a*0.02 and x[1] > -r_a*0.05 and on_boundary else False 
    POmega=POmega()


    # Create mesh functions over the cell facets (Verify Boundary Classes)
    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    sub_domains.set_all(0)
    omega0.mark(sub_domains, 2)
    omega1.mark(sub_domains, 3)
    #POmega.mark(sub_domains, 4)

    #plot(sub_domains, interactive=False, scalarbar = False)
    #quit()

    #Define Boundary Parts
    boundary_parts = FacetFunction("size_t", mesh)
    omega0.mark(boundary_parts,0)
    omega1.mark(boundary_parts,1)
    ds = Measure("ds")[boundary_parts]


    # Discretization  parameters
    family = "CG"; dfamily = "DG"; rich = "Bubble"
    shape = "triangle"; order = 2

    #mesh.ufl_cell()



    V_s = VectorElement(family, mesh.ufl_cell(), order)       # Elements
    Z_c = VectorElement(family, mesh.ufl_cell(),  order, 3)
    Z_s = VectorElement(dfamily, mesh.ufl_cell(),  order-1, 3)
    Z_d = VectorElement(dfamily, mesh.ufl_cell(),  order-2, 3)
    Z_se = VectorElement(rich, mesh.ufl_cell(),  order+1, 3)
    Q_s = FiniteElement(family, mesh.ufl_cell(), order-1)
    Q_p = FiniteElement(rich, mesh.ufl_cell(), order+1, 3)
    Z_e = EnrichedElement(Z_c,Z_se)                 # Enriched Elements
    Q_rich = EnrichedElement(Q_s,Q_p)


    W = FunctionSpace(mesh,V_s*Z_d)             # F.E. Spaces 
    V = FunctionSpace(mesh,V_s)

    Z = FunctionSpace(mesh,Z_s)
    Zd = FunctionSpace(mesh,Z_d)
    #Ze = FunctionSpace(mesh,Z_e)
    Zc = FunctionSpace(mesh,Z_c)
    Q = FunctionSpace(mesh,Q_s)
    Qt = FunctionSpace(mesh, "DG", order-2)
    Qr = FunctionSpace(mesh,Q_s)


    # Define trial and test functions [TAYLOR GALERKIN Method]
    rho=TrialFunction(Q)
    p = TrialFunction(Q)
    T = TrialFunction(Q)
    q = TestFunction(Q)
    r = TestFunction(Q)

    p0=Function(Q)       # Pressure Field t=t^n
    p1=Function(Q)       # Pressure Field t=t^n+1
    rho0=Function(Q)
    rho1=Function(Q)
    T0=Function(Q)       # Temperature Field t=t^n
    T1=Function(Q)       # Temperature Field t=t^n+1


    theta1 = T1-T_0/(T_h-T_0)


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


    I = Expression((('1.0','0.0'),
                    ('0.0','1.0')), degree=2)


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



    # Default Nondimensional Parameters





    # Steady State Method (Re-->10Re)
    Ret = Expression('Re*(1.0+0.5*(1.0+tanh(0.5*t-3.5))*9.0)', t=0.0, Re=Re, degree=2)
    Wet = Expression('We*0.5*(1.0+tanh(0.5*t-3.5))', t=0.0, We=We, degree=2)

    Rey=Re


    # Define boundary conditions
    td= Constant('5')
    e = Constant('6')
    w = Expression(('(0.5*(1.0+tanh(8*(t-0.5))))*(x[1]-y1)/r_a' , '-(0.5*(1.0+tanh(8*(t-0.5))))*(x[0]-x1)/r_a' ), degree=2, r_a=r_a, x1=x_1, y1=y_1, e=e , t=0.0)


    # Define unit Normal/tangent Vector at inner and outer Boundary (Method 2)
    n0 =  Expression(('(x[0]-x1)/r_a' , '(x[1]-y1)/r_a' ), degree=2, r_a=r_a, x1=x_1, y1=y_1)
    n1 =  Expression(('(x[0]-x2)/r_b' , '(x[1]-y2)/r_b' ), degree=2, r_b=r_b, x2=x_2, y2=y_2)
    t0 =  Expression(('(x[1]-y1)/r_a' , '-(x[0]-x1)/r_a' ), degree=2, r_a=r_a, x1=x_1, y1=y_1)
    t1 =  Expression(('(x[1]-y2)/r_b' , '-(x[0]-x2)/r_b' ), degree=2, r_b=r_b, x2=x_2, y2=y_2)

    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    tang = as_vector([n[1], -n[0]])

     # Dirichlet Boundary Conditions  (LID DRIVEN CAVITY)

    spin =  DirichletBC(W.sub(0), w, omega0)  #The inner cylinder will be rotated with constant angular velocity w_a
    noslip  = DirichletBC(W.sub(0), (0.0, 0.0), omega1) #The outer cylinder remains fixed with zero velocity 
    temp0 =  DirichletBC(Q, T_h, omega0)    #Temperature on Omega0 

    #Collect Boundary Conditions
    bcu = [noslip, spin]
    bcp = []
    bcT = [temp0]
    bctau = []



    # Print Parameters of flow simulation
    t = 0.0                  #Time
    e=6
    print '############# Journal Bearing Length Ratios ############'
    print'Eccentricity (m):' ,ec
    print'Radius DIfference (m):',c
    print'Eccentricity Ratio:',ecc

    print '############# TIME SCALE ############'
    print 'Timestep size (s):', dt
    print 'Finish Time (s):', Tf

    print '############# Scalings & Nondimensional Parameters ############'
    print 'Characteristic Length (m):', r_b-r_a
    print 'Characteristic Velocity (m/s):', w_j*r_a
    print 'Speed of sound (m/s):', c0
    print 'Cylinder Speed (t=0) (m/s):', w_j*r_a*(1.0+tanh(e*t-3.0))
    print 'Mach Number', Ma
    print 'Nondimensionalised Cylinder Speed (t=0) (m/s):', (1.0+tanh(e*t-3.0))
    print 'Reynolds Number:', Re
    print 'Weissenberg Number:', We
    print 'Viscosity Ratio:', betav
    print 'Diffusion Number:' ,Di
    print 'Viscous Heating Number:', Vh

    Np= len(p0.vector().get_local())
    Nv= len(w0.vector().get_local())   
    Ntau= len(tau0_vec.vector().get_local())
    dof= 3*Nv+2*Ntau+Np
    print '############# Discrete Space Characteristics ############'
    print 'Degree of Elements', order
    print('Size of Pressure Space = %d ' % Np)
    print('Size of Velocity/DEVSS Space = %d ' % Nv)
    print('Size of Stress Space = %d ' % Ntau)
    print('Degrees of Freedom = %d ' % dof)
    print 'Number of Cells:', mesh.num_cells()
    print 'Number of Vertices:', mesh.num_vertices()
    print 'Minimum Cell Diamter:', mesh.hmin()
    print 'Maximum Cell Diamter:', mesh.hmax()
    print '############# Stabilisation Parameters ############'
    print 'DEVSS Momentum Term:', th

    quit()

    # Initial Density Field
    rho_initial = Expression('1.0', degree=1)
    rho_initial_guess = project(1.0, Q)
    rho0.assign(rho_initial_guess)


    # Initial Conformation Tensor
    I_vec = Expression(('1.0','0.0','1.0'), degree=2)
    initial_guess_conform = project(I_vec, Zc)
    assign(tau0_vec, initial_guess_conform)         # Initial guess for conformation tensor is Identity matrix

    tau0 = as_matrix([[tau0_vec[0], tau0_vec[1]],
                      [tau0_vec[1], tau0_vec[2]]])        # Stress 

    # Initial Temperature Field
    T_initial_guess = project(T_0, Q)
    T0.assign(T_initial_guess)



    #Define Variable Parameters, Strain Rate and other tensors
    gamdots = inner(Dincomp(u1),grad(u1))
    gamdots12 = inner(Dincomp(u12),grad(u12))
    gamdotp = inner(tau1,grad(u1))
    gamdotp12 = inner(tau12,grad(u12))
    thetal = (T)/(T_h-T_0)
    thetar = (T_0)/(T_h-T_0)
    thetar = project(thetar,Q)
    theta0 = (T0-T_0)/(T_h-T_0)

    # Stabilisation

    # Ernesto Castillo 2016 p.
    """F1R = dot(u1,grad(tau1)) - dot(grad(u1),tau1) - dot(tau1,tgrad(u1)) + div(u1)*tau1  #Compute the residual in the STRESS EQUATION
    F1R_vec = as_vector([F1R[0,0], F1R[1,0], F1R[1,1]])
    Dcomp1_vec = as_vector([Dcomp(u1)[0,0], Dcomp(u1)[1,0], Dcomp(u1)[1,1]])
    restau = We*F1R_vec - 2*(1-betav)*Dcomp1_vec
    res_test = project(restau0, Zd)
    res_orth = project(restau0-res_test, Zc) 
    Fv = dot(u1,grad(Rt)) - dot(grad(u1),Rt) - dot(Rt,tgrad(u1)) + div(u1)*Rt
    Fv_vec = as_vector([Fv[0,0], Fv[1,0], Fv[1,1]])
    Dv_vec =  as_vector([Dcomp(v)[0,0], Dcomp(v)[1,0], Dcomp(v)[1,1]])                              
    osgs_stress = inner(res_orth, We*Fv_vec - 2*(1-betav)*Dv_vec)*dx"""

    # LPS Projection
    """F1R = dot(u1,grad(tau1)) - dot(grad(u1),tau1) - dot(tau1,tgrad(u1)) + div(u1)*tau1  #Compute the residual in the STRESS EQUATION
    F1R_vec = as_vector([F1R[0,0], F1R[1,0], F1R[1,1]])
    Dcomp1_vec = as_vector([Dcomp(u1)[0,0], Dcomp(u1)[1,0], Dcomp(u1)[1,1]])
    restau0 = We/dt*(tau1_vec-tau0_vec) + We*F1R_vec + tau1_vec - 2*(1-betav)*Dcomp1_vec 
    res_test = project(restau0, Zd)
    res_orth = project(restau0-res_test, Zc)                                
    res_orth_norm_sq = project(inner(res_orth,res_orth), Qt)     # Project residual norm onto discontinuous space
    res_orth_norm = np.power(res_orth_norm_sq, 0.5)
    tau_stab = as_matrix([[res_orth[0]*tau_vec[0], res_orth[1]*tau_vec[1]],
                          [res_orth[1]*tau_vec[1], res_orth[2]*tau_vec[2]]])
    tau_stab1 = as_matrix([[res_orth[0]*tau1_vec[0], res_orth[1]*tau1_vec[1]],
                          [res_orth[1]*tau1_vec[1], res_orth[2]*tau1_vec[2]]])
    Rt_stab = as_matrix([[res_orth[0]*Rt_vec[0], res_orth[1]*Rt_vec[1]],
                          [res_orth[1]*Rt_vec[1], res_orth[2]*Rt_vec[2]]]) 
    kapp = project(res_orth_norm, Qt)
    LPSl_stress = inner(kapp*h*c1*grad(tau),grad(Rt))*dx + inner(kapp*h*c2*div(tau),div(Rt))*dx  # Stress Stabilisation"""

    # DEVSS Stabilisation

    
    DEVSSl_u12 = 2*(1-betav)*inner(Dcomp(u),Dincomp(v))*dx    
    DEVSSr_u12 = 2*inner(D0,Dincomp(v))*dx   
    DEVSSl_u1 = 2*(1-betav)*inner(Dcomp(u),Dincomp(v))*dx    
    DEVSSr_u1 = 2*inner(D12,Dincomp(v))*dx 

    DEVSSl_T1 = (1.-Di)*inner(grad(thetal), grad(r))*dx
    DEVSSr_T1 = inner((1.-Di)*(grad(thetar) + grad(theta0)), grad(r))*dx

    # DEVSS-G Stabilisation
    
    DEVSSGl_u12 = 2.0*(1.-betav)*inner(Dincomp(u),Dincomp(v))*dx    
    DEVSSGr_u12 = (1-betav)*inner(D0 + transpose(D0),Dincomp(v))*dx   
    DEVSSGl_u1 = 2.0*(1.-betav)*inner(Dincomp(u),Dincomp(v))*dx    
    DEVSSGr_u1 = (1.-betav)*inner(D12 + transpose(D12),Dincomp(v))*dx


    #Folder To Save Plots for Paraview
    #fv=File("Velocity Results Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"theta"+str(theta)+"c0="+str(c0)+"/velocity "+str(t)+".pvd")
 
    #Lists for Energy Values
    x = list()
    y = list()
    ee = list()
    ek = list()
    z = list()
    zx = list()
    xx = list()
    yy = list()
    zz = list()
    xxx = list()
    yyy = list()

    conerr=list()
    deferr=list()
    tauerr=list()

    # Set up Krylov Solver 

    # Use amg preconditioner if available
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

    # Use nonzero guesses - essential for CG with non-symmetric BC
    parameters['krylov_solver']['nonzero_initial_guess'] = True
    parameters['krylov_solver']['monitor_convergence'] = False
    
    solveru = KrylovSolver("bicgstab", "default")
    solvertau = KrylovSolver("bicgstab", "default")
    solverp = KrylovSolver('cg', prec)

    # Time-stepping
    t = 0.0
    iter = 0            # iteration counter
    maxiter = 1000000000
    while t < Tf + DOLFIN_EPS and iter < maxiter:
        iter += 1
        print"t = %s,  Iteration = %d, Convergence Failures = %s, Loop = %s" %(t, iter, jj, j)

        w.t=t
        Ret.t=t
        Wet.t=t

  
        #if jj==1:
            # Update LPS Term
            #F1R = Fdef(u1, tau1)  #Compute the residual in the STRESS EQUATION
            #F1R_vec = as_vector([F1R[0,0], F1R[1,0], F1R[1,1]])
            #dissipation = We*0.5*(phi_def(u1, lambda_d)-1.)*(tau1*Dincomp(u1) + Dincomp(u1)*tau1) 
            #diss_vec = as_vector([dissipation[0,0], dissipation[1,0], dissipation[1,1]])
            #restau0 = We/dt*(tau1_vec-tau0_vec) + We*F1R_vec + fene_func(tau0, b)*tau1_vec - diss_vec - I_vec
            #res_test = project(restau0, Zd)
            #res_orth = project(restau0-res_test, Zc)                                
            #res_orth_norm_sq = project(inner(res_orth,res_orth), Qt)     # Project residual norm onto discontinuous space
            #res_orth_norm = np.power(res_orth_norm_sq, 0.5)
            #kapp = project(res_orth_norm, Qt)
            #kapp = absolute(kapp)
            #LPSl_stress = inner(kapp*h*0.05*grad(tau),grad(Rt))*dx + inner(kapp*h*0.01*div(tau),div(Rt))*dx  # Stress Stabilisation"""

        LPSl_stress = 0
        # Update SU Term
        alpha_supg = h/(magnitude(u1)+0.0000000001)
        SU = inner(dot(u1, grad(tau)), alpha_supg*dot(u1,grad(Rt)))*dx  

       
        DEVSSr_u12 = 2*(1.-betav)*inner(D0,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS


     
        U12 = 0.5*(u1 + u0)    
        # Update Solutions
        if iter > 1:
            w0.assign(w1)
            T0.assign(T1)
            rho0.assign(rho1)
            p0.assign(p1)
            tau0_vec.assign(tau1_vec)
             

        # DEVSS/DEVSS-G STABILISATION
        (u0, D0_vec) = w0.split()  
        D0 = as_matrix([[D0_vec[0], D0_vec[1]],
                        [D0_vec[1], D0_vec[2]]])                   
        DEVSSGr_u1 = (1.-betav)*inner(D0 + transpose(D0),Dincomp(v))*dx            # Update DEVSS-G Stabilisation RHS


        U = 0.5*(u + u0)              
        """# VELOCITY HALF STEP
        lhsFu12 = Re*rho0*(2.0*(u - u0) / dt + conv*dot(u0, nabla_grad(u0)))
        Fu12 = dot(lhsFu12, v)*dx + \
               + inner(2.0*betav*Dincomp(U), Dincomp(v))*dx - inner(2.0/3*betav*div(U),div(v))*dx - ((1. - betav)/We)*inner(div(tau0), v)*dx + inner(grad(p0),v)*dx\
               + inner(D-Dcomp(u),R)*dx 
        a1 = lhs(Fu12)
        L1 = rhs(Fu12)

            #DEVSS Stabilisation
        a1+= th*DEVSSl_u12                     
        L1+= th*DEVSSr_u12 

        A1 = assemble(a1)
        b1= assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, w12.vector(), b1, "bicgstab", "default")
        end()

        (u12, D12_vec) = w12.split()
        D12 = as_matrix([[D12_vec[0], D12_vec[1]],
                        [D12_vec[1], D12_vec[2]]])"""


        """# STRESS Half Step
        F12 = dot(u12,grad(tau)) - dot(grad(u12),tau) - dot(tau,tgrad(u12) + div(u12)*tau) # Convection/Deformation Terms
        lhs_tau12 = (We/dt+1.0)*tau + We*F12                             # Left Hand Side
        rhs_tau12= (We/dt)*tau0 + 2.0*(1.0-betav)*Dcomp(u0)                     # Right Hand Side

        a3 = inner(lhs_tau12,Rt)*dx                                 # Weak Form
        L3 = inner(rhs_tau12,Rt)*dx

        a3 += SUPGl3             # SUPG Stabilisation LHS
        L3 += SUPGr3             # SUPG / SU Stabilisation RHS
        A3=assemble(a3)
        b3=assemble(L3)
        [bc.apply(A3, b3) for bc in bctau]
        solve(A3, tau12_vec.vector(), b3, "bicgstab", "default")
        end()"""

        #Temperature Half Step
        #A8 = assemble(a8)
        #b8 = assemble(L8)
        #[bc.apply(A8, b8) for bc in bcT]
        #solve(A8, T12.vector(), b8, "bicgstab", "default")
        #end()
        
       #Predicted U* Equation
        lhsFus = Re*rho0*((u - u0)/dt + conv*dot(u0, nabla_grad(U)))
        Fus = dot(lhsFus, v)*dx + \
               + inner(2.0*betav*phi_s(theta0,A_0)*Dincomp(U), Dincomp(v))*dx + (1.0/3)*inner(betav*phi_s(theta0,A_0)*div(U),div(v))*dx \
                - ((1.-betav)/(We+DOLFIN_EPS))*inner(div( tau0-Identity(len(u)) ), v)*dx + inner(grad(p0),v)*dx\
               + inner(D-grad(u),R)*dx     
              
        a2= lhs(Fus)
        L2= rhs(Fus)

            # Stabilisation
        a2+= th*DEVSSGl_u1   #[th*DEVSSl_u12]                     
        L2+= th*DEVSSGr_u1    #[th*DEVSSr_u12]

        A2 = assemble(a2)        
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcu]
        solve(A2, ws.vector(), b2, "bicgstab", "default")
        end()
        (us, Ds_vec) = ws.split()


        #Continuity Equation 1
        lhs_p_1 = (Ma*Ma/(dt))*p
        rhs_p_1 = (Ma*Ma/(dt))*p0 - Re*div(rho0*us)

        lhs_p_2 = dt*grad(p)
        rhs_p_2 = dt*grad(p0)
        
        a5=inner(lhs_p_1,q)*dx + inner(lhs_p_2,grad(q))*dx   
        L5=inner(rhs_p_1,q)*dx + inner(rhs_p_2,grad(q))*dx

        A5 = assemble(a5)
        b5 = assemble(L5)
        [bc.apply(A5, b5) for bc in bcp]
        #[bc.apply(p1.vector()) for bc in bcp]
        solve(A5, p1.vector(), b5, "bicgstab", "default")
        end()


        #Continuity Equation 2
        rho1 = rho0 + (Ma*Ma/Re)*(p1-p0)
        rho1 = project(rho1,Q)


        #Velocity Update
        lhs_u1 = (Re/dt)*rho1*u                                          # Left Hand Side
        rhs_u1 = (Re/dt)*rho0*us                                         # Right Hand Side

        a7=inner(lhs_u1,v)*dx + inner(D-grad(u),R)*dx                                           # Weak Form
        L7=inner(rhs_u1,v)*dx - 0.5*inner( grad(p1-p0),v )*dx 

        a7+= 0 #th*DEVSSGl_u1                                                   #DEVSS Stabilisation
        L7+= 0 #th*DEVSSGr_u1   

        A7 = assemble(a7)
        b7 = assemble(L7)
        [bc.apply(A7, b7) for bc in bcu]
        solve(A7, w1.vector(), b7)
        end()
        (u1, D1_vec) = w1.split()
        D1 = as_matrix([[D1_vec[0], D1_vec[1]],
                        [D1_vec[1], D1_vec[2]]])

        U12 = 0.5*(u1 + u0)


        #phifunc = project(phi_ewm(tau1, theta1, k_ewm, B),Q)
        #print phifunc.vector().get_local().max()

        #Temperature Full Step
        gamdot = inner(sigmacon(u0, p0, tau0),grad(u0))
        lhs_temp1 = (1.0/dt)*rho1*thetal + rho1*dot(u1,grad(thetal))
        difflhs_temp1 = Di*grad(thetal)
        rhs_temp1 = (1.0/dt)*rho1*thetar + rho1*dot(u1,grad(thetar)) + (1.0/dt)*rho1*theta0 + Vh*phi_ewm(tau1, theta1, k_ewm, B)*gamdot
        diffrhs_temp1 = Di*grad(thetar)
        a9 = inner(lhs_temp1,r)*dx + inner(difflhs_temp1,grad(r))*dx 
        L9 = inner(rhs_temp1,r)*dx + inner(diffrhs_temp1,grad(r))*dx - Di*Bi*inner(theta0,r)*ds(1) \

        a9+= 0.0*th*DEVSSl_T1                                                #DEVSS Stabilisation
        L9+= 0.0*th*DEVSSr_T1 

        A9 = assemble(a9)
        b9 = assemble(L9)
        [bc.apply(A9, b9) for bc in bcT]
        solve(A9, T1.vector(), b9, "bicgstab", "default")
        end()   

        theta1 = T1-T_0/(T_h-T_0)

        # Stress Full Step
        lhs_tau1 = (We*phi_ewm(tau0, theta1, k_ewm, B)/dt+1.0)*tau  +  We*phi_ewm(tau0, theta1, k_ewm, B)*FdefG(u1, D1, tau)                           # Left Hand Side
        rhs_tau1= (We*phi_ewm(tau0, theta1, k_ewm, B)/dt)*tau0 + Identity(len(u)) 

        Ftau = inner(lhs_tau1,Rt)*dx - inner(rhs_tau1,Rt)*dx
        a4 = lhs(Ftau)
        L4 = rhs(Ftau) 

            # SUPG / SU / LPS Stabilisation (User Choose One)

        a4 += SU + LPSl_stress  # [SUPGl4, SUl4, LPSl_stab, LPSl_stress, diff_stab, 0]
        L4 += 0  # [SUPGr4, SUr4, LPSr_stab, LPS_res_stab, 0]   


        A4=assemble(a4)                                     # Assemble System
        b4=assemble(L4)
        [bc.apply(A4, b4) for bc in bctau]
        solve(A4, tau1_vec.vector(), b4, "bicgstab", "default")
        end()


        #taudiff = np.abs(tau1_vec.vector().get_local() - tau0_vec.vector().get_local()).max()
        #udiff = np.abs(w1.vector().get_local() - w0.vector().get_local()).max()



        # Energy Calculations
        E_k=assemble(0.5*rho1*dot(u1,u1)*dx)
        E_e=assemble((tau1_vec[0]+tau1_vec[2]-2.0)*dx)

        # Save Plot to Paraview Folder
        """if jjj==1 or jjj==3: 
           if j==1 or j==5:
                if iter % frames == 0:
                   fv << u1
                   fT << T1"""

        # Record Torque Data 
        #x.append(t)
        #y.append(assemble(innertorque))
        #zx.append(assemble(innerforcex))
        #z.append(assemble(innerforcey))

        sigma0 = dot(sigmacon(u1, p1, tau1), tang)
        sigma1 = dot(sigmacon(u1, p1, tau1), tang)

        omegaf0 = dot(sigmacon(u1, p1, tau1), n)  #Nomral component of the stress 
        omegaf1 = dot(sigmacon(u1, p1, tau1), n)


        innerforcex = inner(Constant((1.0, 0.0)), omegaf0)*ds(0)
        innerforcey = inner(Constant((0.0, 1.0)), omegaf0)*ds(0)

        innertorque = -inner(n, sigma0)*ds(0)
        outertorque = -inner(n, sigma1)*ds(1)



        # Record Error Data
        x.append(t)




        # Record Elastic & Kinetic Energy Values & Torque (Method 1)
        if j==1:
            x1.append(t)
            ek1.append(E_k)
            ee1.append(E_e)
            y1.append(assemble(innertorque))
            zx1.append(assemble(innerforcex))
            z1.append(assemble(innerforcey))
        if j==2:
            x2.append(t)
            ek2.append(E_k)
            ee2.append(E_e)
            y2.append(assemble(innertorque))
            zx2.append(assemble(innerforcex))
            z2.append(assemble(innerforcey))
        if j==3:
            x3.append(t)
            ek3.append(E_k)
            ee3.append(E_e)
            y3.append(assemble(innertorque))
            zx3.append(assemble(innerforcex))
            z3.append(assemble(innerforcey))
        if j==4:
            x4.append(t)
            ek4.append(E_k)
            ee4.append(E_e)
            y4.append(assemble(innertorque))
            zx4.append(assemble(innerforcex))
            z4.append(assemble(innerforcey))
        if j==5:
            x5.append(t)
            ek5.append(E_k)
            ee5.append(E_e)
            y5.append(assemble(innertorque))
            zx5.append(assemble(innerforcex))
            z5.append(assemble(innerforcey))


        

        # Save Plot to Paraview Folder 
        #for i in range(5000):
            #if iter== (0.02/dt)*i:
               #fv << u1
        #ft << T1

        # Break Loop if code is diverging

        if max(norm(tau1_vec.vector(), 'linf'),norm(w1.vector(), 'linf')) > 10E6 or np.isnan(sum(tau1_vec.vector().get_local())):
            print 'FE Solution Diverging'   #Print message 
            #with open("DEVSS Weissenberg Compressible Stability.txt", "a") as text_file:
                 #text_file.write("Iteration:"+str(j)+"--- Re="+str(Rey)+", We="+str(We)+", t="+str(t)+", dt="+str(dt)+'\n')
            if j==1:           # Clear Lists
               x1=list()
               ek1=list()
               ee1=list()
            if j==2:
               x2=list()
               ek2=list()
               ee2=list()
            if j==3:
               x3=list()
               ek3=list()
               ee3=list()
            if j==4:
               x4=list()
               ek4=list()
               ee4=list()
            if j==5:
               x5=list()
               ek5=list()
               ee5=list() 
            dt=dt/2                        # Use Smaller timestep 
            j-=1                            # Extend loop
            jj+= 1                          # Convergence Failures
            Tf= (iter-10)*dt
            # Reset Functions
            rho0 = Function(Q)
            rho1 = Function(Q)
            p0=Function(Q)       # Pressure Field t=t^n
            p1=Function(Q)       # Pressure Field t=t^n+1
            T0=Function(Q)       # Temperature Field t=t^n
            T1=Function(Q)       # Temperature Field t=t^n+1
            tau0_vec=Function(Zc)     # Stress Field (Vector) t=t^n
            tau12_vec=Function(Zc)    # Stress Field (Vector) t=t^n+1/2
            tau1_vec=Function(Zc)     # Stress Field (Vector) t=t^n+1
            w0= Function(W)
            w12= Function(W)
            ws= Function(W)
            w1= Function(W)
            (u0, D0_vec)=w0.split()
            (u12, D12_vec)=w0.split()
            (us, Ds_vec)=w0.split()
            (u1, D1_vec)=w0.split()
            #quit()
            break


        # Plot solution
        #if t>0.2:
            #plot(kapp, title="tau_xy Stress", rescale=True, interactive=False)
            #plot(tau1[0,0], title="tau_xx Stress", rescale=True, interactive=False)
            #plot(p1, title="Pressure", rescale=True)
            #plot(rho1, title="Density", rescale=True)
            #plot(u1, title="Velocity", rescale=True, mode = "auto")
            #plot(T1, title="Temperature", rescale=True)
           

        # Move to next time step (Continuation in Reynolds Number)
        t += dt





    # Plot Torque/Load for different Wessinberg Numbers
    if max(norm(tau1_vec.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==3 or j==2 or j==1:
        # Plot Torque Data
        plt.figure(0)
        plt.plot(x1, y1, 'r-', label=r'$M1$')
        plt.plot(x2, y2, 'b-', label=r'$M2$')
        plt.plot(x3, y3, 'c-', label=r'$M3$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$C$')
        plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Mesh_Torque_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()
        plt.figure(1)
        plt.plot(x1, zx1, 'r-', label=r'$M1$')
        plt.plot(x2, zx2, 'b-', label=r'$M2$')
        plt.plot(x3, zx3, 'c-', label=r'$M3$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$F_x$')
        plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Mesh_Horizontal_Load_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()
        plt.figure(2)
        plt.plot(x1, z1, 'r-', label=r'$M1$')
        plt.plot(x2, z2, 'b-', label=r'$M2$')
        plt.plot(x3, z3, 'c-', label=r'$M3$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$F_y$')
        plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Mesh_Vertical_Load_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()
        plt.figure(3)
        plt.plot(zx1, z1, 'r-', label=r'$M1$')
        plt.plot(zx2, z2, 'b-', label=r'$M2$')
        plt.plot(zx3, z3, 'c-', label=r'$M3$')
        plt.legend(loc='best')
        plt.xlabel('$F_x$')
        plt.ylabel('$F_y$')
        plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Mesh_Force_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()


        #Plot Kinetic and elasic Energies for different Weissenberg numbers at Re FIXED 
    if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==3 or j==2 or j==1:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'$M1$')
        plt.plot(x2, ek2, 'b--', label=r'$M2$')
        plt.plot(x3, ek3, 'c:', label=r'$M3$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$E_{kinetic}$')
        plt.savefig("Compressible Viscoelastic Flow Results/Energy/Mesh_KineticEnergyRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
        plt.clf()
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'$M1$')
        plt.plot(x2, ee2, 'b--', label=r'$M2$')
        plt.plot(x3, ee3, 'c:', label=r'$M3$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$E_{elastic}$')
        plt.savefig("Compressible Viscoelastic Flow Results/Energy/Mesh_ElasticEnergyRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
        plt.clf()




    """if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==3 or j==1:
        # Plot First Normal Stress Difference
        tau_xx=project(tau1[0,0],Q)
        mplot(tau_xx)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Mesh_tau_xxRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf() 
        tau_xy=project(tau1[1,0],Q)
        mplot(tau_xy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Mesh_tau_xyRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf() 
        tau_yy=project(tau1[1,1],Q)
        mplot(tau_yy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Mesh_tau_yyRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf() 

        # Matlab Plot of the Solution at t=Tf
        mplot(rho1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Mesh_DensityRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()
        mplot(p1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Mesh_PressureRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()
        mplot(T1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Mesh_TemperatureRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()
        divu=project(div(u1),Q)
        mplot(divu)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Mesh_CompressionRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()
        psi = stream_function(u1)
        mplot(psi)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Mesh_Stream_functionRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()



        #Plot PRESSURE Contours USING MATPLOTLIB
        # Scalar Function code

        #Set Values for inner domain as -infty
        Q1=FunctionSpace(meshc, "CG", 1)

        x = Expression('x[0]', degree=2)  #GET X-COORDINATES LIST
        y = Expression('x[1]', degree=2)  #GET Y-COORDINATES LIST
        pj=Expression('0', degree=2) #Expression for the 'pressure' in the domian
        pjq=interpolate(pj, Q1)
        pjvals=pjq.vector().get_local()

        xyvals=meshc.coordinates()
        xqalsv = interpolate(x, Q1)
        yqalsv= interpolate(y, Q1)

        xqals= xqalsv.vector().get_local()
        yqals= yqalsv.vector().get_local()

        pvals = p1.vector().get_local() # GET SOLUTION u= u(x,y) list
        xyvals = mesh.coordinates() # CLEAN THIS UP!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        xvalsv = interpolate(x, Q)#xyvals[:,0]
        yvalsv= interpolate(y, Q)#xyvals[:,1]

        xvals = xvalsv.vector().get_local()
        yvals = yvalsv.vector().get_local()

        pvals = np.concatenate([pvals, pjvals])  #Merge two arrays for pressure values
        xvals = np.concatenate([xvals, xqals])   #Merge two arrays for x-coordinate values
        yvals = np.concatenate([yvals, yqals])   #Merge two arrays for y-coordinate values

        xx = np.linspace(-1.5*r_b,1.5*r_b, num=250)
        yy = np.linspace(-1.5*r_b,1.5*r_b, num=250)
        XX, YY = np.meshgrid(xx,yy)   # (x,y) coordinate data formatted so that it can be used by plt.contour()
        pp = mlab.griddata(xvals, yvals, pvals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 

        plt.contour(XX, YY, pp, 30)
        plt.colorbar()
        plt.title('Pressure Contours')   # PRESSURE CONTOUR PLOT
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Mesh_Pressure_Contours Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()


        #Plot TEMPERATURE Contours USING MATPLOTLIB
        # Scalar Function code

        #Set Values for inner domain as ZERO


        Tj=Expression('0', degree=1) #Expression for the 'pressure' in the domian
        Tjq=interpolate(Tj, Q1)
        Tjvals=Tjq.vector().get_local()

        Tvals = T1.vector().get_local() # GET SOLUTION T= T(x,y) list
        Tvals = np.concatenate([Tvals, Tjvals])  #Merge two arrays for Temperature values

        TT = mlab.griddata(xvals, yvals, Tvals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 

        plt.contour(XX, YY, TT, 30)
        plt.colorbar()
        plt.title('Temperature Contours')   # TEMPERATURE CONTOUR PLOT
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Mesh_Temperature_Contours Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()



        # Plot Velocity Field Contours (MATPLOTLIB)

        # Set Velocity on the bearing to zero

        V1=VectorFunctionSpace(meshc, "CG", 2)

        vj=Expression(('0.','0.'), degree=2) #Expression for the 'pressure' in the domian
        vjq=interpolate(vj, V1)

        ujvals = project(vjq[0], Q1) # Project u_x onto (Q1)
        vjvals = project(vjq[1], Q1) # Project u_y onto (Q1)
        ujvals = ujvals.vector().get_local()
        vjvals = vjvals.vector().get_local()



        xy = Expression(('x[0]','x[1]'), degree=2)  #GET MESH COORDINATES LIST

        xyvalsv = interpolate(xy, V1)

        xvalsj = project(xyvalsv[0], Q1) # Project x-coordinate onto (Q1)
        yvalsj = project(xyvalsv[1], Q1) # Project y-coordinate onto (Q1)
        xvalsj = xvalsj.vector().get_local() # Get x-coordinate array (Q1)
        yvalsj = yvalsj.vector().get_local() # Get y-coordinate array (Q1)


        xyvalsvj = interpolate(xy, V)

        xvalsv = project(xyvalsvj[0], Q) # Project x-coordinate onto (Q)
        yvalsv = project(xyvalsvj[1], Q) # Project y-coordinate onto (Q)
        xvals = xvalsv.vector().get_local() # Get x-coordinate array (Q)
        yvals = yvalsv.vector().get_local() # Get y-coordinate array (Q)

        #Plot Velocity Streamlines USING MATPLOTLIB
        u1_q = project(u1[0],Q)
        uvals = u1_q.vector().get_local()
        v1_q = project(u1[1],Q)
        vvals = v1_q.vector().get_local()

            # Interpoltate velocity field data onto matlab grid
        #uu = mlab.griddata(xvals, yvals, uvals, xx, yy, interp='nn') 
        #vv = mlab.griddata(xvals, yvals, vvals, xx, yy, interp='nn') 
     

        #Merge arrays
        uvals = np.concatenate([uvals, ujvals])  #Merge two arrays for velocity values
        vvals = np.concatenate([vvals, vjvals])  #Merge two arrays for velocity values
        xvals = np.concatenate([xvals, xvalsj])   #Merge two arrays for x-coordinate values
        yvals = np.concatenate([yvals, yvalsj])   #Merge two arrays for y-coordinate values


        uu = mlab.griddata(xvals, yvals, uvals, xx, yy, interp='nn') 
        vv = mlab.griddata(xvals, yvals, vvals, xx, yy, interp='nn') 

        #Determine Speed 
        speed = np.sqrt(uu*uu+ vv*vv)

        plot3 = plt.figure()
        plt.streamplot(XX, YY, uu, vv,  
                       density=5,              
                       color=speed/speed.max(),  
                       cmap=cm.gnuplot,                         # colour map
                       linewidth=0.5*speed/speed.max()+0.5)       # line thickness
        plt.colorbar()
        plt.title('Journal Bearing Problem')
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Mesh_Velocity_Contours_Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")   
        plt.clf()"""                                                                     # display the plot



    plt.close()


    if dt < tol:
       j=loopend+1
       break

    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and abs(E_k) < 10:
        Tf=T_f   

    if j==3:
        jjj+=1
        if jjj==1:
            Ma = 0.001
        if jjj==2:
            Ma = 0.01  
        if jjj==3:
            Ma = 0.1 
        j=0
        x1=list()
        x2=list()
        x3=list()
        x4=list()
        x5=list()
        y=list()
        z=list()
        zz=list()
        zzz=list()
        zl=list()
        ek1=list()
        ek2=list()
        ek3=list()
        ek4=list()
        ee1=list()
        ee2=list()
        ee3=list()
        ee4=list()
        ek5=list()
        ee5=list() 
        y1 = list()
        zx1 = list()
        z1 = list()
        y2 = list()
        zx2 = list()
        z2 = list()
        y3 = list()
        zx3 = list()
        z3 = list()
        y4 = list()
        zx4 = list()
        z4 = list()
        y5 = list()
        zx5 = list()
        z5 = list()

    if jjj==4:
        quit()


    # Reset Functions
    rho0 = Function(Q)
    rho1 = Function(Q)
    p0 = Function(Q)       # Pressure Field t=t^n
    p1 = Function(Q)       # Pressure Field t=t^n+1
    T0 = Function(Q)       # Temperature Field t=t^n
    T1 = Function(Q)       # Temperature Field t=t^n+1
    tau0_vec = Function(Zc)     # Stress Field (Vector) t=t^n
    tau12_vec = Function(Zc)    # Stress Field (Vector) t=t^n+1/2
    tau1_vec = Function(Zc)     # Stress Field (Vector) t=t^n+1
    w0 = Function(W)
    w12 = Function(W)
    ws = Function(W)
    w1 = Function(W)
    (u0, D0_vec) = w0.split()
    (u12, D12_vec) = w0.split()
    (us, Ds_vec) = w0.split()
    (u1, D1_vec) = w0.split()
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



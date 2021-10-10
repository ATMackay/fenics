"""Inompressible Lid Driven Cavity Problem for an COMPRESSIBLE Oldroyd-B Fluid"""
"""Solution Method: Finite Element Method using DOLFIN (FEniCS)"""


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


# FEM Solution Convergence/Energy Plot
x1=list()
x2=list()
x3=list()
x4=list()
x5=list()
x6=list()
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
ek6=list()
ee6=list()


# Experiment Run Time
dt = 0.005  #Timestep
T_f = 10.0
Tf = T_f
tol = 0.0001

conv = 1
U = 1
betav = 0.5     
Re = 2                            #Reynolds Number
We = 0.5                          #Weisenberg NUmber
Di = 0.005                         #Diffusion Number
Vh = 0.005
T_0 = 300
T_h = 350
Bi = 0.2
c0 = 1500
Ma = 0.001 
c1 = 0.1
c2 = 0.01
c3 = 0.01

alph1 = 1.0
alph2 = 0.00
alph3 = 0.000
th = 1.0               # DEVSS
#c1 = alph*h_ska        # SUPG / SU

# Loop Experiments
loopend = 4
j=0
jj=0
while j < loopend:
    j+=1

    t=0.0
    """ mesh refinemment prescribed in code"""
    # Mesh Refinement 
    if j==1:
       mm=80
    elif j==2:
       mm=100
    elif j==3:
       mm=120
    elif j==4:
       mm=140




    # Define Geometry
    B=1
    L=1
    x_0 = 0
    y_0 = 0
    x_1 = B
    y_1 = L

    # Mesh refinement comparison Loop
    nx=mm*B
    ny=mm*L

    c = min(x_1-x_0,y_1-y_0)
    base_mesh = RectangleMesh(Point(x_0,y_0), Point(x_1, y_1), nx, ny)


    # Create Unstructured mesh

    u_rec=Rectangle(Point(0.0,0.0),Point(1.0,1.0))
    mesh0=generate_mesh(u_rec, mm)



    #SKEW MESH FUNCTION

    # MESH CONSTRUCTION CODE

    nv= base_mesh.num_vertices()
    nc= base_mesh.num_cells()
    coorX = base_mesh.coordinates()[:,0]
    coorY = base_mesh.coordinates()[:,1]
    cells0 = base_mesh.cells()[:,0]
    cells1 = base_mesh.cells()[:,1]
    cells2 = base_mesh.cells()[:,2]


    # Skew Mapping EXPONENTIAL
    N=4.0
    def expskewcavity(x,y):
        xi = 0.5*(1+np.tanh(2*N*(x-0.5)))
        ups= 0.5*(1+np.tanh(2*N*(y-0.5)))
        return(xi,ups)

    # Skew Mapping
    pi=3.14159265359

    def skewcavity(x,y):
        xi = 0.5*(1-np.cos(x*pi))**1
        ups =0.5*(1-np.cos(y*pi))**1
        return(xi,ups)

    # OLD MESH COORDINATES -> NEW MESH COORDINATES
    r=list()
    l=list()
    for i in range(nv):
        r.append(skewcavity(coorX[i], coorY[i])[0])
        l.append(skewcavity(coorX[i], coorY[i])[1])

    r=np.asarray(r)
    l=np.asarray(l)

    # MESH GENERATION (Using Mesheditor)
    mesh1 = Mesh()
    editor = MeshEditor()
    editor.open(mesh1,2,2)
    editor.init_vertices(nv)
    editor.init_cells(nc)
    for i in range(nv):
        editor.add_vertex(i, r[i], l[i])
    for i in range(nc):
        editor.add_cell(i, cells0[i], cells1[i], cells2[i])
    editor.close()

    # Mesh Refine Code (UNSTRUCTURED MESH)

    for i in range(0):
          g = (max(x_1,y_1)-max(x_0,y_0))*0.02/(i+1)
          cell_domains = CellFunction("bool", mesh0)
          cell_domains.set_all(False)
          for cell in cells(mesh0):
              x = cell.midpoint()
              if  (x[0] < x_0+g or x[1] < y_0+g) or (x[0] > x_1-g or x[1] > y_1-g): # or (x[0] < x0+g and x[1] < y0+g)  or (x[0] > x1-g and x[1] < g): 
                  cell_domains[cell]=True
          #plot(cell_domains, interactive=True)
          mesh0 = refine(mesh0, cell_domains, redistribute=True)

    for i in range(0):
          g = (max(x_1,y_1)-max(x_0,y_0))*0.05/(i+1)
          cell_domains = CellFunction("bool", mesh1)
          cell_domains.set_all(False)
          for cell in cells(mesh1):
              x = cell.midpoint()
              if  (x[1] > y_1-g and x[0] > 0.6): # or (x[0] < x0+g and x[1] < y0+g)  or (x[0] > x1-g and x[1] < g): 
                  cell_domains[cell]=True
          #plot(cell_domains, interactive=True)
          mesh1 = refine(mesh1, cell_domains, redistribute=False)

    #plot(mesh0)
    #plot(mesh)
    #plot(mesh1,interactive=True)

    #mplot(mesh0)
    #plt.savefig("fine_unstructured_grid.png")
    #plt.clf() 
    #mplot(mesh)
    #plt.savefig("fine_structured_grid.png")
    #plt.clf() 
    #mplot(mesh1)
    #plt.savefig("fine_skewed_grid.png")
    #plt.clf()
    #quit()

        # Choose Mesh to Use
    if j==1 or j==2 or j==3 or j==4:
       mesh = mesh1


    #Define Boundaries 

    top_bound = 0.5*(1+tanh(N)) 

    class No_slip(SubDomain):
          def inside(self, x, on_boundary):
              return True if on_boundary else False 
                                                                              
    class Lid(SubDomain):
          def inside(self, x, on_boundary):
              return True if x[1] > L*(top_bound - DOLFIN_EPS) and on_boundary  else False   

    no_slip = No_slip()
    lid = Lid()


    # MARK SUBDOMAINS (Create mesh functions over the cell facets)
    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    sub_domains.set_all(5)
    no_slip.mark(sub_domains, 0)
    lid.mark(sub_domains, 2)


    #plot(sub_domains, interactive=False)        # DO NOT USE WITH RAVEN
    #quit()

    #Define Boundary Parts
    boundary_parts = FacetFunction("size_t", mesh)
    no_slip.mark(boundary_parts,0)
    lid.mark(boundary_parts,1)
    ds = Measure("ds")[boundary_parts]

    # Define function spaces (P2-P1)

    # Discretization  parameters
    family = "CG"; dfamily = "DG"; rich = "Bubble"
    shape = "triangle"; order = 2

    #mesh.ufl_cell()

    V_s = VectorElement(family, mesh.ufl_cell(), order)       # Elements
    V_d = VectorElement(dfamily, mesh.ufl_cell(), order-1) 
    Z_c = VectorElement(family, mesh.ufl_cell(),  order, 3)
    Z_s = VectorElement(dfamily, mesh.ufl_cell(),  order-1, 3)
    Z_d = VectorElement(dfamily, mesh.ufl_cell(),  order-2, 3)
    Q_s = FiniteElement(family, mesh.ufl_cell(), order-1)
    Q_p = FiniteElement(rich, mesh.ufl_cell(), order+1, 3)
    #Z_e = EnrichedElement(Z_c,Z_se)                 # Enriched Elements
    Q_rich = EnrichedElement(Q_s,Q_p)


    W = FunctionSpace(mesh,V_s*Z_d)             # F.E. Spaces 
    V = FunctionSpace(mesh,V_s)
    Vd = FunctionSpace(mesh,V_d)
    Z = FunctionSpace(mesh,Z_s)
    Zd = FunctionSpace(mesh,Z_d)
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
    T0=Function(Qt)       # Temperature Field t=t^n
    T1=Function(Qt)       # Temperature Field t=t^n+1


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


    # Some Useful Functions
    def  tgrad (w):
        """ Returns  transpose  gradient """
        return  transpose(grad(w))
    def Dincomp (w):
        """ Returns 2* the  rate of  strain  tensor """
        return (grad(w) + tgrad(w))/2
    def Dcomp (w):
        """ Returns 2* the  rate of  strain  tensor """
        return ((grad(w) + tgrad(w))-(2.0/3)*div(w)*Identity(len(u)))/2

    def sigma(u, p, Tau):
        return 2*betav*Dcomp(u) - p*Identity(len(u)) + Tau

    def normalize_solution(u):
        "Normalize u: return u divided by max(u)"
        u_array = u.vector().array()
        u_max = np.max(np.abs(u_array))
        u_array /= u_max
        u.vector()[:] = u_array
        #u.vector().set_local(u_array)  # alternative
        return u

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

    # The  projected  rate -of-strain
    D_proj_vec = Function(Zd)
    D_proj = as_matrix([[D_proj_vec[0], D_proj_vec[1]],
                        [D_proj_vec[1], D_proj_vec[2]]])

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





    # Define boundary/stabilisation FUNCTIONS

    ulidreg=Expression(('8*(1.0+tanh(8*t-4.0))*(x[0]*(L-x[0]))*(x[0]*(L-x[0]))','0'), degree=2, t=0.0, L=L, e=e, T_0=T_0, T_h=T_h) # Lid Speed 
    ulid=Expression(('0.5*(1.0+tanh(8*t-4.0))','0'), degree=2, t=0.0, T_0=T_0, T_h=T_h) # Lid Speed 
    T_bl = Expression('T_0+(T_h-T_0)*sin((x[0]+x[1])*2*pi/L)', degree=2, T_0=T_0, T_h=T_h, pi=pi, L=L, B=B)
    T_bb = Expression('T_0+(T_h-T_0)*sin((x[0]+x[1])*2*pi/B)', degree=2, T_0=T_0, T_h=T_h, pi=pi, L=L, B=B)
    h_sk = Expression('cos(pi*x[0])-cos(pi*(x[0]+1/mm))','cos(pi*x[1])-cos(pi*(x[1]+1/mm))', degree=2, pi=pi, mm=mm, L=L, B=B)             # Mesh size function
    h_k = Expression(('1/mm','1/mm'), degree=2, mm=mm, L=L, B=B)
    h_m = Expression('0.5*h', degree=2, h=mesh.hmin())
    h_ka = Expression('0.5*1/mm', degree=2, mm=mm, L=L, B=B)
    h_ska= Expression('0.5*(cos(pi*x[0])-cos(pi*(x[0]+1/mm))+cos(0.5*pi*x[1])-cos(0.5*pi*(x[1]+1/mm)))', degree=2, pi=pi, mm=mm, L=L, B=B)
    rampd=Expression('0.5*(1+tanh(8*(2.0-t)))', degree=2, t=0.0)
    rampu=Expression('0.5*(1+tanh(16*(t-2.0)))', degree=2, t=0.0)

    # Set Boundary Function Time = 0
    rampd.t=t
    ulid.t=t
    ulidreg.t=t



    # Interpolate Stabilisation Functions
    #h_k = interpolate(h_k, V)
    #h_m = interpolate(h_m, Q)
    #h_ka = interpolate(h_ka, Q)
    #h_ska = interpolate(h_ska, Q)

    # Define unit Normal/tangent Vector at inner and outer Boundary (Method 2)
    n0 =  Expression(('-1' , '0'), degree=2)
    n1 =  Expression(('0' , '1' ), degree=2)
    n2 =  Expression(('1' , '0' ), degree=2)
    n3 =  Expression(('0' , '-1'), degree=2)

    # Dirichlet Boundary Conditions  (LID DRIVEN CAVITY)
    noslip  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), no_slip)  # No Slip boundary conditions on the left wall
    drive  =  DirichletBC(W.sub(0), ulidreg, lid)  # No Slip boundary conditions on the upper wall
    #slip  = DirichletBC(V, sl, omega0)  # Slip boundary conditions on the second part of the flow wall 
    #temp0 =  DirichletBC(Qt, T_0, omega0)    #Temperature on Omega0 
    #temp2 =  DirichletBC(Qt, T_0, omega2)    #Temperature on Omega2 
    #temp3 =  DirichletBC(Qt, T_0, omega3)    #Temperature on Omega3 
    #Collect Boundary Conditions
    bcu = [noslip, drive]
    bcp = []
    bcT = [] # temp0, temp2
    bctau = []



    # Set Stabilisation Parameters

    h = CellSize(mesh)
    #h_k = project(h/mesh.hmax(), Qt)
    n = FacetNormal(mesh) 
        

    # Continuation in Reynolds/Weissenberg Number Number (Re-->20Re/We-->20We)
    Ret=Expression('Re*(1.0+19.0*0.5*(1.0+tanh(0.7*t-4.0)))', t=0.0, Re=Re, degree=2)
    Rey=Re
    Wet=Expression('(We/100)*(1.0+99.0*0.5*(1.0+tanh(0.7*t-5.0)))', t=0.0, We=We, degree=2)


    print '############# TIME SCALE ############'
    print 'Timestep size (s):', dt
    print 'Finish Time (s):', Tf

    print '############# Scalings & Nondimensional Parameters ############'
    print 'Characteristic Length (m):', L
    print 'Characteristic Velocity (m/s):', 1.0
    print 'Lid velocity:', (0.5*(1.0+tanh(e*t-3.0)),0)
    print 'Speed of sound (m/s):', c0
    print 'Mach Number', Ma
    print 'Reynolds Number:', Rey
    print 'Non-inertial parameter:', conv
    print 'Weissenberg Number:', We
    print 'Viscosity Ratio:', betav
    print 'Diffusion Number:' ,Di
    print 'Viscous Heating Number:', Vh

    #Np= len(p0.vector().array())
    #Nv= len(w0.vector().array())   
    #Ntau= len(tau0_vec.vector().array())
    #ND = len(D_proj_vec.vector().array())
    #dof= 3*Nv+2*Ntau+Np
    print '############# Discrete Space Characteristics ############'
    print 'Degree of Elements', order
    print 'Mesh: %s x %s' %(mm, mm)
    #print('Size of Pressure Space = %d ' % Np)
    #print('Size of Velocity Space = %d ' % Nv)
    #print('Size of Stress Space = %d ' % Ntau)
    #print('Size of DEVSS Space = %d ' % ND)
    #print('Degrees of Freedom = %d ' % dof)
    print 'Number of Cells:', mesh.num_cells()
    print 'Number of Vertices:', mesh.num_vertices()
    print 'Minimum Cell Diamter:', mesh.hmin()
    print 'Maximum Cell Diamter:', mesh.hmax()
    print '############# Stabilisation Parameters ############'
    print 'DEVSS Parameter:', th
    print 'SUPG/SU Parameter:', alph1
    print 'LPS Parameter c2:', alph2 
    print 'LPS Parameter c3:', alph3

    
     # Initial Density Field
    #rho_array = rho0.vector().array()
    #for i in range(len(rho_array)):  
    #    rho_array[i] = 1.0
    #rho0.vector()[:] = rho_array 

    # Initial Temperature Field
    #T_array = T0.vector().array()
    #for i in range(len(T_array)):  
    #    T_array[i] = T_0
    #T0.vector()[:] = T_array

    # Identity Tensor   
    I = Expression((('1.0','0.0'),
                    ('0.0','1.0')), degree=2)


    #Define Variable Parameters, Strain Rate and other tensors
    gamdots = inner(Dincomp(u1),grad(u1))
    gamdots12 = inner(Dincomp(u12),grad(u12))
    gamdotp = inner(tau1,grad(u1))
    gamdotp12 = inner(tau12,grad(u12))
    thetal = (T)/(T_h-T_0)
    thetar = (T_0)/(T_h-T_0)
    thetar = project(thetar,Qt)
    theta0 = (T0-T_0)/(T_h-T_0)




    # STABILISATION TERMS
    F1 = dot(u1,grad(tau)) - dot(grad(u1),tau) - dot(tau,tgrad(u1) + div(u1)*tau)                   # Convection/Deformation Terms
    F12 = dot(u12,grad(tau)) - dot(grad(u12),tau) - dot(tau,tgrad(u12)) + div(u12)*tau               # Convection/Deformation Terms


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
    

    # DEVSS Stabilisation
    devss_Z = inner(D-Dcomp(u),R)*dx                        # L^2 Projection of rate-of strain
    
    DEVSSl_u12 = 2*(1-betav)*inner(Dcomp(u),Dincomp(v))*dx    
    DEVSSr_u12 = 2*(1-betav)*inner(D0,Dincomp(v))*dx   
    DEVSSl_u1 = 2*(1-betav)*inner(Dcomp(u),Dincomp(v))*dx    
    DEVSSr_u1 = 2*(1-betav)*inner(D12,Dincomp(v))*dx 

    #DEVSSl_temp1 = (1-Di)*inner(grad(theta),grad(r))
    #DEVSSr_temp1 = (1-Di)*inner(grad(theta),grad(r))

    #Lists for Energy Values
    x=list()
    ee=list()
    ek=list()

    #ftau=File("Incompressible Viscoelastic Flow Results/Paraview/Stress_th"+str(th)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"/stress "+str(t)+".pvd")
    #fv=File("Incompressible Viscoelastic Flow Results/Paraview/Velocity_th"+str(th)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"/velocity "+str(t)+".pvd")


    # Time-stepping
    t = dt
    iter = 0            # iteration counter
    maxiter = 1000000
    while t < Tf + DOLFIN_EPS and iter < maxiter:
        iter += 1
        # Update progress bar
        #if (iter-1)%100 == 0:
        print"t = %s,  Iteration = %d, Convergence Failures = %s, Loop = %s" %(t, iter, jj, j)

        rampd.t=t
        ulid.t=t
        ulidreg.t=t
        Ret.t=t
        Wet.t=t

 
        # Update Stabilisation (Copy and Paste Stabilisation Technique from above)

        F1R = dot(u1,grad(tau1)) - dot(grad(u1),tau1) - dot(tau1,tgrad(u1)) + div(u1)*tau1  #Compute the residual in the STRESS EQUATION
        F1R_vec = as_vector([F1R[0,0], F1R[1,0], F1R[1,1]])
        Dcomp1_vec = as_vector([Dcomp(u1)[0,0], Dcomp(u1)[1,0], Dcomp(u1)[1,1]])
        restau0 = We/dt*(tau1_vec-tau0_vec) + We*F1R_vec + tau1_vec - 2*(1-betav)*Dcomp1_vec 
        res_test = project(restau0, Zd)
        res_orth = project(restau0-res_test, Zc)                                
        res_orth_norm_sq = project(inner(res_orth,res_orth), Qt)     # Project residual norm onto discontinuous space
        res_orth_norm = np.power(res_orth_norm_sq, 0.5)
        kapp = project(res_orth_norm, Qt)

        #resv = (Re/dt)*rho1*(u1-u0) + rho1*grad(u1)*u1 + 0.5*(grad(p1) + grad(p0)) - div(tau0) - betav*(div(grad(u1)) + 1.0/3*grad(div(u1)))        
        #resv_test = project(resv, Vd)
        #resv_orth = project(resv - resv_test, V)
        #resv_orth_norm_sq = project(inner(resv_orth,resv_orth), Qt)     # Project residual norm onto discontinuous space
        #resv_orth_norm = np.power(resv_orth_norm_sq, 0.5)
        #kappv = project(resv_orth_norm, Qt)


        LPSl_stress = inner(h*c1*kapp*grad(tau),grad(Rt))*dx + inner(h*c2*kapp*div(tau),div(Rt))*dx  # Stress Stabilisation
        #LPSl_vel = inner(kappv*c3*h*Dincomp(u),Dincomp(v))*dx                                             # Velocity Stabilisation
       

        # Update Solutions
        if iter > 1:
            w0.assign(w1)
            T0.assign(T1)
            rho0.assign(rho1)
            p0.assign(p1)
            tau0_vec.assign(tau1_vec)
 

        (u0, D0_vec) = w0.split()  
        DEVSSr_u12 = 2*(1-betav)*inner(D0,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS


        U = 0.5*(u + u0)              
        # VELOCITY HALF STEP
        lhsFu12 = Re*rho0*(2.0*(u - u0) / dt + conv*dot(u0, nabla_grad(u0)))
        Fu12 = dot(lhsFu12, v)*dx + \
               + inner(sigma(U, p0, tau0), Dincomp(v))*dx \
               + dot(p0*n, v)*ds - dot(betav*nabla_grad(U)*n, v)*ds - (1.0/3)*betav*dot(div(U)*n,v)*ds \
               - dot(tau0*n, v)*ds \
               +  inner(D-Dincomp(u),R)*dx
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
                        [D12_vec[1], D12_vec[2]]])

        DEVSSr_u1 = 2*(1.0-betav)*inner(D12,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS

        """# STRESS Half Step
        F12 = dot(u12,grad(tau)) - dot(grad(u12),tau) - dot(tau,tgrad(u12)) + div(u12)*tau # Convection/Deformation Terms
        lhs_tau12 = (We/dt+1.0)*tau + We*F12                             # Left Hand Side
        rhs_tau12= (We/dt)*tau0 + 2.0*(1.0-betav)*Dcomp(u0) #+ F_tau                    # Right Hand Side

        a3 = inner(lhs_tau12,Rt)*dx                                 # Weak Form
        L3 = inner(rhs_tau12,Rt)*dx

        a3 += LPSl_stress             # SUPG Stabilisation LHS
        L3 += 0             # SUPG / SU Stabilisation RHS
        A3=assemble(a3)
        b3=assemble(L3)
        [bc.apply(A3, b3) for bc in bctau]
        solve(A3, tau12_vec.vector(), b3, "bicgstab", "default")
        end()"""

      
        #Predicted U* Equation
        lhsFus = Re*rho0*((u - u0)/dt + conv*dot(u0, nabla_grad(u0)))
        Fus = dot(lhsFus, v)*dx + \
              + inner(sigma(U, p0, tau0), Dincomp(v))*dx \
              + 0.5*dot(p0*n, v)*ds - dot(betav*nabla_grad(U)*n, v)*ds - (1.0/3)*betav*dot(div(U)*n,v)*ds \
              - dot(tau0*n, v)*ds\
              +  inner(D-Dincomp(u),R)*dx     
              
        a2= lhs(Fus)
        L2= rhs(Fus)

            # Stabilisation
        a2+= 0   #[th*DEVSSl_u12]                     
        L2+= 0   #[th*DEVSSr_u12]
        A2 = assemble(a2)        
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcu]
        solve(A2, ws.vector(), b2, "bicgstab", "default")
        end()

        (us, Ds_vec) = ws.split()



        #Continuity Equation 1
        lhs_p_1 = (Ma*Ma/(dt*Re))*p
        rhs_p_1 = (Ma*Ma/(dt*Re))*p0 - rho0*div(us) + dot(grad(rho0),us)

        lhs_p_2 = 0.5*dt*grad(p)
        rhs_p_2 = 0.5*dt*grad(p0)
        
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
        lhs_u1 = (Re/dt)*rho0*u                                          # Left Hand Side
        rhs_u1 = (Re/dt)*rho0*us                                         # Right Hand Side

        a7=inner(lhs_u1,v)*dx  + inner(D-Dcomp(u),R)*dx  # Weak Form
        L7=inner(rhs_u1,v)*dx + 0.5*inner(p1-p0,div(v))*dx + 0.5*dot(p1*n, v)*ds

        a7+= th*DEVSSl_u1   #[th*DEVSSl_u1]                                                #DEVSS Stabilisation
        L7+= th*DEVSSr_u1   #[th*DEVSSr_u1] 

        A7 = assemble(a7)
        b7 = assemble(L7)
        [bc.apply(A7, b7) for bc in bcu]
        solve(A7, w1.vector(), b7, "bicgstab", "default")
        end()

        (u1, D1_vec) = w1.split()
        D1 = as_matrix([[D1_vec[0], D1_vec[1]],
                        [D1_vec[1], D1_vec[2]]])

        U12 = 0.5*(u1 + u0)   

        # Stress Full Step
        F1 = dot(u1,grad(tau)) - dot(grad(u1),tau) - dot(tau,tgrad(u1)) + div(u1)*tau # Convection/Deformation Terms t^{n+1}
        #F12 = dot(U12,grad(tau12)) - dot(grad(U12),tau12) - dot(tau12,tgrad(U12)) + div(U12)*tau # Convection/Deformation Terms t^{n+1/2}
        lhs_tau1 = (We/dt+1.0)*tau  +  We*F1                            # Left Hand Side
        rhs_tau1= (We/dt)*tau0 + 2.0*(1.0-betav)*Dcomp(U12) #+  F_tau       # Right Hand Side

        A = inner(lhs_tau1,Rt)*dx - inner(rhs_tau1,Rt)*dx
        a4 = lhs(A)
        L4 = rhs(A) 

            # SUPG / SU / LPS Stabilisation (User Choose One)

        a4 += LPSl_stress  # [SUPGl4, SUl4, LPSl_stab, LPSl_stress, diff_stab, 0]
        L4 += 0  # [SUPGr4, SUr4, LPSr_stab, LPS_res_stab, 0]   


        A4=assemble(a4)                                     # Assemble System
        b4=assemble(L4)
        [bc.apply(A4, b4) for bc in bctau]
        solve(A4, tau1_vec.vector(), b4, "bicgstab", "default")
        end()


        #Temperature Full Step
        #lhs_temp1 = (1.0/dt)*rho1*thetal + rho1*dot(u1,grad(thetal))
        #difflhs_temp1 = Di*grad(thetal)
        #rhs_temp1 = (1.0/dt)*rho1*thetar + rho1*dot(u1,grad(thetar)) + (1.0/dt)*rho1*theta0 + Vh*(gamdots12 + gamdotp12 - p1*div(u1))
        #diffrhs_temp1 = Di*grad(thetar)
        #a9 = inner(lhs_temp1,r)*dx + inner(difflhs_temp1,grad(r))*dx 
        #L9 = inner(rhs_temp1,r)*dx + inner(diffrhs_temp1,grad(r))*dx - Di*Bi*inner(theta0,r)*ds(1) \

        #a9+= th*DEVSSl_T1                                                #DEVSS Stabilisation
        #L9+= th*DEVSSr_T1 

        #A9 = assemble(a9)
        #b9 = assemble(L9)
        #[bc.apply(A9, b9) for bc in bcT]
        #solve(A9, T1.vector(), b9, "bicgstab", "default")
        #end()


        # Energy Calculations
        E_k=assemble(0.5*rho1*dot(u1,u1)*dx)
        E_e=assemble((tau1_vec[0]+tau1_vec[2])*dx)


        # Record Elastic & Kinetic Energy Values (Method 1)
        if j==1:
           x1.append(t)
           ek1.append(E_k)
           ee1.append(E_e)
        if j==2:
           x2.append(t)
           ek2.append(E_k)
           ee2.append(E_e)
        if j==3:
           x3.append(t)
           ek3.append(E_k)
           ee3.append(E_e)
        if j==4:
           x4.append(t)
           ek4.append(E_k)
           ee4.append(E_e)
        if j==5:
           x5.append(t)
           ek5.append(E_k)
           ee5.append(E_e)

        # Record Error Data
        err = project(c1*h*kapp,Qt)
        x.append(t)
        ee.append(norm(err.vector(),'linf'))
        ek.append(norm(tau1_vec.vector(),'linf'))
        

        # Save Plot to Paraview Folder 
        #for i in range(5000):
            #if iter== (0.02/dt)*i:
               #fv << u1
        #ft << T1

        # Break Loop if code is diverging

        if max(norm(tau1_vec.vector(), 'linf'),norm(w1.vector(), 'linf')) > 10E6 or np.isnan(sum(tau1_vec.vector().array())):
            print t
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
            Tf= (iter-40)*dt
            # Reset Functions
            rho0 = Function(Q)
            rho1 = Function(Q)
            p0=Function(Q)       # Pressure Field t=t^n
            p1=Function(Q)       # Pressure Field t=t^n+1
            T0=Function(Qt)       # Temperature Field t=t^n
            T1=Function(Qt)       # Temperature Field t=t^n+1
            tau0_vec=Function(Z)     # Stress Field (Vector) t=t^n
            tau12_vec=Function(Z)    # Stress Field (Vector) t=t^n+1/2
            tau1_vec=Function(Z)     # Stress Field (Vector) t=t^n+1
            w0= Function(W)
            w12= Function(W)
            ws= Function(W)
            w1= Function(W)
            (u0, D0_vec)=w0.split()
            (u12, D12_vec)=w0.split()
            (us, Ds_vec)=w0.split()
            (u1, D1_vec)=w0.split()
            break


        # Plot solution
        #if t > 0.0:
            #plot(kapp, title="fluctuation", rescale=True, interactive=False)
            #plot(tau1[0,0], title="tau_xx Stress", rescale=True, interactive=False)
            #plot(p1, title="Pressure", rescale=True)
            #plot(rho1, title="Density", rescale=True)
            #plot(u1, title="Velocity", rescale=True, mode = "auto")
            #plot(T1, title="Temperature", rescale=True)
           

        # Move to next time step (Continuation in Reynolds Number)
        t += dt


    # Plot DEVSS Data
    D_proj_cont_xx = project(Dcomp(u1)[0,0], Q)
    D_proj_xx = project(D1_vec[0], Qt)
    Diff_loc_xx = project((D_proj_cont_xx - D_proj_xx)/D_proj_xx.vector().array().max(), Q)
    D_proj_cont_xy = project(Dcomp(u1)[0,1], Qt)
    D_proj_xy = project(D1_vec[1], Q)
    Diff_loc_xy = project((D_proj_cont_xy - D_proj_xy)/D_proj_xy.vector().array().max(), Q)
    D_proj_cont_yy = project(Dcomp(u1)[1,1], Q)
    D_proj_yy = project(D1_vec[2], Qt)
    Diff_loc_yy = project((D_proj_cont_yy - D_proj_yy)/D_proj_yy.vector().array().max(), Q)
    mplot(Diff_loc_xx)
    plt.colorbar()
    plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/DEVSS_xxRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
    plt.clf()
    mplot(Diff_loc_xy)
    plt.colorbar()
    plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/DEVSS_xyRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
    plt.clf()
    mplot(Diff_loc_yy)
    plt.colorbar()
    plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/DEVSS_yyRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
    plt.clf()



    # Plot Mesh Convergence Data 
    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and j==1 or j==4:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'M1')
        plt.plot(x2, ek2, 'b--', label=r'M2')
        plt.plot(x3, ek3, 'c-', label=r'M3')
        plt.plot(x4, ek4, 'm--', label=r'M4')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_k')
        plt.savefig("Compressible Viscoelastic Flow Results/Mesh Convergence/Mesh_KineticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
        plt.clf()
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'M1')
        plt.plot(x2, ee2, 'b--', label=r'M2')
        plt.plot(x3, ee3, 'c-', label=r'M3')
        plt.plot(x4, ee4, 'm--', label=r'M4')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Compressible Viscoelastic Flow Results/Mesh Convergence/Mesh_ElasticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
        plt.clf()




    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and abs(E_k) < 10 and j==4 or j==1:

        # Plot Stress/Normal Stress Difference
        tau_xx=project(tau1[0,0],Q)
        mplot(tau_xx)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Mesh Convergence/tau_xxRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"SUPG"+str(alph1)+".png")
        plt.clf() 
        tau_xy=project(tau1[1,0],Q)
        mplot(tau_xy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Mesh Convergence/tau_xyRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"SUPG"+str(alph1)+".png")
        plt.clf() 
        tau_yy=project(tau1[1,1],Q)
        mplot(tau_yy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_yyRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"SUPG"+str(alph1)+".png")
        plt.clf() 

    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E4 and abs(E_k) < 10 and j==4 or j==1:
 
       # Plot Velocity Components
        ux=project(u1[0],Q)
        mplot(ux)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Mesh Convergence/u_xRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"SUPG"+str(alph1)+".png")
        plt.clf()
        uy=project(u1[1],Q)
        mplot(uy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Mesh Convergence/u_yRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"SUPG"+str(alph1)+".png")
        plt.clf()

    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and abs(E_k) < 10 and j==1 or j==4 or j==1:


        # Matlab Plot of the Solution at t=Tf
        #p1=mu_0*(L/U)*p1  #Dimensionalised Pressure
        #p1=project(p1,Q)
        #mplot(p1)
        #plt.colorbar()
        #plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/PressureRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(c1)+".png")
        #plt.clf()


        #Plot Contours USING MATPLOTLIB
        # Scalar Function code


        x = Expression('x[0]', degree=2)  #GET X-COORDINATES LIST
        y = Expression('x[1]', degree=2)  #GET Y-COORDINATES LIST
        pvals = p1.vector().array() # GET SOLUTION p= p(x,y) list
        Tvals = T1.vector().array() # GET SOLUTION T= T(x,y) list
        xyvals = mesh.coordinates()     # CLEAN THIS UP!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        xvalsq = interpolate(x, Q)#xyvals[:,0]
        yvalsq= interpolate(y, Q)#xyvals[:,1]
        xvalsw = interpolate(x, Qt)#xyvals[:,0]
        yvalsw= interpolate(y, Qt)#xyvals[:,1]

        xvals = xvalsq.vector().array()
        yvals = yvalsq.vector().array()


        xx = np.linspace(0,1)
        yy = np.linspace(0,1)
        XX, YY = np.meshgrid(xx,yy)   # (x,y) coordinate data formatted so that it can be used by plt.contour()
        pp = mlab.griddata(xvals, yvals, pvals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 


        plt.contour(XX, YY, pp, 25)
        plt.title('Pressure Contours')   # PRESSURE CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/PressureContoursRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"SUPG"+str(alph1)+".png")
        plt.clf()

        xvals = xvalsw.vector().array()
        yvals = yvalsw.vector().array()




    """if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and abs(E_k) < 10 and j==3 or j==6:

        #Plot Contours USING MATPLOTLIB
        # Vector Function code

        u1=project(u1,V)
        gu=list()
        hu=list()
        n= mesh.num_vertices() 
        for i in range(len(u1.vector().array())/2-1):
            gu.append(u1.vector().array()[2*i+1])
            hu.append(u1.vector().array()[2*i])

        uvals = np.asarray(hu) # GET SOLUTION (u,v) -> u= u(x,y) list
        vvals = np.asarray(gu) # GET SOLUTION (u,v) -> v= v(x,y) list


        xy = Expression(('x[0]','x[1]'), degree=2)  #GET MESH COORDINATES LIST
        xyvalsv = interpolate(xy, V)

        qu=list()
        ru=list()

        for i in range(len(u1.vector().array())/2-1):
           qu.append(xyvalsv.vector().array()[2*i+1])
           ru.append(xyvalsv.vector().array()[2*i])

        xvals = np.asarray(ru)
        yvals = np.asarray(qu)

        # Interpoltate velocity field data onto matlab grid
        uu = mlab.griddata(xvals, yvals, uvals, xx, yy, interp='nn') 
        vv = mlab.griddata(xvals, yvals, vvals, xx, yy, interp='nn') 

        #Determine Speed 
        speed = np.sqrt(uu*uu+ vv*vv)

        plot3 = plt.figure()
        plt.streamplot(XX, YY, uu, vv,  
                       density=3,              
                       color=speed,  
                       cmap=cm.gnuplot,                         # colour map
                       linewidth=0.8)                           # line thickness
                                                                # arrow size
        plt.colorbar()                                          # add colour bar on the right
        plt.title('Lid Driven Cavity Flow')
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/VelocityContoursRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"SUPG"+str(alph1)+".png")   
        plt.clf()                                               # display the plot"""


    plt.close()


    if dt < tol:
       j=loopend+1
       break

    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and abs(E_k) < 10:
        jj=0
        Tf=T_f  
        alph1 = 1.0
        alph2 = 0.01
        alph3 = 0.005
        th = 1.0  





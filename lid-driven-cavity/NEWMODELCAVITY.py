"""Compressible Lid Driven Cavity Problem"""

"""SIMULATION OF NEW FENE-P TYPE MODEL"""
"""Solution Method: COMPRESSIBLE TAYLOR GALERKIN METHOD w/ Finite Element Method using DOLFIN (FEniCS)"""

"""Skewed Finite Element Mesh Used for Improved Perofrmance"""


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

# Define Geometry
B=1
L=1
x0 = 0.0
y0 = 0.0
x1 = B
y1 = L
mm=80
nx=mm*B
ny=mm*L

c = min(x1-x0,y1-y0)

mesh= RectangleMesh(Point(x0,y0), Point(x1, y1), nx, ny)



#SKEW MESH FUNCTION

# MESH CONSTRUCTION CODE

nv= mesh.num_vertices()
nc= mesh.num_cells()
coorX = mesh.coordinates()[:,0]
coorY = mesh.coordinates()[:,1]
cells0 = mesh.cells()[:,0]
cells1 = mesh.cells()[:,1]
cells2 = mesh.cells()[:,2]

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

# Mesh Refine Code

for i in range(0):
      g = (max(x1,y1)-max(x0,y0))*0.1/(i+1)
      cell_domains = CellFunction("bool", mesh)
      cell_domains.set_all(False)
      for cell in cells(mesh):
          x = cell.midpoint()
          if  (x[0] < x0+g and x[1] > y1-g) or (x[0] > x1-g and x[1] > y1-g): # or (x[0] < x0+g and x[1] < y0+g)  or (x[0] > x1-g and x[1] < g): 
              cell_domains[cell]=True
      #plot(cell_domains, interactive=True)
      mesh = refine(mesh, cell_domains, redistribute=True)



#Define Boundaries 


class Omega0(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[0] < DOLFIN_EPS and on_boundary else False 
                                                                          
class Omega1(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[1] > L*(1.0 - DOLFIN_EPS) and on_boundary  else False   


class Omega2(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[0] > B*(1.0 - DOLFIN_EPS)  and on_boundary else False 


class Omega3(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[1] < DOLFIN_EPS and on_boundary else False   

omega1= Omega1()
omega0= Omega0()
omega2= Omega2()
omega3= Omega3()


# MARK SUBDOMAINS (Create mesh functions over the cell facets)
sub_domains = MeshFunction("size_t", mesh1, mesh.topology().dim() - 1)
sub_domains.set_all(5)
omega0.mark(sub_domains, 0)
omega1.mark(sub_domains, 2)
omega2.mark(sub_domains, 3)
omega3.mark(sub_domains, 4)

plot(sub_domains, interactive=False)

# Define function spaces (P2-P1)
d=2

V = VectorFunctionSpace(mesh1, "CG", d)
Q = FunctionSpace(mesh1, "CG", d)
W = FunctionSpace(mesh1, "CG", d)
Z = TensorFunctionSpace(mesh1, "DG", 1)
Zc = TensorFunctionSpace(mesh1, "CG", 1)

"""START LOOP that runs the simulation for a range of parameters"""
"""Ensure that break control parameters are re-adjusted if solution diverges"""
"""ADJUSTIBLE PARAMETERS"""


l1ref=5.0*10E-2
U =  1                    # Characteristic velocity
lambda1=l1ref/U                 # Relaxation Time


#gammah=2*10E-10                  # SUPG Stabilsation Terms
#gam =0.01


thetat = 1.0*10E-20

dt = 0.005  #Time Stepping  
Tf=5
loopend=5
j=0
jj=0
tol=10E-6


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

#Start Solution Loop
while j < loopend:
    j+=1

    # Define trial and test functions
    u = TrialFunction(V)
    rho=TrialFunction(Q)
    p = TrialFunction(Q)
    T = TrialFunction(W)
    v = TestFunction(V)
    q = TestFunction(Q)
    r = TestFunction(W)
    tau = TrialFunction(Z)
    R = TestFunction(Z)


    #Define Discretised Functions
    u00=Function(V) 
    u0=Function(V)       # Velocity Field t=t^n
    us=Function(V)       # Predictor Velocity Field 
    u12=Function(V)      # Velocity Field t=t^n+1/2
    u1=Function(V)       # Velocity Field t=t^n+1
    irho0=Function(Q)
    rho0=Function(Q)     # Density Field t=t^n
    rho1=Function(Q)     # Density Field t=t^n+1
    irho0=Function(Q)
    irho1=Function(Q)
    p00=Function(Q)      # Pressure Field t=t^n-1
    p0=Function(Q)       # Pressure Field t=t^n
    p1=Function(Q)       # Pressure Field t=t^n+1
    mu=Function(W)       # Viscosity Field t=t^n
    T00=Function(W) 
    T0=Function(W)       # Temperature Field t=t^n
    T1=Function(W)       # Temperature Field t=t^n+1
    tau00=Function(Z)
    tau0=Function(Z)     # Stress Field t=t^n
    tau12=Function(Z)    # Stress Field t=t^n+1/2
    tau1=Function(Z)     # Stress Field t=t^n+1

    c0c0=Function(Q)
    tauxx=Function(W)    # Normal Stress

    D=TrialFunction(Zc)   # DEVSS Articficial Diffusion 
    D1=Function(Zc)       # Terms



    #print len(tau1.vector().array())/4
    #print len(tauxx.vector().array())

    #tauxx_vec=tauxx.vector().array()
    #for i in range(len(tauxx.vector().array())):
    #    tauxx_vec[i]=tau1.vector().array()[4*i]
    #tauxx.vector()[:]=tauxx_vec

    #quit()


    boundary_parts = FacetFunction("size_t", mesh1)
    omega0.mark(boundary_parts,0)
    omega1.mark(boundary_parts,1)
    omega2.mark(boundary_parts,2)
    omega3.mark(boundary_parts,3)
    ds = Measure("ds")[boundary_parts]


    # Set parameter values
    h = mesh1.hmin()
    #print(h) 
    #Tf = 5    #Final Time
    Cv = 1000.0
    #Uv=Expression(('0.5*(1+tanh(5*t-4))','0'), t=0.0 ,U=U, d=d, degree=d)
    mu_1 = 10.0*10E-1
    mu_2 = 90.0*10E-1
    mu_0 = mu_1+mu_2
    Rc = 3.33*10E1
    T_0 = 300.0 
    T_h = 350.0      #Reference temperature
    C=250.0 #Sutherland's Constant
    kappa = 2.0
    heatt= 0.00
    rho_0=100.0
    Vh=0.01   #Viscous Heating Number
    #lambda1=2.0*10E-2            #Relaxation Time
    kappa = 2.0
    heatt= 0.1
    beta = 69*10E-2               # Thermal Expansion Coefficient
    betav = mu_1/mu_0
    alpha=1.0/(rho_0*Cv)
    Bi=0.75
    ms=1.0                          # Equation of State Parameter
    Bs=20000.0                       # Equation of State Parameter
    #c0c0=ms*(p0+Bs)*irho0         # Speed of Sound Squared (Dynamic)
    c0=500.0                       # Speed of Sound (Static)
    k = Constant(dt)

    # Nondimensional Parameters

    Re = rho_0*U*c/mu_0                             # Reynolds Number
    We = lambda1*U/c                                # Weisenberg NUmber
    Di=kappa/(rho_0*Cv*U*c)                         # Diffusion Number
    Vh= U*mu_0/(rho_0*Cv*c*(T_h-T_0))               # Viscous Heating Number
    #c0nd=c0/U                                       # Nondimensionalised Speed of Sound
    al=0                                            # Nonisothermal Parameter

    """The Following to are routines for comparing non-inertail flow with inertial flow OR Different Weissenberg Numbers at Re=0"""

    # Comparing different Speed of Sound NUMBERS Numbers (We=0.1,0.2,0.3,0.4,0.5) at Re=0, We=0.1
    """conv=1                                      # Non-inertial Flow Parameter (Re=0)
    We=0.2
    Re=50
    if j==1:
        c0=1500
    elif j==2:
       c0=1250
    elif j==3:
       c0=1000
    elif j==4:
       c0=750
    elif j==5:
       c0=500"""

    # Comparing different WEISSENBERG Numbers (We=0.1,0.2,0.3,0.4,0.5) at Re=50
    conv=10E-20                                      # Non-inertial Flow Parameter (Re=0)
    Re=1.0
    if j==1:
       We=0.1
    elif j==2:
       We=0.2
    elif j==3:
       We=0.3
    elif j==4:
       We=0.4
    elif j==5:
       We=0.5


    # Comparing different REYNOLDS NUMBERS Numbers (Re=0,5,10,25,50) at We=0.5
    """conv=1                                      # Non-inertial Flow Parameter (Re=0)
    We=0.5
    if j==1:
       conv=10E-8
       Re=1
    elif j==2:
       Re=5
    elif j==3:
       Re=10
    elif j==4:
       Re=25
    elif j==5:
       Re=50"""

    # Continuation in Reynolds Number (Re-->10Re)
    Ret=Expression('Re*(1.0+0.5*(1.0+tanh(0.3*t-4.0))*19.0)', t=0.0, Re=Re, d=d, degree=d)
    Rey=Re

    # Stabilisation Parameters
    theta = 1.0*10E-3           # DEVSS Stabilisation Terms
    c1=0.1
    c2=0.1
    c3=0.05
    Wet=Expression('We*0.5*(1.0+tanh(8*t-16.0))', t=0.0, We=We, d=d, degree=d)


    # Define boundary FUNCTIONS
    td= Constant('5')
    e = Constant('6')
    ulid=Expression(('8*(1.0+tanh(8*t-4.0))*(x[0]*(L-x[0]))*(x[0]*(L-x[0]))','0'), d=d, degree=d, t=0.0, U=U, L=L, e=e, T_0=T_0, T_h=T_h) # Lid Speed 
    T_bl = Expression('T_0+(T_h-T_0)*sin((x[0]+x[1])*2*pi/L)', d=d, degree=d, T_0=T_0, T_h=T_h, pi=pi, L=L, B=B)
    T_bb = Expression('T_0+(T_h-T_0)*sin((x[0]+x[1])*2*pi/B)', d=d, degree=d, T_0=T_0, T_h=T_h, pi=pi, L=L, B=B)
    h_skew = Expression('cos(pi*x[0])-cos(pi*(x[0]+1/mm))','cos(pi*x[1])-cos(pi*(x[1]+1/mm))', d=d, degree=d, pi=pi, mm=mm, L=L, B=B)             # Mesh size function
    h_k = Expression('1/mm','1/mm', d=d, degree=d, mm=mm, L=L, B=B)
    

    """NOTE: The Mesh size function uses the skewmesh [0.5*(1-cos(pi*x))]"""

    # Define unit Normal/tangent Vector at inner and outer Boundary (Method 2)
    n0 =  Expression(('-1' , '0' ), d=d, degree=d)
    n1 =  Expression(('0' , '1' ), d=d, degree=d)
    n2 =  Expression(('1' , '0' ), d=d, degree=d)
    n3 =  Expression(('0' , '-1' ), d=d, degree=d)

     # Dirichlet Boundary Conditions  (LID DRIVEN CAVITY)
    noslip0  = DirichletBC(V, (0.0, 0.0), omega0)  # No Slip boundary conditions on the left wall
    drive1  =  DirichletBC(V, ulid, omega1)  # No Slip boundary conditions on the upper wall
    noslip2  = DirichletBC(V, (0.0, 0.0), omega2)  # No Slip boundary conditions on the right part of the flow wall
    noslip3  = DirichletBC(V, (0.0, 0.0), omega3)  # No Slip boundary conditions on the left part of the flow wall
    #slip  = DirichletBC(V, sl, omega0)  # Slip boundary conditions on the second part of the flow wall 
    temp0 =  DirichletBC(W, T_0, omega0)    #Temperature on Omega0 
    temp2 =  DirichletBC(W, T_0, omega2)    #Temperature on Omega2 
    temp3 =  DirichletBC(W, T_0, omega3)    #Temperature on Omega3 


    #Collect Boundary Conditions
    bcu = [noslip0, drive1, noslip2, noslip3]
    bcp = []
    bcT = [temp0, temp2]
    bctau = []

    N= len(p0.vector().array())

    # Print Parameters of flow simulation
    t = 0.0                  #Time
    e=6

    print '############# Fluid Characteristics ############'
    print 'Density', rho_0
    print 'Solvent Viscosity (Pa.s)', mu_1
    print 'Polymeric Viscosity (Pa.s)', mu_2
    print 'Total Viscosity (Pa.s)', mu_0
    print 'Relaxation Time (s)', lambda1
    print 'Heat Capacity', Cv
    print 'Thermal Conductivity', kappa

    print '############# TIME SCALE ############'
    print 'Timestep size (s):', dt
    print 'Finish Time (s):', Tf

    print '############# Scalings & Nondimensional Parameters ############'
    print 'Characteristic Length (m):', L
    print 'Characteristic Velocity (m/s):', U
    print 'Lid velocity:', (U*0.5*(1.0+tanh(e*t-3.0)),0)
    print 'Speed of sound (m/s):', c0
    #print 'Nondimensionalised Speed of Sound', c0nd
    print 'Reynolds Number:', Rey
    print 'Non-inertial parameter:', conv
    print 'Weissenberg Number:', We
    print 'Viscosity Ratio:', betav
    print 'Diffusion Number:' ,Di
    print 'Viscous Heating Number:', Vh


    print '############# Discrete Space Characteristics ############'
    print 'Degree of Elements', d
    print 'Mesh: %s x %s' %(mm, mm)
    print('Size of FE Space = %d x d' % N)
    print 'Number of Cells:', mesh.num_cells()
    print 'Number of Vertices:', mesh.num_vertices()
    print '############# Stabilisation Parameters ############'
    print 'DEVSS Momentum Term:', theta
    print 'DEVSS Temperature Term:', thetat
    quit()

    # Initial Density Field
    rho_array = rho0.vector().array()
    for i in range(len(rho_array)):  
        rho_array[i] = 1.0
    rho0.vector()[:] = rho_array 

    # Initial Reciprocal of Density Field
    irho_array = irho0.vector().array()
    for i in range(len(irho_array)):  
        irho_array[i] = 1/rho_array[i]
    irho0.vector()[:] = irho_array 

    # Initial Temperature Field
    T_array = T0.vector().array()
    for i in range(len(T_array)):  
        T_array[i] = T_0
    T0.vector()[:] = T_array

      
    I = Expression((('1.0','0.0'),
                    ('0.0','1.0')), d=d, degree=d)


    #Define Variable Parameters, Strain Rate and other tensors
    sr0 = 0.5*(grad(u0) + transpose(grad(u0)))
    sr1 = 0.5*(grad(u1) + transpose(grad(u1)))
    sr12 = 0.5*(grad(u12) + transpose(grad(u12)))
    sr = 0.5*(grad(u) + transpose(grad(u)))
    srv = 0.5*(grad(v) + transpose(grad(v)))
    F0 = (grad(u0)*C0 + C0*transpose(grad(u0)))
    F12 = (grad(u12)*C + C*transpose(grad(u12)))
    F12e = (grad(u12)*C12 + C12*transpose(grad(u12)))
    F1 = (grad(u1)*C + C*transpose(grad(u1)))
    gamdots = inner(sr1,grad(u1))
    gamdots12 = inner(sr12,grad(u12))
    gamdotp = inner(tau1,grad(u1))
    gamdotp12 = inner(tau12,grad(u12))
    thetal = (T)/(T_h-T_0)
    thetar = (T_0)/(T_h-T_0)
    thetar = project(thetar,W)
    theta0 = (T0-T_0)/(T_h-T_0)
    alpha = 1.0/(rho*Cv)

    weta = We/dt                                                  #Ratio of Weissenberg number to time step

    # Artificial Diffusion Term
    o= tau1.vector()-tau0.vector()                         # Stress Difference per timestep
    h= p1.vector()-p0.vector()
    m=u1.vector()-u0.vector()                              # Velocity Difference per timestep
    l=T1.vector()-T0.vector()
    alt=norm(o)/(norm(tau1.vector())+10E-10)
    alp=norm(h)/(norm(p1.vector())+10E-10)
    alu=norm(m)/(norm(u1.vector())+10E-10)
    alT=norm(l, 'linf')/(norm(T1.vector(),'linf')+10E-10)
    epstau = alt*betav+10E-8                                    #Stabilisation Parameter (Stress)
    epsp = alp*betav+10E-8                                      #Stabilisation Parameter (Pressure)
    epsu = alu*betav+10E-8                                      #Stabilisation Parameter (Stress)
    epsT = 0.1*alT*kappa+10E-8                                  #Stabilisation Parameter (Temperature)

    # TAYLOR GALERKIN METHOD (COMPRESSIBLE VISCOELASTIC)

    # Weak Formulation (DEVSS in weak formulation)

    """DEVSS Stabilisation used in 'Half step' and 'Velocity Update' stages of the Taylor Galerkin Scheme"""




    # TAYLOR GALERKIN METHOD (COMPRESSIBLE VISCOELASTIC)

    """SUPG Diffusion coefficient used in 

       Finite Element Solution of the Navier-Stokes Equations using a SUPG Formulation
       -Vellando, Puertas Agudo, Marques  (Page 12)"""


    """def al1(x):               # Define Diffusion coeficient function
        return np.tanh(2.0*x)
    f1=Expression(('1','0'), degree=d, d=d)    
    f2=Expression(('0','1'), degree=d, d=d)

  
    speed = dot(u0,u0)        # Determine Speed 
    speed = project(speed,Q)  # Project Speed onto FE Space 



    uvals = dot(f1,u0)
    vvals = dot(f2,u0)
    uval = project(uvals,Q)
    vval = project(vvals,Q)

    hxi = (1.0/mm)                 # Uniform Mesh Length of x-axis side   
    heta = (1.0/mm)                # Uniform Mesh Length of y-axis side

    eta=Function(Q)            # Define Approximation Rule Functions
    xi=Function(Q)

    u_array = uval.vector().array()
    v_array = vval.vector().array()
    eta_array = eta.vector().array()
    xi_array = xi.vector().array()
    for i in range(len(u_array)):  
        eta_array[i] = al1(0.5*u_array[i]/mu_0)
        xi_array[i] = al1(0.5*v_array[i]/mu_0)
    eta.vector()[:] = eta_array  
    xi.vector()[:] = xi_array

    eta=project(eta,Q)
    xi=project(xi,Q)

    gam = gammah*(hxi*xi*uval+heta*eta*vval)/(2.0*speed+10E-8)
    gam = project(gam,Q)"""
    #gam = 0.0

    # SUPG Term
    #vm=v+gam*dot(u0,grad(v))

    # DEVSS STABILISATION

    #Momentum Equation Stabilisation DEVSS

    D=2.0*(sr0)-2.0/3*div(u0)*I
    D=project(D,Zc)

    #Temperature Equation Stabilisation DEVSS
    Dt=grad(T0)
    Dt=project(Dt,V) 

    #Mixed Finite Element Space
    #VZc=FunctionSpace(mesh1, V*Zc)
    #(vm, Rm)=TestFunction(VZc)
    #(u,D)=Function(VZc)

    #gam=0.0


    #Half Step
    a1=(1.0/(dt/2.0))*inner(Rey*rho0*u,v)*dx+betav*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx)#+(1-betav)*c1*inner(h_skew*sr,srv)*dx#\
         #+ theta*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx)
    L1=(1.0/(dt/2.0))*inner(Rey*rho0*u0,v)*dx+inner(p0,div(v))*dx-inner(tau0,grad(v))*dx-conv*inner(Rey*rho0*grad(u0)*u0,v)*dx #\
         #+ theta*inner(D,grad(v))*dx

    #Predicted U* Equation
    a2=(1.0/dt)*inner(Rey*rho0*u,v)*dx + theta*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx)#+(1-betav)*c1*inner(h_skew*sr,srv)*dx
    L2=(1.0/dt)*inner(Rey*rho0*u0,v)*dx-0.5*betav*(inner(grad(u0),grad(v))*dx+1.0/3*inner(div(u0),div(v))*dx) \
        +inner(p0,div(v))*dx-inner(tau0,grad(v))*dx-conv*inner(Rey*rho0*grad(u12)*u12,v)*dx #\
        #+ theta*inner(D,grad(v))*dx

    # Stress Half Step (CONFORMATION TENSOR)
    a3 = (2*We/dt)*inner(C,R)*dx + We*(conv*inner(dot(u12,grad(C)),R)*dx - inner(F12, R)*dx+inner(div(u12)*C,R)*dx)#+We*c2*inner(h_skew*grad(tau),grad(R))*dx+We*c3*inner(h_skew*div(tau),div(R))*dx
    L3 = (2*We/dt-1.0)*inner(C0,R)*dx + 2.0*(1.0-betav)*inner(sr0,R)*dx-(inner(f*C0,R)*dx+inner(I,R)*dx) 

    # Temperature Update (Half Step)
    a8 = (2.0/dt)*inner(rho1*thetal,r)*dx + Di*inner(grad(thetal),grad(r))*dx + inner(rho1*dot(u12,grad(thetal)),r)*dx 
    L8 = (2.0/dt)*inner(rho1*thetar,r)*dx + Di*inner(grad(thetar),grad(r))*dx + inner(rho1*dot(u12,grad(thetar)),r)*dx \
          + (2.0/dt)*inner(rho1*theta0,r)*dx + Vh*(inner(gamdots,r)*dx + inner(gamdotp,r)*dx - inner(p0*div(u0),r)*dx) - Di*Bi*inner(theta0,r)*ds(1) \
          + thetat*(inner(grad(thetar),grad(r))*dx+inner(Dt,grad(r))*dx)
          #+ inner(,r)*dx  #Neumann Condition on the outer bearing is encoded in the weak formulation


    #Continuity Equation 1
    a5=(1.0/(c0*c0*dt))*inner(p,q)*dx+0.5*dt*inner(grad(p),grad(q))*dx   #Using Dynamic Speed of Sound (c=c(x,t))
    L5=(1.0/(c0*c0*dt))*inner(p0,q)*dx+0.5*dt*inner(grad(p0),grad(q))*dx-(inner(rho0*div(us),q)*dx+inner(dot(grad(rho0),us),q)*dx)

    #Continuity Equation 2 
    a6=c0*c0*inner(rho,q)*dx 
    L6=c0*c0*inner(rho0,q)*dx + inner(p1-p0,q)*dx 

    #Velocity Update
    a7=(1.0/dt)*inner(Rey*rho0*u,v)*dx+0.5*betav*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx)#+(1-betav)*c1*inner(h_skew*sr,srv)*dx#\
         #+ theta*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx)
    L7=(1.0/dt)*inner(Rey*rho0*us,v)*dx+0.5*(inner(p1,div(v))*dx-inner(p0,div(v))*dx) #+ theta*inner(D,grad(v))*dx


    # Stress Half Step (CONFORMATION TENSOR)
    a4 = (2*We/dt)*inner(C,R)*dx + We*(conv*inner(dot(u12,grad(C)),R)*dx - inner(F12, R)*dx+inner(div(u12)*C,R)*dx)#+We*c2*inner(h_skew*grad(tau),grad(R))*dx+We*c3*inner(h_skew*div(tau),div(R))*dx
    L4 = (2*We/dt-1.0)*inner(C0,R)*dx + 2.0*(1.0-betav)*inner(sr0,R)*dx-(inner(f*C0,R)*dx+inner(I,R)*dx) 




    # Temperature Update (Full Step)
    a9 = (1.0/dt)*inner(rho1*thetal,r)*dx + Di*inner(grad(thetal),grad(r))*dx + inner(rho1*dot(u1,grad(thetal)),r)*dx 
    L9 = (1.0/dt)*inner(rho1*thetar,r)*dx + Di*inner(grad(thetar),grad(r))*dx + inner(rho1*dot(u1,grad(thetar)),r)*dx \
          + (1.0/dt)*inner(rho1*theta0,r)*dx + Vh*(inner(gamdots12,r)*dx + inner(gamdotp12,r)*dx-inner(p1*div(u1),r)*dx) - Di*Bi*inner(theta0,r)*ds(1) \
          + thetat*(inner(grad(thetar),grad(r))*dx+inner(Dt,grad(r))*dx)
          #+ inner(,r)*dx  #Neumann Condition on the outer bearing is encoded in the weak formulation



    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)
    A4 = assemble(a4)
    A5 = assemble(a5)
    A6 = assemble(a6)
    A7 = assemble(a7)
    A8 = assemble(a8)


    # Use amg preconditioner if available
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

    # Use nonzero guesses - essential for CG with non-symmetric BC
    parameters['krylov_solver']['nonzero_initial_guess'] = True

    #Folder To Save Plots for Paraview
    #fv=File("Velocity Results Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"theta"+str(theta)+"c0="+str(c0)+"/velocity "+str(t)+".pvd")
 
    #Lists for Energy Values
    x=list()
    ee=list()
    ek=list()


    # Time-stepping
    t = dt
    iter = 0            # iteration counter
    maxiter = 100000
    while t < Tf + DOLFIN_EPS and iter < maxiter:
        iter += 1
        print"t = %s,  Iteration = %d, Convergence Failures = %s, Loop = %s" %(t, iter, jj, j)


        """if iter >1:

            speed = dot(u0,u0)        # Determine Speed 
            speed = project(speed,Q)  # Project Speed onto FE Space 
            uvals = dot(f1,u0)
            vvals = dot(f2,u0)
            uval = project(uvals,Q)
            vval = project(vvals,Q)

            u_array = uval.vector().array()
            v_array = vval.vector().array()
            eta_array = eta.vector().array()
            xi_array = xi.vector().array()
            for i in range(len(u_array)):  
                eta_array[i] = al1(u_array[i])
                xi_array[i] = al1(v_array[i])
            eta.vector()[:] = eta_array  
            xi.vector()[:] = xi_array

            eta=project(eta,Q)
            xi=project(xi,Q)

            gam = gammah*(hxi*xi*uval+heta*eta*vval)/(2.0*speed+10E-20)
            gam = project(gam,Q)"""

            #print gam.vector().array()
            
            #print norm(gam.vector(), 'linf')
        D=2.0*(sr0)-2.0/3*div(u0)*I
        D=project(D,Zc)
        

        # Velocity Half Step
        A1 = assemble(a1)
        b1= assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, u12.vector(), b1, "bicgstab", "default")
        end()
        
        D=2.0*(sr12)-2.0/3*div(u12)*I
        D=project(D,Zc)
        
        #Compute Predicted U* Equation
        A2 = assemble(a2)
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcu]
        solve(A2, us.vector(), b2, "bicgstab", "default")
        end()
        #print(norm(us.vector(),'linf'))

        # Stress Half STEP
        #A3=assemble(a3)
        #b3=assemble(L3)
        #[bc.apply(A3, b3) for bc in bctau]
        #solve(A3, tau12.vector(), b3, "bicgstab", "default")
        #end()


        #Continuity Equation 1
        A5 = assemble(a5)
        b5 = assemble(L5)
        [bc.apply(A5, b5) for bc in bcp]
        #[bc.apply(p1.vector()) for bc in bcp]
        solve(A5, p1.vector(), b5, "bicgstab", "default")
        end()


        #Continuity Equation 2
        rho1=rho0+(p1-p0)/(c0*c0)
        rho1=project(rho1,Q)


        #Velocity Update
        A7 = assemble(a7)
        b7 = assemble(L7)
        [bc.apply(A7, b7) for bc in bcu]
        solve(A7, u1.vector(), b7, "bicgstab", "default")
        end()

        # Stress Full Step
        A4=assemble(a4)
        b4=assemble(L4)
        [bc.apply(A4, b4) for bc in bctau]
        solve(A4, tau1.vector(), b4, "bicgstab", "default")
        end()

        #Temperature Half Step
        #A8 = assemble(a8)
        #b8 = assemble(L8)
        #[bc.apply(A8, b8) for bc in bcT]
        #solve(A8, T12.vector(), b9, "bicgstab", "default")
        #end()

        #Temperature Full Step
        #A9 = assemble(a9)
        #b9 = assemble(L9)
        #[bc.apply(A9, b9) for bc in bcT]
        #solve(A9, T1.vector(), b9, "bicgstab", "default")
        #end()

        # First Normal Stress Difference
        #tau_xx=project(tau1[0,0],Q)
        #tau_xy=project(tau1[1,0],Q)
        #tau_yy=project(tau1[1,1],Q)

        #print 'Stress Norm:', norm(tau1.vector(),'linf')
        #print '12 Stress Norm:', norm(tau12.vector(),'linf')
        #print 'Velocity Norm:', norm(u1.vector(),'linf')

        # Energy Calculations
        E_k=assemble(0.5*dot(u1,u1)*dx)
        E_e=assemble((tau1[0,0]+tau1[1,1])*dx)

        


        # Calculate Size of Artificial Term
        o= tau1.vector()-tau0.vector()                         # Stress Difference per timestep
        h= p1.vector()-p0.vector()
        m=u1.vector()-u0.vector()                              # Velocity Difference per timestep
        l=T1.vector()-T0.vector()



        # Record Error Data 
        
        #if iter > 1:
           #x.append(t)
           #y.append(norm(h,'linf')/norm(p1.vector()))
           #z.append(norm(o,'linf')/(norm(tau1.vector())+0.0001))
           #zz.append(norm(m,'linf')/norm(u1.vector()))
           #zzz.append(norm(l,'linf')/(norm(u1.vector())+0.0001))

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

        # Record Elastic & Kinetic Energy Values (Method 2)
        """x.append(t)
        ek.append(E_k)
        ee.append(E_e)"""
        

        # Save Plot to Paraview Folder 
        #for i in range(5000):
            #if iter== (0.02/dt)*i:
               #fv << u1
        #ft << T1

        # Break Loop if code is diverging

        if max(norm(T1.vector(), 'linf'),norm(p1.vector(), 'linf')) > 10E5 or np.isnan(sum(T1.vector().array())) or abs(E_k) > 0.02:
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
            #dt=dt/2                        # Use Smaller timestep 
            j-=1                            # Extend loop
            jj+= 1                          # Convergence Failures
            Tf= (iter-10)*dt
            break


        # Plot solution
        #if t>0.0:
            #plot(tauxx, title="Normal Stress", rescale=True)
            #plot(p1, title="Pressure", rescale=True)
            #plot(rho1, title="Density", rescale=True)
            #plot(u1, title="Velocity", rescale=True, mode = "auto")
            #plot(T1, title="Temperature", rescale=True)
        

           

        # Move to next time step
        u0.assign(u1)
        T0.assign(T1)
        rho0.assign(rho1)
        p0.assign(p1)
        tau0.assign(tau1)
        #Uv.t=t
        ulid.t=t
        Ret.t=t
        Wet.t=t
        t += dt


    # PLOTS


    # Plot Convergence Data 
    """if max(norm(T1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) > 10E6 and dt < 0.01:
        fig1=plt.figure()
        plt.plot(x, y, 'r-', label='Pressure Timestep Error')
        plt.xlabel('time(s)')
        plt.ylabel('||p1-p0||/||p1||')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/PressureTimestepErrorRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"dt="+str(dt)+".png")
        plt.clf()
        plt.plot(x, z, 'r-', label='Stress Timestep Error')
        plt.xlabel('time(s)')
        plt.ylabel('||S1-S0||/||S1||')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/StressCovergenceRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"dt="+str(dt)+".png")
        plt.clf()
        plt.plot(x, zz, 'g-', label='Velocity Field Timestep Error')
        plt.xlabel('time(s)')
        plt.ylabel('||u1-u0||/||u1||')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/VelocityCovergenceRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"dt="+str(dt)+".png")
        plt.clf()
        plt.plot(x, y, 'g-', label='Velocity Field Timestep Error')
        plt.xlabel('time(s)')
        plt.ylabel('||T1-T0||/||T1||')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/TemperatureCovergenceRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"dt="+str(dt)+".png")
        plt.clf() """

        #Plot Kinetic and elasic Energies for different REYNOLDS numbers at constant Weissenberg Number    
    """if max(norm(T1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E6 and j==5:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'$Re=0$')
        plt.plot(x2, ek2, 'b-', label=r'$Re=5$')
        plt.plot(x3, ek3, 'c-', label=r'$Re=10$')
        plt.plot(x4, ek4, 'm-', label=r'$Re=25$')
        plt.plot(x5, ek5, 'g-', label=r'$Re=50$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_k')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/We0p5KineticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"dt="+str(dt)+".png")
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'$Re=0$')
        plt.plot(x2, ee2, 'b-', label=r'$Re=5$')
        plt.plot(x3, ee3, 'c-', label=r'$Re=10$')
        plt.plot(x4, ee4, 'm-', label=r'$Re=25$')
        plt.plot(x5, ee5, 'g-', label=r'$Re=50$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/We0p5ElasticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"dt="+str(dt)+".png")
        plt.clf()"""

        #Plot Kinetic and elasic Energies for different Speed of sound numbers at constant Weissenberg & Reynolds Numbers    
    """if max(norm(T1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E6 and j==5:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'$c_0=1500$')
        plt.plot(x2, ek2, 'b-', label=r'$c_0=1250$')
        plt.plot(x3, ek3, 'c-', label=r'$c_0=1000$')
        plt.plot(x4, ek4, 'm-', label=r'$c_0=750$')
        plt.plot(x5, ek5, 'g-', label=r'$c_0=500$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_k')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/c0KineticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"dt="+str(dt)+".png")
        plt.clf()
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'$c_0=1500$')
        plt.plot(x2, ee2, 'b-', label=r'$c_0=1250$')
        plt.plot(x3, ee3, 'c-', label=r'$c_0=1000$')
        plt.plot(x4, ee4, 'm-', label=r'$c_0=750$')
        plt.plot(x5, ee5, 'g-', label=r'$c_0=500$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/c0ElasticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"dt="+str(dt)+".png")
        plt.clf()"""

        #Plot Kinetic and elasic Energies for different Weissenberg numbers at Re=0 (METHOD 1)  
    """if max(norm(T1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E6:
        if j==1:
           col='r-'
        if j==2:
           col='b-'
        if j==3:
           col='c-'
        if j==4:
           col='m-'
        if j==5:
           col='g-'
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x, ek, col, label=r'$We=%s'%We)
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_k')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/KineticEnergyTf="+str(Tf)+"b="+str(betav)+"c0="+str(c0)+"dt="+str(dt)+".png")
        # Elastic Energy
        plt.figure(1)
        plt.plot(x, ee, col, label=r'$We=%s'%We)
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/ElasticEnergyTf="+str(Tf)+"b="+str(betav)+"c0="+str(c0)+"dt="+str(dt)+".png")
        plt.clf()"""

        #Plot Kinetic and elasic Energies for different Weissenberg numbers at Re=0 (METHOD 2)
    if max(norm(T1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E6 and j==5:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'$We=0.1$')
        plt.plot(x2, ek2, 'b-', label=r'$We=0.2$')
        plt.plot(x3, ek3, 'c-', label=r'$We=0.3$')
        plt.plot(x4, ek4, 'm-', label=r'$We=0.4$')
        plt.plot(x5, ek5, 'g-', label=r'$We=0.5$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_k')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/KineticEnergyTf="+str(Tf)+"b="+str(betav)+"mesh="+str(mm)+"c0="+str(c0)+"dt="+str(dt)+".png")
        plt.clf()
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'$We=0.1$')
        plt.plot(x2, ee2, 'b-', label=r'$We=0.2$')
        plt.plot(x3, ee3, 'c-', label=r'$We=0.3$')
        plt.plot(x4, ee4, 'm-', label=r'$We=0.4$')
        plt.plot(x5, ee5, 'g-', label=r'$We=0.5$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/ElasticEnergyTf="+str(Tf)+"b="+str(betav)+"mesh="+str(mm)+"c0="+str(c0)+"dt="+str(dt)+".png")
        plt.clf()




    """if max(norm(T1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E6:

        # Plot First Normal Stress Difference
        tau_xx=project(tau1[0,0],Q)
        mplot(tau_xx)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_xxRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf() 
        tau_xy=project(tau1[1,0],Q)
        mplot(tau_xy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_xyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf() 
        tau_yy=project(tau1[1,1],Q)
        mplot(tau_yy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_yyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf() 
        N1=project(tau1[0,0]-tau1[1,1],Q)
        mplot(N1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/FirstNormalStressDifferenceRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf()""" 

    """if max(norm(T1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E6:
 
       # Plot Velocity Components
        ux=project(u1[0],Q)
        mplot(ux)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/u_xRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf()
        uy=project(u1[1],Q)
        mplot(uy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/u_yRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf()""" 

    """if max(norm(T1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E6:


        # Matlab Plot of the Solution at t=Tf
        rho1=rho_0*rho1
        rho1=project(rho1,Q)
        #p1=mu_0*(L/U)*p1  #Dimensionalised Pressure
        #p1=project(p1,Q)
        mplot(rho1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/DensityRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf() 
        mplot(p1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/PressureRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf()
        mplot(T1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/TemperatureRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf()



    if max(norm(T1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E6:
        #Plot Contours USING MATPLOTLIB
        # Scalar Function code


        x = Expression('x[0]', d=d, degree=d)  #GET X-COORDINATES LIST
        y = Expression('x[1]', d=d, degree=d)  #GET Y-COORDINATES LIST
        pvals = p1.vector().array() # GET SOLUTION p= p(x,y) list
        Tvals = T1.vector().array() # GET SOLUTION T= T(x,y) list
        rhovals = rho1.vector().array() # GET SOLUTION p= p(x,y) list
        tauxxvals=tauxx.vector().array()
        xyvals = mesh1.coordinates()     # CLEAN THIS UP!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        xvalsq = interpolate(x, Q)#xyvals[:,0]
        yvalsq= interpolate(y, Q)#xyvals[:,1]
        xvalsw = interpolate(x, W)#xyvals[:,0]
        yvalsw= interpolate(y, W)#xyvals[:,1]

        xvals = xvalsq.vector().array()
        yvals = yvalsq.vector().array()


        xx = np.linspace(x0,x1)
        yy = np.linspace(y0,y1)
        XX, YY = np.meshgrid(xx,yy)   # (x,y) coordinate data formatted so that it can be used by plt.contour()
        pp = mlab.griddata(xvals, yvals, pvals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 
        dd = mlab.griddata(xvals, yvals, rhovals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 

        plt.contour(XX, YY, dd, 25)
        plt.title('Density Contours')   # DENSITY CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/DensityContoursRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf()

        plt.contour(XX, YY, pp, 25)
        plt.title('Pressure Contours')   # PRESSURE CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/PressureContoursRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf()

        xvals = xvalsw.vector().array()
        yvals = yvalsw.vector().array()

        TT = mlab.griddata(xvals, yvals, Tvals, xx, yy, interp='nn') 
        plt.contour(XX, YY, TT, 20) 
        plt.title('Temperature Contours')   # TEMPERATURE CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/TemperatureContoursRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf()


        normstress = mlab.griddata(xvals, yvals, tauxxvals, xx, yy, interp='nn')

        plt.contour(XX, YY, normstress, 20) 
        plt.title('Stress Contours')   # NORMAL STRESS CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/StressContoursRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf()


        #Plot Contours USING MATPLOTLIB
        # Vector Function code

        u1=U*u1  # DIMENSIONALISED VELOCITY
        u1=project(u1,V)
        g=list()
        h=list()
        n= mesh1.num_vertices()
        print(u1.vector().array())   # u is the FEM SOLUTION VECTOR IN FUNCTION SPACE 
        for i in range(len(u1.vector().array())/2-1):
            g.append(u1.vector().array()[2*i+1])
            h.append(u1.vector().array()[2*i])

        uvals = np.asarray(h) # GET SOLUTION (u,v) -> u= u(x,y) list
        vvals = np.asarray(g) # GET SOLUTION (u,v) -> v= v(x,y) list


        xy = Expression(('x[0]','x[1]'), d=d, degree=d)  #GET MESH COORDINATES LIST
        xyvalsv = interpolate(xy, V)

        q=list()
        r=list()

        for i in range(len(u1.vector().array())/2-1):
           q.append(xyvalsv.vector().array()[2*i+1])
           r.append(xyvalsv.vector().array()[2*i])

        xvals = np.asarray(r)
        yvals = np.asarray(q)

        # Interpoltate velocity field data onto matlab grid
        uu = mlab.griddata(xvals, yvals, uvals, xx, yy, interp='nn') 
        vv = mlab.griddata(xvals, yvals, vvals, xx, yy, interp='nn') 

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
                       linewidth=1.0)                           # line thickness
                                                                # arrow size
        plt.colorbar()                                          # add colour bar on the right
        plt.title('Lid Driven Cavity Flow')
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/uVelocityContoursRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")   
        plt.clf()                                             # display the plot"""


    #plt.close()


    if jj==30:
       j=loopend+1
       break


    # Update Control Variables 
    """if max(norm(u1.vector(), 'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E6 and np.isfinite(sum(u1.vector().array())):
        #with open("Weissenberg Compressible Stability.txt", "a") as text_file:
        #     text_file.write("Solution Converges Re"+str(Rey)+", We="+str(We)+", dt="+str(dt)+'\n')
        #dt = 0.002  #Time Stepping                        #Go back to Original Timestep
        #gammah=10E0
        jj=0
        if U==200.0:
            U=2.5*U
            lambda1=l1ref/U 
        elif U==100.0:
            U=2.0*U
            lambda1=l1ref/U 
        elif U==50.0:
            U=2*U
            lambda1=l1ref/U 
        elif U==25.0:
            U=2.0*U
            lambda1=l1ref/U 
        elif U==10.0:
            U=2.5*U
            lambda1=l1ref/U 
        elif U==5.0:
            U=2.0*U
            lambda1=l1ref/U 
        elif U==2.5:
            U=5.0
            lambda1=l1ref/U 
        elif U==1.0:
            U=5
            lambda1=l1ref/U 
        elif U==0.5:
            U=1.0
            lambda1=l1ref/U 
        elif U==0.1:
            U=0.5
            lambda1=l1ref/U"""

         

    """if max(norm(T1.vector(), 'linf'),norm(p1.vector(), 'linf')) < 10E6:
       with open("DEVSS Weissenberg Compressible Stability.txt", "a") as text_file:
            text_file.write("Solution Converges Re"+str(Rey)+", We="+str(We)+", dt="+str(dt)+'\n')
       dt = 0.002
       jj=0
       if lambda1==5.0*10E1:
            lambda1=2.0*lambda1
       elif lambda1==2.0*10E1:
            lambda1=2.5*lambda1
       elif lambda1==1.0*10E1:
            lambda1=2.0*lambda1
       elif lambda1==4.0*10E0:
            lambda1=2.5*lambda1
       elif lambda1==2.0*10E0:
            lambda1=2*lambda1
       elif lambda1==1.0*10E0:
            lambda1=2*lambda1
       elif lambda1==4.0*10E-1:
            lambda1=2.5*lambda1
       elif lambda1==2.0*10E-1:
            lambda1=2*lambda1
       elif lambda1==2.0*10E-2:
            lambda1=10*lambda1
       elif lambda1==2.0*10E-3:
            lambda1=10*lambda1
       elif lambda1==1.0*10E-2:
            lambda1=2*lambda1"""









"""
MIT License

Copyright (c) 2020 Alexander Mackay

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# This code geenrates results for 2D flow of a nonisothermal Oldroyd-B liquid between 
# two eccentrically rotating cylinders. Various geomtric and fluid parameters can be 
# changed 


import time, sys
from fenics_base import *

# Progress Bar
def update_progress(job_title, progress):
    length = 20 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 4))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()
update_progress("Simulation", 0)

# SET TIMESTEPPING PARAMTER
T_f = 2.0
Tf = T_f

# SET LOOPING PARAMETER
loopend = 2
primary_loop = 0                              
adaptive_loop = False
secondary_loop = 0
max_refine = 2
err_count = 0
conv_fail = 0
tol = 10E-6
defpar = 1.0



conv = 1                                      # Non-inertial Flow Parameter (Re=0)
We = 0.01
betav = 0.5
Re = 10
Ma = Re*0.0001
c0 = 1.0/Ma

# Default Nondimensional Parameters
U = 1.0
w_j = 1.0

T_0 = 300
T_h = 350
Di = 0.0025              # Diffusion Number
Vh = 0.0000069
Bi = 0.2
rho_0 = 1.0
al = 0.001                # Nonisothermal Parameter between 0 and 1


# Parameters hard-coded for Oldroyd-B 
A_0 = 0.0 # Solvent viscosity thinning
k_ewm = 0.0 # Shear thinning 
B = 0.0 # Polymeric viscosity thinning 
K_0 = 0.0


# Steady State Method (Re-->10Re)
Ret = Expression('Re*(1.0+0.5*(1.0+tanh(0.5*t-3.5))*9.0)', t=0.0, Re=Re, degree=2)
Wet = Expression('We*0.5*(1.0+tanh(0.5*t-3.5))', t=0.0, We=We, degree=2)

alph1 = 1.0
c1 = 0.1
c2 = 0.01
c3 = 0.1
th = 1.0               # DEVSS



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

while primary_loop < loopend:

    primary_loop += 1
    t = 0.0

    r_a = 1.0 #Journal Radius
    r_b = 2.0
    y_1 = 0.0
    x_2 = 0.0
    y_2 = 0.0
    if adaptive_loop == False and err_count == 0:
        # Define mesh using JBP_base functions
        # HOLOLOW CYLINDER MESH
        x_1 = - 0.90 -(0.01 * secondary_loop) #x_1 = -0.80      
        mesh_resolution = 35
        mesh = JBP_mesh(mesh_resolution, x_1, x_2, y_1, y_2, r_a, r_b)

    # Plot Mesh
    mplot(mesh)
    plt.savefig("JBP_mesh_"+str(mesh_resolution)+".png")
    plt.clf() 
    plt.close()

    # Generate internal mesh for hollow cylinder
    c3 = Circle(Point(x_1,y_1), 0.99*r_a, 256)  # Empty hole in mesh
    gdim = mesh.geometry().dim() # Mesh Geometry
    meshc= generate_mesh(c3, 15)

    # Timestepping
    dt = 10*mesh.hmin()**2

    # Reset Mesh Dependent Functions (Reset mesh)
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)
    tang = as_vector([n[1], -n[0]])

    # Finite Element Spaces
    V_s = VectorElement(family, mesh.ufl_cell(), order)       # Velocity Elements
    V_d = VectorElement(dfamily, mesh.ufl_cell(), order-1)
    V_se = VectorElement(rich, mesh.ufl_cell(),  order+1)
     
    Z_c = VectorElement(family, mesh.ufl_cell(),  order, 3)     # Stress Elements
    Z_s = VectorElement(dfamily, mesh.ufl_cell(),  order-1, 3)
    Z_se = VectorElement(rich, mesh.ufl_cell(),  order+1, 3)
    Z_d = VectorElement(dfamily, mesh.ufl_cell(),  order-2, 3)

    Q_s = FiniteElement(family, mesh.ufl_cell(), order-1)   # Pressure/Density Elements
    Q_p = FiniteElement(rich, mesh.ufl_cell(), order+1, 3)


    # Function spaces
    W = FunctionSpace(mesh,V_s*Z_d)             # F.E. Spaces 
    V = FunctionSpace(mesh,V_s)
    Vd = FunctionSpace(mesh,V_d)
    Z = FunctionSpace(mesh,Z_s)
    #Ze = FunctionSpace(mesh,Z_e)               #FIX!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Zd = FunctionSpace(mesh,Z_d)
    Zc = FunctionSpace(mesh,Z_c)
    Q = FunctionSpace(mesh,Q_s)
    Qt = FunctionSpace(mesh, "DG", order-2)
    Qr = FunctionSpace(mesh,Q_s)

    # Reset Trial/Test and Solution Functions

    # Trial Functions
    rho=TrialFunction(Q)
    p = TrialFunction(Q)
    T = TrialFunction(Q)
    tau_vec = TrialFunction(Zc)
    (u, D_vec) = TrialFunctions(W)
    D =  as_matrix([[D_vec[0], D_vec[1]],
                    [D_vec[1], D_vec[2]]])
    tau = as_matrix([[tau_vec[0], tau_vec[1]],
                     [tau_vec[1], tau_vec[2]]]) 


    # Test Functions
    q = TestFunction(Q)
    r = TestFunction(Q)
    Rt_vec = TestFunction(Zc)        # Conformation Stress    
    (v, R_vec) = TestFunctions(W)    # Velocity/DEVSS Space
    R = as_matrix([[R_vec[0], R_vec[1]],
                   [R_vec[1], R_vec[2]]])
    Rt = as_matrix([[Rt_vec[0], Rt_vec[1]],
                    [Rt_vec[1], Rt_vec[2]]])        # DEVSS Space


    #Solution Functions
    rho0 = Function(Q)
    rho12 = Function(Q)
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
    uu0 = Function(V)
    (u0, D0_vec) = w0.split()
    (u12, D12_vec) = w12.split()
    (us, Ds_vec) = ws.split()
    (u1, D1_vec) = w1.split()
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
    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    sub_domains.set_all(0)
    omega0.mark(sub_domains, 2)
    omega1.mark(sub_domains, 3)


    #Define Boundary Parts
    boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    omega0.mark(boundary_parts,0)
    omega1.mark(boundary_parts,1)
    ds = Measure("ds")[boundary_parts]


    if adaptive_loop == True: 
        w = Expression(('(0.5*(1.0+tanh(8*(t-0.5))))*(x[1]-y1)/r_a' , '-(0.5*(1.0+tanh(8*(t-0.5))))*(x[0]-x1)/r_a' ), degree=2, r_a=r_a, x1=x_1, y1=y_1 , t=0.0)
            
    if adaptive_loop == False:
        w = Expression(('(x[1]-y1)/r_a' , '-(x[0]-x1)/r_a' ), degree=2, r_a=r_a, x1=x_1, y1=y_1 , t=0.0)

   
    spin =  DirichletBC(W.sub(0), w, omega0) 
    noslip  = DirichletBC(W.sub(0), (0.0, 0.0), omega1) #The outer cylinder remains fixed with zero velocity 
    temp0 =  DirichletBC(Q, T_h, omega0)    #Temperature on Omega0 

    #Collect Boundary Conditions
    bcu = [noslip, spin]
    bcp = []
    bcT = [temp0]
    bctau = []

    # Incompressible + eccentricity range test
    betav = 1.0 - DOLFIN_EPS
    Re = 10
    We = 0.000001
    Ma = 0.000001



    # Print Parameters of flow simulation
    t = 0.0                  #Time

    if adaptive_loop == False and err_count < max_refine:
        print('############# ADAPTIVE MESH REFINEMENT STAGE ################')   
        print( 'Number of Refinements:', err_count )
        print( "Adaptive loops complete: ", adaptive_loop)
    print('############# Journal Bearing Length Ratios ############')
    ec = np.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)
    c = r_b - r_a
    ecc = ec/c
    print('Eccentricity (m):' , )
    print('Radius DIfference (m):', r_b - r_a)
    print('Eccentricity Ratio:',ecc)

    print('############# TIME SCALE ############')
    print('Timestep size (s):', dt)
    print( 'Finish Time (s):', Tf)
    print( 'Number of Steps:', int(Tf/dt))

    print( '############# Scalings & Nondimensional Parameters ############')
    print( 'Characteristic Length (m):', r_b-r_a)
    print( 'Characteristic Velocity (m/s):', w_j*r_a)
    print( 'Speed of sound (m/s):', c0)
    print( 'Cylinder Speed (t=0) (m/s):', w_j*r_a*(1.0+np.tanh(8.0*t-4.0)))
    print( 'Mach Number', Ma)
    print( 'Nondimensionalised Cylinder Speed (t=0) (m/s):', (1.0+np.tanh(8.0*t-4.0)))
    print( 'Reynolds Number:', Re)
    print('Weissenberg Number:', We)
    print( 'Viscosity Ratio:', betav)
    print( 'Temperature Thinning:', al)
    print( 'Diffusion Number:' ,Di)
    print( 'Viscous Heating Number:', Vh)

    Np= len(p0.vector().get_local())
    Nv= len(w0.vector().get_local())  
    Nvel = len(uu0.vector().get_local()) 
    Ntau= len(tau0_vec.vector().get_local())
    dof= 3*Nv+2*Ntau+Np
    print( '############# Discrete Space Characteristics ############')
    print( 'Degree of Elements', order)
    print('Size of Pressure Space = %d ' % Np)
    print('Size of Velocity Space = %d ' % Nvel)    
    print('Size of Velocity/DEVSS Space = %d ' % Nv)
    print('Size of Stress Space = %d ' % Ntau)
    print('Degrees of Freedom = %d ' % dof)
    print( 'Number of Cells:', mesh.num_cells())
    print( 'Number of Vertices:', mesh.num_vertices())
    print( 'Minimum Cell Diamter:', mesh.hmin())
    print( 'Maximum Cell Diamter:', mesh.hmax())
    print( '############# Stabilisation Parameters ############')
    print( 'DEVSS Momentum Term:', th)

    print( 'Loop:', secondary_loop, '-', primary_loop)

    #quit()

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
    theta1 = T1-T_0/(T_h-T_0)

    # Stabilisation


    # DEVSS Stabilisation

    
    DEVSSl_u12 = 2*(1-betav)*inner(Dcomp(u),Dincomp(v))*dx    
    DEVSSr_u12 = 2*inner(D0,Dincomp(v))*dx   
    DEVSSl_u1 = 2*(1-betav)*inner(Dcomp(u),Dincomp(v))*dx    
    DEVSSr_u1 = 2*inner(D12,Dincomp(v))*dx 

    DEVSSl_T1 = (1.-Di)*inner(grad(thetal), grad(r))*dx
    DEVSSr_T1 = inner((1.-Di)*(grad(thetar) + grad(theta0)), grad(r))*dx

    # DEVSS-G Stabilisation
    
    DEVSSGl_u12 = 2.0*(1.-betav)*inner(Dincomp(u),Dincomp(v))*dx    
    DEVSSGr_u12 = (1-betav)*inner(D0 + D0.T,Dincomp(v))*dx   
    DEVSSGl_u1 = 2.0*(1.-betav)*inner(Dincomp(u),Dincomp(v))*dx    
    DEVSSGr_u1 = (1.-betav)*inner(D12 + D12.T,Dincomp(v))*dx

 
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
    maxiter = 100000
    if adaptive_loop == False and err_count < max_refine:
       maxiter = 15
       dt = mesh.hmin()**2
    frames = int((Tf/dt)/1000)
    while t < Tf + DOLFIN_EPS and iter < maxiter:
        update_progress("Simulation "+str(secondary_loop ), t/Tf) # Update progress bar
        iter += 1
        w.t=t
        Ret.t=t
        Wet.t=t

        (u0, D0_vec)=w0.split()   
        # Update SU Term
        alpha_supg = h/(magnitude(u1) + 0.0000000001)
        SU = inner(dot(u1, grad(tau)), alpha_supg*dot(u1,grad(Rt)))*dx
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
        DEVSSGr_u1 = (1.-betav)*inner(D0 + D0.T,Dincomp(v))*dx            # Update DEVSS-G Stabilisation RHS
        U = 0.5*(u + u0)    


        # TAYLOR-GALERKIN TIME MARCHING ALGORITHM


        # Density Half Step
        rho_eq = 2.0*(rho - rho0)/dt + dot(u0, grad(rho0)) - rho0*div(u0)
        rho_weak = inner(rho_eq,q)*dx

        a0 = lhs(rho_weak)
        L0 = rhs(rho_weak)
        A0 = assemble(a0)
        b0 = assemble(L0)
        [bc.apply(A0, b0) for bc in bcp]
        solve(A0, rho12.vector(), b0, "bicgstab", "default")
        end()
          




        # Velocity half step
        lhsv_eq = 2.0*Re*((rho12*u - rho0*u0)/dt + conv*dot(u0, nabla_grad(U)))
        v_weak12 = dot(lhsv_eq, v)*dx + \
               + inner(2.0*betav*phi_s(theta0,A_0)*Dincomp(U), Dincomp(v))*dx + (1.0/3)*inner(betav*phi_s(theta0,A_0)*div(U),div(v))*dx \
                - ((1.-betav)/(We+DOLFIN_EPS))*inner(div(tau0-rho0*Identity(len(u)) ), v)*dx + inner(grad(p0),v)*dx\
               + inner(D-grad(u),R)*dx   

        a1 = lhs(v_weak12)
        L1 = rhs(v_weak12)

            #DEVSS Stabilisation
        a1+= th*DEVSSl_u12                     
        L1+= th*DEVSSr_u12 

        A1 = assemble(a1)
        b1 = assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, w12.vector(), b1, "bicgstab", "default")
        end()

        (u12, D12_vec) = w12.split()
        D12 = as_matrix([[D12_vec[0], D12_vec[1]],
                        [D12_vec[1], D12_vec[2]]])


        """# STRESS Half Step
        F12 = dot(u12,grad(tau)) - dot(grad(u12),tau) - dot(tau,tgrad(u12) + div(u12)*tau) # Convection/Deformation Terms
        lhs_tau12 = (We/dt+1.0)*tau + We*F12                        # Left Hand Side
        rhs_tau12= (We/dt)*tau0 + 2.0*(1.0-betav)*Dcomp(u0)         # Right Hand Side

        a3 = inner(lhs_tau12,Rt)*dx                                 # Weak Form
        L3 = inner(rhs_tau12,Rt)*dx

        a3 += SUPGl3             # SUPG Stabilisation LHS
        L3 += SUPGr3             # SUPG / SU Stabilisation RHS
        A3=assemble(a3)
        b3=assemble(L3)
        [bc.apply(A3, b3) for bc in bctau]
        solve(A3, tau12_vec.vector(), b3, "bicgstab", "default")
        end()"""


       #Predicted U* Equation
        lhsFus = Re*rho0*((u - u0)/dt + conv*dot(u0, nabla_grad(U)))
        Fus = dot(lhsFus, v)*dx + \
               + inner(2.0*betav*phi_s(theta0,A_0)*Dincomp(U), Dincomp(v))*dx + (1.0/3)*inner(betav*phi_s(theta0,A_0)*div(U),div(v))*dx \
                - ((1.-betav)/(We+DOLFIN_EPS))*inner(div( tau0-rho0*Identity(len(u)) ), v)*dx + inner(grad(p0),v)*dx\
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


        #Continuity Equation 

        lhs_p_1 = (Ma*Ma/(dt))*p
        rhs_p_1 = (Ma*Ma/(dt))*p0 - Re*(1+al*theta0)*div(rho0*us)

        lhs_p_2 = (1+al*theta0)*dt*grad(p)
        rhs_p_2 = (1+al*theta0)*dt*grad(p0)
        
        a5=inner(lhs_p_1,q)*dx + inner(lhs_p_2,grad(q))*dx   
        L5=inner(rhs_p_1,q)*dx + inner(rhs_p_2,grad(q))*dx

        A5 = assemble(a5)
        b5 = assemble(L5)
        [bc.apply(A5, b5) for bc in bcp]
        #[bc.apply(p1.vector()) for bc in bcp]
        solve(A5, p1.vector(), b5, "bicgstab", "default")
        end()


        # Update SUPG density Term
        alpha_supg = h/(magnitude(u12)+0.0000000001)
        SU_rho = inner(dot(u12, grad(rho)), alpha_supg*dot(u12,grad(q)))*dx

        # Density Update
        rho_eq = (rho - rho0)/dt + dot(u12, grad(rho12)) - rho12*div(u12) # this is possibly highly unstable. Get code to measure error norm
        rho_weak = inner(rho_eq,q)*dx

        a6 = lhs(rho_weak)
        L6 = rhs(rho_weak)
        
        a6+= SU_rho

        A6 = assemble(a6)
        b6 = assemble(L6)
        [bc.apply(A6, b6) for bc in bcp]
        solve(A6, rho1.vector(), b6, "bicgstab", "default")
        end()


        #Velocity Update
        lhs_u1 = (Re/dt)*rho1*u                                          # Left Hand Side
        rhs_u1 = (Re/dt)*rho0*us                                         # Right Hand Side

        a7=inner(lhs_u1,v)*dx + inner(D-grad(u),R)*dx                    # Weak Form
        L7=inner(rhs_u1,v)*dx - 0.5*inner( grad(p1-p0),v )*dx 

        a7+= 0 #th*DEVSSGl_u1                                                  
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



        #Temperature Full Step
        gamdot = inner(sigmacon(u0, p0, tau0, betav, We),grad(u0))
        lhs_temp1 = (1.0/dt)*rho1*thetal + rho1*dot(u1,grad(thetal))
        difflhs_temp1 = Di*grad(thetal)
        rhs_temp1 = (1.0/dt)*rho1*thetar + rho1*dot(u1,grad(thetar)) + (1.0/dt)*rho1*theta0 + Vh*phi_ewm(tau1, theta1, k_ewm, B)*gamdot
        diffrhs_temp1 = Di*grad(thetar)
        a9 = inner(lhs_temp1,r)*dx + inner(difflhs_temp1,grad(r))*dx 
        L9 = inner(rhs_temp1,r)*dx + inner(diffrhs_temp1,grad(r))*dx - Di*Bi*inner(theta0,r)*ds(1) \

        a9+= th*DEVSSl_T1                                                
        L9+= th*DEVSSr_T1 

        A9 = assemble(a9)
        b9 = assemble(L9)
        [bc.apply(A9, b9) for bc in bcT]
        solve(A9, T1.vector(), b9, "bicgstab", "default")
        end()   

        theta1 = T1-T_0/(T_h-T_0)

        if We > 0.00001 or adaptive_loop == False and err_count < max_refine:
            # Do not compute stress if We is v. small
            # Stress Full Step
            lhs_tau1 = (We*phi_ewm(tau0, theta1, k_ewm, B)/dt+1.0)*tau  +  We*phi_ewm(tau0, theta1, k_ewm, B)*FdefG(u1, D1, tau)                           # Left Hand Side
            rhs_tau1= (We*phi_ewm(tau0, theta1, k_ewm, B)/dt)*tau0 + rho0*Identity(len(u)) 

            Ftau = inner(lhs_tau1,Rt)*dx - inner(rhs_tau1,Rt)*dx
            a4 = lhs(Ftau)
            L4 = rhs(Ftau) 

                # SUPG / SU / LPS Stabilisation (User Choose One)

            a4_stab = a4 + SU  # [SUPGl4, SUl4, LPSl_stab, LPSl_stress, diff_stab, 0]
            L4_stab = L4  # [SUPGr4, SUr4, LPSr_stab, LPS_res_stab, 0]   


            A4=assemble(a4_stab)                                     # Assemble System
            b4=assemble(L4_stab)
            [bc.apply(A4, b4) for bc in bctau]
            solve(A4, tau1_vec.vector(), b4, "bicgstab", "default")
            end()


        taudiff = np.abs(tau1_vec.vector().get_local() - tau0_vec.vector().get_local()).max()
        udiff = np.abs(w1.vector().get_local() - w0.vector().get_local()).max()



            # Solution Calculations

        E_k=assemble(0.5*rho1*dot(u1,u1)*dx)
        E_e=assemble((tau1_vec[0]+tau1_vec[2]-2.0)*dx)

        sigma0 = dot(sigmacon(u1, p1, tau1, betav, We), tang)
        sigma1 = dot(sigmacon(u1, p1, tau1, betav, We), tang)

        omegaf0 = dot(sigmacon(u1, p1, tau1, betav, We), n)  #Nomral component of the stress 
        omegaf1 = dot(sigmacon(u1, p1, tau1, betav, We), n)


        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
        # Dimensional Force Calculations
        innerforcex = inner(Constant((1.0, 0.0)), omegaf0)*ds(0) 
        innerforcey = -inner(Constant((0.0, 1.0)), omegaf0)*ds(0) 

        # dimensional Torque
        innertorque = -inner(n, sigma0)*ds(0)
        outertorque = -inner(n, sigma1)*ds(1)


        # Record Elastic & Kinetic Energy Values & Torque (Method 1)
        if primary_loop == 1:
            x1.append(t)
            ek1.append(E_k)
            ee1.append(E_e)
            y1.append(assemble(innertorque))
            zx1.append(0.0)     #assemble(innerforcex)
            z1.append(assemble(innerforcey))
        if primary_loop == 2:
            x2.append(t)
            ek2.append(E_k)
            ee2.append(E_e)
            y2.append(assemble(innertorque))
            zx2.append(assemble(innerforcex))
            z2.append(assemble(innerforcey))
        if primary_loop == 3:
            x3.append(t)
            ek3.append(E_k)
            ee3.append(E_e)
            y3.append(assemble(innertorque))
            zx3.append(assemble(innerforcex))
            z3.append(assemble(innerforcey))
        if primary_loop == 4:
            x4.append(t)
            ek4.append(E_k)
            ee4.append(E_e)
            y4.append(assemble(innertorque))
            zx4.append(assemble(innerforcex))
            z4.append(assemble(innerforcey))
        if primary_loop == 5:
            x5.append(t)
            ek5.append(E_k)
            ee5.append(E_e)
            y5.append(assemble(innertorque))
            zx5.append(assemble(innerforcex))
            z5.append(assemble(innerforcey))


        

        # Break Loop if code is diverging

        if max(norm(tau1_vec.vector(), 'linf'),norm(w1.vector(), 'linf')) > 10E6 or np.isnan(sum(tau1_vec.vector().get_local())):
            print( 'FE Solution Diverging')   #Print message 
            if primary_loop == 1:           # Clear Lists
               x1=list()
               ek1=list()
               ee1=list()
            if primary_loop == 2:
               x2=list()
               ek2=list()
               ee2=list()
            if primary_loop == 3:
               x3=list()
               ek3=list()
               ee3=list()
            if primary_loop == 4:
               x4=list()
               ek4=list()
               ee4=list()
            if primary_loop == 5:
               x5=list()
               ek5=list()
               ee5=list() 
            err_count+= 1                          # Convergence Failures
            Tf= (iter-10)*dt
            adaptive_loop = False
            #quit()
            break
           

        # Move to next time step (Continuation in Reynolds Number)
        t += dt


    if adaptive_loop == True:
        # Minimum of stream function (Eye of Rotation)
        u1 = project(u1, V)
        psi = comp_stream_function(rho1, u1)
        psi_max = max(psi.vector().get_local())
        max_loc = max_location(psi, mesh)
        with open("OldroydStream-Function.txt", "a") as text_file:
             text_file.write("Re="+str(Re)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+"----- psi_min="+str(psi_max)+"---"+str(max_loc)+'\n')

        # Data on Kinetic/Elastic Energies
        with open("OldroydEnergy.txt", "a") as text_file:
             text_file.write("Re="+str(Re)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+", E_k="+str(E_k)+", E_e="+str(E_e)+'\n')


        # THESE WILL HAVE TO CHANGE
        F_y = assemble(innerforcey) * (0.01**2)/(0.005*25)
        torque_journal = assemble(innertorque) * (0.01**3)/(0.005*25)
        # Data on Stability Measure
        with open("Oldroydresults.txt", "a") as text_file:
             text_file.write("Re="+str(Re)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+"ecc="+str(ecc)+", F_y = "+str(F_y)+", C ="+str(torque_journal)+'\n')


        # Not properly labelled!!
        # Plot Torque/Load for different Wessinberg Numbers

        # Plot Torque/Load for Incompresssible Newtonian vd Viscoelastic
    if max(norm(tau1_vec.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 and primary_loop == loopend or primary_loop == 2 or primary_loop == 1:
        # Plot Torque Data
        plt.figure(0)
        plt.plot(x1, y1, 'r-', label=r'Incompressible Newtonian')
        #plt.plot(x2, y2, 'b-', label=r'Oldroyd-B $(We=0.1, Ma=0.005)$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$C$')
        plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Torque_We="+str(We)+"Re="+str(Re*conv)+"b="+str(betav)+"Ma="+str(Ma)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()
        plt.figure(1)
        plt.plot(x1, zx1, 'r-', label=r'Incompressible Newtonian')
        #plt.plot(x2, zx2, 'b-', label=r'Oldroyd-B $(We=0.1, Ma=0.005)$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$F_x$')
        plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Horizontal_Load_We="+str(We)+"Re="+str(Re*conv)+"b="+str(betav)+"Ma="+str(Ma)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()
        plt.figure(2)
        plt.plot(x1, z1, 'r-', label=r'Incompressible Newtonian')
        #plt.plot(x2, z2, 'b-', label=r'Oldroyd-B $(We=0.1, Ma=0.005)$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$F_y$')
        plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Vertical_Load_We="+str(We)+"Re="+str(Re*conv)+"b="+str(betav)+"Ma="+str(Ma)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()
        plt.figure(3)
        plt.plot(zx1, z1, 'r-', label=r'Incompressible Newtonian')
        #plt.plot(zx2, z2, 'b-', label=r'Oldroyd-B $(We=0.1, Ma=0.005)$')
        plt.legend(loc='best')
        plt.xlabel('$F_x$')
        plt.ylabel('$F_y$')
        plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Force_Evolution_We="+str(We)+"Re="+str(Re*conv)+"b="+str(betav)+"Ma="+str(Ma)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()




    if dt < tol:
       primary_loop = loopend + 1
       break



    if primary_loop == loopend:
        secondary_loop +=1
        update_progress("Simulation"+str(secondary_loop), 1)
        primary_loop = 0
        #adaptive_loop = False
        #err_count = 0
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

    if secondary_loop == 5:
        quit()


    if adaptive_loop == False: 

        adaptive_loop = True
        # Calculate Stress Residual 
        F1R = Fdef(u1, tau1)  
        F1R_vec = as_vector([F1R[0,0], F1R[1,0], F1R[1,1]])
        restau0 = We/dt*(tau1_vec-tau0_vec) + We*F1R_vec + tau1_vec - I_vec #- diss_vec 
        res_test = inner(restau0,restau0)                            

        kapp = project(res_test, Qt) # Error Function
        norm_kapp = normalize_solution(kapp) # normalised error function

        ratio = 0.2/(1*err_count + 1.0) # Proportion of cells that we want to refine
        tau_average = project((tau1_vec[0]+tau1_vec[1]+tau1_vec[2])/3.0 , Qt)
        error_rat = project(kapp/(tau_average + DOLFIN_EPS) , Qt)
        error_rat = absolute(error_rat)

 

        if error_rat.vector().get_local().max() > 0.01 and err_count < max_refine:
           err_count+=1
           mesh = adaptive_refinement(mesh, norm_kapp, ratio)
           #mplot(mesh)
           #plt.savefig("adaptive-mesh.png")
           #plt.clf()
           adaptive_loop = False
           conv_fail = 0

        # Reset Parameters
        corr = 1    
        primary_loop = 0
        dt = 10*mesh.hmin()**2
        Tf = T_f
        th = 1.0
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






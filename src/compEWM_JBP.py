# This code geenrates results for 2D flow of a nonisothermal FENE-P-MP liquid between 
# two eccentrically rotating variables. Various geomtric and fluid parameters can be 
# changed 


import time, sys
from JBP_base import *

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
dt = 5*mesh.hmin()**2  #Time Stepping  
T_f = 8.0
Tf = T_f

# SET LOOPING PARAMETER
loopend = 5
total_loops = (loopend*(Tf/dt))
j = 0                              
jj = 1
jjj = 0
err_count = 0
conv_fail = 0
tol = 10E-6
defpar = 1.0

conv = 1                                      # Non-inertial Flow Parameter (Re=0)
We = 0.25
betav = 0.5
Re = 100
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

A_0 = 120 # Solvent viscosity thinning
k_ewm = -0.7 # Shear thinning (EWM)
B = 0.1 # Polymeric viscosity thinning (EWM)
K_0 = 0.01


# Steady State Method (Re-->10Re)
Ret = Expression('Re*(1.0+0.5*(1.0+tanh(0.5*t-3.5))*9.0)', t=0.0, Re=Re, degree=2)
Wet = Expression('We*0.5*(1.0+tanh(0.5*t-3.5))', t=0.0, We=We, degree=2)

Rey=Re

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
while j < loopend:
    j+=1
    t=0.0


    # Reset Mesh Dependent Functions
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


    # Create mesh functions over the cell facets (Verify Boundary Classes)
    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    sub_domains.set_all(0)
    omega0.mark(sub_domains, 2)
    omega1.mark(sub_domains, 3)
    #POmega.mark(sub_domains, 4)


    #plot(sub_domains, interactive=False, scalarbar = False)
    #quit()

    #Define Boundary Parts
    boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    #boundary_parts = FacetFunction("size_t", mesh)
    omega0.mark(boundary_parts,0)
    omega1.mark(boundary_parts,1)
    ds = Measure("ds")[boundary_parts]


    if jj==1: 
        w = Expression(('(0.5*(1.0+tanh(8*(t-0.5))))*(x[1]-y1)/r_a' , '-(0.5*(1.0+tanh(8*(t-0.5))))*(x[0]-x1)/r_a' ), degree=2, r_a=r_a, x1=x_1, y1=y_1 , t=0.0)
            
    if jj==0:
        w = Expression(('(x[1]-y1)/r_a' , '-(x[0]-x1)/r_a' ), degree=2, r_a=r_a, x1=x_1, y1=y_1 , t=0.0)

   
    spin =  DirichletBC(W.sub(0), w, omega0) 
    noslip  = DirichletBC(W.sub(0), (0.0, 0.0), omega1) #The outer cylinder remains fixed with zero velocity 
    temp0 =  DirichletBC(Q, T_h, omega0)    #Temperature on Omega0 

    #Collect Boundary Conditions
    bcu = [noslip, spin]
    bcp = []
    bcT = [temp0]
    bctau = []

    

    # Comparing different WEISSENBERG Numbers (We=0.1,0.2,0.3,0.4,0.5) at Re=__
    betav = 0.5
    Ma = 0.05
    if j==1:
       betav = 1.0 - DOLFIN_EPS
       We = 0.001
    elif j==2:
       We = 0.1
    elif j==3:
       We = 0.5
    elif j==4:
       We = 0.75
    elif j==5:
       We = 1.0

    #Second loop comparing Reynolds numbers

    if jjj == 0:
       Re = 25
       Ma = 0.05
    if jjj == 1:
       Re = 50
       Ma = 0.1
    if jjj == 2:
       Re = 100
       Ma = 0.1
    

    # Adaptive Mesh Refinement Step
    if jj==0 and err_count < 1: # 0 = on, 1 = off
       We = 1.0
       betav = 0.5
       Tf = 1.5*(1 + 2*err_count*0.25)
       dt = 10*mesh.hmin()**2 
       th = 0.0

    Rey = Re*conv

    # Print Parameters of flow simulation
    t = 0.0                  #Time
    e=6

    if jj==0:
        print('############# ADAPTIVE MESH REFINEMENT STAGE ################')   
        print( 'Number of Refinements:', err_count )
    print('############# Journal Bearing Length Ratios ############')
    print('Eccentricity (m):' ,ec)
    print('Radius DIfference (m):',c)
    print('Eccentricity Ratio:',ecc)

    print('############# TIME SCALE ############')
    print('Timestep size (s):', dt)
    print( 'Finish Time (s):', Tf)
    print( 'Number of Steps:', int(Tf/dt))

    print( '############# Scalings & Nondimensional Parameters ############')
    print( 'Characteristic Length (m):', r_b-r_a)
    print( 'Characteristic Velocity (m/s):', w_j*r_a)
    print( 'Speed of sound (m/s):', c0)
    print( 'Cylinder Speed (t=0) (m/s):', w_j*r_a*(1.0+np.tanh(e*t-3.0)))
    print( 'Mach Number', Ma)
    print( 'Nondimensionalised Cylinder Speed (t=0) (m/s):', (1.0+np.tanh(e*t-3.0)))
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

    print( 'Loop:', jjj, '-', j)

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
    maxiter = 10000000
    if jj==0:
       maxiter = 25
    frames = int((Tf/dt)/1000)
    while t < Tf + DOLFIN_EPS and iter < maxiter:
        update_progress("Simulation "+str(jjj), t/Tf) # Update progress bar
        iter += 1
        w.t=t
        Ret.t=t
        Wet.t=t

        (u0, D0_vec)=w0.split()   
        # Update SU Term
        alpha_supg = h/(magnitude(u1)+0.0000000001)
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



        # Energy Calculations
        E_k=assemble(0.5*rho1*dot(u1,u1)*dx)
        E_e=assemble((tau1_vec[0]+tau1_vec[2]-2.0)*dx)

        sigma0 = dot(sigmacon(u1, p1, tau1), tang)
        sigma1 = dot(sigmacon(u1, p1, tau1), tang)

        omegaf0 = dot(sigmacon(u1, p1, tau1), n)  #Nomral component of the stress 
        omegaf1 = dot(sigmacon(u1, p1, tau1), n)


        innerforcex = inner(Constant((1.0, 0.0)), omegaf0)*ds(0)
        innerforcey = -inner(Constant((0.0, 1.0)), omegaf0)*ds(0)

        innertorque = -inner(n, sigma0)*ds(0)
        outertorque = -inner(n, sigma1)*ds(1)


        # Record Elastic & Kinetic Energy Values & Torque (Method 1)
        if j==1:
            x1.append(t)
            ek1.append(E_k)
            ee1.append(E_e)
            y1.append(assemble(innertorque))
            zx1.append(0.0)     #assemble(innerforcex)
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


        

        # Break Loop if code is diverging

        if max(norm(tau1_vec.vector(), 'linf'),norm(w1.vector(), 'linf')) > 10E6 or np.isnan(sum(tau1_vec.vector().get_local())):
            print( 'FE Solution Diverging')   #Print message 
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
            err_count+= 1                          # Convergence Failures
            Tf= (iter-10)*dt
            jj=0
            #quit()
            break
           

        # Move to next time step (Continuation in Reynolds Number)
        t += dt


    if jj == 1:
        # Minimum of stream function (Eye of Rotation)
        u1 = project(u1, V)
        psi = comp_stream_function(rho1, u1)
        psi_max = max(psi.vector().get_local())
        max_loc = max_location(psi)
        with open("EWMStream-Function.txt", "a") as text_file:
             text_file.write("Re="+str(Re*conv)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+"----- psi_min="+str(psi_max)+"---"+str(max_loc)+'\n')

        # Data on Kinetic/Elastic Energies
        with open("EWMEnergy.txt", "a") as text_file:
             text_file.write("Re="+str(Rey*conv)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+", E_k="+str(E_k)+", E_e="+str(E_e)+'\n')


        chi = assemble(innerforcex)/assemble(innerforcey)
        torque_journal = assemble(innertorque)
        # Data on Stability Measure
        with open("EWMStability.txt", "a") as text_file:
             text_file.write("Re="+str(Rey*conv)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+"ecc="+str(ecc)+", Stability="+str(chi)+'\n')

        with open("EWMTorque.txt", "a") as text_file:
             text_file.write("Re="+str(Rey*conv)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+"ecc="+str(ecc)+", Stability="+str(torque_journal)+'\n')




        # Plot Torque/Load for different Wessinberg Numbers
        if max(norm(tau1_vec.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==loopend or j==1:
            # Plot Torque Data
            plt.figure(0)
            plt.plot(x1, y1, 'r-', label=r'$We=0$')
            plt.plot(x2, y2, 'b-', label=r'$We=0.1$')
            plt.plot(x3, y3, 'c-', label=r'$We=0.5$')
            plt.plot(x4, y4, 'm-', label=r'$We=0.75$')
            plt.plot(x5, y5, 'g-', label=r'$We=1.0$')
            plt.legend(loc='best')
            plt.xlabel('$t$', fontsize=16)
            plt.ylabel('$C$', fontsize=16)
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Torque_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
            plt.clf()
            plt.figure(1)
            plt.plot(x1, zx1, 'r-', label=r'$We=0$')
            plt.plot(x2, zx2, 'b-', label=r'$We=0.1$')
            plt.plot(x3, zx3, 'c-', label=r'$We=0.5$')
            plt.plot(x4, zx4, 'm-', label=r'$We=0.75$')
            plt.plot(x5, zx5, 'g-', label=r'$We=1.0$')
            plt.legend(loc='best')
            plt.xlabel('$t$', fontsize=16)
            plt.ylabel('$F_x$', fontsize=16)
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Horizontal_Load_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
            plt.clf()
            plt.figure(2)
            plt.plot(x1, z1, 'r-', label=r'$We=0$')
            plt.plot(x2, z2, 'b-', label=r'$We=0.1$')
            plt.plot(x3, z3, 'c-', label=r'$We=0.5$')
            plt.plot(x4, z4, 'm-', label=r'$We=0.75$')
            plt.plot(x5, z5, 'g-', label=r'$We=1.0$')
            plt.legend(loc='best')
            plt.xlabel('$t$', fontsize=16)
            plt.ylabel('$F_y$', fontsize=16)
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Vertical_Load_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"al="+str(al)+"t="+str(t)+".png")
            plt.clf()
            plt.figure(3)
            plt.plot(zx1, z1, 'r-', label=r'$We=0$')
            plt.plot(zx2, z2, 'b-', label=r'$We=0.1$')
            plt.plot(zx3, z3, 'c-', label=r'$We=0.5$')
            plt.plot(zx4, z4, 'm-', label=r'$We=0.75$')
            plt.plot(zx5, z5, 'g-', label=r'$We=1.0$')
            plt.legend(loc='best')
            plt.xlabel('$F_x$', fontsize=16)
            plt.ylabel('$F_y$', fontsize=16)
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Force_Evolution_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
            plt.clf()

        # Plot Torque/Load for Incompresssible Newtonian vd Viscoelastic
        """if max(norm(tau1_vec.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==5 or j==2 or j==1:
            # Plot Torque Data
            plt.figure(0)
            plt.plot(x1, y1, 'r-', label=r'Incompressible Newtonian')
            plt.plot(x2, y2, 'b-', label=r'Oldroyd-B $(We=0.1, Ma=0.005)$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$C$')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Torque_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"al="+str(al)+"t="+str(t)+".png")
            plt.clf()
            plt.figure(1)
            plt.plot(x1, zx1, 'r-', label=r'Incompressible Newtonian')
            plt.plot(x2, zx2, 'b-', label=r'Oldroyd-B $(We=0.1, Ma=0.005)$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$F_x$')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Horizontal_Load_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"al="+str(al)+"t="+str(t)+".png")
            plt.clf()
            plt.figure(2)
            plt.plot(x1, z1, 'r-', label=r'Incompressible Newtonian')
            plt.plot(x2, z2, 'b-', label=r'Oldroyd-B $(We=0.1, Ma=0.005)$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$F_y$')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Vertical_Load_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"al="+str(al)+"t="+str(t)+".png")
            plt.clf()
            plt.figure(3)
            plt.plot(zx1, z1, 'r-', label=r'Incompressible Newtonian')
            plt.plot(zx2, z2, 'b-', label=r'Oldroyd-B $(We=0.1, Ma=0.005)$')
            plt.legend(loc='best')
            plt.xlabel('$F_x$')
            plt.ylabel('$F_y$')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Force_Evolution_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"al="+str(al)+"t="+str(t)+".png")
            plt.clf()"""

        # Plot Torque/Load for different Wessinberg Numbers
        """if max(norm(tau1_vec.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==5 or j==3 or j==1:
            # Plot Torque Data
            plt.figure(0)
            plt.plot(x1, y1, 'r-', label=r'$\alpha=0.1$')
            plt.plot(x2, y2, 'b-', label=r'$\alpha=0.5$')
            plt.plot(x3, y3, 'c-', label=r'$\alpha=0.9$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('Torque')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Torque_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()
            plt.figure(1)
            plt.plot(x1, zx1, 'r-', label=r'$\alpha=0.1$')
            plt.plot(x2, zx2, 'b-', label=r'$\alpha=0.5$')
            plt.plot(x3, zx3, 'c-', label=r'$\alpha=0.9$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('Horzontal Load Force')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Horizontal_Load_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()
            plt.figure(2)
            plt.plot(x1, z1, 'r-', label=r'$\alpha=0.1$')
            plt.plot(x2, z2, 'b-', label=r'$\alpha=0.5$')
            plt.plot(x3, z3, 'c-', label=r'$\alpha=0.9$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('Vertical Load Force')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Vertical_Load_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()
            plt.figure(3)
            plt.plot(zx1, z1, 'r-', label=r'$\alpha=0.1$')
            plt.plot(zx2, z2, 'b-', label=r'$\alpha=0.5$')
            plt.plot(zx3, z3, 'c-', label=r'$\alpha=0.9$')
            plt.legend(loc='best')
            plt.xlabel('Fx')
            plt.ylabel('Fy')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Force_Evolution_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()"""

        # Newtonian Vs Non-Newtonian
        """if max(norm(tau1_vec.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==2 or j==1:
            # Plot Torque Data
            plt.figure(0)
            plt.plot(x1, y1, 'r-', label=r'Newtonian')
            plt.plot(x2, y2, 'b-', label=r'Oldroyd-B ($We=0.5$)')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('Torque')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Torque_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()
            plt.figure(1)
            plt.plot(x1, zx1, 'r-', label=r'Newtonian')
            plt.plot(x2, zx2, 'b-', label=r'Oldroyd-B ($We=0.5$)')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('Horzontal Load Force')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Horizontal_Load_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()
            plt.figure(2)
            plt.plot(x1, z1, 'r-', label=r'Newtonian')
            plt.plot(x2, z2, 'b-', label=r'Oldroyd-B ($We=0.5$)')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('Vertical Load Force')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Vertical_Load_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()
            plt.figure(3)
            plt.plot(zx1, z1, 'r-', label=r'Newtonian')
            plt.plot(zx2, z2, 'b-', label=r'Oldroyd-B ($We=0.5$)')
            plt.legend(loc='best')
            plt.xlabel('Fx')
            plt.ylabel('Fy')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Force_Evolution_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()"""

        

        # Plot Torque/Load for different Reynolds Numbers Numbers
        """if max(norm(tau1_vec.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==4 or j==2:
            # Plot Torque Data
            plt.figure(0)
            plt.plot(x1, y1, 'r-', label=r'$Re=25$')
            plt.plot(x2, y2, 'b-', label=r'$Re=50$')
            plt.plot(x3, y3, 'c-', label=r'$Re=100$')
            plt.plot(x4, y4, 'm-', label=r'$Re=200$')
            plt.legend(loc='best')
            plt.xlabel('t')
            plt.ylabel('$C$')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Torque_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()
            plt.figure(1)
            plt.plot(x1, zx1, 'r-', label=r'$Re=25$')
            plt.plot(x2, zx2, 'b-', label=r'$Re=50$')
            plt.plot(x3, zx3, 'c-', label=r'$Re=100$')
            plt.plot(x4, zx4, 'm-', label=r'$Re=200$')
            plt.legend(loc='best')
            plt.xlabel('t')
            plt.ylabel('$F_x$')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Horizontal_Load_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()
            plt.figure(2)
            plt.plot(x1, z1, 'r-', label=r'$Re=25$')
            plt.plot(x2, z2, 'b-', label=r'$Re=50$')
            plt.plot(x3, z3, 'c-', label=r'$Re=100$')
            plt.plot(x4, z4, 'm-', label=r'$Re=200$')
            plt.legend(loc='best')
            plt.xlabel('t)')
            plt.ylabel('$F_y$')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Vertical_Load_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()
            plt.figure(3)
            plt.plot(zx1, z1, 'r-', label=r'$Re=25$')
            plt.plot(zx2, z2, 'b-', label=r'$Re=50$')
            plt.plot(zx3, z3, 'c-', label=r'$Re=100$')
            plt.plot(zx4, z4, 'm-', label=r'$Re=200$')
            plt.legend(loc='best')
            plt.xlabel('$F_x$')
            plt.ylabel('$F_y$')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Force_Evolution_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()"""

        # Plot Torque/Load for different Mach Numbers
        """if max(norm(tau1_vec.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==1 or j==3:
            # Plot Torque Data
            plt.figure(0)
            plt.plot(x1, y1, 'r-', label=r'$Ma=0.0005$')
            plt.plot(x2, y2, 'b-', label=r'$Ma=0.001$')
            plt.plot(x3, y3, 'c-', label=r'$Ma=0.01$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('Torque')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/TorqueWe="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()
            plt.figure(1)
            plt.plot(x1, zx1, 'r-', label=r'$Ma=0.0005$')
            plt.plot(x2, zx2, 'b-', label=r'$Ma=0.001$')
            plt.plot(x3, zx3, 'c-', label=r'$Ma=0.01$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('Horzontal Load Force')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Horizontal_Load We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()
            plt.figure(2)
            plt.plot(x1, z1, 'r-', label=r'$Ma=0.0005$')
            plt.plot(x2, z2, 'b-', label=r'$Ma=0.001$')
            plt.plot(x3, z3, 'c-', label=r'$Ma=0.01$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('Vertical Load Force')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Vertical_Load We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()
            plt.figure(3)
            plt.plot(zx1, z1, 'r-', label=r'$Ma=0.0005$')
            plt.plot(zx2, z2, 'b-', label=r'$Ma=0.001$')
            plt.plot(zx3, z3, 'c-', label=r'$Ma=0.01$')
            plt.legend(loc='best')
            plt.xlabel('Fx')
            plt.ylabel('Fy')
            plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Force_We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()"""

            #Plot Kinetic and elasic Energies for different Weissenberg numbers at Re FIXED 
        """if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==5 or j==3 or j==1:
            # Kinetic Energy
            plt.figure(0)
            plt.plot(x1, ek1, 'r-', label=r'$Ma=0.0005$')
            plt.plot(x2, ek2, 'b-', label=r'$Ma=0.001$')
            plt.plot(x3, ek3, 'c-', label=r'$Ma=0.01$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$E_{kinetic}$')
            plt.savefig("Compressible Viscoelastic Flow Results/Energy/KineticEnergyRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()
            # Elastic Energy
            plt.figure(1)
            plt.plot(x1, ee1, 'r-', label=r'$Ma=0.0005$')
            plt.plot(x2, ee2, 'b-', label=r'$Ma=0.001$')
            plt.plot(x3, ee3, 'c-', label=r'$Ma=0.01$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$E_{elastic}$')
            plt.savefig("Compressible Viscoelastic Flow Results/Energy/ElasticEnergyRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()"""

            #Plot Kinetic and elasic Energies for different Weissenberg numbers at Re FIXED 
        if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==loopend or j==1:
            # Kinetic Energy
            plt.figure(0)
            plt.plot(x1, ek1, 'r-', label=r'$We=0$')
            plt.plot(x2, ek2, 'b-', label=r'$We=0.1$')
            plt.plot(x3, ek3, 'c-', label=r'$We=0.5$')
            plt.plot(x4, ek4, 'm-', label=r'$We=0.75$')
            plt.plot(x5, ek5, 'g-', label=r'$We=1.0$')
            plt.legend(loc='best')
            plt.xlabel('$t$', fontsize=16)
            plt.ylabel('$E_{k}$', fontsize=16)
            plt.savefig("Compressible Viscoelastic Flow Results/Energy/KineticEnergyRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()
            # Elastic Energy
            plt.figure(1)
            plt.plot(x1, ee1, 'r-', label=r'$We=0$')
            plt.plot(x2, ee2, 'b-', label=r'$We=0.1$')
            plt.plot(x3, ee3, 'c-', label=r'$We=0.5$')
            plt.plot(x4, ee4, 'm-', label=r'$We=0.75$')
            plt.plot(x5, ee5, 'g-', label=r'$We=1.0$')
            plt.legend(loc='best')
            plt.xlabel('$t$', fontsize=16)
            plt.ylabel('$E_{e}$', fontsize=16)
            plt.savefig("Compressible Viscoelastic Flow Results/Energy/ElasticEnergyRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()

            #Plot Kinetic and elasic Energies for different Thermal Expansion Coefficients 
        """if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==5 or j==3 or j==1:
            # Kinetic Energy
            plt.figure(0)
            plt.plot(x1, ek1, 'r-', label=r'$\alpha=0.1$')
            plt.plot(x2, ek2, 'b-', label=r'$\alpha=0.5$')
            plt.plot(x3, ek3, 'c-', label=r'$\alpha=0.9$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$E_{kinetic}$')
            plt.savefig("Compressible Viscoelastic Flow Results/Energy/KineticEnergyRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()
            # Elastic Energy
            plt.figure(1)
            plt.plot(x1, ee1, 'r-', label=r'$\alpha=0.1$')
            plt.plot(x2, ee2, 'b-', label=r'$\alpha=0.5$')
            plt.plot(x3, ee3, 'c-', label=r'$\alpha=0.9$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$E_{elastic}$')
            plt.savefig("Compressible Viscoelastic Flow Results/Energy/ElasticEnergyRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()"""

            #Plot Kinetic and elasic Energies for different Weissenberg numbers at Re FIXED 
        """if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==4 or j==2:
            # Kinetic Energy
            plt.figure(0)
            plt.plot(x1, ek1, 'r-', label=r'$Re=25$')
            plt.plot(x2, ek2, 'b-', label=r'$Re=50$')
            plt.plot(x3, ek3, 'c-', label=r'$Re=100$')
            plt.plot(x4, ek4, 'm-', label=r'$Re=200$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$E_{kinetic}$')
            plt.savefig("Compressible Viscoelastic Flow Results/Energy/KineticEnergyRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()
            # Elastic Energy
            plt.figure(1)
            plt.plot(x1, ee1, 'r-', label=r'$Re=25$')
            plt.plot(x2, ee2, 'b-', label=r'$Re=50$')
            plt.plot(x3, ee3, 'c-', label=r'$Re=100$')
            plt.plot(x4, ee4, 'm-', label=r'$Re=200$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$E_{elastic}$')
            plt.savefig("Compressible Viscoelastic Flow Results/Energy/ElasticEnergyRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()"""

        """fv = File("Paraview_Results/Velocity Results Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"DEVSS"+str(th)+"/velocity "+str(t)+".pvd")
        fv_x = File("Paraview_Results/Velocity Results Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"DEVSS"+str(th)+"/u_x "+str(t)+".pvd")
        fv_y = File("Paraview_Results/Velocity Results Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"DEVSS"+str(th)+"/u_y "+str(t)+".pvd")
        fmom = File("Paraview_Results/Velocity Results Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"DEVSS"+str(th)+"/mom "+str(t)+".pvd")
        fp = File("Paraview_Results/Pressure Results Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"DEVSS"+str(th)+"/pressure "+str(t)+".pvd")
        ftau_xx = File("Paraview_Results/Stress Results Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"DEVSS"+str(th)+"/tua_xx "+str(t)+".pvd")
        ftau_xy = File("Paraview_Results/Stress Results Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"DEVSS"+str(th)+"/tau_xy "+str(t)+".pvd")
        ftau_yy = File("Paraview_Results/Stress Results Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"DEVSS"+str(th)+"/tau_yy "+str(t)+".pvd")
        f_N1 = File("Paraview_Results/Stress Results Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"DEVSS"+str(th)+"/N1"+str(t)+".pvd")"""


        if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==1 or j==loopend:
            # Plot Stress/Normal Stress Difference
            sigma_xx = project(sigmacon(u1, p1, tau1)[0,0], Q)
            mplot(sigma_xx)
            plt.colorbar()
            plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/sigma_xxRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf() 
            tau_xx=project(tau1[0,0],Q)
            mplot(tau_xx)
            plt.colorbar()
            plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_xxRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf() 
            tau_xy=project(tau1[1,0],Q)
            mplot(tau_xy)
            plt.colorbar()
            plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_xyRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf() 
            tau_yy=project(tau1[1,1],Q)
            mplot(tau_yy)
            plt.colorbar()
            plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_yyRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf() 
            N1=project(tau1[0,0]-tau1[1,1],Q)
            mplot(N1)
            plt.colorbar()
            plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/FirstNormalStressDifferenceRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()


            # Matlab Plot of the Solution at t=Tf
            mplot(rho1)
            plt.colorbar()
            plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/DensityRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
            plt.clf()
            mom_mag = project(rho1*magnitude(u1), Q)
            mplot(mom_mag)
            plt.colorbar()
            plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/MomentumRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
            plt.clf()
            mplot(p1)
            plt.colorbar()
            plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/PressureRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
            plt.clf()
            theta0_Q = project(theta0, Q)
            mplot(theta0_Q)
            plt.colorbar()
            plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/TemperatureRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
            plt.clf()
            divu=project(div(u1),Q)
            mplot(divu)
            plt.colorbar()
            plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/CompressionRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
            plt.clf()
            mplot(psi)
            plt.colorbar()
            plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/stream_functionRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
            plt.clf()
            therm_visc = project(betav + (1.-betav)*phi_ewm(tau1, theta1, k_ewm, B), Q)
            mplot(therm_visc)
            plt.colorbar()
            plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/viscosity_functionRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
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
            pp = mlab.griddata(xvals, yvals, pvals, xx, yy, interp='linear') # u(x,y) data so that it can be used by 

            plt.contour(XX, YY, pp, 30)
            plt.colorbar()
            plt.title('Pressure')   # PRESSURE CONTOUR PLOT
            plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Pressure Contours Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
            plt.clf()


            #Plot TEMPERATURE Contours USING MATPLOTLIB
            # Scalar Function code

            #Set Values for inner domain as ZERO


            Tj=Expression('0', degree=1) #Expression for the 'pressure' in the domian
            Tjq=interpolate(Tj, Q1)
            Tjvals=Tjq.vector().get_local()

            Tvals = theta0_Q.vector().get_local() # GET SOLUTION T= T(x,y) list
            Tvals = np.concatenate([Tvals, Tjvals])  #Merge two arrays for Temperature values

            TT = mlab.griddata(xvals, yvals, Tvals, xx, yy, interp='linear') # u(x,y) data so that it can be used by 

            plt.contour(XX, YY, TT, 30)
            plt.colorbar()
            plt.title('Temperature')   # TEMPERATURE CONTOUR PLOT
            plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Temperature Contours Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
            plt.clf()



            # Plot Velocity Field Contours (MATPLOTLIB)

            # Set Velocity on the bearing to zero
            Q1=FunctionSpace(meshc, "CG", 1)
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


            uu = mlab.griddata(xvals, yvals, uvals, xx, yy, interp='linear') 
            vv = mlab.griddata(xvals, yvals, vvals, xx, yy, interp='linear') 

            #Determine Speed 
            speed = np.sqrt(uu*uu+ vv*vv)

            plot3 = plt.figure()
            plt.streamplot(XX, YY, uu, vv,  
                           density=5,              
                           color=speed/speed.max(),  
                           cmap=cm.gnuplot,                         # colour map
                           linewidth=0.5*speed/speed.max()+0.5)       # line thickness
            plt.colorbar()
            plt.title('Isostreams')
            plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Velocity Contours Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")   
            plt.clf()                                                                    # display the plot



        plt.close()


        if dt < tol:
           j=loopend+1
           break


        if j==loopend:
            jjj+=1
            update_progress("Simulation"+str(jjj), 1)
            j = 0
            jj = 1
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

        if jjj==3:
            quit()


    if jj == 0: 
        # Calculate Stress Residual 
        F1R = Fdef(u1, tau1)  
        F1R_vec = as_vector([F1R[0,0], F1R[1,0], F1R[1,1]])
        restau0 = We/dt*(tau1_vec-tau0_vec) + We*F1R_vec + tau1_vec - I_vec #- diss_vec 
        res_test = inner(restau0,restau0)                            

        kapp = project(res_test, Qt) # Error Function
        norm_kapp = normalize_solution(kapp) # normalised error function

        ratio = 0.3/(1*err_count + 1.0) # Proportion of cells that we want to refine
        tau_average = project((tau1_vec[0]+tau1_vec[1]+tau1_vec[2])/3.0 , Qt)
        error_rat = project(kapp/(tau_average + 0.000001) , Qt)
        error_rat = absolute(error_rat)

        jj=1 

        if error_rat.vector().get_local().max() > 0.01 and err_count < 1:
           err_count+=1
           mesh = adaptive_refinement(mesh, norm_kapp, ratio)
           #mplot(error_rat)
           #plt.colorbar()
           #plt.savefig("adaptive-error-function.eps")
           #plt.clf()
           #mplot(mesh)
           #plt.savefig("adaptive-mesh.eps")
           #plt.clf()
           #jj=0
           conv_fail = 0

        # Reset Parameters
        corr=1    
        j = 0
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



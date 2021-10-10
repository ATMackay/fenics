"""Inompressible Double Glazing Problem """

from fenics_fem import *  # Import Base Code for LDC Problem


def main(input_csv,mesh_resolution,simulation_time, mesh_refinement):
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
    nus1=list()
    nus2=list()
    nus3=list()
    nus4=list()
    nus5=list()

    # Define Geometry/Mesh
    B=1
    L=1
    x_0 = 0
    y_0 = 0
    x_1 = 1
    y_1 = 1

    # Mesh refinement comparison Loop
    mm = 48
    


    # Choose Mesh to Use

    mesh = DGP_Mesh(mm)
    gdim = mesh.geometry().dim() # Mesh Geometry

    #mesh = refine_walls(mesh, 2)

    mplot(mesh)
    plt.savefig("fine_skewed_grid.png")
    plt.clf()
    plt.close()
    #quit()

    #Define Boundaries 

    N=20
    bottom_bound = 0.5*(1+tanh(-N)) 
    top_bound = 0.5*(1+tanh(N)) 

    # Experiment Run Time
    dt = 0.001  #Timestep

    T_f = 10.0
    Tf = T_f 
    tol = 0.0001

    tol = 0.0001
    U = 1
    betav = 0.5    
    Ra = 1000                           #Rayleigh Number
    Pr = 2.
    We = 0.01                          #Weisenberg NUmber
    Vh = 0.005
    T_0 = 300
    T_h = 350
    Bi = 0.1

    c1 = 0.1
    c2 = 0.0001
    th = 1.0              # DEVSS

    no_slip = No_slip()
    left = Left()
    right = Right()
    top = Top()


    # MARK SUBDOMAINS (Create mesh functions over the cell facets)
    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    sub_domains.set_all(5)
    no_slip.mark(sub_domains, 0)
    left.mark(sub_domains, 2)
    right.mark(sub_domains, 3)
    top.mark(sub_domains, 4)


    plot(sub_domains, interactive=False)        # DO NOT USE WITH RAVEN
    #quit()

    #Define Boundary Parts

    boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) #FacetFunction("size_t", mesh)
    no_slip.mark(boundary_parts,0)
    left.mark(boundary_parts,1)
    right.mark(boundary_parts,2)
    top.mark(boundary_parts,3)
    ds = Measure("ds")[boundary_parts]

    # Define function spaces (P2-P1)

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


    W = FunctionSpace(mesh,V_s*Z_s)             # F.E. Spaces 
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


    # Initial Conformation Tensor
    I_vec = Expression(('1.0','0.0','1.0'), degree=2)
    initial_guess_conform = project(I_vec, Zc)




    # The  projected  rate -of-strain
    D_proj_vec = Function(Zc)
    D_proj = as_matrix([[D_proj_vec[0], D_proj_vec[1]],
                        [D_proj_vec[1], D_proj_vec[2]]])





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
    U = 1
    betav = 0.5     
    Ra = 1000                           #Rayleigh Number
    Pr = 1
    We = 0.5                           #Weisenberg NUmber
    Di = 0.005                         #Diffusion Number
    Vh = 0.005
    T_0 = 300
    T_h = 350
    Bi = 0.2
    c0 = 0
    Ma = 0 




    # Define boundary/stabilisation FUNCTIONS
    td= Constant('5')
    e = Constant('6')
    T_bl = Expression('T_0+(T_h-T_0)*sin((x[0]+x[1])*2*pi/L)', degree=2, T_0=T_0, T_h=T_h, pi=pi, L=L, B=B)
    T_bb = Expression('T_0+(T_h-T_0)*sin((x[0]+x[1])*2*pi/B)', degree=2, T_0=T_0, T_h=T_h, pi=pi, L=L, B=B)
    rampd = Expression('0.5*(1+tanh(8*(2.0-t)))', degree=2, t=0.0)
    rampu = Expression('0.5*(1+tanh(16*(t-2.0)))', degree=2, t=0.0)
    ramped_T = Expression('0.5*(1+tanh(8*(t-0.5)))*(T_h-T_0)+T_0', degree=2, t=0.0, T_0=T_0, T_h=T_h)
    f = Expression(('0','-1'), degree=2)


    # Interpolate Stabilisation Functions
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)


    # Default Stabilisation Parameters
    alph = 0
    th = 0                  # DEVSS
    c1 = 0*h                # SUPG / SU
    c2 = 0                  # Artificial Diffusion

    # Define unit Normal/tangent Vector at inner and outer Boundary (Method 2)
    n0 =  Expression(('-1' , '0'), degree=2)
    n1 =  Expression(('0' , '1' ), degree=2)
    n2 =  Expression(('1' , '0' ), degree=2)
    n3 =  Expression(('0' , '-1'), degree=2)



    # Dirichlet Boundary Conditions  (LID DRIVEN CAVITY)
    noslip0  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), no_slip)  # No Slip boundary conditions on the left wall
    noslip1 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), left)  # No Slip boundary conditions on the left wall
    noslip2 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), right)  # No Slip boundary conditions on the left wall
    noslip3 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), top)  # No Slip boundary conditions on the left wall
    temp_left =  DirichletBC(Q, ramped_T, left)    #Temperature on Omega0 
    temp_right =  DirichletBC(Q, T_0, right)    #Temperature on Omega2 

    #Collect Boundary Conditions
    bcu = [noslip0, noslip1, noslip2, noslip3]
    bcp = []
    bcT = [temp_left, temp_right]    #temp0, temp2
    bctau = []


    # Loop Experiments
    loopend = 5
    j=0
    jjj=0
    jj=0
    while j < loopend:
        j+=1
        t=0.0
    
        if jjj==0:
            Ra=1000
        if jjj==1:
            Ra=2000  
        if jjj==2:
            Ra=5000
        if jjj==3:
            Ra=10000   
        if jjj==4:
            Ra=10000


        # Comparing different WEISSENBERG Numbers (We=0.1,0.2,0.3,0.4,0.5) at Re=__
        betav=0.5
        if j==1:
        betav = 0.5
        We=0.05
        elif j==2:
        We=0.1
        elif j==3:
        We=0.5
        elif j==4:
        We=1.0
        elif j==5:
        We=2.0



        # Comparing different REYNOLDS NUMBERS Numbers (Re=0,5,10,25,50) at We=0.5
        """conv=1                                      # Non-inertial Flow Parameter (Re=0)
        We=0.5
        if j==1:
        Ra=100
        elif j==2:
        Ra=500
        elif j==3:
        Ra=1000
        elif j==4:
        Ra=5000
        elif j==5:
        Ra=10000"""


        # Comparing Effect of DEVSS/ SUPG Stabilisation Parameter
        """alph = 0.125
        th=10E-16
        c1=alph*h_ska    #SUPG Stabilisation
        We=0.5
        conv=10E-15
        Re=1
        if j==1:
            th=0
        elif j==2:
            th=0.1*(1.0-betav)
        elif j==3:
            th=0.5*(1.0-betav)"""



        # Comparing Effect of Diffusion Stabilisation Parameter
        """c1=h_ka     #SUPG Stabilisation
        th=0.1*(1.0-betav)          #DEVSS Stabilisation
        We=0.5
        conv=10E-15
        Ra=1
        if j==1:
            c2=10E-6*h_ka
        elif j==2:
            c2=rampd*0.1*h_ka"""

        # Comparing the Effect of SUPG Stabilisation
        """th=10E-16        #DEVSS Stabilisation
        c2=10E-6*h_ka    #Diffusion Stabilisation
        We=0.5
        Ra=10
        if j==1:
            c1=h_ka*10E-10
        elif j==2:
            c1=0.1*h_ka
        elif j==3:
            c1=h_ka"""
        
            

        # Continuation in Reynolds/Weissenberg Number Number (Re-->20Re/We-->20We)
        Wet=Expression('(We/100)*(1.0+99.0*0.5*(1.0+tanh(0.7*t-5.0)))', t=0.0, We=We, degree=2)


        print '############# TIME SCALE ############'
        print 'Timestep size (s):', dt
        print 'Finish Time (s):', Tf
        print 'Steps', int(Tf/dt)

        print '############# Scalings & Nondimensional Parameters ############'
        print 'Characteristic Length (m):', L
        print 'Characteristic Velocity (m/s):', 1.0
        print 'Rayleigh Number:', Ra
        print 'Prandtl Number:', Pr
        print 'Weissenberg Number:', We
        print 'Viscosity Ratio:', betav
        print 'Diffusion Number:' ,Di
        print 'Viscous Heating Number:', Vh

        Np= len(p0.vector().get_local())
        Nv= len(w0.vector().get_local())   
        Ntau= len(tau0_vec.vector().get_local())
        dof= 2*Nv+Ntau+Np
        print '############# Discrete Space Characteristics ############'
        print 'Degree of Elements', order
        print 'Mesh: %s x %s' %(mm, mm)
        print('Size of Pressure Space = %d ' % Np)
        print('Size of Velocity/DEVSS Space = %d ' % Nv)
        print('Size of Stress Space = %d ' % Ntau)
        print('Degrees of Freedom = %d ' % dof)
        print 'Number of Cells:', mesh.num_cells()
        print 'Number of Vertices:', mesh.num_vertices()
        print 'Minimum Cell Diamter:', mesh.hmin()
        print 'Maximum Cell Diamter:', mesh.hmax()
        print '############# Stabilisation Parameters ############'
        print 'DEVSS Parameter:', th


    
        # Initial Conformation Tensor
        I_vec = Expression(('1.0','0.0','1.0'), degree=2)
        initial_guess_conform = project(I_vec, Zc)
        assign(tau0_vec, initial_guess_conform)         # Initial guess for conformation tensor is Identity matrix   

        # Initial Temperature Field
        T_initial_guess = project(T_0, Q)
        T0.assign(T_initial_guess)


        


        #Define Variable Parameters, Strain Rate and other tensors
        sr = (grad(u) + transpose(grad(u)))
        srg = grad(u)
        #gamdots = inner(Dincomp(u1),grad(u1))
        #gamdotp = inner(tau1,grad(u1))

        gamdot = inner(sigma(u1, p1, tau1), grad(u1))


        # Nondimensionalised Temperature
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
        
        DEVSSl_u12 = 2*(1-betav)*inner(Dincomp(u),Dincomp(v))*dx    
        DEVSSr_u12 = 2*inner(D0,Dincomp(v))*dx   
        DEVSSl_u1 = 2*(1-betav)*inner(Dincomp(u),Dincomp(v))*dx    
        DEVSSr_u1 = 2*inner(D0,Dincomp(v))*dx 



        # Set up Krylov Solver 

        # Use amg preconditioner if available
        prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

        # Use nonzero guesses - essential for CG with non-symmetric BC
        parameters['krylov_solver']['nonzero_initial_guess'] = True
        parameters['krylov_solver']['monitor_convergence'] = False
        
        solveru = KrylovSolver("bicgstab", "default")
        solvertau = KrylovSolver("bicgstab", "default")
        solverp = KrylovSolver("bicgstab", "default")

        #Folder To Save Plots for Paraview
        #fv=File("Velocity Results Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"/velocity "+str(t)+".pvd")
    

        #Lists for Energy Values
        x=list()
        ee=list()
        ek=list()

        #ftau=File("Incompressible Viscoelastic Flow Results/Paraview/Stress_th"+str(th)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"/stress "+str(t)+".pvd")
        #fv=File("Incompressible Viscoelastic Flow Results/Paraview/Velocity_th"+str(th)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"/velocity "+str(t)+".pvd")

        # Time-stepping
        t = 0.0
        iter = 0            # iteration counter
        maxiter = 10000000
        while t < Tf + DOLFIN_EPS and iter < maxiter:
            #tic = time.time()
            iter += 1
            print"t = %s,  Iteration = %d, Convergence Failures = %s, Loop = %s - %s" %(t, iter, jj, jjj, j)

            # Set Function timestep
            ramped_T.t = t

            # Update Stabilisation (Copy and Paste Stabilisation Technique from above)
            F1R = Fdef(u1, tau1)  #Compute the residual in the STRESS EQUATION
            F1R_vec = as_vector([F1R[0,0], F1R[1,0], F1R[1,1]])
            Dincomp1_vec = as_vector([Dincomp(u1)[0,0], Dincomp(u1)[1,0], Dincomp(u1)[1,1]])
            restau0 = We/dt*(tau1_vec-tau0_vec) + We*F1R_vec + tau1_vec - I_vec
            res_test = project(restau0, Zd)
            res_orth = project(restau0-res_test, Zc)                                
            res_orth_norm_sq = project(inner(res_orth,res_orth), Qt)     # Project residual norm onto discontinuous space
            res_orth_norm = np.power(res_orth_norm_sq, 0.5)
            kapp = project(res_orth_norm, Qt)
            kapp = absolute(kapp)
            LPSl_stress = inner(kapp*h*c1*grad(tau),grad(Rt))*dx + inner(kapp*h*c2*div(tau),div(Rt))*dx  # Stress Stabilisation
                    
            U12 = 0.5*(u1 + u0)    
            # Update Solutions
            if iter > 1:
                w0.assign(w1)
                T0.assign(T1)
                rho0.assign(rho1)
                p0.assign(p1)
                tau0_vec.assign(tau1_vec)

            (u0, D0_vec)=w0.split()

            #DEVSSr_u12 = 2.*Pr*(1.0-betav)*inner(D0,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS
        
            U = 0.5*(u + u0)              
            # VELOCITY HALF STEP
            Du12Dt = (2.0*(u - u0) / dt + dot(u0, nabla_grad(u0)))
            Fu12 = dot(Du12Dt, v)*dx + \
                + inner(sigma(U, p0, tau0), Dincomp(v))*dx + Ra*Pr*inner(theta0*f,v)*dx \
                + dot(p0*n, v)*ds - dot(Pr*betav*nabla_grad(U)*n, v)*ds\
                - (Pr*(1-betav)/We)*dot(tau0*n, v)*ds\
                + inner(D-Dincomp(u),R)*dx 
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

            DEVSSr_u1 = 2*Pr*(1.-betav)*inner(D12,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS     

            """# STRESS Half Step
            F12 = dot(u12,grad(tau)) - dot(grad(u12),tau) - dot(tau,tgrad(u12)) # Convection/Deformation Terms
            lhs_tau12 = (We/dt+1.0)*tau + We*F12                             # Left Hand Side
            rhs_tau12= (We/dt)*tau0 + 2.0*(1.0-betav)*Dincomp(u0)                     # Right Hand Side

            a3 = inner(lhs_tau12,Rt)*dx                                 # Weak Form
            L3 = inner(rhs_tau12,Rt)*dx

            a3 += SUPGl3             # SUPG Stabilisation LHS
            L3 += SUPGr3             # SUPG / SU Stabilisation RHS
            A3=assemble(a3)
            b3=assemble(L3)
            [bc.apply(A3, b3) for bc in bctau]
            solve(A3, tau12_vec.vector(), b3, "bicgstab", "default")
            end()"""


            #Predictor Step [U*]
            lhsFus = ((u - u0)/dt + dot(u12, nabla_grad(U)))
            Fus = dot(lhsFus, v)*dx + inner(D-Dincomp(u),R)*dx  + \
                + inner(sigma(U, p0, tau0), Dincomp(v))*dx + Ra*Pr*inner(theta0*f,v)*dx \
                + dot(p0*n, v)*ds - Pr*betav*(dot(nabla_grad(U)*n, v)*ds) \
                - Pr*((1.0-betav)/We)*dot(tau0*n, v)*ds\
                    
                
            a2= lhs(Fus)
            L2= rhs(Fus) 
            a2_stab = a2 + th*DEVSSl_u1 
            L2_stab = L2 + th*DEVSSr_u1
            A2 = assemble(a2_stab)
            b2 = assemble(L2_stab)
            [bc.apply(A2, b2) for bc in bcu]
            solve(A2, ws.vector(), b2, "bicgstab", "default")
            end()
            (us, Ds_vec) = ws.split()


            # Pressure Correction
            a5=inner(grad(p),grad(q))*dx 
            L5=inner(grad(p0),grad(q))*dx + (1.0/dt)*inner(us,grad(q))*dx
            A5 = assemble(a5)
            b5 = assemble(L5)
            [bc.apply(A5, b5) for bc in bcp]
            #[bc.apply(p1.vector()) for bc in bcp]
            solve(A5, p1.vector(), b5, "bicgstab", "default")
            end()
            
            #Velocity Update
            lhs_u1 = (1./dt)*u                                          # Left Hand Side
            rhs_u1 = (1./dt)*us                                         # Right Hand Side
            a7=inner(lhs_u1,v)*dx + inner(D-Dincomp(u),R)*dx                                           # Weak Form
            L7=inner(rhs_u1,v)*dx + 0.5*inner(p1-p0,div(v))*dx - 0.5*dot((p1-p0)*n, v)*ds
                #DEVSS Stabilisation
            a1+= th*DEVSSl_u1                     
            L1+= th*DEVSSr_u1 
            A7 = assemble(a7)
            b7 = assemble(L7)
            [bc.apply(A7, b7) for bc in bcu]
            solve(A7, w1.vector(), b7)
            end()
            (u1, D1_vec) = w1.split()

            D1 = as_matrix([[D1_vec[0], D1_vec[1]],
                            [D1_vec[1], D1_vec[2]]])

            U12 = 0.5*(u1 + u0)   

            # Stress Full Step
            """F1 = dot(u1,grad(tau)) - dot(grad(u1),tau) - dot(tau,tgrad(u1)) # Convection/Deformation Terms
            lhs_tau1 = (We/dt+1.0)*tau + We*F1                             # Left Hand Side
            rhs_tau1= (We/dt)*tau0 + 2.0*(1.0-betav)*Dincomp(u12)          # Right Hand Side

            a4 = inner(lhs_tau1,Rt)*dx
            L4 = inner(rhs_tau1,Rt)*dx

            a4 += SUPGl4       # SUPG / SU Stabilisation
            L4 += SUPGr4    


            A4=assemble(a4)                                     # Assemble System
            b4=assemble(L4)
            [bc.apply(A4, b4) for bc in bctau]
            solve(A4, tau1_vec.vector(), b4, "bicgstab", "default")
            end()"""

            if betav < 0.95:
                # Stress Full Step
                stress_eq = (We/dt+1.0)*tau  +  We*Fdef(u1,tau) - (We/dt)*tau0 - Identity(len(u))
                A = inner(stress_eq,Rt)*dx
                a4 = lhs(A)
                L4 = rhs(A) 
                a4_stab = a4 + LPSl_stress  # [SUPGl4, SUl4, LPSl_stab, LPSl_stress, diff_stab, 0]
                L4 += 0            # [SUPGr4, SUr4, LPSr_stab, LPS_res_stab, 0]   

                A4=assemble(a4_stab)                                     # Assemble System
                b4=assemble(L4)
                [bc.apply(A4, b4) for bc in bctau]
                solvertau.solve(A4, tau1_vec.vector(), b4)
                end()


            # Temperature Update (FIRST ORDER)
            gamdot = inner(sigma(u1, p1, tau1), grad(u1))
            lhs_theta1 = (1.0/dt)*thetal + dot(u1,grad(thetal))
            rhs_theta1 = (1.0/dt)*thetar + dot(u1,grad(thetar)) + (1.0/dt)*theta0 + Vh*gamdot
            a8 = inner(lhs_theta1,r)*dx + inner(grad(thetal),grad(r))*dx + inner(We*tau1*grad(thetal),grad(r))*dx
            L8 = inner(rhs_theta1,r)*dx + inner(grad(thetar),grad(r))*dx + Bi*inner(grad(theta0),n*r)*ds(3) + inner(We*tau1*grad(thetar),grad(r))*dx

            A8=assemble(a8)                                     # Assemble System
            b8=assemble(L8)
            [bc.apply(A8, b8) for bc in bcT]
            solve(A8, T1.vector(), b8)
            end()


            # Energy Calculations
            E_k=assemble(0.5*dot(u1,u1)*dx)
            E_e=assemble((tau1[0,0]+tau1[1,1]-2.0)*dx)

            # Nusselt Number 
            theta1 = project((T1-T_0)/(T_h-T_0), Q)
            Tdx = inner(grad(theta1),n) 
            Nus = assemble(-Tdx*ds(2))

            #print 'Nu', Nus
            

            
            
            # Record Elastic & Kinetic Energy Values (Method 1)
            if j==1:
            x1.append(t)
            ek1.append(E_k)
            ee1.append(E_e)
            nus1.append(Nus)
            if j==2:
            x2.append(t)
            ek2.append(E_k)
            ee2.append(E_e)
            nus2.append(Nus)
            if j==3:
            x3.append(t)
            ek3.append(E_k)
            ee3.append(E_e)
            nus3.append(Nus)
            if j==4:
            x4.append(t)
            ek4.append(E_k)
            ee4.append(E_e)
            nus4.append(Nus)
            if j==5:
            x5.append(t)
            ek5.append(E_k)
            ee5.append(E_e)
            nus5.append(Nus)



            if norm(w1.vector(), 'linf') > 10E10 or np.isnan(sum(w1.vector().get_local())):
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
                j-=1                            # Extend loop
                jj+= 1                          # Convergence Failures
                if jj>0:
                    Tf= max((iter-25)*dt,1.0)

                #Reset Functions

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

                (u0, D0_vec) = w0.split()
                (u12, D12_vec) = w12.split()
                (u1, D1_vec) = w1.split()
                (us, Ds_vec) = ws.split()
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
                break


            # Plot solution
            #if t>0.0:
                #plot(tau1[0,0], title="Normal Stress", rescale=True)
                #plot(p1, title="Pressure", rescale=True)
                #plot(u1, title="Velocity", rescale=True, mode = "auto")
                #plot(T1, title="Temperature", rescale=True)
                    

            # Move to next time step
            t += dt
            #elapsed = time.time() - tic
            #print elapsed

        mplot(kapp)
        plt.colorbar()
        plt.savefig("kappa.png")
        plt.clf()
        f_corr = File("Paraview_Results/Stabilisation_Ra="+str(Ra)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"/kappa "+str(t)+".pvd")
        #f_corr << kapp

        u1 = project(u1, V)
        psi = stream_function(u1)
        psi_min = min(psi.vector().get_local())
        min_loc = min_location(psi)
        with open("Incompressible Stream-Function.txt", "a") as text_file:
            text_file.write("Ra="+str(Ra)+", We="+str(We)+", t="+str(t)+"----- psi_min="+str(psi_min)+"---"+str(min_loc)+'\n')

        """# Data on Kinetic/Elastic Energies
        with open("Incompressible ConformEnergy.txt", "a") as text_file:
            text_file.write("Ra="+str(Ra)+", We="+str(We)+", t="+str(t)+", E_k="+str(E_k)+", E_e="+str(E_e)+'\n')

        # Data on Max Flow Speed
        u_abs = magnitude(u1)
        u_abs = project(u_abs, Q)
        u_max = u_abs.vector().get_local().max()
        with open("Incompressible Max Flow Speed.txt", "a") as text_file:
            text_file.write("Ra="+str(Ra)+", We="+str(We)+", t="+str(t)+", u_max"+str(u_max)+'\n')"""

        # Nusslet Number data
        if j==1:
        Nus_max = max(nus1)
        if j==2:
        Nus_max = max(nus2)
        if j==3:
        Nus_max = max(nus3)
        if j==4:
        Nus_max = max(nus4)
        if j==5:
        Nus_max = max(nus5) 
        with open("Nusslet Number.txt", "a") as text_file:
            text_file.write("Ra="+str(Ra)+", We="+str(We)+", t="+str(t)+":  Nu="+str(Nus)+\
                            "  Max Nu="+str(Nus_max)+'\n'+'\n')



        if j==3:
            peakEk1 = max(ek1)
            peakEk2 = max(ek2)
            peakEk3 = max(ek3)
            with open("Incompressible ConformEnergy.txt", "a") as text_file:
                text_file.write("Ra="+str(Ra)+", We="+str(We)+"-------Peak Kinetic Energy: "+str(peakEk3)+"Incomp Kinetic En"+str(peakEk1)+'\n')

        # Plot Mesh Convergence Data 
        """if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E6 and j==5:
            # Kinetic Energy
            plt.figure(0)
            plt.plot(x1, ek1, 'r-', label=r'M1')
            plt.plot(x2, ek2, 'b-', label=r'M2')
            plt.plot(x3, ek3, 'c-', label=r'M3')
            plt.plot(x4, ek4, 'm-', label=r'M4')
            plt.plot(x5, ek5, 'g-', label=r'M5')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('E_k')
            plt.savefig("Incompressible Viscoelastic Flow Results/Stability-Convergence/Mesh_KineticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
            # Elastic Energy
            plt.figure(1)
            plt.plot(x1, ee1, 'r-', label=r'M1')
            plt.plot(x2, ee2, 'b-', label=r'M2')
            plt.plot(x3, ee3, 'c-', label=r'M3')
            plt.plot(x4, ee4, 'm-', label=r'M4')
            plt.plot(x5, ee5, 'g-', label=r'M5')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('E_e')
            plt.savefig("Incompressible Viscoelastic Flow Results/Stability-Convergence/Mesh_ElasticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
            plt.clf()"""

            #Plot Kinetic and elasic Energies for different REYNOLDS numbers at constant Weissenberg Number    
        """if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E6 and j==loopend or j==3 or j==1:
            # Kinetic Energy
            plt.figure(0)
            plt.plot(x1, ek1, 'r-', label=r'$Ra=100$')
            plt.plot(x2, ek2, 'b-', label=r'$Ra=500$')
            plt.plot(x3, ek3, 'c-', label=r'$Ra=1000$')
            plt.plot(x4, ek4, 'm-', label=r'$Ra=5000$')
            plt.plot(x5, ek5, 'g-', label=r'$Ra=10000$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('E_k')
            plt.savefig("Incompressible Viscoelastic Flow Results/Energy/Fixed_We_KineticEnergyRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
            # Elastic Energy
            plt.figure(1)
            plt.plot(x1, ee1, 'r-', label=r'$Ra=100$')
            plt.plot(x2, ee2, 'b-', label=r'$Ra=500$')
            plt.plot(x3, ee3, 'c-', label=r'$Ra=1000$')
            plt.plot(x4, ee4, 'm-', label=r'$Ra=5000$')
            plt.plot(x5, ee5, 'g-', label=r'$Ra=10000$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('E_e')
            plt.savefig("Incompressible Viscoelastic Flow Results/Energy/Fixed_We_ElasticEnergyRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
            plt.clf()"""

        plt.close()

            #Plot Kinetic and elasic Energies for different Weissenberg numbers at Re=0 (METHOD 2)
        if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E6 and j==loopend or j==1 or j==2:
            # Kinetic Energy
            plt.figure(0)
            plt.plot(x1, ek1, 'r-', label=r'$We=0$ (Newtonian)')
            plt.plot(x2, ek2, 'b-', label=r'$We=0.1$')
            plt.plot(x3, ek3, 'c-', label=r'$We=0.5$')
            plt.plot(x4, ek4, 'm-', label=r'$We=1.0$')
            plt.plot(x5, ek5, 'g-', label=r'$We=2.0$')
            plt.legend(loc='best')
            plt.xlabel('$t$')
            plt.ylabel('$E_k$')
            plt.savefig("Incompressible Viscoelastic Flow Results/Energy/KineticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
            plt.clf()
            # Elastic Energy
            plt.figure(1)
            plt.plot(x1, ee1, 'r-', label=r'$We=0$ (Newtonian)')
            plt.plot(x2, ee2, 'b-', label=r'$We=0.1$')
            plt.plot(x3, ee3, 'c-', label=r'$We=0.5$')
            plt.plot(x4, ee4, 'm-', label=r'$We=1.0$')
            plt.plot(x5, ee5, 'g-', label=r'$We=2.0$')
            plt.legend(loc='best')
            plt.xlabel('$t$')
            plt.ylabel('$E_e$')
            plt.savefig("Incompressible Viscoelastic Flow Results/Energy/ElasticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
            plt.clf()
            # Nusslet Number
            plt.figure(1)
            plt.plot(x1, nus1, 'r-', label=r'$We=0$ (Newtonian)')
            plt.plot(x2, nus2, 'b-', label=r'$We=0.1$')
            plt.plot(x3, nus3, 'c-', label=r'$We=0.5$')
            plt.plot(x4, nus4, 'm-', label=r'$We=1.0$')
            plt.plot(x5, nus5, 'g-', label=r'$We=2.0$')
            plt.legend(loc='best')
            plt.xlabel('$t$')
            plt.ylabel('$Nu$')
            plt.savefig("Incompressible Viscoelastic Flow Results/Energy/NussletnumberTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
            plt.clf()

        """if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E6 and j==1:
            # Kinetic Energy
            plt.figure(0)
            plt.plot(x1, ek1, 'r-', label=r'$We=0$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('E_k')
            plt.savefig("Incompressible Viscoelastic Flow Results/Energy/KineticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
            plt.clf()
            # Elastic Energy
            plt.figure(1)
            plt.plot(x1, ee1, 'r-', label=r'$We=0$')
            plt.xlabel('time(s)')
            plt.ylabel('E_e')
            plt.savefig("Incompressible Viscoelastic Flow Results/Energy/ElasticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
            plt.clf()"""

        plt.close()


            # Comparing Kinetic & Elastic Energies for different Stablisation parameters
        """if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and abs(E_k) < 10 and j==3 or j==1:
            # Kinetic Energy
            plt.figure(0)
            plt.plot(x1, ek1, 'r-', label=r'$\theta=0$')
            plt.plot(x2, ek2, 'b-', label=r'$\theta=(1-\beta)/10$')
            plt.plot(x3, ek3, 'c-', label=r'$\theta=1-\beta$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('E_k')
            plt.savefig("Incompressible Viscoelastic Flow Results/Energy/KineticEnergyTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"We="+str(We)+"dt="+str(dt)+".png")
            plt.clf()
            # Elastic Energy
            plt.figure(1)
            plt.plot(x1, ee1, 'r-', label=r'$\theta=0$')
            plt.plot(x2, ee2, 'b-', label=r'$\theta=(1-\beta)/10$')
            plt.plot(x3, ee3, 'c-', label=r'$\theta=\beta$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('E_e')
            plt.savefig("Incompressible Viscoelastic Flow Results/Energy/ElasticEnergyTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"We="+str(We)+"dt="+str(dt)+".png")
            plt.clf()"""

        fv = File("Paraview_Results/Velocity Results Ra="+str(Ra)+"We="+str(We)+"b="+str(betav)+"DEVSS"+str(th)+"/velocity "+str(t)+".pvd")
        fv_x = File("Paraview_Results/Velocity Results Ra="+str(Ra)+"We="+str(We)+"b="+str(betav)+"DEVSS"+str(th)+"/u_x "+str(t)+".pvd")
        fv_y = File("Paraview_Results/Velocity Results Ra="+str(Ra)+"We="+str(We)+"b="+str(betav)+"DEVSS"+str(th)+"/u_y "+str(t)+".pvd")
        ftheta = File("Paraview_Results/Velocity Results Ra="+str(Ra)+"We="+str(We)+"b="+str(betav)+"DEVSS"+str(th)+"/mom "+str(t)+".pvd")
        fp = File("Paraview_Results/Pressure Results Ra="+str(Ra)+"We="+str(We)+"b="+str(betav)+"DEVSS"+str(th)+"/pressure "+str(t)+".pvd")
        ftau_xx = File("Paraview_Results/Stress Results Ra="+str(Ra)+"We="+str(We)+"b="+str(betav)+"DEVSS"+str(th)+"/tua_xx "+str(t)+".pvd")
        ftau_xy = File("Paraview_Results/Stress Results Ra="+str(Ra)+"We="+str(We)+"b="+str(betav)+"DEVSS"+str(th)+"/tau_xy "+str(t)+".pvd")
        ftau_yy = File("Paraview_Results/Stress Results Ra="+str(Ra)+"We="+str(We)+"b="+str(betav)+"DEVSS"+str(th)+"/tau_yy "+str(t)+".pvd")
        f_N1 = File("Paraview_Results/Stress Results Ra="+str(Ra)+"We="+str(We)+"b="+str(betav)+"DEVSS"+str(th)+"/N1"+str(t)+".pvd")

        if j==1 or j==2 or j==4 or j==loopend:

            # Plot Stress/Normal Stress Difference
            tau_xx=project(tau1[0,0],Q)
            ftau_xx << tau_xx
            mplot(tau_xx)
            plt.colorbar()
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/tau_xxRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(th)+".png")
            plt.clf() 
            tau_xy=project(tau1[1,0],Q)
            ftau_xy << tau_xy
            mplot(tau_xy)
            plt.colorbar()
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/tau_xyRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(th)+".png")
            plt.clf() 
            tau_yy=project(tau1[1,1],Q)
            ftau_yy << tau_yy
            mplot(tau_yy)
            plt.colorbar()
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/tau_yyRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(th)+".png")
            plt.clf() 
            theta0 = project(theta0, Q)
            ftheta << theta0
            mplot(theta0)
            plt.colorbar()
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/TemperatureRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()

        """if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E6 and abs(E_k) < 10:
    
        # Plot Velocity Components
            ux=project(u1[0],Q)
            mplot(ux)
            plt.colorbar()
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/u_xRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(c1)+".png")
            plt.clf()
            uy=project(u1[1],Q)
            mplot(uy)
            plt.colorbar()
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/u_yRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(c1)+".png")
            plt.clf()"""

        if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==loopend or j==1:
        #Plot Contours USING MATPLOTLIB
            # Scalar Function code


            x = Expression('x[0]', degree=2)     #GET X-COORDINATES LIST
            y = Expression('x[1]', degree=2)     #GET Y-COORDINATES LIST
            pvals = p1.vector().get_local()          # GET SOLUTION p= p(x,y) list
            #theta0 = project(theta0, Q)
            Tvals = theta0.vector().get_local() 
            psiq = project(psi, Q)
            psivals = psiq.vector().get_local() 
            tauxx = project(tau1_vec[0], Q)
            tauxxvals = tauxx.vector().get_local()
            xyvals = mesh.coordinates()     # CLEAN THIS UP!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            xvalsq = interpolate(x, Q)#xyvals[:,0]
            yvalsq= interpolate(y, Q)#xyvals[:,1]
            xvalsw = interpolate(x, Qt)#xyvals[:,0]
            yvalsw= interpolate(y, Qt)#xyvals[:,1]

            xvals = xvalsq.vector().get_local()
            yvals = yvalsq.vector().get_local()


            xx = np.linspace(0.01,0.99)
            yy = np.linspace(0.01,0.99)
            XX, YY = np.meshgrid(xx,yy)   # (x,y) coordinate data formatted so that it can be used by plt.contour()
            pp = mlab.griddata(xvals, yvals, pvals, xx, yy, interp='nn') 
            TT = mlab.griddata(xvals, yvals, Tvals, xx, yy, interp='nn')
            psps = mlab.griddata(xvals, yvals, psivals, xx, yy, interp='nn')  


            plt.contour(XX, YY, pp, 25)
            plt.title('Pressure Contours')   # PRESSURE CONTOUR PLOT
            plt.colorbar() 
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/PressureContoursRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()


            plt.contour(XX, YY, TT, 20)
            plt.title('Temperature Contours')   # PRESSURE CONTOUR PLOT
            plt.colorbar() 
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/TemperatureContoursRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()

            plt.contour(XX, YY, psps, 15)
            plt.title('Streamline Contours')   # PRESSURE CONTOUR PLOT
            plt.colorbar() 
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/StreamlineContoursRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()


            #Plot Velocity Streamlines USING MATPLOTLIB
            u1_q = project(u1[0],Q)
            uvals = u1_q.vector().get_local()
            v1_q = project(u1[1],Q)
            vvals = v1_q.vector().get_local()

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
            plt.title('Natural Convection')
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/VelocityContoursRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")   
            plt.clf()                                             # display the plot


        plt.close()



        if dt < tol:
        j=loopend+1
        break

        if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E6 and abs(E_k) < 10:
            Tf=T_f   
    

        if jjj==4:
        quit()

        if j==loopend:
            jjj+=1
            j=0
            jj=0
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
            nus1=list()
            nus2=list()
            nus3=list()
            nus4=list()
            nus5=list()



        #Reset Functions

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

        (u0, D0_vec) = w0.split()
        (u12, D12_vec) = w12.split()
        (u1, D1_vec) = w1.split()
        (us, Ds_vec) = ws.split()
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

        





if __name__ == "__main__":
    # Execute simulations loop with parameters from "parameters.csv"
    main("parameters-ewm.csv", mesh_resolution=40, simulation_time=15, mesh_refinement=False)
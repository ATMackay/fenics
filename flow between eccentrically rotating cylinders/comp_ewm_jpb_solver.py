"""
Flow Between Eccentrically Rotating Cylinders - Alex Mackay 2018
This Python module contains functions for computing Non Newtonian flow between rotating cylinders using the finite element method.
...

"""
import time, sys
import csv
from fenics_base import *

# Progress Bar
def update_progress(job_title, progress):
    length = 50 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 4))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()





def taylor_galerkin_solver():
    # Main time-marching scheme here
    # TODO
    return 0 


def main(input_csv,mesh_resolution,simulation_time, mesh_refinement):
    """
    This function solves time-dependent Navier-Stokes Equations governining 
    Non-Newtonian, non-isothermal flow between eccentrically rotating cylinders
    and saves solution data along with plots to directories plots/ and results/
    """

    update_progress("Compressible EWM Flow", 0)

    # SET TIMESTEPPING PARAMTER
    T_f = simulation_time
    Tf = T_f

    # SET LOOPING PARAMETER
    loopend = 4
    j = 0                            
    err_count = 0
    jjj = 0
    tol = 10E-6
    defpar = 1.0

    # Default Nondimensional Fluid Parameters
    U = 1.0
    w_j = 1.0

    T_0 = 300               # Inner boundary temperature
    T_h = 350               # Outer boundary temperature
    Di = 0.005              # Diffusion Number
    Vh = 0.0000069          # Viscous heating number
    Bi = 0.2                # Biot number 
    al = 0.001              # Nonisothermal parameter  (0, 1)
    c0 = 1000               # reference Mach number

    A_0 = 120 # Solvent viscosity thinning
    k_ewm = -0.7 # Shear thinning (EWM)
    B = 0.1 # Polymeric viscosity thinning (EWM)
    K_0 = 0.01

    alph1 = 1.0
    c1 = 0.1
    c2 = 0.01
    c3 = 0.1
    th = 1.0               # DEVSS



    # FEM Solution Convergence/Energy Plot
    # TODO FIX this so that fewer arrays are being defined
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

        # DEFINE HOLOLOW CYLINDER MESH

        # Parameters
        r_a = 1.0 #Journal Radius
        r_b = 2.0#1.25 #Bearing Radius
        x_1 = -0.80 #-0.2
        y_1 = 0.0
        x_2 = 0.0
        y_2 = 0.0
        mesh = JBP_mesh(mesh_resolution, x_1, x_2, y_1, y_2, r_a, r_b)

        # Generate internal mesh for hollow cylinder
        c3 = Circle(Point(x_1,y_1), 0.99*r_a, 256)  # Empty hole in mesh
        gdim = mesh.geometry().dim() # Mesh Geometry
        meshc= generate_mesh(c3, 15)

        # Timestepping
        dt = 0.00125#10*mesh.hmin()**2
        total_loops = (loopend*(Tf/dt))

        # Reset Mesh Dependent Functions
        h = CellDiameter(mesh)
        n = FacetNormal(mesh)
        tang = as_vector([n[1], -n[0]])

        # Finite Element Spaces

        # Discretization  parameters
        order = 2
        # Function spaces
        W, V, Vd, Z, Zd, Zc, Q, Qt, Qr = function_spaces(mesh, order)

        # Trial Functions
        rho, p, T, tau_vec, u, D_vec, D, tau = trial_functions(Q, Zc, W)


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
        uu0 = Function(V)   # Velocity space function
        rho0, rho12, rho1, p0, p1, T0, T1, u0, u12, us, u1, D0_vec, D12_vec, Ds_vec, D1_vec, w0, w12, ws, w1, tau0_vec, tau12_vec, tau1_vec = solution_functions(Q, W, V, Zc)
        D0, D12, Ds, D1, tau0, tau12, tau1 = reshape_elements(D0_vec, D12_vec, Ds_vec, D1_vec, tau0_vec, tau12_vec, tau1_vec)


        #Jounral Boundary                                                                              
        class Omega0(SubDomain):
            def inside(self, x, on_boundary):
                return True if (x[0]-x_1)**2+(x[1]-y_1)**2 < (0.9*r_a**2+0.1*r_b**2) and on_boundary  else False  # and 


        # Bearing Boundary
        class Omega1(SubDomain):
            def inside(self, x, on_boundary):
                return True if (x[0]-x_2)**2 + (x[1]-y_2)**2 > (0.1*r_a**2+0.9*r_b**2) and on_boundary else False  #


        # Subdomian for the pressure boundary condition at (r_a,0)
        class POmega(SubDomain):
            def inside(self, x, on_boundary):
                return True if x[0] < 0.5*(r_a+r_b) and x[0] > 0 and x[1] < r_a*0.02 and x[1] > -r_a*0.05 and on_boundary else False 

        omega0= Omega0()
        omega1= Omega1()
        pomega=POmega()

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


        if mesh_refinement == True: 
            w = Expression(('(x[1]-y1)/r_a' , '-(x[0]-x1)/r_a' ), degree=2, r_a=r_a, x1=x_1, y1=y_1 , t=0.0)
        else:
            w = Expression(('(0.5*(1.0+tanh(8*(t-0.5))))*(x[1]-y1)/r_a' , '-(0.5*(1.0+tanh(8*(t-0.5))))*(x[0]-x1)/r_a' ), degree=2, r_a=r_a, x1=x_1, y1=y_1 , t=0.0) 

    
        spin =  DirichletBC(W.sub(0), w, omega0) 
        noslip  = DirichletBC(W.sub(0), (0.0, 0.0), omega1) #The outer cylinder remains fixed with zero velocity 
        temp0 =  DirichletBC(Q, T_h, omega0)    #Temperature on Omega0 

        #Collect Boundary Conditions
        bcu = [noslip, spin]
        bcp = []
        bcT = [temp0]
        bctau = []

        

        # READ IN FLUID PARAMETERS FROM CSV ---------------------------------------------------------------------------
        # March 2021 

        # Import csv using csv
        with open(input_csv, 'r') as csvfile:
            spamreader = csv.reader(csvfile)
            my_csv_data = list(spamreader)
            re_row = my_csv_data[0]
            we_row = my_csv_data[1]
            ma_row = my_csv_data[2]


        # Set parameters for secondary loop -----------------------------------------------------------------------

        betav = 0.5
        Ma = float(ma_row[jjj+1])

        # Set parameters for primary loop ------------------------------------------------        
        if j==1:
            Re = float(re_row[1])
            We = float(we_row[1])
        elif j==2:
            Re = float(re_row[1])
            We = float(we_row[2])
        elif j==3:
            Re = float(re_row[1])
            We = float(we_row[4])
        elif j==4:
            Re = float(re_row[2])
            We = float(we_row[4])


        
        # Adaptive Mesh Refinement Step
        if mesh_refinement == True and err_count < 1: # 0 = on, 1 = off
            We = 1.0
            betav = 0.5
            Tf = 1.5*(1 + 2*err_count*0.25)
            dt = 0.001
            th = 0.0

        conv = 1                # Non-inertial Flow Parameter (Re=0)

        # Print Parameters of flow simulation
        t = 0.0                  #Time


        if mesh_refinement == True:
            print('############# ADAPTIVE MESH REFINEMENT STAGE ################')   
            print( 'Number of Refinements:', err_count )
        print('############# Journal Bearing Length Ratios ############')
        ec = np.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)
        c = r_b - r_a
        ecc = ec/c
        print('Eccentricity (m):' , ecc)
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
        #print( 'Minimum Cell Diamter:', mesh.hmin())
        #print( 'Maximum Cell Diamter:', mesh.hmax())
        print( '############# Stabilisation Parameters ############')
        print( 'DEVSS Momentum Term:', th)

        print( 'Loop:', jjj, '-', j)


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

        
        DEVSSl_u12 = 2*(1.-betav)*inner(Dcomp(u),Dincomp(v))*dx    
        DEVSSr_u12 = 2*inner(D0,Dincomp(v))*dx   
        DEVSSl_u1 = 2*(1.-betav)*inner(Dcomp(u),Dincomp(v))*dx    
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
        z = list()
        xx = list()
        yy = list()
        zz = list()


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
        if mesh_refinement == True:
            maxiter = 25
        else :
            maxiter = 100000000
        while t < Tf + DOLFIN_EPS and iter < maxiter:
            flow_description = "eccentric cyclinder flow: loop: " +str(jjj) + ", Re: "+str(Re)+", We: "+str(We)+", Ma: "+str(Ma)
            update_progress(flow_description, t/Tf) # Update progress bar
            iter += 1
            w.t=t

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
            a2+= th*DEVSSGl_u1                       
            L2+= th*DEVSSGr_u1    

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
            alpha_supg = h/(magnitude(u12)+DOLFIN_EPS)
            SU_rho = inner(dot(u12, grad(rho)), alpha_supg*dot(u12,grad(q)))*dx

            # Density Update
            rho_eq = (rho - rho0)/dt + dot(u12, grad(rho12)) - rho12*div(u12) # this is possibly highly unstable. Get code to measure error norm
            rho_weak = inner(rho_eq,q)*dx

            a6 = lhs(rho_weak)
            L6 = rhs(rho_weak)
            
            a6+= SU_rho       # SUPG stabilisation for equation of state

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
            gamdot = inner(sigmacon(u0, p0, tau0, betav, We),grad(u0))
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

            sigma0 = dot(sigmacon(u1, p1, tau1, betav, We), tang)
            sigma1 = dot(sigmacon(u1, p1, tau1, betav, We), tang)

            omegaf0 = dot(sigmacon(u1, p1, tau1, betav, We), n)  #Nomral component of the stress 
            omegaf1 = dot(sigmacon(u1, p1, tau1, betav, We), n)


            innerforcex = inner(Constant((1.0, 0.0)), omegaf0)*ds(0)
            innerforcey = inner(Constant((0.0, -1.0)), omegaf0)*ds(0)

            innertorque = -inner(n, sigma0)*ds(0)
            outertorque = -inner(n, sigma1)*ds(1)


            # Record Elastic & Kinetic Energy Values & Torque (Method 1)
            if j==1:
                x1.append(t)
                ek1.append(E_k)
                ee1.append(E_e)
                y1.append(assemble(innertorque))
                zx1.append(assemble(innerforcex))     # 0.0
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
                break
            

            # Move to next time step (Continuation in Reynolds Number)
            t += dt

        if mesh_refinement == True: 
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

            mesh_refinement = False

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
            dt = 0.001
            Tf = T_f
            th = 1.0
            # TODO FIX this so that fewer arrays are being defined
            x1 = x2 = x3 = x4 = x5 = list()
            y = y1 = y3 = y4 = y5 = list()
            z = z1 = z2 = z3 = z4 = z5 = list()
            zx1 = zx2 = zx3 = zx4 = zx5 = list()
            z = zz = zzz = zl = list()
            ek1 = ek2 = ek3 = ek4 = ek5 = list()
            ee1 = ee2 = ee3 = ee4 = ee5 = list()


        if mesh_refinement == False:
            # Save FE solutions in HDF5 format
            ufile = "hd5/velocity-solution, Re="+str(Re)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+".h5"
            save_solution(u1, ufile)
            pfile = "hd5/pressure-solution, Re="+str(Re)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+".h5"
            save_solution(p1, pfile)
            rhofile = "hd5/density-solution, Re="+str(Re)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+".h5"
            save_solution(rho1, rhofile)
            thetafile = "hd5/temperature-solution, Re="+str(Re)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+".h5"
            save_solution(T1, thetafile)
            # Minimum of stream function (Eye of Rotation)
            u1 = project(u1, V)
            psi = comp_stream_function(rho1, u1)
            psi_max = max(psi.vector().get_local())
            max_loc = max_location(psi, mesh)
            with open("results/EWMStream-Function.txt", "a") as text_file:
                text_file.write("Re="+str(Re)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+"----- psi_min="+str(psi_max)+"---"+str(max_loc)+'\n')


            # Maximum shear rate
            u1 = project(u1, V)
            gamma = shear_rate(u1)
            gamma_max = max(gamma.vector().get_local())
            max_loc = max_location(gamma, mesh)
            with open("results/EWMShear-Rate.txt", "a") as text_file:
                text_file.write("Re="+str(Re)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+"----- shear_rate="+str(gamma_max)+"---"+str(max_loc)+'\n')

            # Data on Kinetic/Elastic Energies
            with open("results/EWMEnergy.txt", "a") as text_file:
                text_file.write("Re="+str(Re)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+", E_k="+str(E_k)+", E_e="+str(E_e)+'\n')


            chi = assemble(innerforcex)/assemble(innerforcey)
            torque_journal = assemble(innertorque)
            # Data on Stability Measure
            with open("results/EWMStability.txt", "a") as text_file:
                text_file.write("Re="+str(Re)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+"ecc="+str(ecc)+", Stability="+str(chi)+'\n')

            with open("results/EWMTorque.txt", "a") as text_file:
                text_file.write("Re="+str(Re)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+"ecc="+str(ecc)+", Stability="+str(torque_journal)+'\n')




            # Plot Torque/Load and energy metrics
            if max(norm(tau1_vec.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==loopend or j==1:
                # Plot Torque Data
                plt.figure(0)
                plt.plot(x1, y1, 'r--', label=r'$We=0$, $Re=100$')
                plt.plot(x2, y2, 'b--', label=r'$We=0.1$, $Re=100$')
                plt.plot(x3, y3, 'c--', label=r'$We=1.0$, $Re=100$')
                plt.plot(x4, y4, 'm--', label=r'$We=1.0$, $Re=25$')
                plt.legend(loc='best')
                plt.xlabel('$t$', fontsize=16)
                plt.ylabel('$C$', fontsize=16)
                plt.savefig("plots/Load-Torque/Torque_We="+str(We)+"Re="+str(Re)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
                plt.clf()
                # Horizontal load
                plt.figure(1)
                plt.plot(x1, zx1, 'r--', label=r'$We=0$, $Re=100$')
                plt.plot(x2, zx2, 'b--', label=r'$We=0.1$, $Re=100$')
                plt.plot(x3, zx3, 'c--', label=r'$We=1.0$, $Re=100$')
                plt.plot(x4, zx4, 'm--', label=r'$We=1.0$, $Re=25$')
                plt.legend(loc='best')
                plt.xlabel('$t$', fontsize=16)
                plt.ylabel('$F_x$', fontsize=16)
                plt.savefig("plots/Load-Torque/Horizontal_Load_We="+str(We)+"Re="+str(Re)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
                plt.clf()
                # Vertical load
                plt.figure(2)
                plt.plot(x1, z1, 'r--', label=r'$We=0$, $Re=100$')
                plt.plot(x2, z2, 'b--', label=r'$We=0.1$, $Re=100$')
                plt.plot(x3, z3, 'c--', label=r'$We=1.0$, $Re=100$')
                plt.plot(x4, z4, 'm--', label=r'$We=1.0$, $Re=25$')
                plt.plot(x5, z5, 'g--', label=r'$We=1.0$')
                plt.legend(loc='best')
                plt.xlabel('$t$', fontsize=16)
                plt.ylabel('$F_y$', fontsize=16)
                plt.savefig("plots/Load-Torque/Vertical_Load_We="+str(We)+"Re="+str(Re)+"b="+str(betav)+"Ma="+str(Ma)+"al="+str(al)+"t="+str(t)+".png")
                plt.clf()
                # Force evolution
                plt.figure(3)
                plt.plot(zx1, z1, 'r--', label=r'$We=0$, $Re=100$')
                plt.plot(zx2, z2, 'b--', label=r'$We=0.1$, $Re=100$')
                plt.plot(zx3, z3, 'c--', label=r'$We=1.0$, $Re=100$')
                plt.plot(zx4, z4, 'm--', label=r'$We=1.0$, $Re=25$')
                plt.legend(loc='best')
                plt.xlabel('$F_x$', fontsize=16)
                plt.ylabel('$F_y$', fontsize=16)
                plt.savefig("plots/Load-Torque/Force_Evolution_We="+str(We)+"Re="+str(Re)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
                plt.clf()
                # Kinetic Energy
                plt.figure(4)
                plt.plot(x1, ek1, 'r--', label=r'$We=0$, $Re=100$')
                plt.plot(x2, ek2, 'b--', label=r'$We=0.1$, $Re=100$')
                plt.plot(x3, ek3, 'c--', label=r'$We=1.0$, $Re=100$')
                plt.plot(x4, ek4, 'm--', label=r'$We=1.0$, $Re=25$')
                plt.legend(loc='best')
                plt.xlabel('$t$', fontsize=16)
                plt.ylabel('$E_{k}$', fontsize=16)
                plt.savefig("plots/Energy/KineticEnergyWe="+str(We)+"Re="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
                plt.clf()
                # Elastic Energy
                plt.figure(5)
                plt.plot(x1, ee1, 'r--', label=r'$We=0$, $Re=100$')
                plt.plot(x2, ee2, 'b--', label=r'$We=0.1$, $Re=100$')
                plt.plot(x3, ee3, 'c--', label=r'$We=1.0$, $Re=100$')
                plt.plot(x4, ee4, 'm--', label=r'$We=1.0$, $Re=25$')
                plt.legend(loc='best')
                plt.xlabel('$t$', fontsize=16)
                plt.ylabel('$E_{e}$', fontsize=16)
                plt.savefig("plots/Energy/ElasticEnergyWe="+str(We)+"Re="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
                plt.clf()

            # (x,y) plot of pressure, stress, momenetum etc
            if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6:
                # Plot Stress/Normal Stress Difference
                sigma_xx = project(sigmacon(u1, p1, tau1, betav, We)[0,0], Q)
                mplot(sigma_xx)
                plt.colorbar()
                plt.savefig("plots/contours/sigma_xxRe="+str(Re)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
                plt.clf() 
                tau_xx=project(tau1[0,0],Q)
                mplot(tau_xx)
                plt.colorbar()
                plt.savefig("plots/contours/tau_xxRe="+str(Re)+"We="+str(We)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
                plt.clf() 
                tau_xy=project(tau1[1,0],Q)
                mplot(tau_xy)
                plt.colorbar()
                plt.savefig("plots/contours/tau_xyRe="+str(Re)+"We="+str(We)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
                plt.clf() 
                tau_yy=project(tau1[1,1],Q)
                mplot(tau_yy)
                plt.colorbar()
                plt.savefig("plots/contours/tau_yyRe="+str(Re)+"We="+str(We)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
                plt.clf() 
                N1=project(tau1[0,0]-tau1[1,1],Q)
                mplot(N1)
                plt.colorbar()
                plt.savefig("plots/contours/FirstNormalStressDifferenceRe="+str(Re)+"We="+str(We)+"Tf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
                plt.clf()
                # Plot density
                mplot(rho1)
                plt.colorbar()
                plt.savefig("plots/contours/DensityRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
                plt.clf()
                mom_mag = project(rho1*magnitude(u1), Q)
                mplot(mom_mag)
                plt.colorbar()
                plt.savefig("plots/contours/MomentumRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
                plt.clf()
                # Plot pressure
                mplot(p1)
                plt.colorbar()
                plt.savefig("plots/contours/PressureRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
                plt.clf()
                # Plot temperature
                theta0_Q = project(theta0, Q)
                mplot(theta0_Q)
                plt.colorbar()
                plt.savefig("plots/contours/TemperatureRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
                plt.clf()
                divu=project(div(u1),Q)
                mplot(divu)
                plt.colorbar()
                plt.savefig("plots/contours/CompressionRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
                plt.clf()
                mplot(psi)
                plt.colorbar()
                plt.savefig("plots/contours/stream_functionRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
                plt.clf()
                therm_visc = project(betav + (1.-betav)*phi_ewm(tau1, theta1, k_ewm, B), Q)
                mplot(therm_visc)
                plt.colorbar()
                plt.savefig("plots/contours/viscosity_functionRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
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

                # Create 2D array with mesh coordinate points
                points = np.vstack((xvals, yvals)).T

                xx = np.linspace(-1.5*r_b,1.5*r_b, num=250)
                yy = np.linspace(-1.5*r_b,1.5*r_b, num=250)
                XX, YY = np.meshgrid(xx,yy)   # (x,y) coordinate data formatted so that it can be used by plt.contour()

                # Pressure
                pp = sci.griddata(points, pvals, (XX, YY), method='linear') # u(x,y) data so that it can be used by 
                pp = np.reshape(pp, (len(xx), len(yy))) # Reshape to 2D array

                plt.contour(XX, YY, pp, 30)
                plt.colorbar()
                plt.title('Pressure')   # PRESSURE CONTOUR PLOT
                plt.savefig("plots/contours/Pressure Contours Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
                plt.clf()

                # Temperature

                Tj=Expression('0', degree=1) #Expression for the 'pressure' in the domian
                Tjq=interpolate(Tj, Q1)
                Tjvals=Tjq.vector().get_local()

                Tvals = theta0_Q.vector().get_local() # GET SOLUTION T= T(x,y) list
                Tvals = np.concatenate([Tvals, Tjvals])  #Merge two arrays for Temperature values

                TT = sci.griddata(points, Tvals, (XX, YY), method='linear') # u(x,y) data so that it can be used by 
                TT = np.reshape(TT, (len(xx), len(yy))) # Reshape to 2D array

                plt.contour(XX, YY, TT, 30)
                plt.colorbar()
                plt.title('Temperature')   # TEMPERATURE CONTOUR PLOT
                plt.savefig("plots/contours/Temperature Contours Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"k="+str(k_ewm)+"t="+str(t)+".png")
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

            

                #Merge arrays
                uvals = np.concatenate([uvals, ujvals])  #Merge two arrays for velocity values
                vvals = np.concatenate([vvals, vjvals])  #Merge two arrays for velocity values

                xvals = np.concatenate([xvals, xvalsj])   #Merge two arrays for x-coordinate values
                yvals = np.concatenate([yvals, yvalsj])   #Merge two arrays for y-coordinate values
                points = np.vstack((xvals, yvals)).T

                uu = sci.griddata(points, uvals, (XX, YY), method='linear') 
                vv = sci.griddata(points, vvals, (XX, YY), method='linear') 

                # Set all nan values to DOLFIN_EPS (close to zero)
                uu = np.nan_to_num(uu, nan = 0)
                vv = np.nan_to_num(vv, nan = 0)
                # Reshape
                uu = np.reshape(uu, (len(yy), len(xx))) # Reshape to 2D array
                vv = np.reshape(vv, (len(yy), len(xx))) # Reshape to 2D array

                speed = np.sqrt(uu*uu+ vv*vv)

                plt.figure()
                plt.streamplot(XX, YY, uu, vv,  
                            density=4,              
                            color=speed/speed.max(),  
                            cmap=cm.gnuplot,                           # colour map
                            linewidth=speed/speed.max()+0.5)       # line thickness
                plt.colorbar()
                plt.title('Isostreams')
                plt.savefig("plots/contours/Velocity Contours Re="+str(Re)+\
                            "We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")   
                plt.clf()
                
            plt.close()                                                            
            

            if j==loopend:
                jjj+=1
                update_progress(flow_description+str(jjj), 1)
                j = 0
                mesh_refinement = False
                x1 = x2 = x3 = x4 = x5 = list()
                y = y1 = y3 = y4 = y5 = list()
                z = z1 = z2 = z3 = z4 = z5 = list()
                zx1 = zx2 = zx3 = zx4 = zx5 = list()
                z = zz = zzz = zl = list()
                ek1 = ek2 = ek3 = ek4 = ek5 = list()

            if jjj==3:
                quit()




if __name__ == "__main__":
    # Execute simulations loop with parameters from "parameters.csv"
    main("parameters-ewm.csv", mesh_resolution=40, simulation_time=40, mesh_refinement=False)


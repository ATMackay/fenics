"""Inompressible Double Glazing Problem """

import csv
from fenics_fem import *  # Import FEniCS helper functions
import datetime


def main(input_csv,mesh_resolution,simulation_time, mesh_refinement):

    update_progress("incompressible bouyancy-driven flow in the unit square", 0)
    # SET TIMESTEPPING PARAMTER
    T_f = simulation_time
    Tf = T_f

    # Experiment Run Time
    dt = 0.001  #Timestep

    # Nondimensional flow parameters
    B, L = 1, 1 # Length
    U = 1
    Ra = 10000                           #Rayleigh Number
    Pr = 1.0
    We = 0.01                          #Weisenberg NUmber
    Vh = 0.005
    T_0 = 300
    T_h = 350
    Bi = 0.0+DOLFIN_EPS
    Di = 0.005                         #Diffusion Number
    al = 1.0

    c1 = 0.05
    c2 = 0.001
    th = 0.5             # DEVSS
    C = 200.


    # SET LOOPING PARAMETER
    loopend = 3
    j = 0                            
    err_count = 0
    jjj = 0


      
    while j < loopend:
        j+=1
        t=0.0

        # DEFINE THE COMPUTATION GRID
        # Choose Mesh to Use
        if j==1:
            mesh_resolution = 40
            label_1 = "M1"#mesh="+str(mesh_resolution)
        elif j==2:
            mesh_resolution = 60
            label_2 = "M2"#mesh="+str(mesh_resolution)
        elif j==3:
            mesh_resolution = 80  # <--- 65
            label_3 = "M3"#mesh="+str(mesh_resolution)
        #elif j==4:
        #    mesh_resolution = 70  # <--- 65
        #    label_4 = "M4"#mesh="+str(mesh_resolutio
        mesh = DGP_Mesh(mesh_resolution, B, L)

        mplot(mesh)
        plt.savefig("skewed_grid-"+str(mesh_resolution)+".png")
        plt.clf()
        plt.close()


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
        rho0, rho12, rho1, p0, p1, T0, T1, u0, u12, us, u1, D0_vec, D12_vec, Ds_vec, D1_vec, w0, w12, ws, w1, tau0_vec, tau12_vec, tau1_vec = solution_functions(Q, W, V, Zc)
        D0, D12, Ds, D1, tau0, tau12, tau1 = reshape_elements(D0_vec, D12_vec, Ds_vec, D1_vec, tau0_vec, tau12_vec, tau1_vec)
       
        bottom_bound = 0.0 #0.5*(1+tanh(-N)) 
        top_bound = 1.0 #0.5*(1+tanh(N)) 
       

        class No_slip(SubDomain):
            def inside(self, x, on_boundary):
                return True if on_boundary else False 
                                                                                
        class Left(SubDomain):
            def inside(self, x, on_boundary):
                return True if x[0] < bottom_bound + DOLFIN_EPS and on_boundary  else False  

        class Right(SubDomain):
            def inside(self, x, on_boundary):
                return True if x[0] > top_bound - DOLFIN_EPS and on_boundary  else False   

        class Top(SubDomain):
            def inside(self, x, on_boundary):
                return True if x[1] > top_bound - DOLFIN_EPS and on_boundary  else False 

        class Bottom(SubDomain):
            def inside(self, x, on_boundary):
                return True if x[1] < bottom_bound + DOLFIN_EPS and on_boundary  else False 

        no_slip = No_slip()
        left = Left()
        right = Right()
        top = Top()
        bottom = Bottom()


        # MARK SUBDOMAINS (Create mesh functions over the cell facets)
        sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        sub_domains.set_all(5)
        no_slip.mark(sub_domains, 0)
        bottom.mark(sub_domains, 1)
        left.mark(sub_domains, 2)
        right.mark(sub_domains, 3)
        top.mark(sub_domains, 4)


        plot(sub_domains, interactive=False)        # DO NOT USE WITH RAVEN
        #quit()

        #Define Boundary Parts

        boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) #FacetFunction("size_t", mesh)
        no_slip.mark(boundary_parts,0)
        bottom.mark(boundary_parts, 1)
        left.mark(boundary_parts, 2)
        right.mark(boundary_parts, 3)
        top.mark(boundary_parts, 4)
        ds = Measure("ds")[boundary_parts]


        # Define boundary/stabilisation FUNCTIONS
        # ramped thermal boundary condition
        #ramp_function = Expression('0.5*(1+tanh(8*(t-0.5)))*(T_h-T_0)+T_0', degree=2, t=0.0, T_0=T_0, T_h=T_h)
        ramp_function = Expression('0.5*(1+tanh(4*(t-1.5)))*(T_h-T_0)+T_0', degree=2, t=0.0, T_0=T_0, T_h=T_h)
        # direction of gravitational force (0,-1)
        f = Expression(('0','-1'), degree=2)

        # Mesh functions
        h = CellDiameter(mesh)
        n = FacetNormal(mesh)


        # Define unit Normal/tangent Vector at inner and outer Boundary (Method 2)
        n0 =  Expression(('-1' , '0'), degree=2)
        n1 =  Expression(('0' , '1' ), degree=2)
        n2 =  Expression(('1' , '0' ), degree=2)
        n3 =  Expression(('0' , '-1'), degree=2)



        # Dirichlet Boundary Conditions  (Bouyancy driven flow)
        noslip0  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), no_slip)  # No Slip boundary conditions on the left wall
        noslip1 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), left)  # No Slip boundary conditions on the left wall
        noslip2 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), right)  # No Slip boundary conditions on the left wall
        noslip3 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), top)  # No Slip boundary conditions on the left wall
        temp_left =  DirichletBC(Q, ramp_function, left)    #Temperature on Omega0 
        temp_right =  DirichletBC(Q, T_0, right)    #Temperature on Omega2 

        #Collect Boundary Conditions
        bcu = [noslip0, noslip1, noslip2, noslip3]
        bcp = []
        bcT = [temp_left, temp_right] 
        bctau = []

        
        # READ IN FLUID PARAMETERS FROM CSV ---------------------------------------------------------------------------
        # March 2021 

        # Import csv using csv
        with open(input_csv, 'r') as csvfile:
            spamreader = csv.reader(csvfile)
            my_csv_data = list(spamreader)
            ra_row = my_csv_data[0]
            we_row = my_csv_data[1]

        # Set parameters for primary loop ------------------------------------------------        
        betav = 0.5
        #Ra = float(ra_row[jjj+1])
        #We = float(we_row[j])

        data_tag = "incomp-flow-"

        # Set parameters for primary loop ------------------------------------------------        
        #if j==1:
        #    We = float(we_row[1])
        #    Ra = float(ra_row[3])
        #    label_1 = "Ra="+str(Ra)+",We="+str(We)
        #elif j==2:
        #    We = float(we_row[4])
        #    Ra = float(ra_row[3])
        #    label_2 = "Ra="+str(Ra)+",We="+str(We)
        #elif j==3:
        #    We = float(we_row[5])
        #    Ra = float(ra_row[3])
        #    label_3 = "Ra="+str(Ra)+",We="+str(We)
        #elif j==4:
        #    We = float(we_row[1])
        #    Ra = float(ra_row[5])
        #    label_4 = "Ra="+str(Ra)+",We="+str(We)
        #elif j==5:
        #    We = float(we_row[4])
        #    Ra = float(ra_row[5])
        #    label_5 = "Ra="+str(Ra)+",We="+str(We)
        #elif j==6:
        #    We = float(we_row[5])
        #    Ra = float(ra_row[5])
        #    label_6 = "Ra="+str(Ra)+",We="+str(We)

        We = float(we_row[1])
        Ra = float(ra_row[2]) 
        

        h_min = mesh.hmin()
        dt = h_min**2
        dt = float(np.format_float_positional(dt, precision=1, unique=False, fractional=False, trim='k'))
        dt = 0.001

        print('############# TIME SCALE ############')
        print('Timestep size (s):', dt)
        print('Finish Time (s):', Tf)

        print('############# Scalings & Nondimensional Parameters ############')
        print('Characteristic Length (m):', L)
        print('Characteristic Velocity (m/s):', 1.0)
        print('Rayleigh Number:', Ra)
        print('Prandtl Number:', Pr)
        print('Weissenberg Number:', We)
        print('Viscosity Ratio:', betav)
        print('Diffusion Number:' , Di)
        print('Viscous Heating Number:', Vh)

        Np= len(p0.vector().get_local())
        (u0, _) = w0.split() 
        Nv= len(u0.vector().get_local())  
        Ntau= len(tau0_vec.vector().get_local())
        dof= 3*Nv+2*Ntau+Np
        mm = int(np.sqrt(Np))
        print('############# Discrete Space Characteristics ############')
        print('Degree of Elements', order)
        print('Mesh: %s x %s' %(mm, mm))
        print('Size of Pressure Space = %d ' % Np)
        print('Size of Velocity Space = %d ' % Nv)
        print('Size of Stress Space = %d ' % Ntau)
        print('Degrees of Freedom = %d ' % dof)
        print('Number of Cells:', mesh.num_cells())
        print('Number of Vertices:', mesh.num_vertices())
        print('Minimum cell diameter:', mesh.hmin())
        print('Maximum cell diameter:', mesh.hmax())
        print('############# Stabilisation Parameters ############')
        print('DEVSS Parameter:', th)

        # Initial Conformation Tensor
        I_vec = Expression(('1.0','0.0','1.0'), degree=2)
        initial_guess_conform = project(I_vec, Zc)
        assign(tau0_vec, initial_guess_conform)         # Initial guess for conformation tensor is Identity matrix   

        # Initial Temperature Field
        T_initial_guess = project(T_0, Q)
        T0.assign(T_initial_guess)


        gamdot = inner(sigma(u1, p1, tau1, We, Pr, betav), grad(u1))


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
        
        DEVSSl_u12 = 2*(1-betav+DOLFIN_EPS)*inner(Dincomp(u),Dincomp(v))*dx    
        DEVSSr_u12 = 2*inner(D0,Dincomp(v))*dx   
        DEVSSl_u1 = 2*(1-betav+DOLFIN_EPS)*inner(Dincomp(u),Dincomp(v))*dx    
        DEVSSr_u1 = 2*inner(D12,Dincomp(v))*dx 



        # Set up Krylov Solver 

        # Use amg preconditioner if available
        prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

        # Use nonzero guesses - essential for CG with non-symmetric BC
        parameters['krylov_solver']['nonzero_initial_guess'] = True
        parameters['krylov_solver']['monitor_convergence'] = False
        
        solveru = KrylovSolver("bicgstab", "default")
        solvertau = KrylovSolver("bicgstab", "default")
        solverp = KrylovSolver("bicgstab", "default")

        # FEM Solution Convergence Plot
        t_array=list()
        ek_array=list()
        ee_array=list()
        nus_array=list()


        # Time-stepping
        t = 0.0
        start, elapsed, total_elapsed = 0.0, 0.0, 0.0
        iter = 0            # iteration counter
        while t < Tf + DOLFIN_EPS:
            iter += 1
            start = time.process_time()
            time_left = (Tf-t)/dt * (elapsed) 
            flow_description = "incompressible bouyancy-driven flow: "
            flow_description += "loop: "+str(jjj) +"-"+str(j)+ ", Ra: "+str(Ra)+", We: "+str(We)+", Pr: "+str(Pr)+", al: "+str(al)+", betav: "+str(betav)
            flow_description += ", time taken: " + str(datetime.timedelta(seconds= total_elapsed))
            flow_description += ", (est) time to completion: " + str(datetime.timedelta(seconds= time_left))
            update_progress(flow_description, t/Tf) # Update progress bar
            # Set Function timestep
            ramp_function.t = t

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
                + inner(sigma(U, p0, tau0, We, Pr, betav), Dincomp(v))*dx + Ra*Pr*inner(theta0*f,v)*dx \
                + inner(D-Dincomp(u),R)*dx \
                - dot(sigma_n(U, p0, tau0, n, We, Pr, betav), v)*ds\
 
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

            DEVSSr_u1 = 2*Pr*(1.-betav+DOLFIN_EPS)*inner(D12,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS     

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
                + inner(sigma(U, p0, tau0, We, Pr, betav), Dincomp(v))*dx + Ra*Pr*inner(theta0*f,v)*dx \
                 - dot(sigma_n(U, p0, tau0, n, We, Pr, betav), v)*ds
                    
                
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



            if betav < 0.99:
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
            gamdot = inner(sigma(u1, p1, tau1, We, Pr, betav), grad(u1))
            lhs_theta1 = (1.0/dt)*thetal + dot(u1,grad(thetal))
            rhs_theta1 = (1.0/dt)*thetar + dot(u1,grad(thetar)) + (1.0/dt)*theta0 + Vh*gamdot
            a8 = inner(lhs_theta1,r)*dx + inner(grad(thetal),grad(r))*dx + inner(We*tau1*grad(thetal),grad(r))*dx
            L8 = inner(rhs_theta1,r)*dx + inner(grad(thetar),grad(r))*dx + Bi*inner(grad(theta0),n*r)*ds(1)  + Bi*inner(grad(theta0),n*r)*ds(4) + inner(We*tau1*grad(thetar),grad(r))*dx

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
            Nus = assemble(Tdx*ds(2))
            
            # Record Elastic & Kinetic Energy Values 
            t_array.append(t)
            ek_array.append(E_k)
            ee_array.append(E_e)
            nus_array.append(Nus)

            # Move to next time step
            t += dt

            elapsed = (time.process_time() - start)
            total_elapsed += elapsed

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
                conv_fail = 0

            # Reset Parameters
            corr=1    
            j = 0
            dt = 0.001
            Tf = T_f
        else:
            # PLOTS
            # Save array data to file
            save_energy_arrays(t_array, ek_array, ee_array, jjj, j, data_tag)
            save_data_array(nus_array, jjj, j, data_tag)
            mplot(kapp)
            plt.colorbar()
            plt.savefig("plots/kappa-"+"mesh="+str(mm)+".png")
            plt.clf()
            plt.close()

            u1 = project(u1, V)
            psi = stream_function(u1)
            psi_min = min(psi.vector().get_local())
            min_loc = min_location(psi, mesh)
            with open("results/Incompressible-Stream-Function.txt", "a") as text_file:
                text_file.write("Ra="+str(Ra)+", We="+str(We)+", t="+str(t)+"----- psi_min="+str(psi_min)+"---"+str(min_loc)+'\n')

            gamma = shear_rate(u1)
            gamma_max = max(gamma.vector().get_local())
            max_loc = max_location(gamma, mesh)
            with open("results/Incompressible-Shear-Rate.txt", "a") as text_file:
                text_file.write("Re="+str(Ra)+", We="+str(We)+", t="+str(t)+"----- shear_rate="+str(gamma_max)+"---"+str(max_loc)+'\n')

            # Max steady-state flow speed data
            u_abs = magnitude(u1)
            u_abs = project(u_abs, Q)
            u_max = u_abs.vector().get_local().max()
            with open("results/IncompressibleMaxFlowSpeed.txt", "a") as text_file:
                text_file.write("Ra="+str(Ra)+", We="+str(We)+", t="+str(t)+", u_max"+str(u_max)+'\n')

            # Nusslet Number data
            Nus_max = max(nus_array)
            with open("results/IncompressibleNussletNumber.txt", "a") as text_file:
                text_file.write("Ra="+str(Ra)+", We="+str(We)+", t="+str(t)+":  Nu="+str(Nus)+\
                                "  Max Nu="+str(Nus_max)+'\n'+'\n')


            peakEk = max(ek_array)
            # Data on Kinetic/Elastic Energies
            with open("results/IncompressibleEnergy.txt", "a") as text_file:
                text_file.write("Ra="+str(Ra)+", We="+str(We)+", t="+str(t)+", E_k="+str(E_k)+", E_e="+str(E_e)+", max(E_k)="+str(peakEk)+'\n')



                #Plot Kinetic and elasic Energies for different Weissenberg numbers at Re=0 (METHOD 2)
            if j==loopend:
                x1, ek1, ee1 = load_energy_arrays(jjj, 1, data_tag)
                nus1 = load_data_array(jjj, 1, data_tag)
                x2, ek2, ee2 = load_energy_arrays(jjj, 2, data_tag)
                nus2 = load_data_array(jjj, 2, data_tag)
                x3, ek3, ee3 = load_energy_arrays(jjj, 3, data_tag)
                nus3 = load_data_array(jjj, 3, data_tag)
                #x4, ek4, ee4 = load_energy_arrays(jjj, 4, data_tag)
                #nus4 = load_data_array(jjj, 4, data_tag)
                #x5, ek5, ee5 = load_energy_arrays(jjj, 5, data_tag)
                #nus5 = load_data_array(jjj, 5, data_tag)
                #x6, ek6, ee6 = load_energy_arrays(jjj, 6, data_tag)
                #nus6 = load_data_array(jjj, 6, data_tag)
                # Kinetic Energy
                plt.figure(0)
                plt.plot(x1, ek1, 'r-', label=r'%s' % label_1)
                plt.plot(x2, ek2, 'b--', label=r'%s' % label_2)
                plt.plot(x3, ek3, 'c-', label=r'%s' % label_3)
                #plt.plot(x4, ek4, 'm-', label=r'%s' % label_4)
                #plt.plot(x5, ek5, 'r--', label=r'%s' % label_5)
                #plt.plot(x6, ek6, 'b--', label=r'%s' % label_6)
                plt.legend(loc='best')
                plt.xlabel('$t$')
                plt.ylabel('$E_k$')
                plt.savefig("plots/incompressible-flow/KineticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
                plt.clf()
                # Elastic Energy
                plt.figure(1)
                plt.plot(x1, ee1, 'r-', label=r'%s' % label_1)
                plt.plot(x2, ee2, 'b--', label=r'%s' % label_2)
                plt.plot(x3, ee3, 'c-', label=r'%s' % label_3)
                #plt.plot(x4, ee4, 'm-', label=r'%s' % label_4)
                #plt.plot(x5, ee5, 'r--', label=r'%s' % label_5)
                #plt.plot(x6, ee6, 'b--', label=r'%s' % label_6)
                plt.legend(loc='best')
                plt.xlabel('$t$')
                plt.ylabel('$E_e$')
                plt.savefig("plots/incompressible-flow/ElasticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
                plt.clf()
                # Kinetic and Elastic Energy
                plt.figure(2)
                plt.plot(x1, ek1, 'r-', label=r'%s' % label_1)
                plt.plot(x2, ek2, 'b--', label=r'%s' % label_2)
                plt.plot(x3, ek3, 'c-', label=r'%s' % label_3)
                #plt.plot(x4, ek4, 'm-', label=r'%s' % label_4)
                #plt.plot(x5, ek5, 'r--', label=r'%s' % label_5)
                #plt.plot(x6, ek6, 'b--', label=r'%s' % label_6)
                plt.plot(x1, ee1, 'r-', label=r'%s' % label_1)
                plt.plot(x2, ee2, 'b--', label=r'%s' % label_2)
                plt.plot(x3, ee3, 'c-', label=r'%s' % label_3)
                #plt.plot(x4, ee4, 'm-', label=r'%s' % label_4)
                #plt.plot(x5, ee5, 'r--', label=r'%s' % label_5)
                #plt.plot(x6, ee6, 'b--', label=r'%s' % label_6)
                plt.legend(loc='best')
                plt.xlabel('$t$')
                plt.ylabel('$E_e$')
                plt.savefig("plots/incompressible-flow/KineticAndElasticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
                plt.clf()
                # Nusslet Number
                plt.figure(2)
                plt.plot(x1, nus1, 'r-', label=r'%s' % label_1)
                plt.plot(x2, nus2, 'b--', label=r'%s' % label_2)
                plt.plot(x3, nus3, 'c-', label=r'%s' % label_3)
                #plt.plot(x4, nus4, 'm-', label=r'%s' % label_4)
                #plt.plot(x5, nus5, 'r--', label=r'%s' % label_5)
                #plt.plot(x6, nus6, 'b--', label=r'%s' % label_6)
                plt.legend(loc='best')
                plt.xlabel('$t$')
                plt.ylabel('$Nu$')
                plt.savefig("plots/incompressible-flow/NussletnumberTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
                plt.clf()

            """if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E6 and j==1:
                # Kinetic Energy
                plt.figure(0)
                plt.plot(x1, ek1, 'r-', label=r'$We=0$')
                plt.legend(loc='best')
                plt.xlabel('time(s)')
                plt.ylabel('E_k')
                plt.savefig("plots/incompressible-flow/KineticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
                plt.clf()
                # Elastic Energy
                plt.figure(1)
                plt.plot(x1, ee1, 'r-', label=r'$We=0$')
                plt.xlabel('time(s)')
                plt.ylabel('E_e')
                plt.savefig("plots/incompressible-flow/ElasticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
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
                plt.savefig("plots/incompressible-flow/KineticEnergyTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"We="+str(We)+"dt="+str(dt)+".png")
                plt.clf()
                # Elastic Energy
                plt.figure(1)
                plt.plot(x1, ee1, 'r-', label=r'$\theta=0$')
                plt.plot(x2, ee2, 'b-', label=r'$\theta=(1-\beta)/10$')
                plt.plot(x3, ee3, 'c-', label=r'$\theta=\beta$')
                plt.legend(loc='best')
                plt.xlabel('time(s)')
                plt.ylabel('E_e')
                plt.savefig("plots/incompressible-flow/ElasticEnergyTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"We="+str(We)+"dt="+str(dt)+".png")
                plt.clf()"""



            # Plot Stress/Normal Stress Difference
            tau_xx=project(tau1[0,0],Q)

            mplot(tau_xx)
            plt.colorbar()
            plt.savefig("plots/incompressible-flow/tau_xxRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(th)+".png")
            plt.clf() 
            tau_xy=project(tau1[1,0],Q)

            mplot(tau_xy)
            plt.colorbar()
            plt.savefig("plots/incompressible-flow/tau_xyRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(th)+".png")
            plt.clf() 
            tau_yy=project(tau1[1,1],Q)

            mplot(tau_yy)
            plt.colorbar()
            plt.savefig("plots/incompressible-flow/tau_yyRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(th)+".png")
            plt.clf() 
            theta0 = project(theta0, Q)

            mplot(theta0)
            plt.colorbar()
            plt.savefig("plots/incompressible-flow/TemperatureRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()

            psi = stream_function(u1)
            mplot(psi)
            plt.colorbar()
            plt.savefig("plots/incompressible-flow/stream_functionRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()

            mplot(p1)
            plt.colorbar()
            plt.savefig("plots/incompressible-flow/pressureRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()

            # Plot Velocity Components
            ux=project(u1[0],Q)
            mplot(ux)
            plt.colorbar()
            plt.savefig("plots/incompressible-flow/u_xRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(c1)+".png")
            plt.clf()
            uy=project(u1[1],Q)
            mplot(uy)
            plt.colorbar()
            plt.savefig("plots/incompressible-flow/u_yRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(c1)+".png")
            plt.clf()

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


            # Create 2D array with mesh coordinate points
            points = np.vstack((xvals, yvals)).T


            xx = np.linspace(0,1)
            yy = np.linspace(0,1)
            XX, YY = np.meshgrid(xx,yy)   # (x,y) coordinate data formatted so that it can be used by plt.contour()
                # Pressure
            pp = sci.griddata(points, pvals, (XX, YY), method='linear') # u(x,y) data so that it can be used by 
            pp = np.reshape(pp, (len(xx), len(yy))) # Reshape to 2D array
            TT = sci.griddata(points, Tvals, (XX, YY), method='linear')
            TT = np.reshape(TT, (len(xx), len(yy))) # Reshape to 2D array
            ps = sci.griddata(points, psivals, (XX, YY), method='linear')
            ps = np.reshape(ps, (len(xx), len(yy))) # Reshape to 2D array


            plt.contour(XX, YY, pp, 25)
            plt.title('Pressure Contours')   # PRESSURE CONTOUR PLOT
            plt.colorbar() 
            plt.savefig("plots/incompressible-flow/PressureContoursRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()


            plt.contour(XX, YY, TT, 20)
            plt.title('Temperature Contours')   # PRESSURE CONTOUR PLOT
            plt.colorbar() 
            plt.savefig("plots/incompressible-flow/TemperatureContoursRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()

            plt.contour(XX, YY, ps, 15)
            plt.title('Streamline Contours')   # PRESSURE CONTOUR PLOT
            plt.colorbar() 
            plt.savefig("plots/incompressible-flow/StreamlineContoursRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()

            #Plot Velocity Streamlines USING MATPLOTLIB
            u1_q = project(u1[0],Q)
            uvals = u1_q.vector().get_local()
            v1_q = project(u1[1],Q)
            vvals = v1_q.vector().get_local()

            #Merge arrays
            points = np.vstack((xvals, yvals)).T

            uu = sci.griddata(points, uvals, (XX, YY), method='linear') 
            vv = sci.griddata(points, vvals, (XX, YY), method='linear') 

            # Set all nan values to DOLFIN_EPS (close to zero)
            uu = np.nan_to_num(uu, nan = 0)
            vv = np.nan_to_num(vv, nan = 0)
            # Reshape
            uu = np.reshape(uu, (len(yy), len(xx))) # Reshape to 2D array
            vv = np.reshape(vv, (len(yy), len(xx))) # Reshape to 2D array


                #Determine Speed 
            speed = np.sqrt(uu*uu+ vv*vv)
            plt.streamplot(XX, YY, uu, vv,  
                        density=3,              
                        color=speed,  
                        cmap=cm.gnuplot,                         # colour map
                        linewidth=0.8)                           # line thickness
                                                                    # arrow size
            plt.colorbar()                                          # add colour bar on the right
            plt.title('Natural Convection')
            plt.savefig("plots/incompressible-flow/VelocityContoursRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")   
            plt.clf()                                             # display the plot


            plt.close()

            Tf=T_f    

            if j==loopend:
                if jjj==1:
                    quit()
                jjj+=1
                update_progress(flow_description+str(jjj), 1)
                j = 0
                mesh_refinement = False

       


if __name__ == "__main__":
    # Execute simulations loop with parameters from "parameters.csv"
    main("flow-parameters.csv", mesh_resolution=50, simulation_time=15, mesh_refinement=False)
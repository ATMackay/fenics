
"""
MIT License

Copyright (c) 2021 Alexander Mackay

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


"""
Flow in the unit square: the lid driven cavity problem - Alex Mackay 2021
This Python module contains functions for computing compressible and non-Newtonian flow in the unit square using the finite element method.
...

"""

import csv
from fenics_fem import *  # Import FEniCS helper functions 
import datetime

def main(input_csv,simulation_time, mesh_refinement):

    # Experiment Run Time
    T_f = simulation_time
    Tf = T_f

    dt = 0.001  #Time Stepping  
    Tf = T_f

    tol = 10E-6
    defpar = 1.0

    B, L = 1, 1            # Length
    U = 1.0
    conv = 1
    betav = 0.5     
    Di = 0.005                         #Diffusion Number
    Vh = 0.005
    T_0 = 300
    T_h = 350
    Bi = 0.2               
    c0 = 1000
    c0 = 1500
    Ma = U/c0 
    rho_0 = 1.0

    alph1 = 0.0
    c1 = 0.1
    c2 = 0.01
    c3 = 0.1
    th = 1.0               # DEVSS

    # SET LOOPING PARAMETER
    loopend=4
    j = 0
    err_count = 0
    jjj = 0


    # FEM Solution Convergence/Energy Plot
    x_axis=list()
    y_axis=list()
    u_xg = list()
    u_yg = list()
    tau_xxg = list()
    tau_xyg = list()
    tau_yyg = list()
    #Start Solution Loop
    while j < loopend:
        j+=1
        t = 0.0
        t=0.0
        """ mesh refinemment prescribed in code"""
        # Mesh Refinement 
        if j==1:
            mm=20
        elif j==2:
            mm=30
        elif j==3:
            mm=40
        elif j==4:
            mm=45
            dt=0.0005


        # Define Geometry
        B=1
        L=1

        # DEFINE THE COMPUTATION GRID
        # Choose Mesh to Use

        #mesh = LDC_Regular_Mesh(mm, B, L)
        #mesh = refine_boundary(0, 0, B, L, mesh, 1)

        mesh = Skew_Mesh(mm, B, L)
        mesh = refine_top(0, 0, B, L, mesh, 1, 0.025)
        mplot(mesh)
        plt.savefig("fine_skewed_grid-"+str(mm)+".png")
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

        #Define Boundary Parts
        boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) #FacetFunction("size_t", mesh)
        no_slip.mark(boundary_parts,0)
        left.mark(boundary_parts,1)
        right.mark(boundary_parts,2)
        top.mark(boundary_parts,3)
        ds = Measure("ds")[boundary_parts]

        # Define boundary/stabilisation FUNCTIONS
        # ramped thermal boundary condition
        ramp_function = Expression('0.5*(1+tanh(8*(t-0.5)))*(T_h-T_0)+T_0', degree=2, t=0.0, T_0=T_0, T_h=T_h)

        # Define boundary/stabilisation FUNCTIONS

        ulidreg=Expression(('8*(1.0+tanh(8*t-4.0))*(x[0]*(L-x[0]))*(x[0]*(L-x[0]))','0'), degree=2, t=0.0, L=L, T_0=T_0, T_h=T_h) # Lid Speed 
        ulid=Expression(('0.5*(1.0+tanh(8*t-4.0))','0'), degree=2, t=0.0, T_0=T_0, T_h=T_h) # Lid Speed 
        rampd=Expression('0.5*(1+tanh(8*(2.0-t)))', degree=2, t=0.0)
        rampu=Expression('0.5*(1+tanh(16*(t-2.0)))', degree=2, t=0.0)
        # Mesh functions
        h = CellDiameter(mesh)
        n = FacetNormal(mesh)



        # Dirichlet Boundary Conditions  (Bouyancy driven flow)
        noslip0  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), no_slip)  # No Slip boundary conditions on the left wall
        noslip1 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), left)  # No Slip boundary conditions on the left wall
        noslip2 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), right)  # No Slip boundary conditions on the left wall
        noslip3 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), top)  # No Slip boundary conditions on the left wall
        temp_left =  DirichletBC(Q, T_h, left)    #Temperature on Omega0 
        temp_right =  DirichletBC(Q, T_0, right)    #Temperature on Omega2 

        # Dirichlet Boundary Conditions  (LID DRIVEN CAVITY)
        noslip  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), no_slip)  # No Slip boundary conditions on the left wall
        drive1  =  DirichletBC(W.sub(0), ulidreg, top)  # No Slip boundary conditions on the upper wall
        #slip  = DirichletBC(V, sl, omega0)  # Slip boundary conditions on the second part of the flow wall 
        #temp0 =  DirichletBC(Qt, T_0, omega0)    #Temperature on Omega0 
        #temp2 =  DirichletBC(Qt, T_0, omega2)    #Temperature on Omega2 
        #temp3 =  DirichletBC(Qt, T_0, omega3)    #Temperature on Omega3 
        #Collect Boundary Conditions
        bcu = [noslip, drive1]
        bcp = []
        bcT = []    #temp0, temp2
        bctau = []

        # READ IN FLUID PARAMETERS FROM CSV ------------------------------------------------------------------------

        # Import csv using csv
        with open(input_csv, 'r') as csvfile:
            spamreader = csv.reader(csvfile)
            my_csv_data = list(spamreader)
            re_row = my_csv_data[0]
            we_row = my_csv_data[1]
            ma_row = my_csv_data[2]


        # Set parameters for secondary loop -----------------------------------------------------------------------

        betav = 0.5
        Ma = float(ma_row[1])
        Re = float(re_row[2])
        We = float(we_row[3])


        # Continuation in Reynolds/Weissenberg Number Number (Re-->10Re)
        Ret=Expression('Re*(1.0+0.5*(1.0+tanh(0.7*t-4.0))*19.0)', t=0.0, Re=Re, degree=2)
        Wet=Expression('(0.1+(We-0.1)*0.5*(1.0+tanh(500*(t-2.5))))', t=0.0, We=We, degree=2)


        print('############# FLOW PARAMETERS ############')
        print('Timestep size (s):', dt)
        print('Finish Time (s):', Tf)

        print('############# Scalings & Nondimensional Parameters ############')
        print('Characteristic Length (m):', L)
        print('Characteristic Velocity (m/s):', 1.0)
        print('Reynolds Number:', Re)
        print('Mach Number:', Ma)
        print('Weissenberg Number:', We)
        print('Viscosity Ratio:', betav)
        print('Diffusion Number:' , Di)
        print('Viscous Heating Number:', Vh)

        Np= len(p0.vector().get_local())
        Nv= len(w0.vector().get_local())   
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
        print('############# Stabilisation Parameters ############')
        print('DEVSS Parameter:', th)

        #quit()

        # Initial Density Field
        rho_initial_guess = project(1.0, Q)
        rho0.assign(rho_initial_guess)

        # Initial Temperature Field
        T_initial_guess = project(T_0, Q)
        T0.assign(T_initial_guess)

        # Identity Tensor   
        I = Expression((('1.0','0.0'),
                        ('0.0','1.0')), degree=2)
        I_vec = Expression(('1.0','0.0','1.0'), degree=2)


        #Define Variable Parameters, Strain Rate and other tensors
        gamdots = inner(Dincomp(u1),grad(u1))
        gamdots12 = inner(Dincomp(u12),grad(u12))
        gamdotp = inner(tau1,grad(u1))
        gamdotp12 = inner(tau12,grad(u12))
        thetal = (T)/(T_h-T_0)
        thetar = (T_0)/(T_h-T_0)
        thetar = project(thetar,Qt)
        theta0 = (T0-T_0)/(T_h-T_0)
        #alpha = 1.0/(rho*Cv)


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
        devss_Z = inner(D-Dcomp(u),R)*dx                        # L^2 Projection of rate-of strain
        
        DEVSSl_u12 = 2*(1-betav)*inner(Dcomp(u),Dincomp(v))*dx    
        DEVSSr_u12 = 2*inner(D0,Dincomp(v))*dx   
        DEVSSl_u1 = 2*(1-betav)*inner(Dcomp(u),Dincomp(v))*dx    
        DEVSSr_u1 = 2*inner(D12,Dincomp(v))*dx 

        #DEVSSl_temp1 = (1-Di)*inner(grad(theta),grad(r))
        #DEVSSr_temp1 = (1-Di)*inner(grad(theta),grad(r))


        # Use amg preconditioner if available
        prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

        # Use nonzero guesses - essential for CG with non-symmetric BC
        parameters['krylov_solver']['nonzero_initial_guess'] = True

        # Array for storing for energy data
        t_array=list()
        ek_array=list()
        ee_array=list()

        conerr=list()
        deferr=list()
        tauerr=list()


        # Time-stepping
        t = dt
        start, elapsed = 0.0, 0.0
        iter = 0
        while t < Tf + DOLFIN_EPS:
            iter += 1
            start = time.process_time()
            time_left = (Tf-t)/dt * (elapsed) 

            flow_description = "compressible lid-driven cavity flow: loop: " +str(jjj) + ":"+str(j) + ", Re: "+str(Re)+", We: "+str(We)+", Ma: "+str(Ma)+", betav: "+str(betav) + ", (est) time to completion: " + str(datetime.timedelta(seconds= time_left))
            update_progress(flow_description, t/Tf) # Update progress bar
            # Set Function timestep
            ramp_function.t = t
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
            LPSl_stress = inner(kapp*h*c1*grad(tau),grad(Rt))*dx + inner(kapp*h*c2*div(tau),div(Rt))*dx  # Stress Stabilisation
            
            U12 = 0.5*(u1 + u0)    
     
    

            (u0, D0_vec)=w0.split()  

            D0 = as_matrix([[D0_vec[0], D0_vec[1]],
                            [D0_vec[1], D0_vec[2]]])        #DEVSS STABILISATION
            DEVSSr_u12 = 2*(1-betav)*inner(D0,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS

            U = 0.5*(u + u0)  

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

            
            # compute u^{n+1/2} velocity half-step
            lhsFu12 = Re*rho0*(2.0*(u - u0) / dt + conv*dot(u0, nabla_grad(u0)))
            Fu12 = dot(lhsFu12, v)*dx + \
                + inner(sigma(U, p0, tau0, We, betav), Dincomp(v))*dx \
                + dot(p0*n, v)*ds - dot(betav*nabla_grad(U)*n, v)*ds - (1.0/3)*betav*dot(div(U)*n,v)*ds \
                - dot(tau0*n, v)*ds \
                +  inner(D-Dcomp(u),R)*dx
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

            DEVSSr_u1 = 2*(1-betav)*inner(D12,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS

            
            # Compute u* 
            lhsFus = Re*rho0*((u - u0)/dt + conv*dot(u12, nabla_grad(u12)))
            Fus = dot(lhsFus, v)*dx + \
                + inner(sigma(U, p0, tau0, We, betav), Dincomp(v))*dx \
                + 0.5*dot(p0*n, v)*ds - dot(betav*nabla_grad(U)*n, v)*ds - (1.0/3)*betav*dot(div(U)*n,v)*ds \
                - dot(tau0*n, v)*ds\
                +  inner(D-Dcomp(u),R)*dx     
                
            a2= lhs(Fus)
            L2= rhs(Fus)

                # Stabilisation
            a2+= th*DEVSSl_u1   #[th*DEVSSl_u12]                     
            L2+= th*DEVSSr_u1    #[th*DEVSSr_u12]

            A2 = assemble(a2)        
            b2 = assemble(L2)
            [bc.apply(A2, b2) for bc in bcu]
            solve(A2, ws.vector(), b2, "bicgstab", "default")
            end()

            (us, Ds_vec) = ws.split()



            # compute p^{n+1} using the continutity equation
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


            # compute \rho^{n+1} using the equations of state 
            rho1 = rho0 + (Ma*Ma/Re)*(p1-p0)
            rho1 = project(rho1,Q)


            # compute u^{n+1} 
            lhs_u1 = (Re/dt)*rho1*u                                          # Left Hand Side
            rhs_u1 = (Re/dt)*rho1*us                                         # Right Hand Side

            a7=inner(lhs_u1,v)*dx  + inner(D-Dcomp(u),R)*dx  # Weak Form
            L7=inner(rhs_u1,v)*dx + 0.5*inner(p1-p0,div(v))*dx - 0.5*dot(p1*n, v)*ds

            a7+= 0   #[th*DEVSSl_u1]                                                #DEVSS Stabilisation
            L7+= 0   #[th*DEVSSr_u1] 

            A7 = assemble(a7)
            b7 = assemble(L7)
            [bc.apply(A7, b7) for bc in bcu]
            solve(A7, w1.vector(), b7, "bicgstab", "default")
            end()

            (u1, D1_vec) = w1.split()
            D1 = as_matrix([[D1_vec[0], D1_vec[1]],
                            [D1_vec[1], D1_vec[2]]])

            U12 = 0.5*(u1 + u0)   

            # compute \tau^{n+1}
            lhs_tau1 = (We/dt+1.0)*tau  +  We*Fdef(u1,tau)                            # Left Hand Side
            rhs_tau1= (We/dt)*tau0 + 2.0*(1.0-betav)*Dcomp(u1) #+  F_tau       # Right Hand Side

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


            # Energy Calculations
            E_k=assemble(0.5*rho1*dot(u1,u1)*dx)
            E_e=assemble((tau1_vec[0]+tau1_vec[2])*dx)


            t_array.append(t)
            ek_array.append(E_k)
            ee_array.append(E_e)

            
             # Update Solutions
            w0.assign(w1)
            T0.assign(T1)
            rho0.assign(rho1)
            p0.assign(p1)
            tau0_vec.assign(tau1_vec)
            # Move to next time step (Continuation in Reynolds Number)
            t += dt
            elapsed = (time.process_time() - start)

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

            # Reset Parameters   
            j = 0
            dt = 0.001
            Tf = T_f
            th = 1.0
            x1 = x2 = x3 = x4 = x5 = list()
            y = y1 = y3 = y4 = y5 = list()
            ek1 = ek2 = ek3 = ek4 = ek5 = list()
            ee1 = ee2 = ee3 = ee4 = ee5 = list()
        else:
            # Save array data to file
            save_energy_arrays(t_array, ek_array, ee_array, j, "mesh")
            # PLOTS
            # Minimum of stream function (Eye of Rotation)
            u1 = project(u1, V)

            # Stream function minimum
            psi = comp_stream_function(rho1, u1)
            psi_min = min(psi.vector().get_local())
            min_loc = min_location(psi, mesh)

            with open("results/Compressible-Stream-Function.txt", "a") as text_file:
                text_file.write("res="+str(mm)+"Ra="+str(Re)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+"----- psi_min="+str(psi_min)+"---"+str(min_loc)+'\n')

            # Shear rate maximum
            gamma = shear_rate(u1)
            gamma_max = max(gamma.vector().get_local())
            max_loc = max_location(gamma, mesh)
            with open("results/Compressible-Shear-Rate.txt", "a") as text_file:
                text_file.write("res="+str(mm)+"Re="+str(Re)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+"----- shear_rate="+str(gamma_max)+"---"+str(max_loc)+'\n')

            # Data on Kinetic/Elastic Energies
            with open("results/Compressible-ConformEnergy.txt", "a") as text_file:
                text_file.write("res="+str(mm)+"Re="+str(Re)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+", E_k="+str(E_k)+", E_e="+str(E_e)+'\n')

            # Plot DEVSS Data
            D_proj_cont_xx = project(Dcomp(u1)[0,0], Q)
            D_proj_xx = project(D1_vec[0], Qt)
            Diff_loc_xx = project((D_proj_cont_xx - D_proj_xx)/D_proj_xx.vector().get_local().max(), Q)
            D_proj_cont_xy = project(Dcomp(u1)[0,1], Qt)
            D_proj_xy = project(D1_vec[1], Q)
            Diff_loc_xy = project((D_proj_cont_xy - D_proj_xy)/D_proj_xy.vector().get_local().max(), Q)
            D_proj_cont_yy = project(Dcomp(u1)[1,1], Q)
            D_proj_yy = project(D1_vec[2], Qt)
            Diff_loc_yy = project((D_proj_cont_yy - D_proj_yy)/D_proj_yy.vector().get_local().max(), Q)
            mplot(Diff_loc_xx)
            plt.colorbar()
            plt.savefig("plots/stability-convergence/DEVSS_xxResolution="+str(mm)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()
            mplot(Diff_loc_xy)
            plt.colorbar()
            plt.savefig("plots/stability-convergence/DEVSS_xyResolution="+str(mm)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()
            mplot(Diff_loc_yy)
            plt.colorbar()
            plt.savefig("plots/stability-convergence/DEVSS_yyResolution="+str(mm)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()

            if j==loopend:
                # Kinetic Energy
                x1, ek1, ee1 = load_energy_arrays(1, "mesh")
                x2, ek2, ee2 = load_energy_arrays(2, "mesh")
                x3, ek3, ee3 = load_energy_arrays(3, "mesh")
                x4, ek4, ee4 = load_energy_arrays(4, "mesh")
                plt.figure(0)
                plt.plot(x1, ek1, 'r-', label=r'M1')
                plt.plot(x2, ek2, 'b--', label=r'M2')
                plt.plot(x3, ek3, 'c-', label=r'M3')
                plt.plot(x4, ek4, 'm--', label=r'M4')
                plt.legend(loc='best')
                plt.xlabel('time(s)')
                plt.ylabel('$E_k$')
                plt.savefig("plots/energy/MeshKineticEnergyRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+"t="+str(t)+".png")
                plt.clf()
                # Elastic Energy
                plt.figure(1)
                plt.plot(x1, ee1, 'r-', label=r'M1')
                plt.plot(x2, ee2, 'b--', label=r'M2')
                plt.plot(x3, ee3, 'c-', label=r'M3')
                plt.plot(x4, ee4, 'm--', label=r'M4')
                plt.legend(loc='best')
                plt.xlabel('time(s)')
                plt.ylabel('$E_e$')
                plt.savefig("plots/energy/MeshElasticEnergyRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+"t="+str(t)+".png")
                plt.clf()
                plt.close()


            # Plot First Normal Stress Difference
            tau_xx=project(tau1[0,0],Q)
            mplot(tau_xx)
            plt.colorbar()
            plt.savefig("plots/flow/tau_xxResolution="+str(mm)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf() 
            tau_xy=project(tau1[1,0],Q)
            mplot(tau_xy)
            plt.colorbar()
            plt.savefig("plots/flow/tau_xyResolution="+str(mm)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf() 
            tau_yy=project(tau1[1,1],Q)
            mplot(tau_yy)
            plt.colorbar()
            plt.savefig("plots/flow/tau_yyResolution="+str(mm)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf() 
            divu = project(div(u1),Q)
            mplot(divu)
            plt.colorbar()
            plt.savefig("plots/flow/div_uResolution="+str(mm)+"Re="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()

            # Plot Velocity Components
            ux=project(u1[0],Q)
            mplot(ux)
            plt.colorbar()
            plt.savefig("plots/flow/u_xResolution="+str(mm)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()
            uy=project(u1[1],Q)
            mplot(uy)
            plt.colorbar()
            plt.savefig("plots/flow/u_yResolution="+str(mm)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()


            # Matlab Plot of the Solution at t=Tf
            rho1=rho_0*rho1
            rho1=project(rho1,Q)
            #p1=mu_0*(L/U)*p1  #Dimensionalised Pressure
            #p1=project(p1,Q)
            mplot(rho1)
            plt.colorbar()
            plt.savefig("plots/flow/DensityResolution="+str(mm)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf() 
            mplot(p1)
            plt.colorbar()
            plt.savefig("plots/flow/PressureResolution="+str(mm)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()
            mplot(T1)
            plt.colorbar()
            plt.savefig("plots/flow/TemperatureResolution="+str(mm)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()

            #Plot Contours USING MATPLOTLIB
            # Scalar Function code
            x = Expression('x[0]', degree=2)     #GET X-COORDINATES LIST
            y = Expression('x[1]', degree=2)     #GET Y-COORDINATES LIST
            pvals = p1.vector().get_local()          # GET SOLUTION p= p(x,y) list
            Tvals = T1.vector().get_local()         # GET SOLUTION T= T(x,y) list
            rhovals = rho1.vector().get_local()     # GET SOLUTION p= p(x,y) list
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
            pp = sci.griddata(points, pvals, (XX, YY), method='linear') # u(x,y) data so that it can be used by 
            pp = np.reshape(pp, (len(xx), len(yy))) # Reshape to 2D array
            TT = sci.griddata(points, Tvals, (XX, YY), method='linear')
            TT = np.reshape(TT, (len(xx), len(yy))) # Reshape to 2D array
            dd = sci.griddata(points, rhovals, (XX, YY), method='linear')
            dd = np.reshape(dd, (len(xx), len(yy))) # Reshape to 2D array
            normstress = sci.griddata(points, tauxxvals, (XX, YY), method='linear')
            normstress = np.reshape(dd, (len(xx), len(yy))) # Reshape to 2D array


            plt.contour(XX, YY, dd, 25)
            plt.title('Density Contours')   # DENSITY CONTOUR PLOT
            plt.colorbar() 
            plt.savefig("plots/flow/DensityContoursResolution="+str(mm)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()

            plt.contour(XX, YY, pp, 25)
            plt.title('Pressure Contours')   # PRESSURE CONTOUR PLOT
            plt.colorbar() 
            plt.savefig("plots/flow/PressureContoursResolution="+str(mm)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()

            #plt.contour(XX, YY, TT, 20) 
            #plt.title('Temperature Contours')   # TEMPERATURE CONTOUR PLOT
            #plt.colorbar() 
            #plt.savefig("plots/flow/TemperatureContoursRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            #plt.clf()

            plt.contour(XX, YY, normstress, 20) 
            plt.title('Stress Contours')   # NORMAL STRESS CONTOUR PLOT
            plt.colorbar() 
            plt.savefig("plots/flow/StressContoursResolution="+str(mm)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()


            #Plot Velocity Streamlines USING MATPLOTLIB
            u1_q = project(u1[0],Q)
            uvals = u1_q.vector().get_local()
            v1_q = project(u1[1],Q)
            vvals = v1_q.vector().get_local()


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
            plt.title('Lid Driven Cavity Flow')
            plt.savefig("plots/flow/VelocityContoursResolution="+str(mm)+"x"+str(mm)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")   
            plt.clf()                                             # display the plot


            plt.close()


            if dt < tol:
                j=loopend+1
                break


if __name__ == "__main__":
    # Execute simulations loop with parameters from "parameters.csv"
    main("flow-parameters.csv", simulation_time=15.0, mesh_refinement=False)

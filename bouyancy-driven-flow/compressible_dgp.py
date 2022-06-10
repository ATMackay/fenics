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
Flow in the unit square - Alex Mackay 2021
This Python module contains functions for computing compressible and non-Newtonian flow in the unit square using the finite element method.
...

"""

import csv
from fenics_fem import *  # Import FEniCS helper functions 


def main(input_csv,mesh_resolution,simulation_time, mesh_refinement):

    update_progress("compressible bouyancy-driven flow in the unit square", 0)

    # Experiment Run Time
    T_f = simulation_time
    Tf = T_f

    # Timestepping parameters
    dt = 0.001  #Timestep

    # Nondimensional flow parameters
    B, L = 1, 1            # Length
    U = 1
    betav = 0.99    
    Ra = 10000             # Rayleigh Number
    Pr = 2.0
    We = 0.01              # Weisenberg NUmber
    Vh = 0.005
    T_0 = 300
    T_h = 350
    Bi = 0.0
    Di = 0.005             #Diffusion Number
    Ma = 0.01
    al = 2.0
    beta_0 = 300
    beta_1 = 0
    beta_2 = 1
    al_1 = (beta_1 + beta_2 * T_0)/beta_0
    al_2 = beta_2*(T_h-T_0)/beta_0

    c1 = 0.05
    c2 = 0.001
    th = 0.5              # DEVSS
    C = 200.


    # SET LOOPING PARAMETER
    loopend = 4
    j = 0                            
    err_count = 0
    jjj = 0

    # FEM Solution Convergence/Energy Plot
    # TODO FIX this so that fewer arrays are being defined
    x1=list()
    x2=list()
    x3=list()
    x4=list()
    x5=list()
    y=list()
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
    nus1 = list()
    nus2 = list()
    nus3 = list()
    nus4 = list()
    nus5 = list() 
    while j < loopend:
        j+=1
        t=0.0

        # DEFINE THE COMPUTATION GRID
        # Choose Mesh to Use

        mesh = DGP_Mesh(mesh_resolution, B, L)
        #gdim = mesh.geometry().dim() # Mesh Geometry

        mplot(mesh)
        plt.savefig("fine_skewed_grid.png")
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
                return True if x[1] < bottom_bound - DOLFIN_EPS and on_boundary  else False 

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

        #Define Boundary Parts

        boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) #FacetFunction("size_t", mesh)
        no_slip.mark(boundary_parts,0)
        left.mark(boundary_parts,1)
        right.mark(boundary_parts,2)
        top.mark(boundary_parts,3)
        ds = Measure("ds")[boundary_parts]

        # Define boundary/stabilisation FUNCTIONS
        # ramped thermal boundary condition
        #ramp_function = Expression('0.5*(1+tanh(8*(t-0.5)))*(T_h-T_0)+T_0', degree=2, t=0.0, T_0=T_0, T_h=T_h)
        ramp_function = Expression('0.5*(1+tanh(4*(t-0.5)))*(T_h-T_0)+T_0', degree=2, t=0.0, T_0=T_0, T_h=T_h)
        # direction of gravitational force (0,-1)
        f = Expression(('0','-1'), degree=2)
        k = Expression(('0','1'), degree=2)


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
            ma_row = my_csv_data[2]


        # Set parameters for secondary loop -----------------------------------------------------------------------

        betav = 0.5
        Ma = float(ma_row[jjj+1])

        # Set parameters for primary loop ------------------------------------------------        
        if j==1:
            Ra = float(ra_row[1])
            We = float(we_row[3])
        elif j==2:
            Ra = float(ra_row[2])
            We = float(we_row[3])
        elif j==3:
            Ra = float(ra_row[3])
            We = float(we_row[3])
        elif j==4:
            Ra = float(ra_row[3])
            We = float(we_row[2])

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

        phi0 = project((T0+C)/(T_0+C)*(T0/T_0)**(3/2),Q)


        # Initial Conformation Tensor
        I_vec = Expression(('1.0','0.0','1.0'), degree=2)
        initial_guess_conform = project(I_vec, Zc)
        assign(tau0_vec, initial_guess_conform)         # Initial guess for conformation tensor is Identity matrix


        # Initial Density Field
        rho_initial = Expression('1.0', degree=1)
        rho_initial_guess = project(1.0, Q)
        rho0.assign(rho_initial_guess)


        # Initial Temperature Field
        T_initial_guess = project(T_0, Q)
        T0.assign(T_initial_guess)





        #Define Variable Parameters, Strain Rate and other tensors
        thetal = (T)/(T_h-T_0)
        thetar = (T_0)/(T_h-T_0)
        thetar = project(thetar,Q)
        theta0 = (T0-T_0)/(T_h-T_0)
        theta1 = (T1-T_0)/(T_h-T_0)



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



        # FEM Solution Convergence Plot
        x=list()
        y=list()


        # Set up Krylov Solver 

        # Use amg preconditioner if available
        prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

        # Use nonzero guesses - essential for CG with non-symmetric BC
        parameters['krylov_solver']['nonzero_initial_guess'] = True
        parameters['krylov_solver']['monitor_convergence'] = False
        
        solveru = KrylovSolver("bicgstab", "default")
        solvertau = KrylovSolver("bicgstab", "default")
        solverp = KrylovSolver("bicgstab", "default")


        # Time-stepping
        t = 0.0
        iter = 0            # iteration counter
        while t < Tf + DOLFIN_EPS:
            flow_description = "compressible bouyancy-driven flow: loop: " +str(j) + ", Ra: "+str(Ra)+", We: "+str(We)+", Ma: "+str(Ma)+", Pr: "+str(Pr)+", al_1: "+str(al_1)+", al_2: "+str(al_2)+", betav: "+str(betav)
            update_progress(flow_description, t/Tf) # Update progress bar
            iter += 1
            # Set Function timestep
            ramp_function.t = t

            # Update Stabilisation (Copy and Paste Stabilisation Technique from above)
            F1R = Fdefcom(u1, tau1)  #Compute the residual in the STRESS EQUATION
            F1R_vec = as_vector([F1R[0,0], F1R[1,0], F1R[1,1]])
            Dcomp1_vec = as_vector([Dcomp(u1)[0,0], Dcomp(u1)[1,0], Dcomp(u1)[1,1]])
            restau0 = We/dt*(tau1_vec-tau0_vec) + We*F1R_vec + tau1_vec - I_vec
            res_test = project(restau0, Zd)
            res_orth = project(restau0-res_test, Zc)                                
            res_orth_norm_sq = project(inner(res_orth,res_orth), Qt)     # Project residual norm onto discontinuous space
            res_orth_norm = np.power(res_orth_norm_sq, 0.5)
            kapp = project(res_orth_norm, Qt)
            LPSl_stress = inner(kapp*h*c1*grad(tau),grad(Rt))*dx + inner(kapp*h*c2*div(tau),div(Rt))*dx  # Stress Stabilisation

            U12 = 0.5*(u1 + u0)  
    
            # Update Solutions
            if iter > 1:
                w0.assign(w1)
                T0.assign(T1)
                rho0.assign(rho1)
                p0.assign(p1)
                tau0_vec.assign(tau1_vec)

            # Use previous solutions to update nonisthermal paramters
            phi0 = project((T0+C)/(T_0+C)*(T0/T_0)**(3/2),Q)   
            theta0 = project((T0-T_0)/(T_h-T_0), Q) 
            al_1 = (beta_1 + beta_2 * T_0)/beta_0
            al_2 = beta_2*(T_h-T_0)/beta_0          
            therm = (al_1+al_2*theta0)
            therm_inv = project(1.0/therm, Q)

            (u0, D0_vec) = w0.split()
            D0 = as_matrix([[D0_vec[0], D0_vec[1]],
                            [D0_vec[1], D0_vec[2]]])                    #DEVSS STABILISATION
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
            Du12Dt = rho0*(2.0*(u - u0) / dt + dot(u0, nabla_grad(u0)))
            Fu12 = dot(Du12Dt, v)*dx + \
                + inner(sigmacom(U, p0, tau0, We, Pr, betav), Dincomp(v))*dx + Ra*Pr*inner(rho0*therm_inv*k,v)*dx \
                + dot(p0*n, v)*ds - betav*Pr*(dot(nabla_grad(U)*n, v)*ds + (1.0/3)*dot(div(U)*n,v)*ds)\
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

            # split solution
            (u12, D12_vec) = w12.split()
            D12 = as_matrix([[D12_vec[0], D12_vec[1]],
                            [D12_vec[1], D12_vec[2]]])
            DEVSSr_u1 = 2*(1-betav)*inner(D12,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS     

            """# compute \tau^{n+1}
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


            # Compute u* 
            lhsFus = rho0*((u - u0)/dt + dot(u12, nabla_grad(U)))
            Fus = dot(lhsFus, v)*dx + \
                + inner(sigmacom(U, p0, tau0, We, Pr, betav), Dincomp(v))*dx + Ra*Pr*inner(rho0*therm_inv*k,v)*dx \
                + dot(p0*n, v)*ds - betav*Pr*(dot(nabla_grad(U)*n, v)*ds + (1.0/3)*dot(div(U)*n,v)*ds) \
                - (Pr*(1.0-betav)/We)*dot(tau0*n, v)*ds\
                + inner(D-Dincomp(u),R)*dx             
            a2= lhs(Fus)
            L2= rhs(Fus)
            # Stabilise solution with DEVSS terms
            a2+= th*DEVSSl_u1                       
            L2+= th*DEVSSr_u1   
            A2 = assemble(a2)        
            b2 = assemble(L2)
            [bc.apply(A2, b2) for bc in bcu]
            solve(A2, ws.vector(), b2, "bicgstab", "default")
            end()
            (us, Ds_vec) = ws.split()
            

            # compute p^{n+1} using the continutity equation
            lhs_p_1 = (Ma*Ma/(dt))*p
            rhs_p_1 = (Ma*Ma/(dt))*p0 - therm*dot(grad(rho0),us) - therm*rho0*div(us)

            lhs_p_2 = therm*0.5*dt*grad(p)
            rhs_p_2 = therm*0.5*dt*grad(p0)
            
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

            # compute \rho^{n+1} using the equations of state 
            rho1 = rho0 + (Ma*Ma)*therm_inv*(p1-p0)
            rho1 = project(rho1,Q)

            # compute u^{n+1} 
            lhs_u1 = (1./dt)*rho1*u                                          # Left Hand Side
            rhs_u1 = (1./dt)*rho0*us                                         # Right Hand Side

            a7=inner(lhs_u1,v)*dx +inner(D-Dincomp(u),R)*dx  
            L7=inner(rhs_u1,v)*dx + 0.5*(inner(p1-p0,div(v))*dx - dot((p1-p0)*n, v)*ds) 
            a7+= 0                      #DEVSS Stabilisation (optional)
            L7+= 0 
            A7 = assemble(a7)
            b7 = assemble(L7)
            [bc.apply(A7, b7) for bc in bcu]
            solve(A7, w1.vector(), b7, "bicgstab", "default")
            end()

            (u1, D1_vec) = w1.split()


            # compute \tau^{n+1}
            stress_eq = ((We/dt)+1.0)*tau  +  We*Fdefcom(u1,tau) - (We/dt)*tau0 - Identity(len(u)) 
            A = inner(stress_eq,Rt)*dx
            a4 = lhs(A)
            L4 = rhs(A) 
            a4 += LPSl_stress  # [SUPGl4, SUl4, LPSl_stab, LPSl_stress, diff_stab, 0]
            L4 += 0  # [SUPGr4, SUr4, LPSr_stab, LPS_res_stab, 0]   
            A4=assemble(a4)                                     # Assemble System
            b4=assemble(L4)
            [bc.apply(A4, b4) for bc in bctau]
            solvertau.solve(A4, tau1_vec.vector(), b4)
            end()

            # compute T^{n+1}
            gamdot = inner(sigmacom(u1, p1, tau1, We, Pr, betav), grad(u1))
            lhs_theta1 = (1.0/dt)*rho1*thetal + rho1*dot(u1,grad(thetal))
            rhs_theta1 = (1.0/dt)*rho0*thetar + rho1*dot(u1,grad(thetar)) + (1.0/dt)*rho0*theta0 + Vh*gamdot
            a8 = inner(lhs_theta1,r)*dx + inner(grad(thetal),grad(r))*dx 
            L8 = inner(rhs_theta1,r)*dx + inner(grad(thetar),grad(r))*dx + Bi*inner(grad(theta0),n*r)*ds(3) + Bi*inner(grad(theta0),n*r)*ds(1) + inner(We*tau1*grad(thetar),grad(r))*dx

            A8=assemble(a8)                                     # Assemble System
            b8=assemble(L8)
            [bc.apply(A8, b8) for bc in bcT]
            solve(A8, T1.vector(), b8, "bicgstab", "default")
            end()


            # Energy Calculations
            E_k=assemble(0.5*rho1*dot(u1,u1)*dx)
            E_e=assemble((tau1[0,0]+tau1[1,1]-2.0)*dx)

            # Nusselt Number 
            theta1 = project((T1-T_0)/(T_h-T_0), Q)
            Tdx = inner(grad(theta1),n) 
            Nus = assemble(-Tdx*ds(2))

            # Record Elastic & Kinetic Energy Values 
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

            # Move to next timestep
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
            x1 = x2 = x3 = x4 = x5 = list()
            y = y1 = y3 = y4 = y5 = list()
            ek1 = ek2 = ek3 = ek4 = ek5 = list()
            ee1 = ee2 = ee3 = ee4 = ee5 = list()
        else:
            # Minimum of stream function (Eye of Rotation)
            u1 = project(u1, V)

            # Stream function minimum
            psi = comp_stream_function(rho1, u1)
            psi_min = min(psi.vector().get_local())
            min_loc = min_location(psi, mesh)

            with open("results/Compressible-Stream-Function.txt", "a") as text_file:
                text_file.write("Ra="+str(Ra)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+"----- psi_min="+str(psi_min)+"---"+str(min_loc)+'\n')

            # Shear rate maximum
            gamma = shear_rate(u1)
            gamma_max = max(gamma.vector().get_local())
            max_loc = max_location(gamma, mesh)
            with open("results/Compressible-Shear-Rate.txt", "a") as text_file:
                text_file.write("Re="+str(Ra)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+"----- shear_rate="+str(gamma_max)+"---"+str(max_loc)+'\n')

            # Data on Kinetic/Elastic Energies
            with open("results/Compressible-ConformEnergy.txt", "a") as text_file:
                text_file.write("Ra="+str(Ra)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+", E_k="+str(E_k)+", E_e="+str(E_e)+'\n')


            if j==loopend:
                peakEk1 = max(ek1)
                peakEk2 = max(ek2)
                peakEk3 = max(ek3)
                with open("results/Compressible-ConformEnergy.txt", "a") as text_file:
                    text_file.write("Ra="+str(Ra)+", We="+str(We)+", Ma="+str(Ma)+"-------peak Kinetic Energy: "+str(peakEk3)+"Incomp Kinetic En"+str(peakEk1)+'\n')

            plt.clf()
            plt.close()
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
            plt.savefig("results/Stability-Convergence/Mesh_KineticEnergyRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
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
            plt.savefig("results/Stability-Convergence/Mesh_ElasticEnergyRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
            plt.clf()"""

            # Plot Kinetic and elasic Energies for different REYNOLDS numbers at constant Weissenberg Number    
            # Kinetic Energy
            plt.figure(0)
            plt.plot(x1, ek1, 'r-', label=r'$Ra=500$,$We=0.25$')
            plt.plot(x2, ek2, 'b-', label=r'$Ra=1000$,$We=0.25$')
            plt.plot(x3, ek3, 'c-', label=r'$Ra=5000$,$We=0.25$')
            plt.plot(x4, ek4, 'm-', label=r'$Ra=5000$,$We=0.1$')
            #plt.plot(x5, ek5, 'g-', label=r'$Ra=50$')
            plt.legend(loc='best')
            plt.xlabel('$t$')
            plt.ylabel('$E_k$')
            plt.savefig("plots/compressible-flow/Fixed_We_KineticEnergyRa="+str(Ra)+"We="+str(We)+", Pr="+str(Ra)+"b="+str(betav)+"dt="+str(dt)+".png")
            plt.clf()
            plt.close()
            # Elastic Energy
            plt.figure(1)
            plt.plot(x1, ee1, 'r-', label=r'$Ra=500$,$We=0.25$')
            plt.plot(x2, ee2, 'b-', label=r'$Ra=1000$,$We=0.25$')
            plt.plot(x3, ee3, 'c-', label=r'$Ra=5000$,$We=0.25$')
            plt.plot(x4, ee4, 'm-', label=r'$Ra=5000$,$We=0.1$')
            #plt.plot(x5, ee5, 'g-', label=r'$Ra=50$')
            plt.legend(loc='best')
            plt.xlabel('$t$')
            plt.ylabel('$E_e$')
            plt.savefig("plots/compressible-flow/Fixed_We_ElasticEnergyRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
            plt.clf()

            plt.close()

            plt.figure(2)
            plt.plot(x1, nus1, 'r-', label=r'$Ra=500$,$We=0.25$')
            plt.plot(x2, nus2, 'b-', label=r'$Ra=1000$,$We=0.25$')
            plt.plot(x3, nus3, 'c-', label=r'$Ra=5000$,$We=0.25$')
            plt.plot(x4, nus4, 'm-', label=r'$Ra=5000$,$We=0.1$')
            #plt.plot(x5, ee5, 'g-', label=r'$We=2.0$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$N_u$')
            plt.savefig("plots/compressible-flow/NussletNumberTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
            plt.clf()

            
            #Plot Kinetic and elasic Energies for different Weissenberg numbers at Re=0 (METHOD 2)
            # Kinetic Energy
            """plt.figure(0)
            plt.plot(x1, ek1, 'r-', label=r'$We=0.0, Ra=500$')
            plt.plot(x2, ek2, 'b-', label=r'$We=0.1, Ra=500$')
            plt.plot(x3, ek3, 'c-', label=r'$We=0.5, Ra=500$')
            plt.plot(x4, ek4, 'm-', label=r'$We=0.5, Ra=750$')
            #plt.plot(x5, ek5, 'g-', label=r'$We=2.0$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$E_k$')
            plt.savefig("plots/compressible-flow/KineticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
            plt.clf()
            # Elastic Energy
            plt.figure(1)
            plt.plot(x1, ee1, 'r-', label=r'$We=0.0, Ra=500$')
            plt.plot(x2, ee2, 'b-', label=r'$We=0.1, Ra=500$')
            plt.plot(x3, ee3, 'c-', label=r'$We=0.5, Ra=500$')
            plt.plot(x4, ee4, 'm-', label=r'$We=0.5, Ra=750$')
            #plt.plot(x5, ee5, 'g-', label=r'$We=2.0$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$E_e$')
            plt.savefig("plots/compressible-flow/ElasticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
            plt.clf()

            plt.figure(2)
            plt.plot(x1, nus1, 'r-', label=r'$We=0.0, Ra=500$')
            plt.plot(x2, nus2, 'b-', label=r'$We=0.1, Ra=500$')
            plt.plot(x3, nus3, 'c-', label=r'$We=0.5, Ra=500$')
            plt.plot(x4, nus4, 'm-', label=r'$We=0.5, Ra=750$')
            #plt.plot(x5, ee5, 'g-', label=r'$We=2.0$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$N_u$')
            plt.savefig("plots/compressible-flow/NussletNumberTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
            plt.clf()"""

            """# Kinetic Energy
            plt.figure(0)
            plt.plot(x1, ek1, 'r-', label=r'$We=0$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('E_k')
            plt.savefig("Inresults/Energy/KineticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
            plt.clf()
            # Elastic Energy
            plt.figure(1)
            plt.plot(x1, ee1, 'r-', label=r'$We=0$')
            plt.xlabel('time(s)')
            plt.ylabel('E_e')
            plt.savefig("Inresults/Energy/ElasticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
            plt.clf()"""


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
            plt.savefig("Inresults/Energy/KineticEnergyTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"We="+str(We)+"dt="+str(dt)+".png")
            plt.clf()
            # Elastic Energy
            plt.figure(1)
            plt.plot(x1, ee1, 'r-', label=r'$\theta=0$')
            plt.plot(x2, ee2, 'b-', label=r'$\theta=(1-\beta)/10$')
            plt.plot(x3, ee3, 'c-', label=r'$\theta=\beta$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('E_e')
            plt.savefig("plotsElasticEnergyTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"We="+str(We)+"dt="+str(dt)+".png")
            plt.clf()"""


            # Plot Stress/Normal Stress Difference
            tau_xx=project(tau1[0,0],Q)
            mplot(tau_xx)
            plt.colorbar()
            plt.savefig("plots/compressible-flow/tau_xxRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(th)+".png")
            plt.clf() 
            tau_xy=project(tau1[1,0],Q)
            mplot(tau_xy)
            plt.colorbar()
            plt.savefig("plots/compressible-flow/tau_xyRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(th)+".png")
            plt.clf() 
            tau_yy=project(tau1[1,1],Q)
            mplot(tau_yy)
            plt.colorbar()
            plt.savefig("plots/compressible-flow/tau_yyRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(th)+".png")
            plt.clf() 
            theta0 = project(theta0, Q)
            mplot(theta0)
            plt.colorbar()
            plt.savefig("plots/compressible-flow/TemperatureRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()

            # Plot Velocity Components
            ux=project(u1[0],Q)
            mplot(ux)
            plt.colorbar()
            plt.savefig("plots/compressible-flow/u_xRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(c1)+".png")
            plt.clf()
            uy=project(u1[1],Q)
            mplot(uy)
            plt.colorbar()
            plt.savefig("plots/compressible-flow/u_yRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(c1)+".png")
            plt.clf()

            # Matlab Plot of the Solution at t=Tf
            rho1=project(rho1,Q)
            mplot(rho1)
            plt.colorbar()
            plt.savefig("plots/compressible-flow/DensityRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf() 

            mplot(p1)
            plt.colorbar()
            plt.savefig("plots/compressible-flow/PressureRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()

            mplot(T1)
            plt.colorbar()
            plt.savefig("plots/compressible-flow/TemperatureRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()

            mplot(psi)
            plt.colorbar()
            plt.savefig("plots/compressible-flow/stream_functionRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()

            mplot(gamma)
            plt.colorbar()
            plt.savefig("plots/compressible-flow/shear_rateRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
            plt.clf()

            #Plot Contours USING MATPLOTLIB
                # Scalar Function code


            x = Expression('x[0]', degree=2)     #GET X-COORDINATES LIST
            y = Expression('x[1]', degree=2)     #GET Y-COORDINATES LIST
            pvals = p1.vector().get_local()          # GET SOLUTION p= p(x,y) list
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
            plt.savefig("plots/compressible-flow/PressureContoursRa="+str(Ra)+"Pr="+str(Pr)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()


            plt.contour(XX, YY, TT, 20)
            plt.title('Temperature Contours')   # PRESSURE CONTOUR PLOT
            plt.colorbar() 
            plt.savefig("plots/compressible-flow/TemperatureContoursRa="+str(Ra)+"Pr="+str(Pr)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()

            plt.contour(XX, YY, ps, 15)
            plt.title('Streamline Contours')   # PRESSURE CONTOUR PLOT
            plt.colorbar() 
            plt.savefig("plots/compressible-flow/StreamlineContoursRa="+str(Ra)+"Pr="+str(Pr)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
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

            speed = np.sqrt(uu*uu+ vv*vv)
            plt.streamplot(XX, YY, uu, vv,  
                        density=3,              
                        color=speed,  
                        cmap=cm.gnuplot,                         # colour map
                        linewidth=0.8)                           # line thickness
                                                                    # arrow size
            plt.colorbar()                                          # add colour bar on the right
            plt.title('bouyancy driven flow')
            plt.savefig("plots/compressible-flow/VelocityContoursRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")   
            plt.clf()                                           # display the plot


            plt.close()

            Tf=T_f    

            if j==loopend:
                jjj+=1
                update_progress(flow_description+str(jjj), 1)
                j = 0
                mesh_refinement = False
                x1 = x2 = x3 = x4 = x5 = list()
                y = y1 = y3 = y4 = y5 = list()
                ek1 = ek2 = ek3 = ek4 = ek5 = list()

            if jjj==3:
                quit()

        
if __name__ == "__main__":
    # Execute simulations loop with parameters from "parameters.csv"
    main("flow-parameters.csv", mesh_resolution=50, simulation_time=15, mesh_refinement=False)




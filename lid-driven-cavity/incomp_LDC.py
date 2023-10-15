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
This Python module contains functions for computing incompressible and non-Newtonian flow in the unit square using the finite element method.
Oldroyd-B constitutive model is used for the constituve stress.
Computational solution is stabilized using a combination of DEVSS and OP methods for the velocity and constitutive stress, respectively.
...

"""

import csv
from fenics_fem import *  # Import FEniCS helper functions 
import datetime

def main(input_csv,mesh_resolution,simulation_time, mesh_refinement):

    # Experiment Run Time
    T_f = simulation_time
    Tf = T_f

    dt = 0.001  #Time Stepping  
    Tf = T_f

    B, L = 1, 1            # Length
    conv = 1
    betav = 0.5            
    c1 = 0.05
    c2 = 0.01

    th = 1.0               # DEVSS

    # SET LOOPING PARAMETER
    loopend=3
    j = 0
    jjj = 1

    label_1, label_2, label_3, label_4 = "", "", "", ""


    # FEM Solution Convergence/Energy Plot
    x_axis=list()
    y_axis=list()
    u_xg = list()
    u_yg = list()
    tau_xxg = list()
    tau_yyg = list()
    # Start Solution Loop
    while j < loopend:
        j+=1
        t = 0.0

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
        We = 0.5 #float(we_row[j])
        Re = 0.5 #float(re_row[2])

        if Re < 1:
            conv = 0.0

        # Set parameters for primary loop ------------------------------------------------     
        #if j==1:
        #    c1, c2 = DOLFIN_EPS, DOLFIN_EPS 
        #    label_1 = "$c_1$= "+str(0)+", $c_2$= "+str(0)
        #elif j==2:
        #    c1, c2 = 0.05, 0.01 
        #    label_2 = "$c_1$= "+str(c1)+", $c_2$= "+str(c2)
        #elif j==3:
        #    c1, c2 = 0.5, 0.5 
        #    label_3 = "$c_1$= "+str(c1)+", $c_2$= "+str(c2)  

        #if j==1:
        #    label_1 = "Re= "+str(Re*conv)#+", We = "+str(We)
        #elif j==2:
        #    label_2 = "Re= "+str(Re*conv)#+", We = "+str(We)
        #elif j==3:
        #    label_3 = "Re= "+str(Re*conv)#+", We = "+str(We)

        # DEFINE THE COMPUTATION GRID
        # Choose Mesh to Use

        ## Compare mesh resolution and timestep
        if j==1:
            mesh_resolution = 30
            label_1 = "M1"#mesh="+str(mesh_resolution)
        elif j==2:
            mesh_resolution = 40
            label_2 = "M2"#mesh="+str(mesh_resolution)
        elif j==3:
            mesh_resolution = 45  # <--- 65
            label_3 = "M3"#mesh="+str(mesh_resolution)
        
        if jjj==1:
            dt = 0.005
        elif jjj==2:
            dt = 0.001
        elif jjj==3:
            dt = 0.0005

        mesh = LDC_Regular_Mesh(mesh_resolution, B, L)
        mesh = refine_top(0, 0, B, L, mesh, 1, 0.025)

        mplot(mesh)
        plt.savefig("grid-"+str(mesh_resolution)+".png")
        plt.clf()
        plt.close()

        # Discretization  parameters
        order = 2
        # Function spaces
        W, V, _, Z, Zd, Zc, Q, Qt, Qr = function_spaces(mesh, order)

        # Trial Functions
        _, p, T, tau_vec, u, D_vec, D, tau = trial_functions(Q, Zc, W)


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
        _, _, _, p0, p1, T0, T1, u0, u12, us, u1, D0_vec, D12_vec, Ds_vec, D1_vec, w0, w12, ws, w1, tau0_vec, tau12_vec, tau1_vec = solution_functions(Q, W, V, Zc)
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
        ulidreg=Expression(('8*(1.0+tanh(8*t-4.0))*(x[0]*(L-x[0]))*(x[0]*(L-x[0]))','0'), degree=2, t=0.0, L=L) # Lid Speed 
        ulid=Expression(('0.5*(1.0+tanh(8*t-4.0))','0'), degree=2, t=0.0) # Lid Speed 
        rampd=Expression('0.5*(1+tanh(8*(2.0-t)))', degree=2, t=0.0)
        # Mesh functions
        h = CellDiameter(mesh)
        n = FacetNormal(mesh)


        # Dirichlet Boundary Conditions  (LID DRIVEN CAVITY)
        noslip  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), no_slip)  # No Slip boundary conditions on the left wall
        drive1  =  DirichletBC(W.sub(0), ulidreg, top)  # No Slip boundary conditions on the upper wall
        #Collect Boundary Conditions
        bcu = [noslip, drive1]
        bcp = []
        bctau = []

        print('############# FLOW PARAMETERS ############')
        print('Timestep size (s):', dt)
        print('Finish Time (s):', Tf)

        print('############# Scalings & Nondimensional Parameters ############')
        print('Characteristic Length (m):', L)
        print('Characteristic Velocity (m/s):', 1.0)
        print('Reynolds Number:', Re)
        print('Weissenberg Number:', We)
        print('Viscosity Ratio:', betav)

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
        print('Minimum cell diameter:', mesh.hmin())
        print('Maximum cell diameter:', mesh.hmax())
        print('############# Stabilisation Parameters ############')
        print('DEVSS Parameter:', th)
        
        DEVSSl_u12 = 2*(1-betav)*inner(Dcomp(u),Dincomp(v))*dx    
        DEVSSr_u12 = 2*inner(D0,Dincomp(v))*dx   
        DEVSSl_u1 = 2*(1-betav)*inner(Dcomp(u),Dincomp(v))*dx    
        DEVSSr_u1 = 2*inner(D12,Dincomp(v))*dx 


        # Use nonzero guesses - essential for CG with non-symmetric BC
        parameters['krylov_solver']['nonzero_initial_guess'] = True

        # Array for storing for energy data
        t_array=list()
        ek_array=list()
        ee_array=list()
        err_array=list()
        data_tag = "incomp-flow"

        # Time-stepping
        t = dt
        start, elapsed, total_elapsed = 0.0, 0.0, 0.0
        iter = 0
        while t < Tf + DOLFIN_EPS:
            iter += 1
            start = time.process_time()
            time_left = (Tf-t)/dt * (elapsed) 

            flow_description = "incompressible lid-driven cavity flow: loop: " +str(jjj) + ":"+str(j) + ", Re: "+str(Re*conv)+", We: "+str(We)+", beta: "+str(betav)
            flow_description += ", time taken: " + str(datetime.timedelta(seconds= total_elapsed))
            flow_description += ", (est) time to completion: " + str(datetime.timedelta(seconds= time_left))
            update_progress(flow_description, t/Tf) # Update progress bar

            # Set Function timestep
            rampd.t=t
            ulid.t=t
            ulidreg.t=t
        

            (u0, D0_vec)=w0.split()


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
            #LPSl_vel = th*inner(kapp*Dincomp(u),Dincomp(v))*dx

    
            DEVSSr_u12 = 2*(1-betav)*inner(D0,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS

                
            # VELOCITY HALF STEP
            visc_u12 = betav*grad(u) 
            lhs_u12 = (Re/(dt/2.0))*u
            rhs_u12 = (Re/(dt/2.0))*u0 - Re*conv*grad(u0)*u0

            a1=inner(lhs_u12,v)*dx + inner(visc_u12,grad(v))*dx + (inner(D-Dincomp(u),R)*dx)
            L1=inner(rhs_u12,v)*dx + inner(p0,div(v))*dx - inner(tau0,grad(v))*dx 

                #DEVSS Stabilisation
            a1+= th*DEVSSl_u12                     
            L1+= th*DEVSSr_u12 

            A1 = assemble(a1)
            b1= assemble(L1)
            [bc.apply(A1, b1) for bc in bcu]
            solve(A1, w12.vector(), b1, "bicgstab", "default")
            end()

            (u12, D12_vec)=w12.split()

            DEVSSr_u1 = 2*(1-betav)*inner(D12,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS

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
            
            #Predicted U* Equation
            lhs_us = (Re/dt)*u
            rhs_us = (Re/dt)*u0 - Re*conv*grad(u12)*u12

            a2=inner(lhs_us,v)*dx + inner(D-Dincomp(u),R)*dx
            L2=inner(rhs_us,v)*dx - 0.5*betav*(inner(grad(u0),grad(v))*dx) + inner(p0,div(v))*dx - inner(tau0,grad(v))*dx 
            A2 = assemble(a2)
            b2 = assemble(L2)
            [bc.apply(A2, b2) for bc in bcu]
            solve(A2, ws.vector(), b2, "bicgstab", "default")
            end()

            (us, Ds_vec) = ws.split()


            #PRESSURE CORRECTION
            a5=inner(grad(p),grad(q))*dx 
            L5=inner(grad(p0),grad(q))*dx + (Re/dt)*inner(us,grad(q))*dx
            A5 = assemble(a5)
            b5 = assemble(L5)
            [bc.apply(A5, b5) for bc in bcp]
            solve(A5, p1.vector(), b5, "bicgstab", "default")
            end()
            
            #VELOCITY UPDATE
            visc_u1 = 0.5*betav*grad(u)

            lhs_u1 = (Re/dt)*u                                          # Left Hand Side
            rhs_u1 = (Re/dt)*us                                         # Right Hand Side

            a7=inner(lhs_u1,v)*dx + inner(visc_u1,grad(v))*dx + inner(D-Dincomp(u),R)*dx  # Weak Form
            L7=inner(rhs_u1,v)*dx + 0.5*inner(p1-p0,div(v))*dx 

                #DEVSS Stabilisation
            a1+= th*DEVSSl_u1                      
            L1+= th*DEVSSr_u1 

            A7 = assemble(a7)
            b7 = assemble(L7)
            [bc.apply(A7, b7) for bc in bcu]
            solve(A7, w1.vector(), b7, "bicgstab", "default")
            end()

            (u1, D1_vec) = w1.split()

            # Stress Full Step
            F1 = dot(u1,grad(tau)) - dot(grad(u1),tau) - dot(tau,tgrad(u1)) # Convection/Deformation Terms
            lhs_tau1 = (We/dt+1.0)*tau + We*F1                             # Left Hand Side
            rhs_tau1= (We/dt)*tau0 + 2.0*(1.0-betav)*Dincomp(u12)          # Right Hand Side

            a4 = inner(lhs_tau1,Rt)*dx
            L4 = inner(rhs_tau1,Rt)*dx

                # SUPG / SU / LPS Stabilisation (User Choose One)

            a4 += LPSl_stress  # [SUPGl4, SUl4, LPSl_stab, LPSl_stress, diff_stab, 0]
            L4 += 0  # [SUPGr4, SUr4, LPSr_stab, LPS_res_stab, 0]   


            A4=assemble(a4)                                     # Assemble System
            b4=assemble(L4)
            [bc.apply(A4, b4) for bc in bctau]
            solve(A4, tau1_vec.vector(), b4, "bicgstab", "default")
            end()

            # Energy Calculations
            E_k=assemble(0.5*dot(u1,u1)*dx)
            E_e=assemble((tau1[0,0]+tau1[1,1])*dx)

            
            
            # record elastic & kinetic Energy values
            t_array.append(t)
            ek_array.append(E_k)
            ee_array.append(E_e)
            t += dt
            elapsed = (time.process_time() - start)
            total_elapsed += elapsed

            # Record Error Data 
            err = project(h*kapp,Qt)
            err_array.append(norm(err.vector(),'l2'))

            # Convergence criteria
            #if t > 2:
            #    if abs(ek_array[-1] - ek_array[-2]) < 1e-8:
            #        break

            # Move to next time step
            w0.assign(w1)
            T0.assign(T1)
            p0.assign(p1)
            tau0_vec.assign(tau1_vec)


        # PLOTS
        # Save array data to file
        save_energy_arrays(t_array, ek_array, ee_array, j, data_tag)
        save_err_array(err_array, j, data_tag)
        # Minimum of stream function (Eye of Rotation)
        u1 = project(u1, V)

        # Stream function minimum
        psi = stream_function(u1)
        psi_min = min(psi.vector().get_local())
        min_loc = min_location(psi, mesh)

        with open("results/Incompressible-Stream-Function.txt", "a") as text_file:
            text_file.write("Re="+str(Re*conv)+", We="+str(We)+", t="+str(t)+"----- psi_min="+str(psi_min)+"---"+str(min_loc)+'\n')

        # Shear rate maximum
        gamma = shear_rate(u1)
        gamma_max = max(gamma.vector().get_local())
        max_loc = max_location(gamma, mesh)
        with open("results/Incompressible-Shear-Rate.txt", "a") as text_file:
            text_file.write("Re="+str(Re*conv)+", We="+str(We)+", t="+str(t)+"----- shear_rate="+str(gamma_max)+"---"+str(max_loc)+'\n')

        # Data on Kinetic/Elastic Energies
        with open("results/Incompressible-Energy.txt", "a") as text_file:
            text_file.write("Re="+str(Re*conv)+", We="+str(We)+", t="+str(t)+", E_k="+str(E_k)+", E_e="+str(E_e)+'\n')

        # Plot Cross Section Flow Values 
        u_x = project(u1[0],Q)      # Project U_x onto scalar function space
        u_y = project(u1[1],Q)      # Project U_y onto scalar function space
        tau_xx = project(tau1_vec[0],Q)
        tau_xy = project(tau1_vec[1],Q)
        tau_yy = project(tau1_vec[2],Q)
        for i in range(mm):
            x_axis.append(0.5*(1.0-cos(i*pi/mm)))
            y_axis.append(0.5*(1.0-cos(i*pi/mm)))
            u_xg.append(u_x([0.5,0.5*(1.0-cos(i*pi/mm))]))   
            u_yg.append(u_y([0.5*(1.0-cos(i*pi/mm)),0.75])) 
            tau_xxg.append(tau_xx([0.5*(1.0-cos(i*pi/mm)), 1.0])) 
            tau_yyg.append(tau_xx([0.5*(1.0-cos(i*pi/mm)), 1.0]))  
        if j==loopend:
            # First Normal Stress
            x1, ek1, ee1 = load_energy_arrays(1, data_tag)
            x2, ek2, ee2 = load_energy_arrays(2, data_tag)
            x3, ek3, ee3 = load_energy_arrays(3, data_tag)
            err1 = load_err_array(1, data_tag)
            err2 = load_err_array(1, data_tag)
            err3 = load_err_array(1, data_tag)
            #x4, ek4, ee4 = load_energy_arrays(4, data_tag)
            # Kinetic Energy
            plt.figure(0)
            plt.plot(x1, ek1, 'r-', label=r'%s' % label_1)
            plt.plot(x2, ek2, 'b--', label=r'%s' % label_2)
            plt.plot(x3, ek3, 'c-', label=r'%s' % label_3)
            #plt.plot(x4, ek4, 'g-', label=r'%s' % label_4)
            plt.legend(loc='best')
            plt.xlabel('$t$')
            plt.ylabel('$E_k$')
            plt.savefig("plots/incompressible/energy/KineticEnergyRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+"t="+str(t)+".png")
            plt.clf()
            # Elastic Energy
            plt.figure(1)
            plt.plot(x1, ee1, 'r-', label=r'%s' % label_1)
            plt.plot(x2, ee2, 'b--', label=r'%s' % label_2)
            plt.plot(x3, ee3, 'c-', label=r'%s' % label_3)
            #plt.plot(x4, ee4, 'g-', label=r'%s' % label_4)
            plt.legend(loc='best')
            plt.xlabel('$t$')
            plt.ylabel('$E_e$')
            plt.savefig("plots/incompressible/energy/ElasticEnergyRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+"t="+str(t)+".png")
            plt.clf()
            plt.close()
            x_axis1 = list(chunks(x_axis, mm))
            y_axis1 = list(chunks(y_axis, mm))
            u_x1 = list(chunks(u_xg, mm))
            u_y1 = list(chunks(u_yg, mm))
            tau_xx1 = list(chunks(tau_xxg, mm))
            tau_yy1 = list(chunks(tau_yyg, mm))
            plt.figure(2)
            plt.plot(x_axis1[0], u_y1[0], 'r-', label=r'%s' % label_1)
            plt.plot(x_axis1[1], u_y1[1], 'b--', label=r'%s' % label_2)
            plt.plot(x_axis1[2], u_y1[2], 'c-', label=r'%s' % label_3)
            #plt.plot(x_axis1[3], u_y1[3], 'g--', label=r'%s' % label_4)
            plt.legend(loc='best')
            plt.xlabel('x')
            plt.ylabel('$u_y(x,0.75)$')
            plt.savefig("plots/incompressible/cross-section/u_yRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"dt="+str(dt)+".png")
            plt.clf()
            plt.figure(3)
            plt.plot(u_x1[0], y_axis1[0], 'r-', label=r'%s' % label_1)
            plt.plot(u_x1[1], y_axis1[1], 'b--', label=r'%s' % label_2)
            plt.plot(u_x1[2], y_axis1[2], 'c-', label=r'%s' % label_3)
            #plt.plot(u_x1[3], y_axis1[3], 'g--', label=r'%s' % label_4)
            plt.legend(loc='best')
            plt.xlabel('$u_x(0.5,y)$')
            plt.ylabel('y')
            plt.savefig("plots/incompressible/cross-section/u_xRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"dt="+str(dt)+".png")
            plt.clf()
            plt.figure(4)
            plt.plot(x_axis1[0], tau_xx1[0], 'r-', label=r'%s' % label_1)
            plt.plot(x_axis1[1], tau_xx1[1], 'b--', label=r'%s' % label_2)
            plt.plot(x_axis1[2], tau_xx1[2], 'c-', label=r'%s' % label_3)
            #plt.plot(x_axis1[3], tau_xx1[3], 'g--', label=r'%s' % label_4)
            plt.legend(loc='best')
            plt.xlabel('x')
            plt.ylabel('$\tau_{xx}(x,1.0)$')
            plt.savefig("plots/incompressible/cross-section/tau_xxRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"dt="+str(dt)+"mesh="+str(mm)+".png")
            plt.clf()
            plt.figure(5)
            plt.plot(x_axis1[0], tau_yy1[0], 'r-', label=r'%s' % label_1)
            plt.plot(x_axis1[1], tau_yy1[1], 'b--', label=r'%s' % label_2)
            plt.plot(x_axis1[2], tau_yy1[2], 'c-', label=r'%s' % label_3)
            #plt.plot(x_axis1[3], tau_yy1[3], 'g--', label=r'%s' % label_4)
            plt.legend(loc='best')
            plt.xlabel('x')
            plt.ylabel('$\tau_{yy}(x,1.0)$')
            plt.savefig("plots/incompressible/cross-section/tau_yyRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"dt="+str(dt)+"mesh="+str(mm)+".png")
            plt.clf()
            plt.figure(6)
            plt.plot(x1, err1, 'r-', label=r'%s' % label_1)
            plt.plot(x2, err2, 'b--', label=r'%s' % label_2)
            plt.plot(x3, err3, 'c-', label=r'%s' % label_3)
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$||\kappa||_{2}$')
            plt.savefig("plots/incompressible/stability-convergence/kappa-normRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+"mesh="+str(mm)+"c1"+str(c1)+"c2"+str(c2)+".png")
            plt.clf()
            plt.close()


        # Plot First Normal Stress Difference
        tau_xx=project(tau1[0,0],Q)
        mplot(tau_xx)
        plt.colorbar()
        plt.savefig("plots/incompressible/flow/tau_xxRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+"t="+str(t)+"c1"+str(c1)+"c2"+str(c2)+".png")
        plt.clf() 
        tau_xy=project(tau1[1,0],Q)
        mplot(tau_xy)
        plt.colorbar()
        plt.savefig("plots/incompressible/flow/tau_xyRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+"t="+str(t)+"c1"+str(c1)+"c2"+str(c2)+".png")
        plt.clf() 
        tau_yy=project(tau1[1,1],Q)
        mplot(tau_yy)
        plt.colorbar()
        plt.savefig("plots/incompressible/flow/tau_yyRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+"t="+str(t)+"c1"+str(c1)+"c2"+str(c2)+".png")
        plt.clf() 

        # Plot Velocity Components
        ux=project(u1[0],Q)
        mplot(ux)
        plt.colorbar()
        plt.savefig("plots/incompressible/flow/u_xRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+"t="+str(t)+"c1"+str(c1)+"c2"+str(c2)+".png")
        plt.clf()
        uy=project(u1[1],Q)
        mplot(uy)
        plt.colorbar()
        plt.savefig("plots/incompressible/flow/u_yRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+"t="+str(t)+"c1"+str(c1)+"c2"+str(c2)+".png")
        plt.clf()

        mplot(p1)
        plt.colorbar()
        plt.savefig("plots/incompressible/flow/PressureRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+"t="+str(t)+"c1"+str(c1)+"c2"+str(c2)+".png")
        plt.clf()

        #Plot Contours USING MATPLOTLIB
        # Scalar Function code
        x = Expression('x[0]', degree=2)     #GET X-COORDINATES LIST
        y = Expression('x[1]', degree=2)     #GET Y-COORDINATES LIST
        pvals = p1.vector().get_local()          # GET SOLUTION p= p(x,y) list
        Tvals = T1.vector().get_local()         # GET SOLUTION T= T(x,y) list
        tauxx = project(tau1_vec[0], Q)
        tauxxvals = tauxx.vector().get_local()
        xvalsq = interpolate(x, Q)#xyvals[:,0]
        yvalsq= interpolate(y, Q)#xyvals[:,1]


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
        normstress = sci.griddata(points, tauxxvals, (XX, YY), method='linear')
        normstress = np.reshape(normstress, (len(xx), len(yy))) # Reshape to 2D array

        plt.contour(XX, YY, pp, 25)
        plt.title('Pressure Contours')   # PRESSURE CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("plots/incompressible/flow/PressureContoursRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+"t="+str(t)+"c1"+str(c1)+"c2"+str(c2)+".png")
        plt.clf()

        plt.contour(XX, YY, normstress, 20) 
        plt.title('Stress Contours')   # NORMAL STRESS CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("plots/incompressible/flow/StressContoursRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+"t="+str(t)+"c1"+str(c1)+"c2"+str(c2)+".png")
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
        plt.title('Velocity Contours')
        plt.savefig("plots/incompressible/flow/VelocityContoursRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+"t="+str(t)+"c1"+str(c1)+"c2"+str(c2)+".png")   
        plt.clf()                                             # display the plot


        plt.close()
        if j==loopend:
            if jjj==3:
                quit()
            jjj+=1
            update_progress(flow_description+str(jjj), 1)
            j = 0
            mesh_refinement = False

if __name__ == "__main__":
    # Execute simulations loop with parameters from "parameters.csv"
    main("flow-parameters.csv", mesh_resolution=40, simulation_time=8, mesh_refinement=False)

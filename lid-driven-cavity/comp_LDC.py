"""Compressible Lid Driven Cavity Problem for an COMPRESSIBLE Oldroyd-B Fluid"""
"""Solution Method: Finite Element Method using DOLFIN (FEniCS)"""

"""COMPRESSIBLE TAYLOR GALERKIN METHOD"""
"""DISCRETE Elastic Viscous Split Stress (DEVSS) Stabilisation used """

"""Governing Equations:
                        Re rho Du/Dt = -grad(p)
"""


from fenics_fem import *  # Import Base Code for LDC Problem



"""START LOOP that runs the simulation for a range of parameters"""
"""Ensure that break control parameters are re-adjusted if solution diverges"""
"""ADJUSTIBLE PARAMETERS"""


dt = 0.0025  #Time Stepping  
T_f = 10.0
Tf = T_f
loopend=5
j = 0
jj = 0
jjj = 0
tol = 10E-6
defpar = 1.0

conv = 0                                      # Non-inertial Flow Parameter (Re=0)
We=0.5
Re=1
c0 = 1000
Ma = 0.001
betav = 0.5

alph1 = 0.0
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


    # Comparing different MACH NUMBERS Numbers (We=0.1,0.2,0.3,0.4,0.5) at Re=0, We=0.1
    if j==1:
       Ma = 0.001
    elif j==2:
       Ma = 0.01
    elif j==3:
       Ma = 0.1


    # Comparing different WEISSENBERG Numbers (We=0.1,0.2,0.3,0.4,0.5) at Re=__
    """if j==1:
       We=0.1
    elif j==2:
       We=0.2
    elif j==3:
       We=0.3
    elif j==4:
       We=0.4
    elif j==5:
       We=0.5"""


    # Comparing different REYNOLDS NUMBERS Numbers (Re=0,5,10,25,50) at We=0.5
    """conv=1                                      # Non-inertial Flow Parameter (Re=0)
    We=0.5        # We=0.1-->0.5 using continuation in We 
    if j==1:
       Tf=4.3
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

    # Continuation in Reynolds/Weissenberg Number Number (Re-->10Re)
    Ret=Expression('Re*(1.0+0.5*(1.0+tanh(0.7*t-4.0))*19.0)', t=0.0, Re=Re, degree=2)
    Rey=Re
    Wet=Expression('(0.1+(We-0.1)*0.5*(1.0+tanh(500*(t-2.5))))', t=0.0, We=We, degree=2)


    print '############# TIME SCALE ############'
    print 'Timestep size (s):', dt
    print 'Finish Time (s):', Tf

    print '############# Scalings & Nondimensional Parameters ############'
    print 'Characteristic Length (m):', L
    print 'Characteristic Velocity (m/s):', U
    print 'Lid velocity:', (U*0.5*(1.0+tanh(e*t-3.0)),0)
    print 'Speed of sound (m/s):', U/Ma
    print 'Mach Number', Ma
    print 'Reynolds Number:', Rey
    print 'Non-inertial parameter:', conv
    print 'Weissenberg Number:', We
    print 'Viscosity Ratio:', betav
    print 'Diffusion Number:' ,Di
    print 'Viscous Heating Number:', Vh

    Np= len(p0.vector().array())
    Nv= len(w0.vector().array())   
    Ntau= len(tau0_vec.vector().array())
    dof= 3*Nv+2*Ntau+Np
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
    print 'DEVSS Momentum Term:', th

    #quit()

    # Initial Density Field
    rho_array = rho0.vector().array()
    for i in range(len(rho_array)):  
        rho_array[i] = 1.0
    rho0.vector()[:] = rho_array 

    # Initial Temperature Field
    T_array = T0.vector().array()
    for i in range(len(T_array)):  
        T_array[i] = T_0
    T0.vector()[:] = T_array

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

    #Folder To Save Plots for Paraview
    #fv=File("Velocity Results Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"theta"+str(theta)+"c0="+str(c0)+"/velocity "+str(t)+".pvd")
 
    #Lists for Energy Values
    x=list()
    ee=list()
    ek=list()
    z=list()

    conerr=list()
    deferr=list()
    tauerr=list()


    # Time-stepping
    t = dt
    iter = 0            # iteration counter
    maxiter = 10000000 
    while t < Tf + DOLFIN_EPS and iter < maxiter:
        iter += 1
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

        D0 = as_matrix([[D0_vec[0], D0_vec[1]],
                        [D0_vec[1], D0_vec[2]]])        #DEVSS STABILISATION
        DEVSSr_u12 = 2*(1-betav)*inner(D0,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS


        U = 0.5*(u + u0)              
        # VELOCITY HALF STEP
        lhsFu12 = Re*rho0*(2.0*(u - u0) / dt + conv*dot(u0, nabla_grad(u0)))
        Fu12 = dot(lhsFu12, v)*dx + \
               + inner(sigma(U, p0, tau0), Dincomp(v))*dx \
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

        """# STRESS Half Step
        F12 = dot(u12,grad(tau)) - dot(grad(u12),tau) - dot(tau,tgrad(u12) + div(u12)*tau) # Convection/Deformation Terms
        lhs_tau12 = (We/dt+1.0)*tau + We*F12                             # Left Hand Side
        rhs_tau12= (We/dt)*tau0 + 2.0*(1.0-betav)*Dcomp(u0)                     # Right Hand Side

        a3 = inner(lhs_tau12,Rt)*dx                                 # Weak Form
        L3 = inner(rhs_tau12,Rt)*dx

        a3 += SUPGl3             # SUPG Stabilisation LHS
        L3 += SUPGr3             # SUPG / SU Stabilisation RHS
        A3=assemble(a3)
        b3=assemble(L3)
        [bc.apply(A3, b3) for bc in bctau]
        solve(A3, tau12_vec.vector(), b3, "bicgstab", "default")
        end()"""

        #Temperature Half Step
        #A8 = assemble(a8)
        #b8 = assemble(L8)
        #[bc.apply(A8, b8) for bc in bcT]
        #solve(A8, T12.vector(), b8, "bicgstab", "default")
        #end()
        
        #Predicted U* Equation
        lhsFus = Re*rho0*((u - u0)/dt + conv*dot(u12, nabla_grad(u12)))
        Fus = dot(lhsFus, v)*dx + \
              + inner(sigma(U, p0, tau0), Dincomp(v))*dx \
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

        # Stress Full Step
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
        #err = project(h*kapp,Qt)
        x.append(t)
        #ee.append(norm(err.vector(),'linf'))
        ek.append(norm(tau1_vec.vector(),'linf'))
        

        # Save Plot to Paraview Folder 
        #for i in range(5000):
            #if iter== (0.02/dt)*i:
               #fv << u1
        #ft << T1

        # Break Loop if code is diverging

        if max(norm(tau1_vec.vector(), 'linf'),norm(w1.vector(), 'linf')) > 10E6 or np.isnan(sum(tau1_vec.vector().array())):
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
        #if t>0.1:
            #plot(Dcomp(u1)[0,0], title="Deformation Grad xx", rescale=True, interactive=False)
            #plot(D1_vec[0], title="Deformation Grad xx", rescale=True, interactive=False)
            #plot(l2_kapp, title="tau_xy Stress", rescale=True, interactive=False)
            #plot(tau1[0,0], title="tau_xx Stress", rescale=True, interactive=False)
            #plot(p1, title="Pressure", rescale=True)
            #plot(rho1, title="Density", rescale=True)
            #plot(u1, title="Velocity", rescale=True, mode = "auto")
            #plot(T1, title="Temperature", rescale=True)
           

        # Move to next time step (Continuation in Reynolds Number)
        t += dt



    # PLOTS
    # Plot Error Control Data
    """plt.figure(0)
    plt.plot(x, ee, 'r-', label=r'$\kappa$')
    plt.plot(x, ek, 'b-', label=r'$||\tau||$')
    plt.legend(loc='best')
    plt.xlabel('time(s)')
    plt.ylabel('$||\cdot||_{\infty}$')
    plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/Error_controlRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
    plt.clf()
    plt.close()"""

    # Data on Kinetic/Elastic Energies
    with open("Energy.txt", "a") as text_file:
         text_file.write("Re="+str(Rey*conv)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+", E_k="+str(E_k)+", E_e="+str(E_e)+'\n')


    if j==3:
        peakEk1 = max(ek1)
        peakEk2 = max(ek2)
        peakEk3 = max(ek3)
        with open("Energy.txt", "a") as text_file:
             text_file.write("Re="+str(Rey*conv)+", We="+str(We)+", Ma="+str(Ma)+"Peak Kinetic Energy"+str(peakEk3)+"Incomp Kinetic En"+str(peakEk1)+'\n')

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
    """if j==3:
        # First Normal Stress
        x_axis1 = list(chunks(x_axis, mm))
        y_axis1 = list(chunks(y_axis, mm))
        u_x1 = list(chunks(u_xg, mm))
        u_y1 = list(chunks(u_yg, mm))
        tau_xx1 = list(chunks(tau_xxg, mm))
        tau_yy1 = list(chunks(tau_yyg, mm))
        plt.figure(0)
        plt.plot(x_axis1[0], u_y1[0], 'r-', label=r'$We=0.1$')
        plt.plot(x_axis1[1], u_y1[1], 'b-', label=r'$We=0.2$')
        plt.plot(x_axis1[2], u_y1[2], 'c-', label=r'$We=0.3$')
        plt.plot(x_axis1[3], u_y1[3], 'm-', label=r'$We=0.4$')
        plt.plot(x_axis1[4], u_y1[4], 'g-', label=r'$We=0.5$')
        plt.legend(loc='best')
        plt.xlabel('x')
        plt.ylabel('$u_y(x,0.75)$')
        plt.savefig("Compressible Viscoelastic Flow Results/Cross Section/u_yRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
        plt.clf()
        plt.figure(1)
        plt.plot(u_x1[0], y_axis1[0], 'r-', label=r'$We=0.1$')
        plt.plot(u_x1[1], y_axis1[1], 'b-', label=r'$We=0.2$')
        plt.plot(u_x1[2], y_axis1[2], 'c-', label=r'$We=0.3$')
        plt.plot(u_x1[3], y_axis1[3], 'm-', label=r'$We=0.4$')
        plt.plot(u_x1[4], y_axis1[4], 'g-', label=r'$We=0.5$')
        plt.legend(loc='best')
        plt.xlabel('$u_x(0.5,y)$')
        plt.ylabel('y')
        plt.savefig("Compressible Viscoelastic Flow Results/Cross Section/u_xRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
        plt.clf()
        plt.figure(2)
        plt.plot(x_axis1[0], tau_xx1[0], 'r-', label=r'$We=0.1$')
        plt.plot(x_axis1[1], tau_xx1[1], 'b-', label=r'$We=0.2$')
        plt.plot(x_axis1[2], tau_xx1[2], 'c-', label=r'$We=0.3$')
        plt.plot(x_axis1[3], tau_xx1[3], 'm-', label=r'$We=0.4$')
        plt.plot(x_axis1[4], tau_xx1[4], 'g-', label=r'$We=0.5$')
        plt.legend(loc='best')
        plt.xlabel('x')
        plt.ylabel('$\tau_{xx}(x,1.0)$')
        plt.savefig("Compressible Viscoelastic Flow Results/Cross Section/tau_xxRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
        plt.clf()
        plt.figure(3)
        plt.plot(x_axis1[0], tau_yy1[0], 'r-', label=r'$We=0.1$')
        plt.plot(x_axis1[1], tau_yy1[1], 'b-', label=r'$We=0.2$')
        plt.plot(x_axis1[2], tau_yy1[2], 'c-', label=r'$We=0.3$')
        plt.plot(x_axis1[3], tau_yy1[3], 'm-', label=r'$We=0.4$')
        plt.plot(x_axis1[4], tau_yy1[4], 'g-', label=r'$We=0.5$')
        plt.legend(loc='best')
        plt.xlabel('x')
        plt.ylabel('$\tau_{yy}(x,1.0)$')
        plt.savefig("Compressible Viscoelastic Flow Results/Cross Section/tau_yyRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
        plt.clf()
        plt.close()"""



    if j==3:
        # First Normal Stress
        x_axis1 = list(chunks(x_axis, mm))
        y_axis1 = list(chunks(y_axis, mm))
        u_x1 = list(chunks(u_xg, mm))
        u_y1 = list(chunks(u_yg, mm))
        tau_xx1 = list(chunks(tau_xxg, mm))
        tau_yy1 = list(chunks(tau_yyg, mm))
        plt.figure(0)
        plt.plot(x_axis1[0], u_y1[0], 'r-', label=r'$Ma=0.001$')
        plt.plot(x_axis1[1], u_y1[1], 'b-', label=r'$Ma=0.01$')
        plt.plot(x_axis1[2], u_y1[2], 'c-', label=r'$Ma=0.1$')
        plt.legend(loc='best')
        plt.xlabel('x')
        plt.ylabel('$u_y(x,0.75)$')
        plt.savefig("Compressible Viscoelastic Flow Results/Cross Section/u_yRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
        plt.clf()
        plt.figure(1)
        plt.plot(u_x1[0], y_axis1[0], 'r-', label=r'$Ma=0.001$')
        plt.plot(u_x1[1], y_axis1[1], 'b-', label=r'$Ma=0.01$')
        plt.plot(u_x1[2], y_axis1[2], 'c-', label=r'$Ma=0.1$')
        plt.legend(loc='best')
        plt.xlabel('$u_x(0.5,y)$')
        plt.ylabel('y')
        plt.savefig("Compressible Viscoelastic Flow Results/Cross Section/u_xRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
        plt.clf()
        plt.figure(2)
        plt.plot(x_axis1[0], tau_xx1[0], 'r-', label=r'$Ma=0.001$')
        plt.plot(x_axis1[1], tau_xx1[1], 'b-', label=r'$Ma=0.01$')
        plt.plot(x_axis1[2], tau_xx1[2], 'c-', label=r'$Ma=0.1$')
        plt.legend(loc='best')
        plt.xlabel('x')
        plt.ylabel('$\tau_{xx}(x,1.0)$')
        plt.savefig("Compressible Viscoelastic Flow Results/Cross Section/tau_xxRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
        plt.clf()
        plt.figure(3)
        plt.plot(x_axis1[0], tau_yy1[0], 'r-', label=r'$Ma=0.001$')
        plt.plot(x_axis1[1], tau_yy1[1], 'b-', label=r'$Ma=0.01$')
        plt.plot(x_axis1[2], tau_yy1[2], 'c-', label=r'$Ma=0.1$')
        plt.legend(loc='best')
        plt.xlabel('x')
        plt.ylabel('$\tau_{yy}(x,1.0)$')
        plt.savefig("Compressible Viscoelastic Flow Results/Cross Section/tau_yyRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
        plt.clf()
        plt.close()

    # Plot Convergence Data 
    """if max(norm(tau1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 or jj>0:
        fig1=plt.figure()
        plt.plot(x, z, 'r-', label='Stress Timestep Error')
        plt.xlabel('time(s)')
        plt.ylabel('||S1-S0||/||S1||')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/StressCovergenceRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"al="+str(al)+"dt="+str(dt)+".png")
        plt.clf()"""


        #Plot Kinetic and elasic Energies for different REYNOLDS numbers at constant Weissenberg Number    
    """if max(norm(tau1_vec.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==1 or j==5:
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
        plt.savefig("Compressible Viscoelastic Flow Results/Energy/We0p5KineticEnergyRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
        plt.clf()
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
        plt.savefig("Compressible Viscoelastic Flow Results/Energy/We0p5ElasticEnergyRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
        plt.clf()"""

        #Plot Kinetic and elasic Energies for different Speed of sound numbers at constant Weissenberg & Reynolds Numbers    
    if max(norm(tau1_vec.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==3:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'$Ma=0.001$')
        plt.plot(x2, ek2, 'b--', label=r'$Ma=0.01$')
        plt.plot(x3, ek3, 'c-', label=r'$Ma=0.1$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$E_k$')
        plt.savefig("Compressible Viscoelastic Flow Results/Energy/MaKineticEnergyRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
        plt.clf()
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'$Ma=0.001$')
        plt.plot(x2, ee2, 'b--', label=r'$Ma=0.01$')
        plt.plot(x3, ee3, 'c-', label=r'$Ma=0.1$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$E_e$')
        plt.savefig("Compressible Viscoelastic Flow Results/Energy/MaElasticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
        plt.clf()
        plt.close()

        #Plot Kinetic and elasic Energies for different Weissenberg numbers at Re=0 (METHOD 1)  
    """if max(norm(tau1_vec.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6:
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
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/KineticEnergyTf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
        # Elastic Energy
        plt.figure(1)
        plt.plot(x, ee, col, label=r'$We=%s'%We)
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/ElasticEnergyTf="+str(Tf)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
        plt.clf()"""

        #Plot Kinetic and elasic Energies for different Weissenberg numbers at Re FIXED 
    """if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==5 or j==3 or j==1:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'$We=0.1$')
        plt.plot(x2, ek2, 'b-', label=r'$We=0.2$')
        plt.plot(x3, ek3, 'c-', label=r'$We=0.3$')
        plt.plot(x4, ek4, 'm-', label=r'$We=0.4$')
        plt.plot(x5, ek5, 'g-', label=r'$We=0.5$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$E_{kinetic}$')
        plt.savefig("Compressible Viscoelastic Flow Results/Energy/KineticEnergyRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"mesh="+str(mm)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
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
        plt.ylabel('$E_{elastic}$')
        plt.savefig("Compressible Viscoelastic Flow Results/Energy/ElasticEnergyRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"mesh="+str(mm)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
        plt.clf()"""



    
    if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==3 or j==1:

        # Plot First Normal Stress Difference
        tau_xx=project(tau1[0,0],Q)
        mplot(tau_xx)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_xxRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf() 
        tau_xy=project(tau1[1,0],Q)
        mplot(tau_xy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_xyRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf() 
        tau_yy=project(tau1[1,1],Q)
        mplot(tau_yy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_yyRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf() 
        divu = project(div(u1),Q)
        mplot(divu)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/div_uRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()

    if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==5 or j==1:
 
       # Plot Velocity Components
        ux=project(u1[0],Q)
        mplot(ux)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/u_xRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()
        uy=project(u1[1],Q)
        mplot(uy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/u_yRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()

    if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==3 or j==1:


        # Matlab Plot of the Solution at t=Tf
        rho1=rho_0*rho1
        rho1=project(rho1,Q)
        #p1=mu_0*(L/U)*p1  #Dimensionalised Pressure
        #p1=project(p1,Q)
        mplot(rho1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/DensityRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf() 
        mplot(p1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/PressureRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()
        mplot(T1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/TemperatureRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()



    if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==3 or j==1:
        #Plot Contours USING MATPLOTLIB
        # Scalar Function code


        x = Expression('x[0]', degree=2)     #GET X-COORDINATES LIST
        y = Expression('x[1]', degree=2)     #GET Y-COORDINATES LIST
        pvals = p1.vector().array()          # GET SOLUTION p= p(x,y) list
        Tvals = T1.vector().array()          # GET SOLUTION T= T(x,y) list
        rhovals = rho1.vector().array()      # GET SOLUTION p= p(x,y) list
        tauxx = project(tau1_vec[0], Q)
        tauxxvals = tauxx.vector().array()
        xyvals = mesh.coordinates()     # CLEAN THIS UP!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        xvalsq = interpolate(x, Q)#xyvals[:,0]
        yvalsq= interpolate(y, Q)#xyvals[:,1]
        xvalsw = interpolate(x, Qt)#xyvals[:,0]
        yvalsw= interpolate(y, Qt)#xyvals[:,1]

        xvals = xvalsq.vector().array()
        yvals = yvalsq.vector().array()


        xx = np.linspace(x_0,x_1)
        yy = np.linspace(y_0,y_1)
        XX, YY = np.meshgrid(xx,yy)   # (x,y) coordinate data formatted so that it can be used by plt.contour()
        pp = mlab.griddata(xvals, yvals, pvals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 
        dd = mlab.griddata(xvals, yvals, rhovals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 

        plt.contour(XX, YY, dd, 25)
        plt.title('Density Contours')   # DENSITY CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/DensityContoursRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()

        plt.contour(XX, YY, pp, 25)
        plt.title('Pressure Contours')   # PRESSURE CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/PressureContoursRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()

        """TT = mlab.griddata(xvals, yvals, Tvals, xx, yy, interp='nn') 
        plt.contour(XX, YY, TT, 20) 
        plt.title('Temperature Contours')   # TEMPERATURE CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/TemperatureContoursRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()"""


        normstress = mlab.griddata(xvals, yvals, tauxxvals, xx, yy, interp='nn')

        """plt.contour(XX, YY, normstress, 20) 
        plt.title('Stress Contours')   # NORMAL STRESS CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/StressContoursRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()"""


        #Plot Velocity Streamlines USING MATPLOTLIB
        u1_q = project(u1[0],Q)
        uvals = u1_q.vector().array()
        v1_q = project(u1[1],Q)
        vvals = v1_q.vector().array()

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
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/VelocityContoursRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")   
        plt.clf()                                             # display the plot


    plt.close()


    if dt < tol:
       j=loopend+1
       break

    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and abs(E_k) < 10:
        Tf=T_f   

    if j==3:
        jjj+=1
        if jjj==1:
            We = 0.01
        if jjj==2:
            We = 0.1        
        #Re = Re/2
        j=0
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


    if jjj==4:
        quit()

    if jjj==3:
        quit()
        jjj+=1
        Re = 1.0
        We = 0.5
        Ma = 0.01
        Tf = 20.0


    # Reset Functions
    rho0 = Function(Q)
    rho1 = Function(Q)
    p0 = Function(Q)       # Pressure Field t=t^n
    p1 = Function(Q)       # Pressure Field t=t^n+1
    T0 = Function(Qt)       # Temperature Field t=t^n
    T1 = Function(Qt)       # Temperature Field t=t^n+1
    tau0_vec = Function(Zc)     # Stress Field (Vector) t=t^n
    tau12_vec = Function(Zc)    # Stress Field (Vector) t=t^n+1/2
    tau1_vec = Function(Zc)     # Stress Field (Vector) t=t^n+1
    w0 = Function(W)
    w12 = Function(W)
    ws = Function(W)
    w1 = Function(W)
    (u0, D0_vec) = w0.split()
    (u12, D12_vec) = w0.split()
    (us, Ds_vec) = w0.split()
    (u1, D1_vec) = w0.split()
    D_proj_vec = Function(Ze)
    D_proj = as_matrix([[D_proj_vec[0], D_proj_vec[1]],
                        [D_proj_vec[1], D_proj_vec[2]]])
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






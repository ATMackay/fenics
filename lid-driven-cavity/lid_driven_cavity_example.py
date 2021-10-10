'''
Solve the driven cavity problem with Chorin method. We consider [0, 1]^2 domain
with no slip boundary conditions everywhere apart from y = 1 where the velocity
is (1, 0). Initial conditions for pressure and velocity are 0. For other pareme-
ters such as kinematic viscosity, time step, simulation time see below. For
testing, we are interested in the stream function at each time instance.
'''

from dolfin import *
import numpy as np

set_log_level(WARNING)

def stream_function(u):
  '''Compute stream function of given 2-d velocity vector.'''
  V = u.function_space().sub(0).collapse()

  if V.mesh().topology().dim() != 2:
    raise ValueError("Only stream function in 2D can be computed.")

  psi = TrialFunction(V)
  phi = TestFunction(V)

  a = inner(grad(psi), grad(phi))*dx
  L = inner(u[1].dx(0) - u[0].dx(1), phi)*dx
  bc = DirichletBC(V, Constant(0.), DomainBoundary())

  A, b = assemble_system(a, L, bc)
  psi = Function(V)
  solve(A, psi.vector(), b)

  return psi

#---------------------------------------------------------------------

def chorin_driven_cavity(n_cells):
  '''Parameter n_cells determins mesh size.'''

  # Test for PETSc
  if not has_linear_algebra_backend("PETSc"):
      info("DOLFIN has not been configured with PETSc. Exiting.")
      exit()

  parameters["linear_algebra_backend"] = "PETSc"

  mesh = UnitSquareMesh(n_cells, n_cells)
  h = mesh.hmin()
 
  #! problem specific
  f = Constant((0., 0.))         # force
  nu = Constant(1./1000)         # kinematic viscosity
  dt = 0.2*h/1.                  # time step CFL with 1 = max. velocity
  k = Constant(dt)               # time step 
  T = 10.0                        # total simulation time
  u0 = Constant((0., 0.))        # initial velocity
  p0 = Constant(0.)              # initial pressure

  no_slip_value = Constant((0., 0.)) 
  
  def no_slip_domain(x, on_boundary):
    return on_boundary and (near(x[1], 0) or near(x[0]*(1 - x[0]), 0))

  driven_value = Constant((1., 0.))

  def driven_domain(x, on_boundary):
    return on_boundary and near(x[1], 1)

  #! solver specific
  V = VectorFunctionSpace(mesh, 'CG', 2)
  Q = FunctionSpace(mesh, 'CG', 1)

  u = TrialFunction(V)
  v = TestFunction(V)
  p = TrialFunction(Q)
  q = TestFunction(Q)

  u0 = interpolate(u0, V)  
  p0 = interpolate(p0, Q)
  us = Function(V)
  u1 = Function(V)
  p1 = Function(Q)

  # tentative velocity
  F0 = (1./k)*inner(u - u0, v)*dx + inner(dot(grad(u0), u0), v)*dx\
       + nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
  a0, L0 = system(F0)

  # projection
  F1 = inner(grad(p), grad(q))*dx + (1./k)*q*div(us)*dx
  a1, L1 = system(F1)

  # finalize
  F2 = (1./k)*inner(u - us, v)*dx + inner(grad(p1), v)*dx 
  a2, L2 = system(F2)

  # boundary conditions
  bc_no_slip = DirichletBC(V, no_slip_value, no_slip_domain)
  bc_driven = DirichletBC(V, driven_value, driven_domain)
  bcs = [bc_no_slip, bc_driven]

  A0 = assemble(a0)
  A1 = assemble(a1)
  A2 = assemble(a2)

  # Create Krylov solver
  solver = PETScKrylovSolver("cg")
  solver.set_operator(A1)
 
  solver02 = KrylovSolver('gmres', 'ilu')

  solver1 = KrylovSolver('cg', 'petsc_amg')
  null_vec = Vector(p0.vector())
  Q.dofmap().set(null_vec, 1.0)
  null_vec *= 1.0/null_vec.norm("l2")
  null_space = VectorSpaceBasis([null_vec])
  as_backend_type(A1).set_nullspace(null_space)
  #solver.set_nullspace(null_space)

  t = 0
  udiff = 1.0
  while t < T and udiff > 10E-4:
    t += dt
    
    b = assemble(L0)
    [bc.apply(A0, b) for bc in bcs]
    solver02.solve(A0, us.vector(), b)

    b = assemble(L1)
    null_space.orthogonalize(b);
    solver.solve(A1, p1.vector(), b)

    b = assemble(L2)
    [bc.apply(A2, b) for bc in bcs]
    solver02.solve(A2, u1.vector(), b)
     
    udiff = np.abs(u1.vector().array()-u0.vector().array()).max()
    print "t = %s,  Error = %.3g" %(t, udiff)
    u0.assign(u1)
    plot(p1)


  phi0 = stream_function(u0)
  return u0, phi0

if __name__ == '__main__':
  u, phi = chorin_driven_cavity(64)

  print phi.vector().min()
  
  plot(u)
  plot(phi)
interactive()

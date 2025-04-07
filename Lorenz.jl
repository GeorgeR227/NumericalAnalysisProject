using OrdinaryDiffEq

function lorenz!(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end
u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 100.0)
# The last arg is for the parameters, these can be changed
# Docs for solve are at: https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/#CommonSolve.solve-Tuple%7BSciMLBase.AbstractDEProblem,%20Vararg%7BAny%7D%7D
prob = ODEProblem(lorenz!, u0, tspan, (10.0,))
sol = solve(prob, Euler(), dt=0.01)
# sol = solve(prob, RK4())

# Get coordinates over time
xs = map(x->x[1], sol.u)
ys = map(y->y[2], sol.u)
zs = map(z->z[3], sol.u)

plot(xs, ys, zs)

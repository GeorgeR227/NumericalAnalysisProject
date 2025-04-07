# lorenz_rk_plots.jl
#Dylan Rautenstrauch 


using Plots  # for plots & 2D plots
plotlyjs()   # for 3D plots
using StaticArrays  # for SVector (creates a stack-allocated immutable vector. Basically, a very fast vector for small fixed sized vectors)

# Lorenz system definition
function lorenz(u, t; σ=10.0, ρ=28.0, β=8/3)
    x, y, z = u
    dx = σ*(y - x)
    dy = x*(ρ - z) - y
    dz = x*y - β*z
    return SVector(dx, dy, dz)
end

# Just a generic RK integrator builder
function make_rk_method(stages, a, b, c)
    function integrator(f, u0, tspan, dt)
        t0, tf = tspan
        ts = collect(t0:dt:tf)
        us = Vector{typeof(u0)}(undef, length(ts))
        us[1] = u0
        for i in 1:length(ts)-1
            t = ts[i]; u = us[i]
            ks = Vector{typeof(u)}(undef, stages)
            for j in 1:stages
                arg = u
                for m in 1:j-1
                    arg += dt * a[j,m] * ks[m]
                end
                ks[j] = f(arg, t + c[j]*dt)
            end
            increment = zero(u)
            for j in 1:stages
                increment += dt * b[j] * ks[j]
            end
            us[i+1] = u + increment
        end
        return ts, us
    end
    return integrator
end

# Coefficient matrices for RK2, RK3, RK4, RK5 (Fehlberg)
# (constant matrices come from "rungekutta.pdf" so I'm not entirely sure where they come from but they can be changed if needed)
# RK2 (midpoint 2nd-order)
a2 = [0.0 0.0;
      1/2 0.0]
b2 = [0.0, 1.0]
c2 = [0.0, 1/2]

# RK3 (Kutta’s 3rd-order)
a3 = [0.0 0.0 0.0;
      1/2 0.0 0.0;
     -1   2.0  0.0]
b3 = [1/6, 2/3, 1/6]
c3 = [0.0, 1/2, 1.0]

# RK4 ("classic" 4th-order)
a4 = [0.0 0.0 0.0 0.0;
      1/2 0.0 0.0 0.0;
      0.0 1/2 0.0 0.0;
      0.0 0.0 1.0 0.0]
b4 = [1/6, 1/3, 1/3, 1/6]
c4 = [0.0, 1/2, 1/2, 1.0]

# RK5 (Fehlberg 4(5) coefficients - 5th order) (Uses 6 "stages")
a5 = [
    0        0        0        0       0       0
    1/4      0        0        0       0       0
    3/32     9/32     0        0       0       0
    1932/2197 -7200/2197 7296/2197 0    0       0
    439/216  -8       3680/513 -845/4104 0    0
    -8/27     2      -3544/2565 1859/4104 -11/40 0
]
b5 = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]
c5 = [0, 1/4, 3/8, 12/13, 1.0, 1/2]

# Build integrators 
rk2 = make_rk_method(2, a2, b2, c2)
rk3 = make_rk_method(3, a3, b3, c3)
rk4 = make_rk_method(4, a4, b4, c4)
rk5 = make_rk_method(6, a5, b5, c5)  # 6 stages for Fehlberg


function main()
    u0 = @SVector [1.0, 1.0, 1.0] # @SVector comes from StaticArrays
    tspan = (0.0, 40.0) # Change the span to whatever desired range, I set it at 40 for quicker runtime.
    dt = 0.01           # Same with time step, I set it to 0.01 for runtime and the plots look good enough @0.01.

    methods = Dict(
      "RK2" => rk2,
      "RK3" => rk3,
      "RK4" => rk4,
      "RK5" => rk5
    )

    for (name, integrator) in methods
        ts, us = integrator(lorenz, u0, tspan, dt)
        xs = getindex.(us, 1)
        ys = getindex.(us, 2)
        zs = getindex.(us, 3)

        # 1D: x(t), y(t), z(t) (plotted together)
        plt1 = plot(ts, xs, label="x(t)", title="$name: x,y,z vs t")
        plot!(plt1, ts, ys, label="y(t)")
        plot!(plt1, ts, zs, label="z(t)")
        savefig(plt1, "$name+_xyz_vs_t.png")    # Keep in mind the plots are saved to the current working directory

        # 2D projections (plotted separately)
        plt2 = plot(xs, ys, xlabel="x", ylabel="y", title="$name: x vs y")
        savefig(plt2, "$name+_x_vs_y.png")
        plt3 = plot(xs, zs, xlabel="x", ylabel="z", title="$name: x vs z")
        savefig(plt3, "$name+_x_vs_z.png")
        plt4 = plot(ys, zs, xlabel="y", ylabel="z", title="$name: y vs z")
        savefig(plt4, "$name+_y_vs_z.png")

        # 3D plot
        plt5 = plot(xs, ys, zs,
                    linetype = :path,
                    xlabel="x", ylabel="y", zlabel="z",
                    title="$name: 3D Trajectory")
        savefig(plt5, "$name+_3D.png")
    end

    # just lets me know the program is done processing
    println("Plots saved to current directory.")
end

main()

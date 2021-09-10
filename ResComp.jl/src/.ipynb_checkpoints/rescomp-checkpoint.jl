using DifferentialEquations
using LinearAlgebra

mutable struct ResComp{T <: Real, N <: Int}
    win::Array{T, N}
    adj::AbstractArray
    wout::Array{T, N}
    γ::T
    σ::T
end

function ResComp(; n=500, m=3, p=.001, γ=10.0, σ=0.014, ρ=0.9)
    win = rand(n, m) .*2 .- 1
    adj = Float64.(rand(n, n) .< p)
    rad = maximum(abs.(eigvals(adj)))
    
    wout = zeros(m, n)
ResComp()

function (rc::ResComp)(x0::Array{<:Real, 1}, t::Array{<:Real, 1})
""" Make a predicition for timesteps t with initial state x0
"""

end

function fit!(rc::ResComp, t::Array{<:Real, 1}, u, r=1e-6)
    internal_states = drive(rc, t, u)
    rc.wout = ridge(internal_states, u, r=r) # Need to get transpose right
end


function drive(rc::ResComp, t::Array{<:Real, 1}, u)
    x0 = u(t[1])
    f = untrained_ode(rc, u)
    driven_states = integrate(F, x0, t)
end

function untrained_ode(rc::ResComp, u)
    """ Make ode governing the node states of the untrained reservoir
    """
    γ, σ, adj, win = rc.γ, rc.σ, rc.adj, rc.win
    f(r::Array{<:Real, 1}, t<:Real) = γ*(-r + tanh(adj * r + σ * win * u(t)))
    return f
end

function trained_ode(rc::ResComp)
    """ Make ode governing the node states of the trained reservoir
    """
    γ, σ, adj, win, wout = rc.γ, rc.σ, rc.adj, rc.win, rc.wout
    f(dr, r::Array{<:Real, 1}, t<:Real) = γ*(-r + tanh(adj * r + σ * win * wout * r)
    return f
end

function parameterized_lorenz(du,u,p,t)
  x,y,z = u
  σ,ρ,β = p
  du[1] = dx = σ*(y-x)
  du[2] = dy = x*(ρ-z) - y
  du[3] = dz = x*y - β*z
end

u0 = [1.0,0.0,0.0]
tspan = (0.0,10.0)
p = [10.0,28.0,8/3]
prob = ODEProblem(parameterized_lorenz,u0,tspan,p)
u = solve(prob, RK4, dt=0.001)

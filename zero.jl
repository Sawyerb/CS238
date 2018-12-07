using POMDPs
using POMDPToolbox
using QMDP
using POMDPModelTools
using ParticleFilters, Distributions
using Random

mutable struct ZeroPolicy{P<:Union{POMDP,MDP}, U<:Updater} <: Policy
    problem::P
    updater::U # set this to use a custom updater, by default it will be a void updater
end

function POMDPs.action(policy::ZeroPolicy, s)
    return 0
end

function POMDPs.action(policy::ZeroPolicy, b::Nothing)
    return 0
end

POMDPs.updater(policy::ZeroPolicy) = policy.updater

mutable struct ZeroSolver <: Solver
    rng::AbstractRNG
end
ZeroSolver(;rng=Base.GLOBAL_RNG) = ZeroSolver(rng)
POMDPs.solve(solver::ZeroSolver, problem::Union{POMDP,MDP}) = ZeroPolicy(solver.rng, problem, VoidUpdater())
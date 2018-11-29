using POMDPs
using QMDP
using POMDPModelTools
using ParticleFilters, Distributions
using Random

# state, action, observation
mutable struct DonationsPOMDP <: POMDP{Tuple{Int64, Int64, Int64}, Int64, Tuple{Int64, Int64, Int64}}
    win_r::Int64 # winning reward
    total_steps::Int64
    initial_supp::Int64
    initial_budg::Int64
    opp_money::Int64 # opponent's money
    pol_money::Int64 # supported politician's money
end

DonationsPOMDP() = DonationsPOMDP(100, 10, 45, 100, 100, 100)

POMDPs.updater(problem::DonationsPOMDP) = ParticleFilters(problem)

POMDPs.actions(::DonationsPOMDP) = Tuple(0:100)

POMDPs.actions(::DonationsPOMDP, s::Tuple{Int64, Int64, Int64}) = Tuple(0:s[3])

POMDPs.actionindex(::DonationsPOMDP, a::Int64) = a + 1

POMDPs.n_actions(::DonationsPOMDP) = 101

function POMDPs.states(pomdp::DonationsPOMDP)
    ret = []
    for num in 1:pomdp.total_steps
        for vote_per in 0:100
            for money in 0:100
                push!(ret, (num, vote_per, money))
            end
        end
    end
    return ret
end

function POMDPs.stateindex(pomdp::DonationsPOMDP, s::Tuple{Int64, Int64, Int64}) 
    a = zeros(pomdp.total_steps, 101, 101)
    return LinearIndices(a)[s[1], s[2], s[3]]
end 

POMDPs.n_states(pomdp::DonationsPOMDP) = pomdp.total_steps*101*101

POMDPs.observations(::DonationsPOMDP) = Tuple(0:100)

POMDPs.obsindex(::DonationsPOMDP, o::Int64) = o + 1

POMDPs.n_observations(::DonationsPOMDP) = 101

POMDPs.discount(p::DonationsPOMDP) = 1

function POMDPs.initialstate(pomdp::DonationsPOMDP, rng::AbstractRNG)
    return (pomdp.total_steps, pomdp.initial_supp, pomdp.initial_budg)
end

function POMDPs.transition(pomdp::DonationsPOMDP, s::Tuple{Int64, Int64, Int64}, a::Int64)
    a = min(a, s[3]) # cannot give more than you have
    agent_money = s[3] - a
    pomdp.pol_money = pomdp.pol_money + a
    # Q how to return a discrete distribution? e.g. 1/3 prob you get a lower amount, 1/3 you get higher, 1/3 you stay the same?
    money_percent = Int(100*pomdp.pol_money/(pomdp.pol_money + pomdp.opp_money))
    support = money_percent # TODO we want noise here
    num_steps = s[1] - 1
    return (num_steps, support, agent_money)
end

function POMDPs.observation(pomdp::DonationsPOMDP, a::Int64, sp::Tuple{Int64, Int64, Int64})
    # Q how to return a discrete distribution? (same problem as above)
    # TODO we want sp[2] to have noise
    return (sp[1], sp[2], sp[3]) 
end

function POMDPs.reward(pomdp::DonationsPOMDP, s::Tuple{Int64, Int64, Int64}, a::Int64, sp::Tuple{Int64, Int64, Int64})
    # reward * (1-num_steps_left)
    r = 0.0
    if pomdp.support > 50
        r += pomdp.win_r # award for winning
    elseif pomdp.support < 50
        r -= pomdp.win_r
    end
    r -= sp[3] # penalty for what you spent
    return r
end


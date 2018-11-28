using POMDPs
using QMDP
using POMDPModelTools
using ParticleFilters, Distributions

# state, action, observation
mutable struct DonationsPOMDP <: POMDP{Tuple{Int64, Int64, Int64}, Int64, Tuple{Int64, Int64, Int64}}
    #o_mean::Float64 # mean of normal dist vote -> poll
    #o_std::Float64 # std of normal dist vote -> poll
    #t_mean::Float64 # money -> vote
    #t_std::Float64 # money -> vote
    win_r::Float64 # winning reward
    total_steps::Int64
    initial_supp::Int64
    initial_budg::Int64
    opp_money::Int64 # opponent's money
    pol_money::Int64 # supported politician's money
end

# Q why is nothing showing up when we want requirements_info? 
DonationsPOMDP() = DonationsPOMDP(0.5, 0.2, 0.5, 0.2, 100, 10, 45, 100, 100, 100)

updater(problem::DonationsPOMDP) = ParticleFilters(problem)

actions(::DonationsPOMDP) = Tuple(0:100)

actionindex(::DonationsPOMDP, a::Int64) = a + 1

n_actions(::DonationsPOMDP) = 101

function states(::DonationsPOMDP)
    ret = []
    for num in 1:total_steps
        for vote_per in 0:100
            for money in 0:100
                push!(ret, (num, vote_per, money))
            end
        end
    end
    return ret
end

function stateindex(::DonationsPOMDP, s::Tuple{Int64, Int64, Int64}) 
    a = zeros(total_steps, 101, 101)
    return LinearIndices(a)[s[1], s[2], s[3]]
end 

n_states(::DonationsPOMDP) = total_steps*101*101

observations(::DonationsPOMDP) = Tuple(0:100)

obsindex(::DonationsPOMDP, o::Int64) = o + 1

n_observations(::DonationsPOMDP) = 101

discount(p::DonationsPOMDP) = 1

# Q why don't we need to specify an initial state for QMDPSolver()?  
# Q is it ok to fix our initial state so we can run analyses of how it impacts things? 
function initialstate(pomdp::DonationsPOMDP, rng::AbstractRNG)
    return (pomdp.total_steps, pomdp.initial_supp, pomdp.initial_budg)
end

function transition(pomdp::DonationsPOMDP, s::Tuple{Int64, Int64, Int64}, a::Int64)
    a = min(a, s[3]) # cannot give more than you have
    agent_money = s[3] - a
    pomdp.pol_money = pomdp.pol_money + a
    # Q how to return a discrete distribution? e.g. 1/3 prob you get a lower amount, 1/3 you get higher, 1/3 you stay the same?
    money_percent = Int(100*pomdp.pol_money/(pomdp.pol_money + pomdp.opp_money))
    support = money_percent # TODO we want noise here
    num_steps = s[1] - 1
    return (num_steps, support, agent_money)
end

# Q Why don't we require an observation function for QMDPSolver() when it seems important for our problem? 
function observation(pomdp::DonationsPOMDP, a::Int64, sp::Tuple{Int64, Int64, Int64})
    # TODO we want sp[2] to have noise
    return (sp[1], sp[2], sp[3])
end

function reward(pomdp::DonationsPOMDP, s::Bool, a::Bool)
    # Q Do we only calculate the win reward if we are winning at the end or along the way? 
    r = 0.0
    if pomdp.support > 50: 
        r += pomdp.win_r # award for winning
    r -= s[3] # penalty for what you spent
    return r
end

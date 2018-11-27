using POMDPs
using POMDPModelTools
using ParticleFilters, Distribution

# state, action, observation
mutable struct DonationsPOMDP <: POMDP{Tuple{Int64, Int64, Int64, Int64}, Int64, Int64}
    o_mean::Float64 # mean of normal dist vote -> poll
    o_std::Float64 # std of normal dist vote -> poll
    t_mean::Float64 # money -> vote
    t_std::Float64 # money -> vote
    win_r::Float64 # winning reward

    num_steps::Int64
    support::Int64 # % support for your politician
    opp_money::Int64 # opponent's money
    pol_money::Int64 # supported politician's money
    agent_money::Int64 # agent's money to spend
end

updater(problem::DonationsPOMDP) = ParticleFilters(problem)

actions(::DonationsPOMDP) = 0:100
actionindex(::DonationsPOMDP, a::Int64) = a + 1
n_actions(::DonationsPOMDP) = 101
function states(::DonationsPOMDP)
    ret = []
    i = 1
    for num in 1:num_steps
        for vote_per in 0:100
            for money in 0:100
                push!(ret, (num, vote_per, money, i))
                i = i + 1
            end
        end
    end
    return ret
end

stateindex(::DonationsPOMDP, s::Tuple{Int64, Int64, Int64, Int64}) = s[4]
n_states(::DonationsPOMDP) = num_steps*101*101

observations(::DonationsPOMDP) = 0:100
obsindex(::DonationsPOMDP, o::Int64) = o + 1
n_observations(::DonationsPOMDP) = 101

initial_belief(::DonationsPOMDP) = DiscreteBelief(2)
support(::BoolDistribution)
pdf(::BoolDistribution, ::Bool)
initialstate_distribution(::DonationsPOMDP)

function transition(pomdp::DonationsPOMDP, s::Bool, a::Bool)
    if a # fed
        return BoolDistribution(0.0)
    elseif s # did not feed when hungry
        return BoolDistribution(1.0)
    else # did not feed when not hungry
        return BoolDistribution(pomdp.p_become_hungry)
    end
end

function observation(pomdp::DonationsPOMDP, a::Bool, sp::Bool)
    if sp # hungry
        return BoolDistribution(pomdp.p_cry_when_hungry)
    else
        return BoolDistribution(pomdp.p_cry_when_not_hungry)
    end
end

function reward(pomdp::DonationsPOMDP, s::Bool, a::Bool)
    r = 0.0
    if s # hungry
        r += pomdp.r_hungry
    end
    if a # feed
        r += pomdp.r_feed
    end
    return r
end

discount(p::DonationsPOMDP) = 1

function generate_o(p::DonationsPOMDP, s::Bool, rng::AbstractRNG)
    d = observation(p, true, s) # obs distrubtion not action dependant
    return rand(rng, d)
end

# # some example policies
# mutable struct Starve <: Policy end
# action(::Starve, ::B) where {B} = false
# updater(::Starve) = NothingUpdater()
#
# mutable struct AlwaysFeed <: Policy end
# action(::AlwaysFeed, ::B) where {B} = true
# updater(::AlwaysFeed) = NothingUpdater()
#
# # feed when the previous observation was crying - this is nearly optimal
# mutable struct FeedWhenCrying <: Policy end
# updater(::FeedWhenCrying) = PreviousObservationUpdater()
# function action(::FeedWhenCrying, b::Union{Nothing, Bool})
#     if b == nothing || b == false # not crying (or null)
#         return false
#     else # is crying
#         return true
#     end
# end
# action(::FeedWhenCrying, b::Bool) = b
# action(p::FeedWhenCrying, b::Missing) = false
# # assume the second argument is a distribution
# action(::FeedWhenCrying, d::Any) = pdf(d, true) > 0.5

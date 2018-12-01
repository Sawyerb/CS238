using POMDPs
using POMDPModels
using POMDPSimulators
using POMDPPolicies
using JLD2, FileIO
using BeliefUpdaters

start = time()
solver = QMDPSolver() 
pomdp = DonationsPOMDP()
planner = solve(solver, pomdp)
println("TIME: ", time() - start)
@save "qmdp_100_10_45_100_100_100_false.jld" planner

println("Starting history.")
b_up = updater(planner)
init_dist = initialstate_distribution(pomdp)

hr = HistoryRecorder(max_steps=10)
hist = simulate(hr, pomdp, planner, b_up, init_dist)
for (s, b, a, r, sp, o) in hist
    @show s, a, r, sp, o
end

#rhist = simulate(hr, pomdp, RandomPolicy(pomdp))
#println("""
#    Cumulative Undiscounted Reward (for 1 simulation)
#        Random: $(undiscounted_reward(rhist))
#        QMDP: $(undiscounted_reward(hist))
#    """)

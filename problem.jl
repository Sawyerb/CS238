using POMDPs
using POMDPModels
using POMDPSimulators
using POMDPPolicies
using JLD

start = time()
solver = QMDPSolver()
pomdp = DonationsPOMDP()
planner = solve(solver, pomdp)
save("qmdp_100_10_45_100_100_100_false.jld", "policy", planner)

println("TIME: ", time() - start)
println("Starting history.")

hr = HistoryRecorder(max_steps=10)
hist = simulate(hr, pomdp, planner, (10, 45, 100))
for (s, b, a, r, sp, o) in hist
    @show s, b, a, r, sp, o
end

rhist = simulate(hr, pomdp, RandomPolicy(pomdp))
println("""
    Cumulative Undiscounted Reward (for 1 simulation)
        Random: $(undiscounted_reward(rhist))
        QMDP: $(undiscounted_reward(hist))
    """)
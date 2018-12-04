using POMDPs
using POMDPModels
using POMDPSimulators
using POMDPPolicies
using JLD2, FileIO
using BeliefUpdaters
using Statistics
using PyPlot

function run_action(win_reward::Int64, total_steps::Int64, initial_supp::Int64, initial_budg::Int64, opp_money::Int64, pol_money::Int64, determin::Bool)
    start = time()
    solver = QMDPSolver() 
    pomdp = DonationsPOMDP(win_reward, total_steps, initial_supp, initial_budg, opp_money, pol_money, determin)
    planner = solve(solver, pomdp)
    println("TIME: ", time() - start)

    println("Starting history.")
    b_up = updater(planner)
    init_dist = initialstate_distribution(pomdp)

    hr = HistoryRecorder(max_steps=total_steps)
    hist = simulate(hr, pomdp, planner, b_up, init_dist)
    a_arr = Float64[]
    for (s, b, a, r, sp, o) in hist
        @show s, a, r, sp, o
        push!(a_arr, a)
    end

    rhist = simulate(hr, pomdp, RandomPolicy(pomdp))
    random_reward = undiscounted_reward(rhist)
    policy_reward = undiscounted_reward(hist)
    println("""
        Cumulative Undiscounted Reward (for 1 simulation)
            Random: $(undiscounted_reward(rhist))
            QMDP: $(undiscounted_reward(hist))
        """)
    return a_arr
end

function run_multiple_reward(win_reward::Int64, total_steps::Int64, initial_supp::Int64, initial_budg::Int64, opp_money::Int64, pol_money::Int64, determin::Bool)
    rr = zeros(10)
    pr = zeros(10)
    for i in 1:5
        start = time()
        solver = QMDPSolver() 
        pomdp = DonationsPOMDP(win_reward, total_steps, initial_supp, initial_budg, opp_money, pol_money, determin)
        planner = solve(solver, pomdp)
        println("TIME: ", time() - start)

        println("Starting history.")
        b_up = updater(planner)
        init_dist = initialstate_distribution(pomdp)

        hr = HistoryRecorder(max_steps=total_steps)
        hist = simulate(hr, pomdp, planner, b_up, init_dist)
        for (s, b, a, r, sp, o) in hist
            @show s, a, r, sp, o
        end

        rhist = simulate(hr, pomdp, RandomPolicy(pomdp))
        random_reward = undiscounted_reward(rhist)
        policy_reward = undiscounted_reward(hist)
        rr[i] = random_reward
        pr[i] = policy_reward
    end
    return (mean(rr), mean(pr))
end

function main()
    # Heatmap of time step vs. initial support w/ action as value
    # Heatmap of initial support vs final win reward w/ % of max possible score as value
    initial_supp_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    root_path = "/Users/lucyli/Documents/Masters/CS 238/CS238/"
    total_steps = 10
    initial_supp = 5 
    initial_budg = 10
    win_reward = 5
    determin = true # are transitions deterministic? 

    #average_rs = run_multiple_reward(win_reward, total_steps, initial_supp, initial_budg, opp_money, pol_money, determin)
    #println(average_rs)

    action_z = zeros(length(initial_supp_range), total_steps)
    for i in 3:4 #1:length(initial_supp_range) # SOMETHING STILL DOESN'T ADD UP
        initial_supp = initial_supp_range[i]
        opp_money = Int(floor(initial_budg*(1-initial_supp/10)))
        pol_money = Int(floor(initial_budg*initial_supp/10))
        println("win_reward=", win_reward, ", total_steps=", total_steps, ", initial_supp=", initial_supp, ", initial_budg=", initial_budg)
        println("opp_money=", opp_money, ", pol_money=", pol_money, ", determin=", determin)

        action_arr = run_action(win_reward, total_steps, initial_supp, initial_budg, opp_money, pol_money, determin)
        action_z[i,:] = action_arr
    end

    filename = string(root_path, "initial_supp.png")
    im = imshow(action_z, cmap="Blues") 
    cbar = colorbar(im)
    cbar[:set_label]("Contribution")
    xticks(0:(total_steps-1), 1:total_steps)
    yticks(0:(length(initial_supp_range)-1), 0:(length(initial_supp_range)-1))
    title("Discrete actions when varying initial candidate support")
    xlabel("Time Step")
    ylabel("Initial Support")
    savefig(filename)
    close() 
end

main()

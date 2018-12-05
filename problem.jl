using POMDPs
using POMDPModels
using POMDPSimulators
using POMDPPolicies
using JLD2, FileIO
using BeliefUpdaters
using Statistics
using PyPlot

function run_action(win_reward::Int64, lose_penalty::Int64, total_steps::Int64, initial_supp::Int64, initial_budg::Int64, opp_money::Int64, pol_money::Int64, determin::Bool)
    start = time()
    solver = QMDPSolver() 
    pomdp = DonationsPOMDP(win_reward, lose_penalty, total_steps, initial_supp, initial_budg, opp_money, pol_money, determin)
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

function run_supp_reward(win_reward::Int64, lose_penalty::Int64, total_steps::Int64, initial_supp::Int64, initial_budg::Int64, opp_money::Int64, pol_money::Int64, determin::Bool)
    initial_supp_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rr_arr = zeros(length(initial_supp_range))
    pr_arr = zeros(length(initial_supp_range))
    for i in 1:length(initial_supp_range) 
        initial_supp = initial_supp_range[i]
        println(initial_supp)
        opp_money = Int(floor(initial_budg*(1-initial_supp/10)))
        pol_money = Int(floor(initial_budg*initial_supp/10))
        rr = zeros(5)
        pr = zeros(5)
        for j in 1:5
            start = time()
            solver = QMDPSolver() 
            pomdp = DonationsPOMDP(win_reward, lose_penalty, total_steps, initial_supp, initial_budg, opp_money, pol_money, determin)
            planner = solve(solver, pomdp)
            println("TIME: ", time() - start)

            b_up = updater(planner)
            init_dist = initialstate_distribution(pomdp)

            hr = HistoryRecorder(max_steps=total_steps)
            hist = simulate(hr, pomdp, planner, b_up, init_dist)

            rhist = simulate(hr, pomdp, RandomPolicy(pomdp))
            random_reward = undiscounted_reward(rhist)
            policy_reward = undiscounted_reward(hist)
            rr[j] = random_reward
            pr[j] = policy_reward
        end
        rr_arr[i] = mean(rr)
        pr_arr[i] = mean(pr)
    end
    return (rr_arr, pr_arr)
end

function support_actions(root_path::String, total_steps::Int64, initial_supp::Int64, initial_budg::Int64, win_reward::Int64, lose_penalty::Int64, determin::Bool)
    #initial_supp_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    initial_supp_range = [7]

    action_z = zeros(length(initial_supp_range), total_steps)
    for i in 1:length(initial_supp_range) 
        initial_supp = initial_supp_range[i]
        opp_money = Int(floor(initial_budg*(1-initial_supp/10)))
        pol_money = Int(floor(initial_budg*initial_supp/10))
        println("win_reward=", win_reward, ", total_steps=", total_steps, ", initial_supp=", initial_supp, ", initial_budg=", initial_budg)
        println("opp_money=", opp_money, ", pol_money=", pol_money, ", determin=", determin)

        action_arr = run_action(win_reward, lose_penalty, total_steps, initial_supp, initial_budg, opp_money, pol_money, determin)
        action_z[i,:] = action_arr
    end

    filename = string(root_path, determin, "_initial_supp.png")
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

function support_reward(root_path::String, total_steps::Int64, initial_supp::Int64, initial_budg::Int64, win_reward::Int64, lose_penalty::Int64, determin::Bool)
    # random, policy, zero
    average_rs = run_supp_reward(win_reward, lose_penalty, total_steps, initial_supp, initial_budg, opp_money, pol_money, determin) 

    filename = string(root_path, "line_support_reward.png")
    plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], average_rs[1], label="random")
    plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], average_rs[2], label="qmdp")
    if lose_penalty > 0
        plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], average_rs[3], label="zero")
    end
    title("Scores achieved when varying initial candidate support")
    ylabel("Score")
    xlabel("Initial Support")
    legend()
    savefig(filename)
    close()
end

function main()

    root_path = "/Users/lucyli/Documents/Masters/CS 238/CS238/discrete_plots/"
    total_steps = 10
    initial_supp = 5 
    initial_budg = 10
    win_reward = 10
    lose_penalty = 0
    determin = true 

    # vary support and plot actions per time step
    #support_actions(root_path, total_steps, initial_supp, initial_budg, win_reward, lose_penalty, determin)

    # plot line graph of score vs initial support for baseline and qmdp
    # policy, random, zero
    support_reward(root_path, total_steps, initial_supp, initial_budg, win_reward, lose_penalty, determin)

    # plot % max score for initial support vs max reward for qmdp

    # plot % max score for initial support vs lose penalty for qmdp

    # plot line graph of score vs initial support for baseline, no contributions, and qmdp
    #support_reward(root_path, total_steps, initial_supp, initial_budg, win_reward, 20, determin)

    # MAYBE vary support and plot actions per time step w/ determin = false 
    # DonationsPOMDP(50, 0, 10, 7, 10, 3, 7, false)
    #support_actions(root_path, total_steps, initial_supp, initial_budg, 500, lose_penalty, false)

    # MAYBE plot line graph of score vs initial support for baseline and qmdp w/ determin = false

end

main()

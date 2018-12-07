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

function run_supp_reward(win_reward::Int64, lose_penalty::Int64, total_steps::Int64, initial_supp::Int64, initial_budg::Int64, determin::Bool, n::Int64)
    initial_supp_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rr_arr = zeros(length(initial_supp_range))
    pr_arr = zeros(length(initial_supp_range))
    zr_arr = zeros(length(initial_supp_range))
    for i in 1:length(initial_supp_range) 
        initial_supp = initial_supp_range[i]
        println(initial_supp)
        opp_money = Int(floor(initial_budg*(1-initial_supp/10)))
        pol_money = Int(floor(initial_budg*initial_supp/10))
        rr = zeros(n)
        pr = zeros(n)
        zr = zeros(n)
        for j in 1:n 
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
            zhist = simulate(hr, pomdp, ZeroPolicy(pomdp, b_up))

            random_reward = undiscounted_reward(rhist)
            policy_reward = undiscounted_reward(hist)
            zero_reward = undiscounted_reward(zhist)
            rr[j] = random_reward
            pr[j] = policy_reward
            zr[j] = zero_reward
        end
        rr_arr[i] = mean(rr)
        pr_arr[i] = mean(pr)
        zr_arr[i] = mean(zr)
    end
    return (rr_arr, pr_arr, zr_arr)
end

function support_actions(root_path::String, total_steps::Int64, initial_supp::Int64, initial_budg::Int64, win_reward::Int64, lose_penalty::Int64, determin::Bool)
    initial_supp_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

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

    filename = string(root_path, determin, win_reward, "_initial_supp.png")
    im = imshow(action_z, cmap="Blues") 
    cbar = colorbar(im)
    cbar[:set_label]("Contribution")
    xticks(0:(total_steps-1), 1:total_steps)
    yticks(0:(length(initial_supp_range)-1), 0:(length(initial_supp_range)-1))
    title(string("Discrete actions with w_r = ", win_reward))
    xlabel("Time Step")
    ylabel("Initial Support")
    savefig(filename)
    close() 
end

function support_reward(root_path::String, total_steps::Int64, initial_supp::Int64, initial_budg::Int64, win_reward::Int64, lose_penalty::Int64, determin::Bool)
     # random, policy, zero
    average_rs = run_supp_reward(win_reward, lose_penalty, total_steps, initial_supp, initial_budg, determin, 5) 

    if lose_penalty > 0
        filename = string(root_path, lose_penalty, "_line_support_reward.png")
    else
        filename = string(root_path, "line_support_reward.png")
    end
    plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], average_rs[1], label="random")
    plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], average_rs[2], label="qmdp")
    plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], average_rs[3], label="zero")
    if lose_penalty > 0
        title(string("Scores with various initial support, lose_penalty = ", lose_penalty))
    else
        title("Scores with various initial support")
    end
    ylabel("Score")
    xlabel("Initial Support")
    legend()
    savefig(filename)
    close()
end

function get_max_reward(win_reward::Int64, total_steps::Int64)
    # always win every step with no money spent
    r = 0.0
    for m in 0:(total_steps-1)
        r += win_reward * ((total_steps-m+1)/(total_steps+1))
    end
    return r
end

function support_reward_heat(root_path::String, total_steps::Int64, initial_supp::Int64, initial_budg::Int64, win_reward::Int64, lose_penalty::Int64, determin::Bool)
    win_reward_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    initial_supp_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    z = zeros(length(initial_supp_range), length(win_reward_range))
    for k in 1:length(win_reward_range) 
        # random, policy, zero
        win_reward = win_reward_range[k]
        average_rs = run_supp_reward(win_reward, lose_penalty, total_steps, initial_supp, initial_budg, determin, 1) 
        max_reward = get_max_reward(win_reward, total_steps)
        qmdp_arr = 100*average_rs[2] / max_reward 
        z[:,k] = qmdp_arr
    end

    filename = string(root_path, "heat_support_reward.png")
    im = imshow(z, cmap="RdYlGn") 
    cbar = colorbar(im)
    cbar[:set_label]("% of Maximum Possible Score Achieved")
    xticks(0:(length(win_reward_range)-1), win_reward_range)
    yticks(0:(length(initial_supp_range)-1), initial_supp_range)
    title("QMDP Model")
    xlabel("Win Reward")
    ylabel("Initial Support")
    savefig(filename)
    close() 
end

function support_penalty_heat(root_path::String, total_steps::Int64, initial_supp::Int64, initial_budg::Int64, win_reward::Int64, lose_penalty::Int64, determin::Bool)
    penalty_range = [0, 10, 20, 30]
    initial_supp_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    z = zeros(length(initial_supp_range), length(penalty_range))
    for k in 1:length(penalty_range) 
        # random, policy, zero
        lose_penalty = penalty_range[k]
        average_rs = run_supp_reward(win_reward, lose_penalty, total_steps, initial_supp, initial_budg, determin, 1) 
        max_reward = get_max_reward(win_reward, total_steps)
        qmdp_arr = 100*average_rs[2] / max_reward 
        z[:,k] = qmdp_arr
    end

    filename = string(root_path, "heat_support_penalty.png")
    im = imshow(z, cmap="RdYlGn") 
    cbar = colorbar(im)
    cbar[:set_label]("% of Maximum Possible Score Achieved")
    xticks(0:(length(penalty_range)-1), penalty_range)
    yticks(0:(length(initial_supp_range)-1), initial_supp_range)
    title("QMDP Model")
    xlabel("Lose Penalty")
    ylabel("Initial Support")
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

    # vary support and plot actions per time step with a different win_reward 
    #support_actions(root_path, total_steps, initial_supp, initial_budg, 20, lose_penalty, determin)

    # plot line graph of score vs initial support for baseline and qmdp
    # policy, random, zero
    #support_reward(root_path, total_steps, initial_supp, initial_budg, win_reward, lose_penalty, determin)

    # plot % max score for initial support vs max reward for qmdp
    #support_reward_heat(root_path, total_steps, initial_supp, initial_budg, win_reward, lose_penalty, determin)

    # plot % max score for initial support vs lose penalty for qmdp
    support_penalty_heat(root_path, total_steps, initial_supp, initial_budg, win_reward, lose_penalty, determin)

    # plot line graph of score vs initial support for baseline, no contributions, and qmdp
    #support_reward(root_path, total_steps, initial_supp, initial_budg, win_reward, 20, determin)

    # vary support and plot actions per time step w/ determin = false 
    # DonationsPOMDP(50, 0, 10, 7, 10, 3, 7, false)
    #support_actions(root_path, total_steps, initial_supp, initial_budg, 50, lose_penalty, false)

end

main()

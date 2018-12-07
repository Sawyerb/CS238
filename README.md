# Campaign Funding Under Uncertainty
Sawyer Birnbaum, Lucy Li, Vivian Hsu

#Insert abstract

Important Files:
1. donations.jl / problem.jl / zero.jl -- define the discrete problem and its solvers
2. election.py / donor.py -- define the continuous problem 
3. continuous_solver.py / continuous_solver_multi.py -- POMDP based solvers for the continuous problem (with and without multiple races)
4. baseline_solver.py / baseline_solver_multi.py -- baseline solver for the continuous problem (with and without multiple races)
5. exp_[1-5].py -- experiments with the continuous problem
6. data_vis.py -- generates visualizations based on the continuous problem's experiments
7. real_data.py -- calculates parameters for distributions governing the Observation and Transition functions 

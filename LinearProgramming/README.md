# Optimization: Linear Programming

This project explores Linear Programming (LP) by using the Simplex method along with the dual Simplex method for constrained optimization problems.

This project uses the `SimplexMethod` class to solve LPs. The primal problem is used for maximization LPs, whereas the dual form is used for minimization problems. The `Optimizer` class provides a wrapper interface so that the user can setup LPs without knowledge of the underlying optimization method. For examples on how to setup and optimize an LP see `solve_linear_programs.cpp`. For example, this code sets up a constrained maximization problem:

```c++
Optimizer optimizer;
optimizer.AddLTConstraint({1, 0}, 4);
optimizer.AddLTConstraint({0, 2}, 12);
optimizer.AddLTConstraint({3, 2}, 18);
optimizer.AddGTConstraint({3, 2}, 2);
optimizer.MaximizeCost({3, 5});
auto result = optimizer.Solve();
``` 
This yields output
```
184 Maximize 3 * x_1 + 5 * x_2
185  such that 
186     1 * x_1 + 0 * x_2 <= 4
187     0 * x_1 + 2 * x_2 <= 12
188     3 * x_1 + 2 * x_2 <= 18
189     -3 * x_1 + -2 * x_2 <= -2
190    
191 ############################################################
192 (1) Optimized solution:  2  6  2  0  0 16 ---> optimum @ 36
193 ############################################################
```

Similarly, we can define a minimization problem:
```c++
Optimizer optimizer;
optimizer.AddGTConstraint({1, 1, 1}, 6);
optimizer.AddGTConstraint({0, 1, 2}, 8);
optimizer.AddGTConstraint({-1, 2, 2}, 4);
optimizer.MinimizeCost({2, 10, 8});
auto result = optimizer.Solve();
PrintSolution(7, result);
```
which yields the output
```
Minimize -2 * x_1 + -10 * x_2 + -8 * x_3
 such that 
	-1 * x_1 + -1 * x_2 + -1 * x_3 <= -6
	-0 * x_1 + -1 * x_2 + -2 * x_3 <= -8
	1 * x_1 + -2 * x_2 + -2 * x_3 <= -4

############################################################
(7) Optimized solution: 0 0 2 2 0 4 ---> optimum @ 36
############################################################
```

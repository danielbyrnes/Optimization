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

#include "constrained_optimization.h"

/// @brief Helper function to pretty print the optimized solution.
/// @param test_number Test ID number.
/// @param result Thje optimized result to print.
void PrintSolution(uint32_t test_number, const std::optional<std::tuple<Eigen::VectorXd, uint32_t>>& result) {
    if (!result) {
        std::cout << "(" << test_number << ") ";
        std::cout << "Failed to find feasible solution" << std::endl;
    }
    Eigen::VectorXd solution = std::get<0>(*result);
    uint32_t optimal_value = std::get<1>(*result);
    std::cout << "############################################################" << std::endl;
    std::cout << "(" << test_number << ") ";
    std::cout << "Optimized solution: " << solution.transpose() << " ---> optimum @ " << optimal_value << std::endl;
    std::cout << "############################################################\n" << std::endl;
}

int main() {
    // Run a series of maximization/minimization problems to test the solver.
    {
        // 1. Test Maximization
        Optimizer optimizer;
        optimizer.AddLTConstraint({1, 0}, 4);
        optimizer.AddLTConstraint({0, 2}, 12);
        optimizer.AddLTConstraint({3, 2}, 18);
        optimizer.AddGTConstraint({3, 2}, 2);
        optimizer.MaximizeCost({3, 5});
        auto result = optimizer.Solve();
        PrintSolution(1, result);
    }
    {
        // 2. Test Maximization
        Optimizer optimizer;
        optimizer.AddLTConstraint({1, 2}, 8);
        optimizer.AddLTConstraint({3, 2}, 12);
        optimizer.AddGTConstraint({1, 3}, 3);
        optimizer.MaximizeCost({1, 1});
        auto result = optimizer.Solve();
        PrintSolution(2, result);
    }
    {
        // 3. Test Maximization
        Optimizer optimizer;
        optimizer.AddLTConstraint({1, 2}, 8);
        optimizer.AddLTConstraint({3, 2}, 12);
        optimizer.AddGTConstraint({1, 3}, 3);
        optimizer.MaximizeCost({1, 1});
        auto result = optimizer.Solve();
        PrintSolution(3, result);
    }
    {
        // 4. Test Maximization
        Optimizer optimizer;
        optimizer.AddLTConstraint({1, 1}, 8);
        optimizer.AddLTConstraint({2, 1}, 12);
        optimizer.AddLTConstraint({1, 2}, 14);
        optimizer.MaximizeCost({2, 3});
        auto result = optimizer.Solve();
        PrintSolution(4, result);
    }
    {
        // 5. Test Minimization
        Optimizer optimizer;
        optimizer.AddGTConstraint({1, 2}, 40);
        optimizer.AddGTConstraint({1, 1}, 30);
        optimizer.MinimizeCost({12, 16});
        auto result = optimizer.Solve();
        PrintSolution(5, result);
    }
    {
        // 6. Test Minimization
        Optimizer optimizer;
        optimizer.AddGTConstraint({1, 1, 1}, 6);
        optimizer.AddGTConstraint({0, 1, 2}, 8);
        optimizer.AddGTConstraint({-1, 2, 2}, 4);
        optimizer.MinimizeCost({2, 10, 8});
        auto result = optimizer.Solve();
        PrintSolution(6, result);
    }
    {
        // Test SimplexMethod class interface directly
        Eigen::MatrixXd C(3,2); // Constraint Matrix
        C << 1, 0, 0, 2, 3, 2;
        Eigen::VectorXd cl(3); // Constraint limits
        cl << 4, 12, 18;
        Eigen::VectorXd cost(2); // Cost function
        cost << 3, 5;
        SimplexMethod opt(true /* maximize */, C, cl, cost);
        auto result = opt.Solve();
        PrintSolution(7, result);
    }
    return 0;
}
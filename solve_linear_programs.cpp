#include "constrained_optimization.h"

void PrintSolution(uint32_t test_number, const std::optional<std::tuple<Eigen::VectorXd, uint32_t>>& result) {
    if (!result) {
        std::cout << "(" << test_number << ") ";
        std::cout << "Failed to find feasible solution" << std::endl;
    }
    Eigen::VectorXd solution = std::get<0>(*result);
    uint32_t optimal_value = std::get<1>(*result);
    std::cout << "###########################################" << std::endl;
    std::cout << "(" << test_number << ") ";
    std::cout << "Optimized solution: " << solution.transpose() << " ---> optimum @ " << optimal_value << std::endl;
    std::cout << "###########################################\n" << std::endl;
}

int main() {
    {
        // Test Optimizer
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
        // Test Optimizer
        Optimizer optimizer;
        optimizer.AddLTConstraint({1, 2}, 8);
        optimizer.AddLTConstraint({3, 2}, 12);
        optimizer.AddGTConstraint({1, 3}, 3);
        optimizer.MaximizeCost({1, 1});
        auto result = optimizer.Solve();
        PrintSolution(2, result);
    }
    {
        // Test Minimization Optimizer
        Optimizer optimizer;
        optimizer.AddGTConstraint({1, 1}, 4);
        optimizer.AddLTConstraint({2, -1}, -1);
        optimizer.MinimizeCost({2, 3});
        auto result = optimizer.Solve();
        PrintSolution(3, result);
    }
    {
        // Test Minimization Optimizer
        Optimizer optimizer;
        optimizer.AddGTConstraint({1, 2}, 40);
        optimizer.AddGTConstraint({1, 1}, 30);
        optimizer.MinimizeCost({12, 16});
        auto result = optimizer.Solve();
        PrintSolution(4, result);
    }
    {
        // Test Simplex Method class directly
        Eigen::MatrixXd C(3,2); // Constraint Matrix
        C << 1, 0, 0, 2, 3, 2;
        Eigen::VectorXd cl(3); // Constraint limits
        cl << 4, 12, 18;
        Eigen::VectorXd cost(2); // Cost function
        cost << 3, 5;
        SimplexMethod opt(true /* maximize */, C, cl, cost);
        auto result = opt.Solve();
        PrintSolution(5, result);
    }
    return 0;
}
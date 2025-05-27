#include "simplex_method.h"

#include <algorithm>
#include <iostream>
#include <variant>
#include <sstream>
#include <iomanip>

/// @brief Interface for optimizer class that maximizes/minimizes linear programs (LP). Uses Simplex method under the hood.
class Optimizer {
    public:
        /// @brief Default constructor.
        Optimizer() = default;

        /// @brief Add maximization cost term to the problem.
        /// @param cost The cost function coefficients.
        void MaximizeCost(const std::vector<double>& cost) {
            cost_function_ = cost;
            maximize_ = true;
        }

        /// @brief Add minimization cost term to the problem.
        /// @param cost The cost function coefficients.
        void MinimizeCost(const std::vector<double>& cost) {
            std::vector<double> negated_cost;
            std::transform(cost.begin(), cost.end(), std::back_inserter(negated_cost), [](double n) { return -n; });
            cost_function_ = negated_cost;
            maximize_ = false;
        }

        /// @brief Adds a <= (less than or equal to) constraint to the optimization problem.
        /// @param constraint Vector of constraint coefficients.
        /// @param constraint_limit Constraint limit.
        void AddLTConstraint(const std::vector<double>& constraint, double constraint_limit) {
            constraints_.push_back(constraint);
            constraint_limits_.push_back(constraint_limit);
        }

        /// @brief Adds a >= (greater than or equal to) constraint to the optimization problem.
        /// @param constraint Vector of constraint coefficients.
        /// @param constraint_limit Constraint limit.
        void AddGTConstraint(const std::vector<double>& constraint, double constraint_limit) {
            // Negate the LHS and RHS
            std::vector<double> negated_constraint;
            std::transform(constraint.begin(), constraint.end(), std::back_inserter(negated_constraint), [](double n) { return -n; });
            constraints_.push_back(negated_constraint);
            constraint_limits_.push_back(-constraint_limit);
        }

        /// @brief Runs the solver to optimize the LP.
        /// @return Optional tuple of the optimized solution and the computed optimum value (if the problem can be solved).
        std::optional<std::tuple<Eigen::VectorXd, uint32_t>> Solve() {
            // First print the optimization probelm
            PrintProblem();
            uint32_t num_variables = cost_function_.size();
            uint32_t num_constraints = constraints_.size();
            Eigen::MatrixXd coefficients(num_constraints, num_variables);
            Eigen::VectorXd limits(num_constraints);
            for (uint32_t r = 0; r < num_constraints; ++r) {
                for (uint32_t c = 0; c < num_variables; ++c) {
                    coefficients(r,c) = constraints_[r][c];
                }
                limits(r) = constraint_limits_[r];
            }
            Eigen::VectorXd cost(num_variables);
            for (uint32_t c = 0; c < num_variables; ++c) {
                cost(c) = cost_function_[c];
            }
            simplex_ = std::make_unique<SimplexMethod>(maximize_, coefficients, limits, cost);
        return simplex_->Solve();
        }

        /// @brief Helper function to print the LP (i.e. the constrained optimization problem).
        void PrintProblem() {
            // Pretty print optimization problem
            if (cost_function_.empty()) {
                // Optimization problem hasn't been fully specified yet
                return;
            }
            std::ostringstream objective_func;
            objective_func << std::fixed << std::setprecision(0);
            if (maximize_) {
                objective_func << "Maximize ";
            } else {
                objective_func << "Minimize ";
            }
            for (uint32_t i = 0; i < cost_function_.size(); ++i) {
                objective_func << cost_function_[i];
                objective_func << " * x_" + std::to_string(i+1);
                if (i < cost_function_.size() - 1) {
                    objective_func << " + ";
                }
            }
            objective_func << "\n such that \n";

            // Add constraints
            assert(constraints_.size() == constraint_limits_.size());
            for (size_t i = 0; i < constraint_limits_.size(); ++i) {
                objective_func << "\t";
                for (uint32_t j = 0; j < constraints_[i].size(); ++j) {
                    objective_func << constraints_[i][j] << " * x_" << std::to_string(j+1);
                    if (j < constraints_[i].size() - 1) {
                        objective_func << " + ";
                    }
                }
                objective_func << " <= " << constraint_limits_[i];
                objective_func << "\n";
            }
            std::cout << objective_func.str() << std::endl;
        }
    private:
        /// @brief Indicates whether the LP is a maximization problem.
        bool maximize_ = true;
        /// @brief The LP cost function.
        std::vector<double> cost_function_;
        /// @brief The LP constraints (LHS).
        std::vector<std::vector<double>> constraints_;
        /// @brief The LP constraint limits (RHS).
        std::vector<double> constraint_limits_;
        /// @brief The Simplex class used to solve the LP.
        std::unique_ptr<SimplexMethod> simplex_;
};
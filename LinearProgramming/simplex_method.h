#pragma once

#include <Eigen/Core>
#include <optional>
#include <tuple>

/// @brief Simplex method for optimizing linear programs.
class SimplexMethod {
    public:
        /// @brief Constructor of SimplexMethod.
        /// @param maximize Boolean indicating whether this is a maximization problem.
        /// @param constraint_coefficients nxm matrix of constraint coefficients.
        /// @param constraint_limits nx1 vector of constraint limits (assumes <= constraints).
        /// @param cost_coefficients mx1 vector of cost coefficients.
        SimplexMethod(bool maximize, 
                      const Eigen::MatrixXd& constraint_coefficients,
                      const Eigen::VectorXd& constraint_limits,
                      const Eigen::VectorXd& cost_coefficients);

        /// @brief Solves the linear program optimization problem.
        /// @return Optional tuple of the vector solution (basic and non-basic variables) and the optimum value found.
        std::optional<std::tuple<Eigen::VectorXd, uint32_t>> Solve();

    private:
        /// @brief Creates an auxiliary tableau for phase I of the 2-phase Simplex method.
        /// @return True if the auxiliary tableau was solved, indicating a feasible solution was found.
        bool ExecuteSimplexPreprocessing();

       /// @brief Runs phase II of the simplex algorithm. This assumes the tableau has been transformed into a maximization problem.
       /// @return Tuple of the solution and optimum value.
       std::tuple<Eigen::VectorXd, uint32_t> ExecuteSimplex(Eigen::MatrixXd& tableau);

       /// @brief Helper function to check if any of the constraint limits are negative, suggesting phase I is needed.
       /// @return True if any of the constraint limits are negative.
       bool AnyNegativeConstraints();
       
        /// @brief Constraints the tableau for the primal problem (maximization) or dual problem (minimization).
        /// @param CC nxm matrix of constraint coefficients.
        /// @param cl nx1 vector of constraint limits. 
        /// @param cost mx1 vector of cost coefficients.
        /// @return The tableau of the primary/dual problem (depending on whether optimizing max or min).
        Eigen::MatrixXd FormTableau(const Eigen::MatrixXd& CC,
                                    const Eigen::VectorXd& cl,
                                    const Eigen::VectorXd& cost);

        /// @brief Indicates whether the dual problem should be optimized instead (e.g. minimization problem)
        bool optimize_dual_ = false;
        /// @brief Number of constraints
        uint32_t nc_;
        /// @brief Number of variables
        uint32_t nv_;
        /// @brief  Tableau of current iteration
        Eigen::MatrixXd tableau_;
        /// @brief Indicates whether to log debug statements
        bool debug_logs_ = false;
};
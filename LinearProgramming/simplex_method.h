#pragma once

#include <Eigen/Core>
#include <optional>
#include <tuple>

/// @brief 
class SimplexMethod {
    public:
        /// @brief  
        /// @param maximize
        /// @param constraint_coefficients 
        /// @param constraint_limits 
        /// @param cost_coefficients 
        SimplexMethod(bool maximize, 
                      const Eigen::MatrixXd& constraint_coefficients,
                      const Eigen::VectorXd& constraint_limits,
                      const Eigen::VectorXd& cost_coefficients);

        /// @brief 
        /// @return 
        std::optional<std::tuple<Eigen::VectorXd, uint32_t>> Solve();

    private:
        /// @brief 
        /// @return 
        bool ExecuteSimplexPreprocessing();

       /// @brief 
       /// @return 
       std::tuple<Eigen::VectorXd, uint32_t> ExecuteSimplex(Eigen::MatrixXd& tableau);

       /// @brief 
       /// @return 
       bool AnyNegativeConstraints();
       
        /// @brief 
        /// @param CC 
        /// @param cl 
        /// @param cost 
        /// @return 
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
#include "simplex_method.h"
#include <iostream>
#include <cassert>
#include <limits>

SimplexMethod::SimplexMethod(bool maximize, 
                             const Eigen::MatrixXd& constraint_coefficients,
                             const Eigen::VectorXd& constraint_limits,
                             const Eigen::VectorXd& cost_coefficients) : 
                                optimize_dual_(!maximize),
                                nc_(constraint_coefficients.rows()),
                                nv_(cost_coefficients.size()),
                                tableau_(FormTableau(constraint_coefficients, constraint_limits, cost_coefficients)) {
    if (debug_logs_) {
        std::cout << "Initial Tableau:\n" << tableau_ << std::endl;
    }
}

std::optional<std::tuple<Eigen::VectorXd, uint32_t>> SimplexMethod::Solve() {
    // If any constraints are negative run phase I simplex.
    if (AnyNegativeConstraints()) {
        if (debug_logs_) {
            std::cout << "Running preprocessing phase" << std::endl;
        }
        if(!ExecuteSimplexPreprocessing()) {
            return {};
        }
    }
    // Run simplex starting from feasible solution.
    return ExecuteSimplex(tableau_);
}

bool SimplexMethod::ExecuteSimplexPreprocessing() {
    assert(tableau_.rows() == nc_ + 1);
    assert(tableau_.cols() == nc_ + nv_ + 1);
    std::vector<uint32_t> artifical_vars;
    for (uint32_t r = 0; r < nc_; ++r) {
        if (tableau_.row(r).tail(1)[0] < 0) {
            artifical_vars.push_back(r);
        }
    }
    // Add artifical variable block
    Eigen::MatrixXd initial_tableau(tableau_.rows(), tableau_.cols() + artifical_vars.size());
    // Copy tableau_ into this new tableau matrix, leaving space for the artificial variable submatrix.
    initial_tableau.block(0, 0, nc_, nc_+nv_) = tableau_.block(0,0,nc_,nc_+nv_);
    initial_tableau.block(0, initial_tableau.cols()-1, nc_, 1) = tableau_.block(0,nc_+nv_,nc_,1);
    // Add artifical variables to create an obviously feasible solution
    for (uint32_t r = 0; r < artifical_vars.size(); ++r) {
        if (initial_tableau(artifical_vars[r], initial_tableau.cols()-1) < 0) {
            // The artificial varialbe must be >= 0; negative constraint if RHS is negative
            initial_tableau.row(artifical_vars[r]) *= -1;
        }
        initial_tableau(artifical_vars[r], nc_ + nv_ + r) = 1;
    }
    // Only the artifical variables should appear in the cost row
    initial_tableau.row(nc_).setZero();
    for (uint32_t r = 0; r < artifical_vars.size(); ++r) {
        initial_tableau(nc_, nc_ + nv_ + r) = 1;
    }
    // Eliminate aritifical variables from cost row; make them basic variables
    for (uint32_t r = 0; r < artifical_vars.size(); ++r) {
        initial_tableau.row(nc_) -= initial_tableau.row(artifical_vars[r]);
    }
    if (debug_logs_) {
        std::cout << "Auxiliary tableau: \n" << initial_tableau << std::endl;
    }
    // Run simplex method on the auxiliary tableau
    auto result = ExecuteSimplex(initial_tableau);
    Eigen::VectorXd solution = std::get<0>(result);
    // Check if the artificial variables are zero, if not then solution is not feasible
    if ((solution.tail(artifical_vars.size()).array() != 0).any()) {
        return false;
    }
    // Replace starting values in tableau_ (except the cost)
    uint32_t last_column = tableau_.cols() - 1;
    tableau_.col(last_column) = initial_tableau.col(initial_tableau.cols() -1);
    tableau_.block(0, 0, nc_, last_column) = initial_tableau.block(0, 0, nc_, last_column);
    if (debug_logs_) {
        std::cout << "solution: " << solution.transpose() << std::endl;
        std::cout << "Updated tableau after preprocessing:" << std::endl;
        std::cout << tableau_ << std::endl;
    }
    return true;
}

std::tuple<Eigen::VectorXd, uint32_t> SimplexMethod::ExecuteSimplex(Eigen::MatrixXd& tableau) {
    uint32_t nc = tableau.rows() - 1;
    uint32_t nv = tableau.cols() - 1;
    while (true) {
        // Find the pivot column
        Eigen::Index pivot_col;
        // For maximization problems select the column with the
        // most negative entry in the objective row
        tableau.row(nc).head(nv).minCoeff(&pivot_col);
        if (tableau(nc, pivot_col) >= 0) {
            break;
        }

        // Find pivot row
        double ratio_limit = std::numeric_limits<double>::max();
        uint32_t pivot_row = 0;
        for (uint32_t row = 0; row < nc; ++row) {
            if (tableau(row, pivot_col) > 0) {
                double ratio = tableau(row, nv) / tableau(row, pivot_col);
                if (ratio >= 0 && ratio < ratio_limit) {
                    ratio_limit = ratio;
                    pivot_row = row;
                }
            }
        }
        if (ratio_limit == std::numeric_limits<double>::max()) {
            std::cout << "Error: No pivot row found ..." << std::endl;
            continue;
        }
        // Perform pivot operation
        double pivot_val = tableau(pivot_row, pivot_col);
        tableau.row(pivot_row) /= pivot_val;
        for (uint32_t r = 0; r < tableau.rows(); ++r) {
            if (r == pivot_row) continue;
            double scale_factor = tableau(r, pivot_col);
            tableau.row(r) -= scale_factor * tableau.row(pivot_row);
        }
        if (debug_logs_) {
            std::cout << "Intermediate tableau: \n" << tableau << std::endl;
        }
    }

    // Extract the solution
    Eigen::VectorXd solution(nv);
    for (uint32_t c = 0; c < tableau.cols()-1; ++c) {
        if (optimize_dual_) {
            // If using the dual then the solution is in the cost function row
            if (tableau.col(c).head(nc).cwiseAbs().sum() != 1 || (tableau.col(c).head(nc).array() == 0).count() != nc - 1) {
                solution(c) = tableau(nc, c);
            }
        } else {
            // Otherwise the solution is in the RHS column
            if (tableau.col(c).head(nc).cwiseAbs().sum() == 1 && (tableau.col(c).head(nc).array() == 0).count() == nc - 1) {
                Eigen::Index row_idx;
                tableau.col(c).maxCoeff(&row_idx);
                solution(c) = tableau(row_idx, nv);
            }
        }
    }
    double optimal_value = tableau(nc, nv);
    if (debug_logs_) {
        std::cout << "optimal solution: " << optimal_value << std::endl;  
    }
    return std::make_tuple(solution, optimal_value);
}

bool SimplexMethod::AnyNegativeConstraints() {
    assert(tableau_.rows() == nc_ + 1);
    for (uint32_t r = 0; r < nc_; ++r) {
        if (tableau_.row(r).tail(1)[0] < 0) return true;
    }
    return false;
}

Eigen::MatrixXd SimplexMethod::FormTableau(const Eigen::MatrixXd& CC,
                            const Eigen::VectorXd& cl,
                            const Eigen::VectorXd& cost) {
    // Use dual simplex method for minimization problems
    if (optimize_dual_) {
        // Form tableau for dual problem
        Eigen::MatrixXd CC_t = CC.transpose();
        uint32_t num_constraints = CC_t.rows();
        uint32_t num_vars = CC_t.cols();
        nc_ = num_constraints;
        nv_ = num_vars;
        // General tableau form dual problem:
        //       CC^T  sl cost
        //       -cl   0  0
        uint32_t sv = nv_; // num variables
        Eigen::MatrixXd dual_tableau = Eigen::MatrixXd::Zero(nv_+1,nv_+nc_+1);
        dual_tableau.block(0,0,nv_,nc_) = -CC_t;
        dual_tableau.block(0,nc_,sv,sv) = Eigen::MatrixXd::Identity(sv, sv);
        dual_tableau.block(0,nc_+sv,nv_,1) = -cost;
        dual_tableau.block(nv_,0,1,nc_) = cl.transpose();
        return dual_tableau;
    }
    uint32_t num_constraints = CC.rows();
    uint32_t num_vars = CC.cols();
    assert(num_constraints == nc_);
    assert(num_vars == nv_);
    uint32_t sv = nc_; // num slack variables
    Eigen::MatrixXd tableau = Eigen::MatrixXd::Zero(nc_+1,nv_+nc_+1);
    // General tableau form (for maximization problem):
    //        CC  sl cl
    //      -cost 0  0
    tableau.block(0,0,nc_,nv_) = CC;
    tableau.block(0,nv_,sv,sv) = Eigen::MatrixXd::Identity(sv, sv);
    tableau.block(0,nv_+sv,nc_,1) = cl;
    tableau.block(nc_,0,1,nv_) = -cost.transpose();
    return tableau;
}

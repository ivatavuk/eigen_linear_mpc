/**
 * @file EigenPTSC.hpp
 * @author Ivo Vatavuk
 * @copyright Released under the terms of the BSD 3-Clause License
 * @date 2022
 * 
 *    Simple wrapper for OsqpEigen QP solver 
 *    The QP problem is of the following form:
 *
 *      min 	1 / 2 * x^T * A_qp * x + b_qp^T * x
 *       x
 *
 *      s.t.	A_eq * x + b_eq = 0
 *            A_ieq * x + b_ieq <= 0
 */
#ifndef OSQP_EIGEN_OPTIMIZATION_HPP_
#define OSQP_EIGEN_OPTIMIZATION_HPP_

#include <iostream>
#include <vector>
#include <OsqpEigen/OsqpEigen.h>

#include "QpProblem.hpp"

using VecNd = Eigen::VectorXd;
using MatNd = Eigen::MatrixXd;

class OsqpEigenOpt 
{    
public:
  OsqpEigenOpt( );
  OsqpEigenOpt(	const SparseQpProblem &sparse_qp_problem, 
                bool verbosity = false );

  void initializeSolver(const SparseQpProblem &sparse_qp_problem, 
                        bool verbosity );

  void setGradientAndInit(VecNd &b_qp); 

  VecNd solveProblem();

  bool checkFeasibility(); 

private:
  OsqpEigen::Solver solver_;

  double alpha_;

  uint32_t n_; //number of optimization variables
  uint32_t m_; //number of constraints

  double inf = 1e100;

  VecNd lower_bound_, upper_bound_;
};

#endif //OSQP_EIGEN_OPTIMIZATION_HPP_
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
  OsqpEigenOpt();
  OsqpEigenOpt(double alpha);
  OsqpEigenOpt(	const DenseQpProblem &dense_qp_problem, 
                  bool verbosity = false );
  ~OsqpEigenOpt();

  void initializeSolver(bool verbosity );
  VecNd solveProblem();

  void setupQP( const DenseQpProblem &qp_problem ); 

  bool checkFeasibility(); 

private:
  OsqpEigen::Solver solver_;

  DenseQpProblem qp_problem_;
  SparseQpProblem sparse_qp_problem_;

  double alpha_;

  uint32_t n_; //number of optimization variables
  uint32_t m_; //number of constraints

  double inf = 1e100;
};

#endif //OSQP_EIGEN_OPTIMIZATION_HPP_
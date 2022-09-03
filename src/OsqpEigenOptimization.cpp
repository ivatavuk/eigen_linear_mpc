/**
 * @file OsqpEigenOptimization.cpp
 * @author Ivo Vatavuk
 * @copyright Released under the terms of the BSD 3-Clause License
 * @date 2022
 */

#include "OsqpEigenOptimization.hpp"

OsqpEigenOpt::OsqpEigenOpt() : alpha_(1.0)
{}

OsqpEigenOpt::OsqpEigenOpt( const DenseQpProblem &dense_qp_problem, 
                            bool verbosity ) : alpha_(1.0)
{
  setupQP(dense_qp_problem);
  initializeSolver(verbosity);
}

OsqpEigenOpt::~OsqpEigenOpt() {}

void OsqpEigenOpt::setupQP( const DenseQpProblem &dense_qp_problem ) 
{
  qp_problem_ = dense_qp_problem;

  n_ = dense_qp_problem.A_qp.rows();
  m_ = dense_qp_problem.A_eq.rows() + dense_qp_problem.A_ieq.rows();

  sparse_qp_problem_ = SparseQpProblem(dense_qp_problem);
}


void OsqpEigenOpt::initializeSolver(bool verbosity) 
{
  solver_.settings()->setVerbosity(verbosity);
  solver_.settings()->setAlpha(alpha_);

  solver_.settings()->setAdaptiveRho(false);

  solver_.data()->setNumberOfVariables(n_);
  solver_.data()->setNumberOfConstraints(m_);

  solver_.data()->clearHessianMatrix();
  solver_.data()->setHessianMatrix(sparse_qp_problem_.A_qp);
  solver_.data()->setGradient(sparse_qp_problem_.b_qp);

  solver_.data()->clearLinearConstraintsMatrix();
  solver_.data()->setLinearConstraintsMatrix(sparse_qp_problem_.A_ieq);

  VecNd lower_bound_eq = -sparse_qp_problem_.b_eq;
  VecNd upper_bound_eq = -sparse_qp_problem_.b_eq;

  VecNd lower_bound_ieq = -inf * VecNd::Ones(sparse_qp_problem_.b_ieq.size());
  VecNd upper_bound_ieq = -sparse_qp_problem_.b_ieq;

  VecNd lower_bound(lower_bound_eq.size() + lower_bound_ieq.size());
  VecNd upper_bound(upper_bound_eq.size() + upper_bound_ieq.size());

  lower_bound << lower_bound_eq, lower_bound_ieq;
  upper_bound << upper_bound_eq, upper_bound_ieq;

  solver_.data()->setBounds(lower_bound, upper_bound);

  solver_.clearSolver();
  solver_.initSolver();
}

VecNd OsqpEigenOpt::solveProblem()
{
  auto exit_flag = solver_.solveProblem();
  return solver_.getSolution();
}

bool OsqpEigenOpt::checkFeasibility() //Call this after calling solve
{
  return !( (int) solver_.getStatus() == OSQP_PRIMAL_INFEASIBLE );
}


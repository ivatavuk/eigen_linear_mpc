/**
 * @file QpProblem.hpp
 * @author Ivo Vatavuk
 * @copyright Released under the terms of the BSD 3-Clause License
 * @date 2022
 *  
 *    The QP problem is of the following form:
 *
 *      min 	1 / 2 * x^T * A_qp * x + b_qp^T * x
 *       x
 *
 *      s.t.	A_eq * x + b_eq = 0
 *            A_ieq * x + b_ieq <= 0
 */

#ifndef OP_PROBLEM_HPP_
#define OP_PROBLEM_HPP_

#include <Eigen/Dense>
#include <Eigen/Sparse>

using VecNd = Eigen::VectorXd;
using MatNd = Eigen::MatrixXd;
using SparseMat = Eigen::SparseMatrix<double>;

//QP description
struct DenseQpProblem {
  MatNd A_qp, A_eq, A_ieq;
  VecNd b_qp, b_eq, b_ieq, upper_bound, lower_bound;
  DenseQpProblem(	MatNd A_qp_in, VecNd b_qp_in, 
                  MatNd A_eq_in, VecNd b_eq_in,
                  MatNd A_ieq_in, VecNd b_ieq_in)
    : A_qp(A_qp_in), b_qp(b_qp_in), 
      A_eq(A_eq_in), b_eq(b_eq_in),
      A_ieq(A_ieq_in), b_ieq(b_ieq_in) {};
};

struct SparseQpProblem {
  SparseMat A_qp, A_eq, A_ieq;
  VecNd b_qp, b_eq, b_ieq, upper_bound, lower_bound;
  SparseQpProblem(SparseMat A_qp_in, VecNd b_qp_in, 
                  SparseMat A_eq_in, VecNd b_eq_in,
                  SparseMat A_ieq_in, VecNd b_ieq_in) 
    : A_qp(A_qp_in), b_qp(b_qp_in), 
      A_eq(A_eq_in), b_eq(b_eq_in),
      A_ieq(A_ieq_in), b_ieq(b_ieq_in) {};
  SparseQpProblem(SparseMat A_qp_in, VecNd b_qp_in, 
                  SparseMat A_eq_in, VecNd b_eq_in,
                  SparseMat A_ieq_in, VecNd b_ieq_in,
                  VecNd lower_bound_in, VecNd upper_bound_in) 
    : A_qp(A_qp_in), b_qp(b_qp_in), 
      A_eq(A_eq_in), b_eq(b_eq_in),
      A_ieq(A_ieq_in), b_ieq(b_ieq_in),
      lower_bound(lower_bound_in),
      upper_bound(upper_bound_in) 
      {
        //TODO check dimensions??
      };

  SparseQpProblem(DenseQpProblem dense_qp_prob) 
  {
    A_qp = sparseMatrixFromDense(dense_qp_prob.A_qp);
    A_eq = sparseMatrixFromDense(dense_qp_prob.A_eq);
    A_ieq = sparseMatrixFromDense(dense_qp_prob.A_ieq);

    b_qp = dense_qp_prob.b_qp;
    b_eq = dense_qp_prob.b_eq;
    b_ieq = dense_qp_prob.b_ieq;
  };

  static SparseMat sparseMatrixFromDense(const MatNd &matrix) //TODO: does Eigen support something like this?
  {
    SparseMat sparse_matrix(matrix.rows(), matrix.cols());
    for(uint32_t j = 0; j < matrix.cols(); j++)
    {
      for(uint32_t i = 0; i < matrix.rows(); i++)
      {
        if(matrix(i, j) != 0)
        {
          sparse_matrix.insert(i, j) = matrix(i, j);	
        }
      }
    }
    return sparse_matrix;
  }
};

#endif //OP_PROBLEM_HPP_
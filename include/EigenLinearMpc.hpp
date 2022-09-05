/*
  Ivo Vatavuk, 2021

  Linear tracking MPC library using Eigen linear algebra library and BPMPD QP solver

  Structures:
    LinearSystem
      - describes a linear system of the following form: 
          x(k+1) =  A * x(k) + B * u(k)
          y(k) =    C * x(k) + D * u(k)

    QP	
      - describes a Quadratic Programming problem of the following form: 
        min 	1/2 * x^T * A_qp * x + b_qp^T * x
         x
          
        s.t.	A_eq * x + b_eq = 0
            A_ieq * x + b_ieq <= 0
  Class:
    MPC
      - sets up a QP problem for:

        - MPC I:
          min  	Q * ||Y - Y_d||^2 + R * ||U||^2
           U
            
          s.t.	(implicit constraints)
              x(k+1) =  A * x(k) + B * u(k)
              y(k) =    C * x(k)

        - MPC II:
          min  	||Y - Y_d||^2 + ||W_u * U||^2 + ||W_x * X||^2
           U
            
          s.t. 	(implicit constraints)
              x(k+1) =  A * x(k) + B * u(k)
              y(k) =    C * x(k)
      

      - produces matrices for the QP solver

      MPC dynamics:
        X = A_mpc * U + B_mpc * x0
      MPC output vector: 
        Y = C_mpc * X 
        (TODO: add support for D_mpc*U)
      where:
          x0 - inital state
          X - vector of N states
          U - vector of N inputs 
          (N - prediction horizon)

  TODO: 
    Add custom constraint support
*/
#ifndef EIGENLINEARMPC_H_
#define EIGENLINEARMPC_H_

#include <iostream>
#include <chrono>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "OsqpEigenOptimization.hpp"

typedef Eigen::VectorXd VecNd;
typedef Eigen::MatrixXd MatNd;
typedef Eigen::SparseMatrix<double> SparseMat;

//typedef Eigen::SparseMatrix<double> SparseMatNd; todo probat ubrzat koristenjem ovoga!

namespace EigenLinearMpc {

struct LinearSystem {
  /*
  LinearSystem( const MatNd &A, const MatNd &B, 
                const MatNd &C, const MatNd &D );
  */
  LinearSystem( const SparseMat &A, const SparseMat &B, 
                const SparseMat &C, const SparseMat &D );

  //throws an error if the system is ill defined
  void checkMatrixDimensions() const;
  
  //system dynamics
  uint32_t n_x; // x vector dimension
  uint32_t n_u; // u vector dimension
  uint32_t n_y; // y vector dimension
  SparseMat A, B, C, D;
};

class MPC {
public:
  MPC(const LinearSystem &linear_system, uint32_t horizon, 
      const VecNd &Y_d, const VecNd &x0, double Q, double R); 
  
  MPC(const LinearSystem &linear_system, uint32_t horizon, 
      const VecNd &Y_d, const VecNd &x0, double Q, double R,
      const VecNd &u_lower_bound, const VecNd &u_upper_bound,
      const VecNd &x_lower_bound, const VecNd &x_upper_bound);

  MPC(const LinearSystem &linear_system, uint32_t horizon, 
      const VecNd &Y_d, const VecNd &x0, const MatNd &w_u, 
      const MatNd &w_x); 
  
  void setYd(const VecNd &Y_d_in); // set Y_d from an Eigen vector Nd

  void initializeSolver();
  
  VecNd calculateX(const VecNd &U_in) const;
  VecNd calculateY(const VecNd &U_in) const;

  SparseQpProblem getQpProblem() const;
  VecNd solve() const;

private:

  uint32_t N_; // mpc prediction horizon
  LinearSystem linear_system_; // linear_system
  
  SparseMat A_mpc_, B_mpc_, C_mpc_; // mpc dynamics matrices

  VecNd Y_d_; //refrence output
  VecNd x0_; //initial state

  SparseQpProblem qp_problem_;
  double Q_, R_; 

  MatNd W_u_, w_u_;
  MatNd W_x_, w_x_;

  // matrices saved for faster QP problem update
  SparseMat C_A_; // C_mpc * A_mpc
  SparseMat C_B_; // C_mpc * B_mpc
  MatNd W_x_A_; // W_x * A_mpc
  MatNd W_x_B_; // W_x * B_mpc

  enum mpc_type
  {
    MPC1 = 0,
    MPC2 = 1,
    MPC1_BOUND_CONSTRAINED = 2,
    MPC2_BOUND_CONSTRAINED = 3
  };

  mpc_type mpc_type_;

  void setWx();
  void setWu();

  //Sets A_mpc, B_mpc, C_mpc
  void setupMpcDynamics();
  // MPC1
  void setupQpMPC1(); 
  void setupQpBoundConstrainedMPC1(); 
  // MPC2
  void setupQpMPC2(); 
  // THIS IS SLOW!! - speed up using sparse matrices
  void updateQpMatrices2(const VecNd &Y_d_in, const VecNd &x0); // update qp problem for new Y_d_in and x0

  void checkMatrixDimensions() const; 
  void checkBoundsDimensions() const; 

  VecNd u_lower_bound_, u_upper_bound_,
        x_lower_bound_, x_upper_bound_;

  std::unique_ptr<OsqpEigenOpt> osqp_eigen_opt_;
};
}
#endif //EIGENLINEARMPC_H_
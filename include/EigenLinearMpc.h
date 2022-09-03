/*
  Ivo Vatavuk, 2021

  Linear tracking MPC library using Eigen linear algebra library and BPMPD QP solver

  Structures:
    LinearSystem
      - describes a linear system of the following form: 
          x(k+1) = A * x(k) + B * u(k)
          y(k) = C * x(k) + D * u(k)

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
              x(k+1) = A * x(k) + B * u(k)
              y(k) = C * x(k)

        - MPC II:
          min  	Q * ||Y - Y_d||^2 + ||W_u * U||^2 + ||W_x * X||^2
          U
            
          s.t. 	(implicit constraints)
              x(k+1) = A * x(k) + B * u(k)
              y(k) = C * x(k)
      

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

typedef Eigen::VectorXd VecNd;
typedef Eigen::MatrixXd MatNd;

//typedef Eigen::SparseMatrix<double> SparseMatNd; todo probat ubrzat koristenjem ovoga!

namespace EigenLinearMpc {

struct LinearSystem {
  LinearSystem() {};
  LinearSystem( 	const MatNd &A, const MatNd &B, 
          const MatNd &C, const MatNd &D);
  ~LinearSystem() {};

  //throws an error if the system is ill defined
  void checkMatrixDimensions() const;
  
  //system dynamics
  uint32_t n_x; // x vector dimension
  uint32_t n_u; // u vector dimension
  uint32_t n_y; // y vector dimension
  MatNd A, B, C, D;
};

struct QpProblem {
  QpProblem() {};
  QpProblem( const MatNd &A_qp_in, const VecNd &b_qp_in, 
    const MatNd &A_eq_in, const VecNd &b_eq_in, 
    const MatNd &A_ieq_in, const VecNd &b_ieq_in);
  ~QpProblem() {};
  
  //updates all QP matrices
  void problemSetup(	const MatNd &A_qp, const VecNd &b_qp, 
            const MatNd &A_eq, const VecNd &b_eq,
            const MatNd &A_ieq, const VecNd &b_ieq );

  void setAqp (const MatNd &A_qp) { this->A_qp = A_qp; };
  void setBqp (const VecNd &b_qp) { this->b_qp = b_qp; };
  void setAeq (const MatNd &A_eq) { this->A_eq = A_eq; };
  void setBeq (const VecNd &b_eq) { this->b_eq = b_eq; };
  void setAieq (const MatNd &A_ieq) { this->A_ieq = A_ieq; };
  void setBieq (const VecNd &b_ieq) { this->b_ieq = b_ieq; };
            
  MatNd A_qp, A_eq, A_ieq;
  VecNd b_qp, b_eq, b_ieq;
};

class MPC {
  public:
    MPC() {};
    // calls setMPCdynamics(LinearSystem linear_system, uint32_t horizon)
    MPC(const LinearSystem &linear_system, uint32_t horizon); 

    ~MPC() {};


    void setLinearSystem(const LinearSystem &linear_system) { linear_system_ = linear_system; };
    void setHorizon(uint32_t horizon) { N_ = horizon; };

    // sets member variables linear_system_, N_, A_mpc, B_mpc, C_mpc
    void setMpcDynamics(const LinearSystem &linear_system, uint32_t horizon);

    // sets A_mpc, B_mpc, C_mpc for current members linear_system_ and N_
    void setMpcDynamics(); 

    void setYd(const std::vector<double> &Y_d_in, uint32_t start_index); // set Y_d from a vector<double>
    void setYd(const VecNd &Y_d_in); // set Y_d from an Eigen vector Nd
    
    void setQandR(double Q_in, double R_in); 
    void setQ(double Q_in); 

    // MPC I
    void setupQpMatrices1(const VecNd &x0); // setup qp problem
    // MPC II
    void setupQpMatrices2(const VecNd &x0); // setup qp problem

    // THIS IS SLOW!! - speed up using sparse matrices
    void updateQpMatrices2(const VecNd &Y_d_in, const VecNd &x0); // update qp problem for new Y_d_in and x0
    
    VecNd calculateX(const VecNd &U_in, const VecNd &x0) const;
    VecNd calculateY(const VecNd &U_in, const VecNd &x0) const;

    void setWx(const MatNd &w_x_in);
    void setWu(const MatNd &w_u_in);

    QpProblem getQpProblem() const ;

  private:

    uint32_t N_; // mpc prediction horizon
    LinearSystem linear_system_; // linear_system
    
    MatNd A_mpc_, B_mpc_, C_mpc_; // mpc dynamics matrices

    VecNd Y_d_; //refrence output
    QpProblem qp_problem_;
    double Q_, R_; 

    MatNd W_u_, w_u_;
    MatNd W_x_, w_x_;

    // matrices saved for faster QP problem update
    MatNd C_A_; // C_mpc * A_mpc
    MatNd C_B_; // C_mpc * B_mpc
    MatNd W_x_A_; // W_x * A_mpc
    MatNd W_x_B_; // W_x * B_mpc
};

}

#endif //EIGENLINEARMPC_H_
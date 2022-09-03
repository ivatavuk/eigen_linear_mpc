#include "EigenLinearMpc.hpp"

// returns input_mat^(power)
MatNd matrixPow(const MatNd &input_mat, uint32_t power) 
{
  MatNd output_mat = MatNd::Identity(input_mat.rows(), input_mat.cols());
  if (power == 0) 
    return output_mat;

  for (int i = 0; i < power; i++) 
    output_mat *= input_mat;

  return output_mat;
}

// -------------- LinearSystem -----------------
EigenLinearMpc::LinearSystem::LinearSystem(const MatNd &A, const MatNd &B, const MatNd &C, const MatNd &D) 
  : A(A), B(B), C(C), D(D), n_x(A.cols()), n_u(B.cols()), n_y(C.rows())
{
  checkMatrixDimensions();
}

void EigenLinearMpc::LinearSystem::checkMatrixDimensions() const 
{
  std::ostringstream msg;

  // Check matrix dimensions
  if ((int)A.rows() != A.cols()) {
    msg << "set_system: Matrix 'A' needs to be a square matrix\n A.dimensions = (" << A.rows() << " x " 
        << A.cols() << ")";
    throw std::logic_error(msg.str());
  }
  if ((int)A.rows() != B.rows()) {
    msg << "set_system: 'A' and 'B' matrices need to have an equal number of rows\n A.rows = " << A.rows() << ", B.rows =  " 
        << B.rows() << ")";
    throw std::logic_error(msg.str());
  }
  if ((int)A.cols() != C.cols()) {
    msg << "set_system: A.cols (" << A.cols() << ") != C.cols (" 
        << C.cols() << ")";
    throw std::logic_error(msg.str());
  }
  if ((int)C.rows() != D.rows()) {
    msg << "set_system: C.rows (" << C.rows() << ") != D.rows (" 
        << D.rows() << ")";
    throw std::logic_error(msg.str());
  }
  if ((int)D.cols() != B.cols()) {
    msg << "set_system: D.cols (" << D.cols() << ") != B.cols (" 
        << B.cols() << ")";
    throw std::logic_error(msg.str());
  }
}

// -------------- MPC -----------------


EigenLinearMpc::MPC::MPC( const LinearSystem &linear_system, uint32_t horizon, 
                          const VecNd &Y_d, const VecNd &x0, double Q, double R ) 
: linear_system_(linear_system), N_(horizon), Y_d_(Y_d), Q_(Q), R_(R), x0_(x0)
{
  mpc_type_ = MPC1;
  setupMpcDynamics();
}

EigenLinearMpc::MPC::MPC( const LinearSystem &linear_system, uint32_t horizon, 
                          const VecNd &Y_d, const VecNd &x0, const MatNd &w_u, 
                          const MatNd &w_x ) 
: linear_system_(linear_system), N_(horizon), Y_d_(Y_d), x0_(x0)
{
  mpc_type_ = MPC2;
  setupMpcDynamics();
  setWu(w_u);
  setWx(w_x);
}

void EigenLinearMpc::MPC::setupMpcDynamics() 
{
  uint32_t n_x = linear_system_.n_x;
  uint32_t n_u = linear_system_.n_u;
  uint32_t n_y = linear_system_.n_y;
  
  B_mpc_ = MatNd::Zero(N_ * n_x, n_x);
  A_mpc_ = MatNd::Zero(N_ * n_x, N_ * n_u);
  C_mpc_ = MatNd::Zero(N_ * n_y, N_ * n_x);

  for (int i = 0; i < N_; i++) 
    B_mpc_.block(i * n_x, 0, n_x, n_x) = matrixPow(linear_system_.A, i+1);
  
  for (int i = 0; i < N_; i++) 
  {
    for (int j = 0; j < N_; j++) 
    {
      if (i == j) 
        A_mpc_.block( n_x * i, n_u * j, n_x, n_u) = linear_system_.B;

      if (i > j) 
      {
        A_mpc_.block( n_x * i, n_u * j, n_x, n_u) = 
        matrixPow(linear_system_.A, i-j) * linear_system_.B;
      }
    }
  }
  for (int i = 0; i < N_; i++) 
  {
    for (int j = 0; j < N_; j++) 
    {
      if (i == j) 
      {
        C_mpc_.block( n_y * i, n_x * j, n_y, n_x) = linear_system_.C;
      }
    }
  }
}

void EigenLinearMpc::MPC::setYd(const VecNd &Y_d_in) 
{
  Y_d_ = Y_d_in;
}

void EigenLinearMpc::MPC::initializeSolver()
{
  if(mpc_type_ == MPC1)
    setupQpMatrices1();
  if(mpc_type_ == MPC2)
    setupQpMatrices2();
}

void EigenLinearMpc::MPC::setupQpMatrices1() 
{
  uint32_t n_u = linear_system_.n_u;
  C_A_ = C_mpc_*A_mpc_;
  C_B_ = C_mpc_*B_mpc_;

  MatNd A_qp = Q_*(C_A_).transpose()*(C_A_) + R_*MatNd::Identity(N_*n_u, N_*n_u);
  VecNd b_qp = (Q_*(C_B_*x0_ - Y_d_).transpose()*C_A_).transpose();
  
  MatNd A_eq = MatNd::Zero(0, N_ * n_u);
  MatNd b_eq = VecNd::Zero(0);
  MatNd A_ieq = MatNd::Zero(0, N_ * n_u);
  MatNd b_ieq = VecNd::Zero(0);

  qp_problem_ = DenseQpProblem(A_qp, b_qp, A_eq, b_eq, A_ieq, b_ieq);
}

void EigenLinearMpc::MPC::setupQpMatrices2() 
{
  // THIS IS SLOW!! - speed up using sparse matrices
  uint32_t n_u = linear_system_.n_u;

  // save temp matrices to reduce redundant computation
  // save these ones!
  C_A_ = C_mpc_*A_mpc_;
  C_B_ = C_mpc_*B_mpc_;
  W_x_B_ = W_x_*B_mpc_;
  W_x_A_ = W_x_*A_mpc_;
  
  //these are temp - dont need those??
  MatNd B_x0 = B_mpc_*x0_;
  MatNd C_B_x0 = C_B_*x0_;
  MatNd W_x_B_x0 = W_x_B_*x0_;
  

  MatNd A_qp = 	Q_*(C_A_).transpose()*(C_A_) 
                + W_u_.transpose()*W_u_ 
                + (W_x_A_).transpose()*(W_x_A_);


  VecNd b_qp = ( Q_*(C_B_x0 - Y_d_).transpose()*C_A_ +
                 (W_x_B_x0).transpose()*(W_x_A_) 
                 ).transpose();

  MatNd A_eq = MatNd::Zero(0, N_ * n_u);
  MatNd b_eq = VecNd::Zero(0);
  MatNd A_ieq = MatNd::Zero(0, N_ * n_u);
  MatNd b_ieq = VecNd::Zero(0);

  qp_problem_ = DenseQpProblem(A_qp, b_qp, A_eq, b_eq, A_ieq, b_ieq);
}

void EigenLinearMpc::MPC::updateQpMatrices2(const VecNd &Y_d_in, const VecNd &x0) 
{
  /* 
    only b_qp vector of the MPC II QP depends on x0 and Y_d 
    (vectors that change from step to step of MPC)
  */
  Y_d_ = Y_d_in;
  VecNd b_qp = (	Q_*(C_B_*x0 - Y_d_).transpose()*C_A_ +
                  (W_x_B_*x0).transpose()*(W_x_A_)
                  ).transpose();
  qp_problem_.b_qp = b_qp;
}

VecNd EigenLinearMpc::MPC::calculateX(const VecNd &U_in) const 
{
  VecNd X = A_mpc_ * U_in + B_mpc_ * x0_;
  return X;
}

VecNd EigenLinearMpc::MPC::calculateY(const VecNd &U_in) const 
{
  VecNd X = calculateX(U_in);
  VecNd Y = C_mpc_ * X;
  return Y;
}

void EigenLinearMpc::MPC::setWu(const MatNd &w_u_in) 
{
  std::ostringstream msg;
  w_u_ = w_u_in;

  uint32_t n_r = w_u_in.rows();
  uint32_t n_c = w_u_in.cols();
  uint32_t n_u = linear_system_.n_u;

  if ((int)n_r != n_c) 
  {
    msg << "set_w_u: Input matrix needs to be a square matrix\n mat.dimensions = (" << n_r << " x " 
        << n_c << ")";
    throw std::logic_error(msg.str());
  }
  if ((int)n_r != n_u) 
  {
    msg << "set_w_u: Input matrix needs to have number of rows equal to n_u\n (" << n_r << " x " 
        << n_u << ")";
    throw std::logic_error(msg.str());
  }

  MatNd W_u_temp = MatNd::Zero(N_*n_u, N_*n_u);

  for (int i = 0; i < N_; i++) 
  {
    for (int j = 0; j < N_; j++) 
    {
      if (i == j) 
      {
        W_u_temp.block( n_r * i, n_c * j, n_r, n_c) = w_u_in;
      }
    }
  }
  W_u_ = W_u_temp;
}

void EigenLinearMpc::MPC::setWx(const MatNd &w_x_in) 
{
  std::ostringstream msg;
  w_x_ = w_x_in;

  uint32_t n_r = w_x_in.rows();
  uint32_t n_c = w_x_in.cols();
  uint32_t n_x = linear_system_.n_x;

  if ((int)n_r != n_c) {
    msg << "set_w_x: Input matrix needs to be a square matrix\n mat.dimensions = (" << n_r << " x " 
        << n_c << ")";
    throw std::logic_error(msg.str());
  }
  if ((int)n_r != n_x) {
    msg << "set_w_x: Input matrix needs to have number of rows equal to n_x\n (" << n_r << " x " 
        << n_x << ")";
    throw std::logic_error(msg.str());
  }

  MatNd W_x_temp = MatNd::Zero(N_*n_x, N_*n_x);

  for (int i = 0; i < N_; i++) 
  {
    for (int j = 0; j < N_; j++) 
    {
      if (i == j) 
      {
        W_x_temp.block( n_r * i, n_c * j, n_r, n_c) = w_x_in;
      }
    }
  }
  W_x_ = W_x_temp;
}

DenseQpProblem EigenLinearMpc::MPC::getQpProblem() const 
{
  return qp_problem_;
}

VecNd EigenLinearMpc::MPC::solve() const 
{
  OsqpEigenOpt osqp_eigen_opt; //TODO how to handle making a solver??..
  osqp_eigen_opt.setupQP(qp_problem_);

  osqp_eigen_opt.initializeSolver(false);
  return osqp_eigen_opt.solveProblem();
} 
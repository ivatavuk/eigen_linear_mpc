#include "EigenLinearMpc.hpp"

void setSparseBlock(Eigen::SparseMatrix<double> &output_matrix, const Eigen::SparseMatrix<double> &input_block,
                    uint32_t i, uint32_t j) 
{
  if((input_block.rows() > output_matrix.rows() - i) || (input_block.cols() > output_matrix.cols() - j))
  {
    std::cout << "input_block.cols() = " << input_block.cols() << "\n";
    std::cout << "input_block.rows() = " << input_block.rows() << "\n";
    std::cout << "output_matrix.cols() - i = " << output_matrix.cols() - i << "\n";
    std::cout << "output_matrix.rows() - j = " << output_matrix.rows() - j << "\n";
    throw std::runtime_error("setSparseBlock: Can't fit block");
  }
  for (int k=0; k < input_block.outerSize(); ++k)
  {
    for (Eigen::SparseMatrix<double>::InnerIterator it(input_block,k); it; ++it)
    {
      output_matrix.insert(it.row() + i, it.col() + j) = it.value();
    }
  }
}

// returns input_mat^(power)
MatNd matrixPow(const MatNd &input_mat, uint32_t power) 
{
  MatNd output_mat = MatNd::Identity(input_mat.rows(), input_mat.cols());

  for (int i = 0; i < power; i++) 
    output_mat = output_mat * input_mat;

  return output_mat;
}
// returns input_mat^(power)
SparseMat matrixPow(const SparseMat &input_mat, uint32_t power) 
{
  SparseMat output_mat(input_mat.rows(), input_mat.cols());
  output_mat.setIdentity();

  for (int i = 0; i < power; i++) 
    output_mat = output_mat * input_mat;
  
  return output_mat;
}

// -------------- LinearSystem -----------------
EigenLinearMpc::LinearSystem::LinearSystem(const SparseMat &A, const SparseMat &B, const SparseMat &C, const SparseMat &D) 
  : A(A), B(B), C(C), D(D), n_x(A.cols()), n_u(B.cols()), n_y(C.rows())
{
  checkMatrixDimensions();
}

void EigenLinearMpc::LinearSystem::checkMatrixDimensions() const 
{
  std::ostringstream msg;

  // Check matrix dimensions
  if ((int)A.rows() != A.cols()) 
  {
    msg << "set_system: Matrix 'A' needs to be a square matrix\n A.dimensions = (" << A.rows() << " x " 
        << A.cols() << ")";
    throw std::logic_error(msg.str());
  }
  if ((int)A.rows() != B.rows()) 
  {
    msg << "set_system: 'A' and 'B' matrices need to have an equal number of rows\n A.rows = " << A.rows() << ", B.rows =  " 
        << B.rows() << ")";
    throw std::logic_error(msg.str());
  }
  if ((int)A.cols() != C.cols()) 
  {
    msg << "set_system: A.cols (" << A.cols() << ") != C.cols (" 
        << C.cols() << ")";
    throw std::logic_error(msg.str());
  }
  if ((int)C.rows() != D.rows()) 
  {
    msg << "set_system: C.rows (" << C.rows() << ") != D.rows (" 
        << D.rows() << ")";
    throw std::logic_error(msg.str());
  }
  if ((int)D.cols() != B.cols()) 
  {
    msg << "set_system: D.cols (" << D.cols() << ") != B.cols (" 
        << B.cols() << ")";
    throw std::logic_error(msg.str());
  }
}

// -------------- MPC -----------------


EigenLinearMpc::MPC::MPC( const LinearSystem &linear_system, uint32_t horizon, 
                          const VecNd &Y_d, const VecNd &x0, double Q, double R ) 
: linear_system_(linear_system), N_(horizon), Y_d_(Y_d), Q_(Q), R_(R), x0_(x0),
  A_mpc_(SparseMat(N_ * linear_system_.n_x, N_ * linear_system_.n_u)),
  B_mpc_(SparseMat(N_ * linear_system_.n_x, linear_system_.n_x)),
  C_mpc_(SparseMat(N_ * linear_system_.n_y, N_ * linear_system_.n_x))
{
  mpc_type_ = MPC1;
  checkMatrixDimensions();
  setupMpcDynamics();
}

EigenLinearMpc::MPC::MPC( const LinearSystem &linear_system, uint32_t horizon, 
                          const VecNd &Y_d, const VecNd &x0, double Q, double R,
                          const VecNd &u_lower_bound, const VecNd &u_upper_bound,
                          const VecNd &x_lower_bound, const VecNd &x_upper_bound ) 
: linear_system_(linear_system), N_(horizon), Y_d_(Y_d), Q_(Q), R_(R), x0_(x0),
  u_lower_bound_(u_lower_bound), u_upper_bound_(u_upper_bound),
  x_lower_bound_(x_lower_bound), x_upper_bound_(x_upper_bound)
{
  mpc_type_ = MPC1_BOUND_CONSTRAINED;
  checkBoundsDimensions();
  checkMatrixDimensions();
  setupMpcDynamics();
}

EigenLinearMpc::MPC::MPC( const LinearSystem &linear_system, uint32_t horizon, 
                          const VecNd &Y_d, const VecNd &x0, const MatNd &w_u, 
                          const MatNd &w_x ) 
: linear_system_(linear_system), N_(horizon), Y_d_(Y_d), x0_(x0), w_u_(w_u), w_x_(w_x)
{
  mpc_type_ = MPC2;
  checkMatrixDimensions();
  setupMpcDynamics();
  setWu();
  setWx();
}

void EigenLinearMpc::MPC::checkMatrixDimensions() const 
{
  std::ostringstream msg;

  // Check matrix dimensions
  if ((int)Y_d_.rows() != N_ * linear_system_.n_y) 
  {
    msg << "MPC: Vector 'Y_d' size error\n Y_d_.rows() = " << Y_d_.rows() 
        << ", needs to be = " << N_ * linear_system_.n_y << "\n";
    throw std::logic_error(msg.str());
  }
  if ((int)x0_.rows() != linear_system_.n_x) 
  {
    msg << "MPC: Vector 'x0' size error\n x0.rows() = " << x0_.rows() 
        << ", needs to be = " << linear_system_.n_x << "\n";
    throw std::logic_error(msg.str());
  }
}

void EigenLinearMpc::MPC::checkBoundsDimensions() const 
{
  std::ostringstream msg;

  if ((int)u_lower_bound_.rows() != linear_system_.n_u) 
  {
    msg << "MPC: Vector 'u_lower_bounds_' size error\n lower_bounds_.rows() = " << u_lower_bound_.rows() 
        << ", needs to be = " << linear_system_.n_u << "\n";
    throw std::logic_error(msg.str());
  }
  if ((int)u_upper_bound_.rows() != linear_system_.n_u) 
  {
    msg << "MPC: Vector 'u_upper_bound_' size error\n lower_bounds_.rows() = " << u_upper_bound_.rows() 
        << ", needs to be = " << linear_system_.n_u << "\n";
    throw std::logic_error(msg.str());
  }
  if ((int)x_lower_bound_.rows() != linear_system_.n_x) 
  {
    msg << "MPC: Vector 'x_lower_bound_' size error\n lower_bounds_.rows() = " << x_lower_bound_.rows() 
        << ", needs to be = " << linear_system_.n_x << "\n";
    throw std::logic_error(msg.str());
  }
  if ((int)x_upper_bound_.rows() != linear_system_.n_x) 
  {
    msg << "MPC: Vector 'x_upper_bound_' size error\n lower_bounds_.rows() = " << x_upper_bound_.rows() 
        << ", needs to be = " << linear_system_.n_x << "\n";
    throw std::logic_error(msg.str());
  }
}

void EigenLinearMpc::MPC::setupMpcDynamics() 
{
  uint32_t n_x = linear_system_.n_x;
  uint32_t n_u = linear_system_.n_u;
  uint32_t n_y = linear_system_.n_y;
  
  for (int i = 0; i < N_; i++) 
  {
    setSparseBlock(B_mpc_, linear_system_.A, n_x * i, 0);
    setSparseBlock(C_mpc_, linear_system_.C, n_y * i, n_x * i);
    for (int j = 0; j <= i; j++) 
    {
      if (i == j) 
      {
        setSparseBlock(A_mpc_, linear_system_.B, n_x * i, n_u * j);
      }
      else
      {
        setSparseBlock(A_mpc_, matrixPow(linear_system_.A, i-j) * linear_system_.B, n_x * i, n_u * j);
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
    setupQpMPC1();
  if(mpc_type_ == MPC2)
    setupQpMPC2();
  if(mpc_type_ == MPC1_BOUND_CONSTRAINED)
    setupQpBoundConstrainedMPC1();
  if(mpc_type_ == MPC2_BOUND_CONSTRAINED){}
}

void EigenLinearMpc::MPC::setupQpMPC1() 
{
  uint32_t n_u = linear_system_.n_u;
  C_A_ = C_mpc_*A_mpc_;
  C_B_ = C_mpc_*B_mpc_;

  SparseMat A_qp = Q_*(C_A_).transpose()*(C_A_) + R_*MatNd::Identity(N_*n_u, N_*n_u);
  VecNd b_qp = (Q_*(C_B_*x0_ - Y_d_).transpose()*C_A_).transpose();
  
  SparseMat A_eq(0, N_ * n_u);
  auto b_eq = VecNd::Zero(0);
  SparseMat A_ieq(0, N_ * n_u);
  auto b_ieq = VecNd::Zero(0);

  qp_problem_ = SparseQpProblem(A_qp, b_qp, A_eq, b_eq, A_ieq, b_ieq); //TODO: make_unique
  osqp_eigen_opt_ = std::make_unique<OsqpEigenOpt>(qp_problem_);
}

void EigenLinearMpc::MPC::setupQpBoundConstrainedMPC1() 
{
  uint32_t n_u = linear_system_.n_u;
  C_A_ = C_mpc_*A_mpc_;
  C_B_ = C_mpc_*B_mpc_;

  SparseMat A_qp = Q_*(C_A_).transpose()*(C_A_) + R_*MatNd::Identity(N_*n_u, N_*n_u);
  VecNd b_qp = (Q_*(C_B_*x0_ - Y_d_).transpose()*C_A_).transpose();
  
  SparseMat A_eq(0, N_ * n_u);
  VecNd b_eq = VecNd::Zero(0);
  SparseMat A_ieq(0, N_ * n_u);
  VecNd b_ieq = VecNd::Zero(0);

  qp_problem_ = DenseQpProblem(A_qp, b_qp, A_eq, b_eq, A_ieq, b_ieq);
  osqp_eigen_opt_ = std::make_unique<OsqpEigenOpt>(qp_problem_);
}

void EigenLinearMpc::MPC::setupQpMPC2() 
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
  

  SparseMat A_qp = 	Q_*(C_A_).transpose()*(C_A_) 
                    + W_u_.transpose()*W_u_ 
                    + (W_x_A_).transpose()*(W_x_A_);


  VecNd b_qp = ( Q_*(C_B_x0 - Y_d_).transpose()*C_A_ +
                 (W_x_B_x0).transpose()*(W_x_A_) 
                 ).transpose();

  SparseMat A_eq(0, N_ * n_u);
  VecNd b_eq = VecNd::Zero(0);
  SparseMat A_ieq(0, N_ * n_u);
  VecNd b_ieq = VecNd::Zero(0);

  qp_problem_ = SparseQpProblem(A_qp, b_qp, A_eq, b_eq, A_ieq, b_ieq);
  osqp_eigen_opt_ = std::make_unique<OsqpEigenOpt>(qp_problem_);
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

void EigenLinearMpc::MPC::setWu() 
{
  std::ostringstream msg;

  uint32_t n_r = w_u_.rows();
  uint32_t n_c = w_u_.cols();
  uint32_t n_u = linear_system_.n_u;

  if ((int)n_r != n_c) 
  {
    msg << "set_w_u: Input matrix needs to be a square matrix\n mat.dimensions = (" << n_r << " != " 
        << n_c << ")";
    throw std::logic_error(msg.str());
  }
  if ((int)n_r != n_u) 
  {
    msg << "set_w_u: Input matrix needs to have number of rows equal to n_u\n (" << n_r << " != " 
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
        W_u_temp.block( n_r * i, n_c * j, n_r, n_c) = w_u_;
      }
    }
  }
  W_u_ = W_u_temp;
}

void EigenLinearMpc::MPC::setWx() 
{
  std::ostringstream msg;

  uint32_t n_r = w_x_.rows();
  uint32_t n_c = w_x_.cols();
  uint32_t n_x = linear_system_.n_x;

  if ((int)n_r != n_c) {
    msg << "set_w_x: Input matrix needs to be a square matrix\n mat.dimensions = (" << n_r << " != " 
        << n_c << ")";
    throw std::logic_error(msg.str());
  }
  if ((int)n_r != n_x) {
    msg << "set_w_x: Input matrix needs to have number of rows equal to n_x\n (" << n_r << " != " 
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
        W_x_temp.block( n_r * i, n_c * j, n_r, n_c) = w_x_;
      }
    }
  }
  W_x_ = W_x_temp;
}

SparseQpProblem EigenLinearMpc::MPC::getQpProblem() const 
{
  return qp_problem_;
}

VecNd EigenLinearMpc::MPC::solve() const 
{
  osqp_eigen_opt_->initializeSolver(false);
  return osqp_eigen_opt_->solveProblem();
} 
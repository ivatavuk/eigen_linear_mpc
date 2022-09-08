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

SparseMat cocatenateMatrices(SparseMat mat_upper, SparseMat mat_lower)
{
  SparseMat M(mat_upper.rows() + mat_lower.rows(), mat_lower.cols());
  M.reserve(mat_upper.nonZeros() + mat_lower.nonZeros());
  for(Eigen::Index i = 0; i < mat_upper.cols(); i++)
  {
      M.startVec(i); // Important: Must be called once for each column before inserting!
      for(SparseMat::InnerIterator itUpper(mat_upper, i); itUpper; ++itUpper)
          M.insertBack(itUpper.row(), i) = itUpper.value();
      for(SparseMat::InnerIterator itLower(mat_lower, i); itLower; ++itLower)
          M.insertBack(itLower.row() + mat_upper.rows(), i) = itLower.value();
  }
  M.finalize();
  return M;
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
    throw std::runtime_error(msg.str());
  }
  if ((int)A.rows() != B.rows()) 
  {
    msg << "set_system: 'A' and 'B' matrices need to have an equal number of rows\n A.rows = " << A.rows() << ", B.rows =  " 
        << B.rows() << ")";
    throw std::runtime_error(msg.str());
  }
  if ((int)A.cols() != C.cols()) 
  {
    msg << "set_system: A.cols (" << A.cols() << ") != C.cols (" 
        << C.cols() << ")";
    throw std::runtime_error(msg.str());
  }
  if ((int)C.rows() != D.rows()) 
  {
    msg << "set_system: C.rows (" << C.rows() << ") != D.rows (" 
        << D.rows() << ")";
    throw std::runtime_error(msg.str());
  }
  if ((int)D.cols() != B.cols()) 
  {
    msg << "set_system: D.cols (" << D.cols() << ") != B.cols (" 
        << B.cols() << ")";
    throw std::runtime_error(msg.str());
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
                          const VecNd &u_lower_bound, const VecNd &u_upper_bound ) 
: linear_system_(linear_system), N_(horizon), Y_d_(Y_d), Q_(Q), R_(R), x0_(x0),
  u_lower_bound_(u_lower_bound), u_upper_bound_(u_upper_bound),
  A_mpc_(SparseMat(N_ * linear_system_.n_x, N_ * linear_system_.n_u)),
  B_mpc_(SparseMat(N_ * linear_system_.n_x, linear_system_.n_x)),
  C_mpc_(SparseMat(N_ * linear_system_.n_y, N_ * linear_system_.n_x))
{
  mpc_type_ = MPC1_BOUND_CONSTRAINED;
  checkBoundsDimensions();
  checkMatrixDimensions();
  setupMpcDynamics();
}

EigenLinearMpc::MPC::MPC( const LinearSystem &linear_system, uint32_t horizon, 
                          const VecNd &Y_d, const VecNd &x0, double W_y, 
                          const SparseMat &w_u, const SparseMat &w_x ) 
: linear_system_(linear_system), N_(horizon), Y_d_(Y_d), x0_(x0), 
  W_y_(W_y), w_u_(w_u), w_x_(w_x),
  A_mpc_(SparseMat(N_ * linear_system_.n_x, N_ * linear_system_.n_u)),
  B_mpc_(SparseMat(N_ * linear_system_.n_x, linear_system_.n_x)),
  C_mpc_(SparseMat(N_ * linear_system_.n_y, N_ * linear_system_.n_x)),
  W_u_(SparseMat(N_ * linear_system_.n_u, N_ * linear_system_.n_u)),
  W_x_(SparseMat(N_ * linear_system_.n_x, N_ * linear_system_.n_x))

{
  mpc_type_ = MPC2;
  checkMatrixDimensions();
  setupMpcDynamics();
  setWeightMatrices();
}

void EigenLinearMpc::MPC::checkMatrixDimensions() const 
{
  std::ostringstream msg;

  // Check matrix dimensions
  if ((int)Y_d_.rows() != N_ * linear_system_.n_y) 
  {
    msg << "MPC: Vector 'Y_d' size error\n Y_d_.rows() = " << Y_d_.rows() 
        << ", needs to be = " << N_ * linear_system_.n_y << "\n";
    throw std::runtime_error(msg.str());
  }
  if ((int)x0_.rows() != linear_system_.n_x) 
  {
    msg << "MPC: Vector 'x0' size error\n x0.rows() = " << x0_.rows() 
        << ", needs to be = " << linear_system_.n_x << "\n";
    throw std::runtime_error(msg.str());
  }
}

void EigenLinearMpc::MPC::checkBoundsDimensions() const 
{
  std::ostringstream msg;

  if ((int)u_lower_bound_.rows() != linear_system_.n_u) 
  {
    msg << "MPC: Vector 'u_lower_bounds_' size error\n lower_bounds_.rows() = " << u_lower_bound_.rows() 
        << ", needs to be = " << linear_system_.n_u << "\n";
    throw std::runtime_error(msg.str());
  }
  if ((int)u_upper_bound_.rows() != linear_system_.n_u) 
  {
    msg << "MPC: Vector 'u_upper_bound_' size error\n lower_bounds_.rows() = " << u_upper_bound_.rows() 
        << ", needs to be = " << linear_system_.n_u << "\n";
    throw std::runtime_error(msg.str());
  }
  /*
  if ((int)x_lower_bound_.rows() != linear_system_.n_x) 
  {
    msg << "MPC: Vector 'x_lower_bound_' size error\n lower_bounds_.rows() = " << x_lower_bound_.rows() 
        << ", needs to be = " << linear_system_.n_x << "\n";
    throw std::runtime_error(msg.str());
  }
  if ((int)x_upper_bound_.rows() != linear_system_.n_x) 
  {
    msg << "MPC: Vector 'x_upper_bound_' size error\n lower_bounds_.rows() = " << x_upper_bound_.rows() 
        << ", needs to be = " << linear_system_.n_x << "\n";
    throw std::runtime_error(msg.str());
  }
  */
}

void EigenLinearMpc::MPC::setupMpcDynamics() 
{
  uint32_t n_x = linear_system_.n_x;
  uint32_t n_u = linear_system_.n_u;
  uint32_t n_y = linear_system_.n_y;
  
  for (int i = 0; i < N_; i++) 
  {
    setSparseBlock(B_mpc_, matrixPow(linear_system_.A, i+1), n_x * i, 0);
    setSparseBlock(C_mpc_, linear_system_.C, n_y * i, n_x * i);
    for (int j = 0; j <= i; j++) 
    {
      if (i == j) 
        setSparseBlock(A_mpc_, linear_system_.B, n_x * i, n_u * j);
      else
        setSparseBlock(A_mpc_, matrixPow(linear_system_.A, i-j) * linear_system_.B, n_x * i, n_u * j);
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
  {
    setupQpConstrainedMPC1();
  }
}

void EigenLinearMpc::MPC::updateSolver(const VecNd &Y_d_in, const VecNd &x0)
{
  Y_d_ = Y_d_in;
  x0_ = x0;
  if(mpc_type_ == MPC1 || mpc_type_ == MPC1_BOUND_CONSTRAINED)
    updateQpMPC1();
  if(mpc_type_ == MPC2 || mpc_type_ == MPC2_BOUND_CONSTRAINED)
    updateQpMPC2();
}

VecNd EigenLinearMpc::MPC::solve() const 
{
  return osqp_eigen_opt_->solveProblem();
} 

void EigenLinearMpc::MPC::setupQpMPC1() 
{
  uint32_t n_u = linear_system_.n_u;
  C_A_ = C_mpc_*A_mpc_;
  C_B_ = C_mpc_*B_mpc_;
  Q_C_A_T_ = Q_*(C_A_).transpose();
  Q_C_A_T_C_B_ = Q_*(C_A_).transpose()*(C_B_);

  SparseMat A_qp = Q_*(C_A_).transpose()*(C_A_) + R_*MatNd::Identity(N_*n_u, N_*n_u); //Sparse identity
  VecNd b_qp = Q_C_A_T_C_B_ * x0_ - Q_C_A_T_*Y_d_;

  SparseMat A_eq(0, N_ * n_u);
  VecNd b_eq = VecNd::Zero(0);
  SparseMat A_ieq(0, N_ * n_u);
  VecNd b_ieq = VecNd::Zero(0);

  qp_problem_ = std::make_unique<SparseQpProblem>(A_qp, b_qp, A_eq, b_eq, A_ieq, b_ieq);
  osqp_eigen_opt_ = std::make_unique<OsqpEigenOpt>(*qp_problem_);
}

void EigenLinearMpc::MPC::updateQpMPC1() 
{
  VecNd b_qp = Q_C_A_T_C_B_ * x0_ - Q_C_A_T_*Y_d_;

  qp_problem_->b_qp = b_qp;
  osqp_eigen_opt_->setGradientAndInit(b_qp);
}

void EigenLinearMpc::MPC::setupQpConstrainedMPC1() 
{
  uint32_t n_u = linear_system_.n_u;
  C_A_ = C_mpc_*A_mpc_;
  C_B_ = C_mpc_*B_mpc_;
  Q_C_A_T_ = Q_*(C_A_).transpose();
  Q_C_A_T_C_B_ = Q_*(C_A_).transpose()*(C_B_);

  SparseMat A_qp = Q_*(C_A_).transpose()*(C_A_) + R_*MatNd::Identity(N_*n_u, N_*n_u); //Sparse identity
  VecNd b_qp = Q_C_A_T_C_B_ * x0_ - Q_C_A_T_*Y_d_;

  SparseMat A_eq(0, N_ * n_u);
  VecNd b_eq = VecNd::Zero(0);
  SparseMat A_ieq(0, N_ * n_u);
  VecNd b_ieq = VecNd::Zero(0);
  
  qp_problem_ = std::make_unique<SparseQpProblem>(A_qp, b_qp, A_eq, b_eq, A_ieq, b_ieq, 
                                                  u_lower_bound_.colwise().replicate(N_), 
                                                  u_upper_bound_.colwise().replicate(N_));
  osqp_eigen_opt_ = std::make_unique<OsqpEigenOpt>(*qp_problem_);
}

void EigenLinearMpc::MPC::setupQpMPC2() 
{
  uint32_t n_u = linear_system_.n_u;
  // save intermediate product matrices to reduce redundant computation
  C_A_ = C_mpc_*A_mpc_;
  C_B_ = C_mpc_*B_mpc_;
  W_x_B_ = W_x_*B_mpc_;
  W_x_A_ = W_x_*A_mpc_;

  SparseMat A_qp = 	W_y_*(C_A_).transpose()*(C_A_) 
                    + W_u_.transpose()*W_u_ 
                    + (W_x_A_).transpose()*(W_x_A_);

  VecNd b_qp = ( W_y_*(C_B_*x0_ - Y_d_).transpose()*C_A_ +
                 (W_x_B_*x0_).transpose()*(W_x_A_) 
                 ).transpose();

  SparseMat A_eq(0, N_ * n_u);
  VecNd b_eq = VecNd::Zero(0);
  SparseMat A_ieq(0, N_ * n_u);
  VecNd b_ieq = VecNd::Zero(0);

  qp_problem_ = std::make_unique<SparseQpProblem>(A_qp, b_qp, A_eq, b_eq, A_ieq, b_ieq);
  osqp_eigen_opt_ = std::make_unique<OsqpEigenOpt>(*qp_problem_);
}

void EigenLinearMpc::MPC::updateQpMPC2() 
{
  VecNd b_qp = (	W_y_*(C_B_*x0_ - Y_d_).transpose()*C_A_ +
                  (W_x_B_*x0_).transpose()*(W_x_A_)
                  ).transpose();
  qp_problem_->b_qp = b_qp;
  osqp_eigen_opt_->setGradientAndInit(b_qp);
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

std::vector< std::vector<double> > EigenLinearMpc::MPC::extractU(const VecNd &U_in) const 
{
  std::vector<std::vector<double>> return_vector_U;
  for(uint32_t i = 0; i < linear_system_.n_u; i++)
    return_vector_U.push_back( std::vector<double>() );

  for(uint32_t i = 0; i < U_in.rows(); i++)
  {
    uint32_t mod = i % linear_system_.n_u;
    return_vector_U[mod].push_back(U_in(i));
  }
  return return_vector_U;
}

std::vector< std::vector<double> > EigenLinearMpc::MPC::extractX(const VecNd &U_in) const
{
  auto X = calculateX(U_in);
  std::vector<std::vector<double>> return_vector_X;
  for(uint32_t i = 0; i < linear_system_.n_x; i++)
    return_vector_X.push_back( std::vector<double>() );

  for(uint32_t i = 0; i < X.rows(); i++)
  {
    uint32_t mod = i % linear_system_.n_x;
    return_vector_X[mod].push_back(X(i));
  }
  return return_vector_X;
} 

void EigenLinearMpc::MPC::setWeightMatrices() 
{
  checkWeightDimensions();
  for (int i = 0; i < N_; i++) 
  {
    setSparseBlock(W_u_, w_u_, linear_system_.n_u * i, linear_system_.n_u * i);
    setSparseBlock(W_x_, w_x_, linear_system_.n_x * i, linear_system_.n_x * i);
  }
}

void EigenLinearMpc::MPC::checkWeightDimensions() const
{
  std::ostringstream msg;
  if ((int)w_u_.rows() != w_u_.cols()) 
  {
    msg << "set_w_u: Input matrix needs to be a square matrix\n mat.dimensions = (" << w_u_.rows() << " != " 
        << w_u_.cols() << ")";
    throw std::runtime_error(msg.str());
  }
  if ((int)w_u_.rows() != linear_system_.n_u) 
  {
    msg << "set_w_u: Input matrix needs to have number of rows equal to n_u\n (" << w_u_.rows() << " != " 
        << linear_system_.n_u << ")";
    throw std::runtime_error(msg.str());
  }
  if ((int)w_x_.rows() != w_x_.cols()) {
    msg << "set_w_x: Input matrix needs to be a square matrix\n mat.dimensions = (" << w_x_.rows() << " != " 
        << w_x_.cols() << ")";
    throw std::runtime_error(msg.str());
  }
  if ((int)w_x_.rows() != linear_system_.n_x) {
    msg << "set_w_x: Input matrix needs to have number of rows equal to n_x\n (" << w_x_.rows() << " != " 
        << linear_system_.n_x << ")";
    throw std::runtime_error(msg.str());
  }
}
#include "EigenLinearMpc.hpp"
#include "matplotlibcpp.hpp"
#include "QpProblem.hpp"
#include "ChronoCall.hpp"

namespace plt = matplotlibcpp;  

Eigen::VectorXd generate_lawnmower_vec(	uint32_t len, uint32_t lawnmower_period, 
                              double lawnmower_max, double lawnmower_min );
std::vector<double> eigen2stdVec(	Eigen::VectorXd eigen_vec );

int main()
{
  /** Define linear system
   * x = [px, dpx]^T
   * u = [ddpx]^T
   * 
   * y = [px]^T
   */
  double T = 0.1;
  Eigen::MatrixXd A(2, 2);
  A <<  1, T,
        0, 1;

  Eigen::MatrixXd B(2, 1);
  B <<  T*T/2.0,
        T;

  Eigen::MatrixXd C(1, 2);
  C <<  1, 0;
  
  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(1, 1);
  
  Eigen::SparseMatrix<double> A_sparse = SparseQpProblem::sparseMatrixFromDense(A);
  Eigen::SparseMatrix<double> B_sparse = SparseQpProblem::sparseMatrixFromDense(B);
  Eigen::SparseMatrix<double> C_sparse = SparseQpProblem::sparseMatrixFromDense(C);
  Eigen::SparseMatrix<double> D_sparse = SparseQpProblem::sparseMatrixFromDense(D);

  EigenLinearMpc::LinearSystem example_system(A_sparse, B_sparse, C_sparse, D_sparse);
  uint32_t horizon = 200;
  double Q = 10000.0;
  double R = 5.0;
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);
  x0 << 1, 0;
  // define lawnmower reference
  Eigen::VectorXd Y_d = generate_lawnmower_vec(horizon, 20, 1.0, 0.0);

  EigenLinearMpc::MPC mpc(example_system, horizon, Y_d, x0, Q, R);
  ChronoCall(
    mpc.initializeSolver();
  );
  auto U_sol = mpc.solve();

  Eigen::VectorXd y;
  ChronoCall(microseconds,
    y = mpc.calculateY(U_sol);
  );

  plt::plot(eigen2stdVec(Y_d));
  plt::plot(eigen2stdVec(y));
  plt::show();
  
  return 0;
}

Eigen::VectorXd generate_lawnmower_vec(	uint32_t len, uint32_t lawnmower_period, 
                                        double lawnmower_max, double lawnmower_min )
{
  Eigen::VectorXd lawnmower_vec(len);

  for(uint32_t i = 0; i < len; i++)
    lawnmower_vec(i) = !((i / lawnmower_period) % 2) ? lawnmower_max : lawnmower_min;

  return lawnmower_vec;
}

std::vector<double> eigen2stdVec(	Eigen::VectorXd eigen_vec )
{
  std::vector<double> ret_vec;
  for(uint32_t i = 0; i < eigen_vec.rows(); i++)
    ret_vec.push_back(eigen_vec(i));
  return ret_vec;
}
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

  // define lawnmower reference
  uint32_t n_simulate_steps = 30;
  uint32_t horizon = 200;
  double Q = 10000.0;
  double R = 3.0;
  Eigen::VectorXd Y_d_full = generate_lawnmower_vec(horizon + n_simulate_steps, 20, 1.0, 0.0);

  /**                   Define linear system
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

  EigenLinearMpc::LinearSystem example_system(SparseQpProblem::sparseMatrixFromDense(A), 
                                              SparseQpProblem::sparseMatrixFromDense(B), 
                                              SparseQpProblem::sparseMatrixFromDense(C), 
                                              SparseQpProblem::sparseMatrixFromDense(D));

  
  Eigen::VectorXd x0(2);
  x0 << 1, 0;
  
  VecNd Y_d = Y_d_full.segment(0, horizon);

  VecNd u_lower_bound(1);
  u_lower_bound << -10;
  VecNd u_upper_bound(1);
  u_upper_bound << 10;

  EigenLinearMpc::MPC mpc(example_system, horizon, Y_d, x0, Q, R, u_lower_bound, u_upper_bound);
  VecNd U_sol;
  for(uint32_t i = 0; i < n_simulate_steps; i++)
  {
    if(i == 0)
    {
      std::cout << "First solver initialization:\n";
      ChronoCall(microseconds,
        mpc.initializeSolver();
      );
    }
    else
    { 
      std::cout << "i = " << i << "\n";
      Y_d = Y_d_full.segment(i, horizon);
      x0 = mpc.calculateX(U_sol).segment(0, 2);
      std::cout << "Updating MPC:\n";
      ChronoCall(microseconds,
        mpc.updateSolver(Y_d, x0);
      );
    }
    
    std::cout << "Solving:\n";
    ChronoCall(microseconds,
      U_sol = mpc.solve();
    );
    plt::plot(eigen2stdVec(U_sol));
    plt::show();

    plt::plot(eigen2stdVec(Y_d));
    plt::plot(eigen2stdVec(mpc.calculateY(U_sol))); //Y does not show current point!!
    plt::show();
  }
  
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
#include "EigenLinearMpc.hpp"
#include "matplotlibcpp.hpp"
#include "QpProblem.hpp"
#include "ChronoCall.hpp"

namespace plt = matplotlibcpp;  

Eigen::VectorXd generate_ramp_vec(	uint32_t len, uint32_t ramp_half_period, double ramp_rate );
std::vector<double> eigen2stdVec(	Eigen::VectorXd eigen_vec );

int main()
{

  // define lawnmower reference
  uint32_t n_simulate_steps = 30;
  uint32_t horizon = 40;
  double Q = 10000.0;
  double R = 1.0;
  Eigen::VectorXd Y_d_full = generate_ramp_vec(horizon + n_simulate_steps, 20, 0.1);

  /**                   Define linear system
   * x = [px, dpx]^T
   * u = [ddpx]^T
   * 
   * y = [px]^T
   */
  double T = 0.1;
  Eigen::MatrixXd A(4, 4);
  A <<  1, 0, T, 0,
        0, 1, 0, T,
        0, 0, 1, 0,
        0, 0, 0, 1;

  Eigen::MatrixXd B(4, 2);
  B <<  T*T/2.0,  0,
        0,        T*T/2.0,
        T,        0,
        0,        T;

  Eigen::MatrixXd C(1, 4);
  C <<  1, 1, 0, 0;
  
  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(1, 2);

  EigenLinearMpc::LinearSystem example_system(A.sparseView(), 
                                              B.sparseView(), 
                                              C.sparseView(), 
                                              D.sparseView());

  
  Eigen::VectorXd x0(4);
  x0 << 0, 0, 0, 0;
  
  VecNd Y_d = Y_d_full.segment(0, horizon);

  VecNd u_lower_bound(2);
  u_lower_bound << -7, -2;
  VecNd u_upper_bound(2);
  u_upper_bound << 7, 2;

  //EigenLinearMpc::MPC mpc(example_system, horizon, Y_d, x0, Q, R, u_lower_bound, u_upper_bound);
  double W_y = 8000;
  double wBddx = 80;
  double wAddx = 8;
  double wAx = 10;

  MatNd w_u(2, 2); 
  w_u <<  wBddx,  0,
          0,      wAddx;

  MatNd w_x(4, 4);
  w_x <<  0, 0,   0, 0,
          0, wAx, 0, 0,
          0, 0,   0, 0,
          0, 0,   0, 0;

  SparseMat w_u_sparse = w_u.sparseView();
  SparseMat w_x_sparse = w_x.sparseView();

  EigenLinearMpc::MPC mpc(example_system, horizon, Y_d, x0, W_y, w_u_sparse, w_x_sparse);
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

    for(auto curr_U : mpc.extractU(U_sol))
      plt::plot(curr_U);

    plt::show();

    auto x_vectors = mpc.extractX(U_sol);
    for(uint32_t i = 0; i < 2; i++)
      plt::plot(x_vectors[i]);

    plt::plot(eigen2stdVec(Y_d));
    plt::plot(eigen2stdVec(mpc.calculateY(U_sol))); //Y does not show current point!!
    plt::show();
  }
  
  return 0;
}

Eigen::VectorXd generate_ramp_vec(	uint32_t len, uint32_t ramp_half_period, double ramp_rate )
{
  Eigen::VectorXd lawnmower_vec(len);

  lawnmower_vec(0) = 0.0;
  for(uint32_t i = 1; i < len; i++)
  {
    lawnmower_vec(i) = !((i / ramp_half_period) % 2) ?  lawnmower_vec(i-1) : lawnmower_vec(i-1) + ramp_rate;
  }

  return lawnmower_vec;
}

std::vector<double> eigen2stdVec(	Eigen::VectorXd eigen_vec )
{
  std::vector<double> ret_vec;
  for(uint32_t i = 0; i < eigen_vec.rows(); i++)
    ret_vec.push_back(eigen_vec(i));
  return ret_vec;
}
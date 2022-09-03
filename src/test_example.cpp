#include "EigenLinearMpc.hpp"
#include "matplotlibcpp.hpp"

#include "ChronoCall.hpp"

namespace plt = matplotlibcpp;  

VecNd generate_lawnmower_vec(	uint32_t len, uint32_t lawnmower_period, 
                              double lawnmower_max, double lawnmower_min );
std::vector<double> eigen2stdVec(	VecNd eigen_vec );

int main()
{
  /** Define linear system
   * x = [px, dpx]^T
   * u = [ddpx]^T
   * 
   * y = [px]^T
   */
  double T = 0.1;
  MatNd A(2, 2);
  A <<  1, T,
        0, 1;

  MatNd B(2, 1);
  B <<  T*T/2.0,
        T;

  MatNd C(1, 2);
  C <<  1, 0;
  
  MatNd D = MatNd::Zero(1, 1);

  EigenLinearMpc::LinearSystem test_system(A, B, C, D);
  uint32_t horizon = 200;
  double Q = 10000.0;
  double R = 5.0;
  VecNd x0 = VecNd::Zero(2);
  // define lawnmower reference
  VecNd Y_d = generate_lawnmower_vec(horizon, 20, 1.0, 0.0);

  EigenLinearMpc::MPC mpc(test_system, horizon, Y_d, x0, Q, R);
  ChronoCall(
    mpc.initializeSolver();
  );
  auto U_sol = mpc.solve();

  VecNd y;
  ChronoCall(microseconds,
    y = mpc.calculateY(U_sol);
  );

  plt::plot(eigen2stdVec(Y_d));
  plt::plot(eigen2stdVec(y));
  plt::show();

  return 0;
}

VecNd generate_lawnmower_vec(	uint32_t len, uint32_t lawnmower_period, 
                              double lawnmower_max, double lawnmower_min )
{
  VecNd lawnmower_vec(len);

  for(uint32_t i = 0; i < len; i++)
    lawnmower_vec(i) = !((i / lawnmower_period) % 2) ? lawnmower_max : lawnmower_min;

  return lawnmower_vec;
}

std::vector<double> eigen2stdVec(	VecNd eigen_vec )
{
  std::vector<double> ret_vec;
  for(uint32_t i = 0; i < eigen_vec.rows(); i++)
    ret_vec.push_back(eigen_vec(i));
  return ret_vec;
}
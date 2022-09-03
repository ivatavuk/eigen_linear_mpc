#include "EigenLinearMpc.hpp"
#include "matplotlibcpp.hpp"

namespace plt = matplotlibcpp;  

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


int main()
{
  std::cout << "testing started...\n";

  uint32_t horizon = 100;
  uint32_t ref_period = 20;
  
  // define lawnmower reference
  VecNd lawnmower_vec = generate_lawnmower_vec(horizon, ref_period, 1.0, 0.0);

  // define linear system
  /**
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

  EigenLinearMpc::MPC mpc(test_system, horizon);
  mpc.setYd(lawnmower_vec); //TODO ref in constructor?
  mpc.setQandR(10000.0, 5.0); 
  VecNd x0 = VecNd::Zero(2);
  mpc.setupQpMatrices1(x0); //TODO check x0 dimension and also in constructor
  auto U_sol = mpc.solve();
  auto y = mpc.calculateY(U_sol, x0);

  plt::plot(eigen2stdVec(lawnmower_vec));
  plt::plot(eigen2stdVec(y));
  plt::show();

  return 0;
}
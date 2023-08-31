# Linear reference tracking MPC library using Eigen and OSQP

Linear reference tracking MPC library using Eigen linear algebra library, OSQP quadratic programming solver and the OsqpEigen wrapper for OSQP.

This is a work in progress.

The library supports two versions of the reference tracking linear MPC:

#### MPC 1:

$$\begin{equation}
\begin{aligned}
& \underset{\boldsymbol{U}}{\text{min}} & &
Q||\boldsymbol{Y} - \boldsymbol{Y}_d||^2 + R||\boldsymbol{U}||^2\\
& \text{s.t.} & & \boldsymbol{x}(k+1) =
\boldsymbol{A} \boldsymbol{x}(k) +
\boldsymbol{B} \boldsymbol{u}(k) \\
& & & \boldsymbol{y}(k) =
\boldsymbol{C} \boldsymbol{x}(k)\\
& & & \boldsymbol{x}(0) =
\boldsymbol{x}_0\\
& & & \underline{\boldsymbol{x}} \leq  \boldsymbol{x}(k) \leq  \overline{\boldsymbol{x}}\\
& & & \underline{\boldsymbol{u}} \leq  \boldsymbol{u}(k) \leq  \overline{\boldsymbol{u}}
\end{aligned}
\end{equation}$$

#### MPC 2:

$$\begin{equation}
\begin{aligned}
& \underset{\boldsymbol{U}}{\text{min}} & &
||\boldsymbol{Y} - \boldsymbol{Y}_d||^2 + ||\boldsymbol{W}_u\boldsymbol{U}||^2 + ||\boldsymbol{W}_x\boldsymbol{X}||^2\\
& \text{s.t.} & & \boldsymbol{x}(k+1) =
\boldsymbol{A} \boldsymbol{x}(k) +
\boldsymbol{B} \boldsymbol{u}(k) \\
& & & \boldsymbol{y}(k) =
\boldsymbol{C} \boldsymbol{x}(k)\\
& & & \boldsymbol{x}(0) =
\boldsymbol{x}_0\\
& & & \underline{\boldsymbol{x}} \leq  \boldsymbol{x}(k) \leq  \overline{\boldsymbol{x}}\\
& & & \underline{\boldsymbol{u}} \leq  \boldsymbol{u}(k) \leq  \overline{\boldsymbol{u}}
\end{aligned}
\end{equation}$$

$$\begin{equation}
\boldsymbol{W}_{u} =
\begin{bmatrix}
\boldsymbol{w} _u & 0 & \cdots & 0\\
0 & \boldsymbol{w} _u & \cdots & 0\\
0 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \boldsymbol{w} _{u}
\end{bmatrix} \ \ \ \ \ \ \
\boldsymbol{W} _{x} =
\begin{bmatrix}
\boldsymbol{w}_x & 0 & \cdots & 0\\
0 & \boldsymbol{w}_x & \cdots & 0\\
0 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \boldsymbol{w}_x
\end{bmatrix}
\end{equation}$$

## 🖥️ Using the library

Eigen matrices and vectors are used to describe the MPC problem:

```cpp
//Define a discrete linear system 
// A, B, C, and D are of type Eigen::MatrixXd
EigenLinearMpc::LinearSystem example_system(A, B, C, D); 
// Construct a MPC object
EigenLinearMpc::MPC mpc(example_system, // Discrete linear system (EigenLinearMpc::LinearSystem)
                        horizon,        // MPC horizon (uint32_t)
                        Y_d,            // System output reference (Eigen::VectorXd)
                        x0,             // Initial system state (Eigen::VectorXd)
                        Q,              // Q weight value (double)
                        R);             // R weight value (double)

mpc.initializeSolver();
// Solve problem
auto U_sol = mpc.solve();
```

The type of the reference tracking problem (`MPC 1` or `MPC 2`) is determined by MPC the object constructor.

For complete examples of the two versions of the MPC problem see `test_example` and `test_example_2`.

## 📄 Dependences

This project depends on [`osqp`](https://github.com/osqp/osqp) and [osqp-eigen](https://github.com/robotology/osqp-eigen)

## 🛠️ Usage

### ⚙️ Build from source

  ```
  git clone https://github.com/ivatavuk/lin_mpc_eigen.git
  cd lin_mpc_eigen
  mkdir build
  cmake ..
  make
  make install
  ```

### Including the library in your project

**lin_mpc_eigen** provides native `CMake` support which allows the library to be easily used in `CMake` projects.
**lin_mpc_eigen** exports a CMake target called `LinMpcEigen::LinMpcEigen` which can be imported using the `find_package` CMake command and used by calling `target_link_libraries` as in the following example:
```cmake
project(myproject)
find_package(LinMpcEigen REQUIRED)
add_executable(example example.cpp)
target_link_libraries(example LinMpcEigen::LinMpcEigen)

## 📝 License

Materials in this repository are distributed under the following license:

> All software is licensed under the BSD 3-Clause License.

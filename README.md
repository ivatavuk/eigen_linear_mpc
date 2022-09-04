# Reference tracking Linear MPC library using Eigen and OSQP QP solver
Reference tracking Linear MPC library using Eigen QSQP solver 

## 🖥️ Using the library

Easy to use library, using Eigen matrices and vectors:

```cpp
//Define a discrete linear system 
// A, B, C, and D are of type Eigen::MatrixXd
EigenLinearMpc::LinearSystem example_system(A, B, C, D); 
// Create mpc object
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


Discrete linear system:
$$
	\boldsymbol{x}(k+1) = \boldsymbol{A}\boldsymbol{x}(k) + \boldsymbol{B}\boldsymbol{u}(k)
$$
$$
	\boldsymbol{y}(k) = \boldsymbol{C}\boldsymbol{x}(k)
$$

MPC 1:

$\begin{equation}
	\begin{aligned}
        & \underset{\boldsymbol{U}}{\text{min}} & & 
        Q||\boldsymbol{Y} - \boldsymbol{Y}_d||^2 + R||\boldsymbol{U}||^2\\
        & \text{s.t.} & &  \boldsymbol{x}(k+1) =
        \boldsymbol{A} \boldsymbol{x}(k) + 
        \boldsymbol{B} \boldsymbol{u}(k) \\
        & & & \boldsymbol{y}(k) =
        \boldsymbol{C} \boldsymbol{x}(k)\\
        & & & \boldsymbol{x}(0) =
        \boldsymbol{x}_0\\
        & & & \underline{\boldsymbol{x}} \leq \boldsymbol{x}(k) \leq \overline{\boldsymbol{x}}\\
        & & & \underline{\boldsymbol{u}} \leq \boldsymbol{u}(k) \leq \overline{\boldsymbol{u}}
    \end{aligned}
\end{equation}$

MPC 2:

$\begin{equation}
	\begin{aligned}
        & \underset{\boldsymbol{U}}{\text{min}} & & 
        ||\boldsymbol{Y} - \boldsymbol{Y}_d||^2 + ||\boldsymbol{W}_u\boldsymbol{U}||^2  + ||\boldsymbol{W}_x\boldsymbol{X}||^2\\
        & \text{s.t.} & &  \boldsymbol{x}(k+1) =
        \boldsymbol{A} \boldsymbol{x}(k) + 
        \boldsymbol{B} \boldsymbol{u}(k) \\
        & & & \boldsymbol{y}(k) =
        \boldsymbol{C} \boldsymbol{x}(k)\\
        & & & \boldsymbol{x}(0) =
        \boldsymbol{x}_0\\
        & & & \underline{\boldsymbol{x}} \leq \boldsymbol{x}(k) \leq \overline{\boldsymbol{x}}\\
        & & & \underline{\boldsymbol{u}} \leq \boldsymbol{u}(k) \leq \overline{\boldsymbol{u}}
    \end{aligned}
\end{equation}$

System output vector $Y$ is calculated from $U$ via the dynamics equations over the entire horizon:
$$
\boldsymbol{X} = \boldsymbol{A}_{mpc} \boldsymbol{U} + \boldsymbol{B}_{mpc} \boldsymbol{x}_0
$$
$$
\boldsymbol{Y} = \boldsymbol{C}_{mpc} \boldsymbol{X}
$$

## 📄 Dependences

This project depends on [`osqp`](https://github.com/ivatavuk/osqp) and [osqp-eigen](https://github.com/ivatavuk/osqp-eigen)

## 🛠️ Usage

### ⚙️ Build from source

1. Clone the repository to a catkin workspace
2. Build it with
   ```
   catkin build eigen_ptsc
   ```

## 📝 License

Materials in this repository are distributed under the following license:

> All software is licensed under the BSD 3-Clause License.

## TODO
### High priority
- [x] Switch to Osqp
- [x] Write example
- [x] Add plotting for example
- [x] Add better constructor, enum type of mpc
- [x] Add better constructor type 2
- [ ] Throw errors for all wrong dimensions
    - [x] Y_d and x0
    - [ ] w_u, w_y
- [ ] Add constraints
- [ ] Write more complex example
- [ ] Remove OsqpEigenOptimization and QpProblem files to different package - larics_qp_common or smth
- [ ] Sparse matrices!
- [ ] Remove ROS - build with cmake
- [ ] Write nice README
    - [ ] Write equations
    - [ ] Write dependencies
    - [ ] Write usage directions
    - [ ] Switch equations to pics
- [ ] Search TODOs in code
- [x] Add license info

### Medium priority
- [ ] Allow continuous systems
- [ ] Allow using both implicit and explicit dynamics constraints
- [ ] Add D matrix
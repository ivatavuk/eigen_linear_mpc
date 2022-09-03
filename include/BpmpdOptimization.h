/*
Ivo Vatavuk, 2021

Eigen C++ wrapper for BPMPD Primal Dual Interior Point QP solver

Prepares matrix sparsity information and calls the BPMPD library

The QP problem is in the form:

  min 	1 / 2 * x^T * A_qp * x + b_qp^T * x
   x

  s.t.	A_eq * x + b_eq = 0
      A_ieq * x + b_ieq <= 0

*/
#ifndef BPMPD_OPTIMIZATION_H_
#define BPMPD_OPTIMIZATION_H_

extern "C" void bpmpd(int *, int *, int *, int *, int *, int *, int *,
       double *, int *, int *, double *, double *, double *, double *,
       double *, double *, double *, int *, double *, int *, double *, int *);

#include <iostream>
#include <vector>
#include <Eigen/Dense>

typedef Eigen::VectorXd VecNd;
typedef Eigen::Vector3d Vec3d;
typedef Eigen::MatrixXd MatNd;
typedef Eigen::Matrix3d Mat3d;

class BpmpdOpt {
  
  public:
    BpmpdOpt();
    ~BpmpdOpt();

    //QP description
    struct QpProblem {
      MatNd A_qp, A_eq, A_ieq;
      VecNd b_qp, b_eq, b_ieq;
    };

    //calls BPMPD
    VecNd optimize();
    //prepares matrices/matrix sparsity information
    void initializeBpmpdMatrices();

    void setupQP( 	const MatNd &A_qp, const VecNd &b_qp, 
            const MatNd &A_eq, const VecNd &b_eq, 
            const MatNd &A_ieq, const VecNd &b_ieq ); 

    //updates only the A_qp matrix
    void updateAqp(const MatNd &A_qp);
    //updates only the b_qp vector
    void updateBqp(const VecNd &b_qp);

    void refreshBpmpdMatrices();	
    void printReturnCode();

  private:
    int m_, n_, nz_, qn_, qnz_, code_, memsiz_;
    int *acolcnt_, *acolidx_, *qcolcnt_, *qcolidx_, *status_;
    double *acolnzs_, *qcolnzs_, *rhs_, *obj_, *lbound_, *ubound_, *primal_, *dual_;
    double big_, opt_;

    QpProblem qp_problem_;

};

#endif //BPMPD_OPTIMIZATION_H_
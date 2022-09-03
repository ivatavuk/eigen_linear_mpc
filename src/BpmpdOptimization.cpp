#include "BpmpdOptimization.h"

BpmpdOpt::BpmpdOpt() : memsiz_(4000000000), big_(1.0e+30)
{
  initializeBpmpdMatrices();
}

BpmpdOpt::~BpmpdOpt() { }

void BpmpdOpt::refreshBpmpdMatrices() {
  initializeBpmpdMatrices();
}

VecNd BpmpdOpt::optimize() {

  bpmpd(&m_, &n_, &nz_, &qn_, &qnz_, acolcnt_, acolidx_, acolnzs_, qcolcnt_, qcolidx_, qcolnzs_,
  rhs_, obj_, lbound_, ubound_, primal_, dual_, status_, &big_, &code_, &opt_, &memsiz_);

  VecNd output = VecNd::Zero(n_);
  for (int i = 0; i < n_; i++) {
    output(i) = primal_[i];
  }
  return output;
}

void BpmpdOpt::initializeBpmpdMatrices() {

  std::vector<int> qcolcnt_vec;
  std::vector<int> qcolidx_vec;
  std::vector<double> qcolnzs_vec;

  std::vector<int> acolcnt_vec;
  std::vector<int> acolidx_vec;
  std::vector<double> acolnzs_vec;

  MatNd A = MatNd::Zero (	qp_problem_.A_eq.rows() + qp_problem_.A_ieq.rows(), 
              qp_problem_.A_eq.cols() );

  int n_equality = qp_problem_.A_eq.rows();
  int n_inequality = qp_problem_.A_ieq.rows(); 

  A << qp_problem_.A_eq, qp_problem_.A_ieq;
  m_ = A.rows();
  n_ = A.cols();

  nz_ = 0;
  
  for(int j = 0; j < A.cols(); j++) {
    int sum_nz_col = 0;
    int sum_nz_col_low_trian = 0;

    for (int i = 0; i < A.rows(); i++) {
      if (A(i,j) != 0) {
        sum_nz_col++;
        nz_++;
        acolnzs_vec.push_back(A(i,j));
        acolidx_vec.push_back(i+1);
      }
    }
    acolcnt_vec.push_back(sum_nz_col);
  }


  acolnzs_ = (double *) malloc(acolnzs_vec.size() * sizeof(double));
  for (int i = 0; i < acolnzs_vec.size(); i++) {
    acolnzs_[i] = acolnzs_vec[i];
  }
  
  acolidx_ = (int *) malloc(acolidx_vec.size() * sizeof(int));
  for (int i = 0; i < acolidx_vec.size(); i++) {
    acolidx_[i] = acolidx_vec[i];
  }

  acolcnt_ = (int *) malloc(acolcnt_vec.size() * sizeof(int));
  for (int i = 0; i < acolcnt_vec.size(); i++) {
    acolcnt_[i] = acolcnt_vec[i];
  }
  
  rhs_ = (double *) malloc((n_equality + n_inequality) * sizeof(double));
  for (int i = 0; i < n_equality + n_inequality; i++) {
    if (i < n_equality) {
      rhs_[i] = -qp_problem_.b_eq(i);
    }
    else {
      rhs_[i] = -qp_problem_.b_ieq( i - n_equality );
    }
  }

  qn_ = 0;
  qnz_ = 0;
  for(int j = 0; j < qp_problem_.A_qp.cols(); j++) {
    int sum_nz_col = 0;
    int sum_nz_col_low_trian = 0;

    for (int i = 0; i < qp_problem_.A_qp.rows(); i++) {
      if (qp_problem_.A_qp(i,j) != 0) {
        sum_nz_col++;
      }	
      
      /*lower triangular*/
      if (j <= i) {
        if (qp_problem_.A_qp(i,j) != 0) {
          sum_nz_col_low_trian++;
          qcolnzs_vec.push_back(qp_problem_.A_qp(i,j));
          qcolidx_vec.push_back(i+1);
        }
      }
    }

    qnz_ += sum_nz_col_low_trian;

    qcolcnt_vec.push_back(sum_nz_col_low_trian);
    if(sum_nz_col != 0) {
      qn_++;
    }
  }

  obj_ = (double *) malloc(n_ * sizeof(double));
  for (int i = 0; i < n_; i++) {
    obj_[i] = qp_problem_.b_qp(i);
  }


  qcolnzs_ = (double *) malloc(qcolnzs_vec.size() * sizeof(double));
  for (int i = 0; i < qcolnzs_vec.size(); i++) {
    qcolnzs_[i] = qcolnzs_vec[i];
  }

  qcolidx_ = (int *) malloc(qcolidx_vec.size() * sizeof(int));
  for (int i = 0; i < qcolidx_vec.size(); i++) {
    qcolidx_[i] = qcolidx_vec[i];
  }

  qcolcnt_ = (int *) malloc(qcolcnt_vec.size() * sizeof(int));
  for (int i = 0; i < qcolcnt_vec.size(); i++) {
    qcolcnt_[i] = qcolcnt_vec[i];
  }
 
  lbound_ = (double *) malloc((n_ + m_) * sizeof(double));
  ubound_ = (double *) malloc((n_ + m_) * sizeof(double));
  for (int i = 0; i < n_ + m_; i++) {
    // optimization variables bound
    if (i < n_) {
      lbound_[i] = -big_;
      ubound_[i] = big_;
    }
    else {
      // equality constraints 
      if (i >= n_ && i < n_ + n_equality) {
        lbound_[i] = 0;
        ubound_[i] = 0;
      }
      //inequality constraints
      else
      {
        lbound_[i] = 0;
        ubound_[i] = big_;
      }
    }
  }

  primal_ = (double *) malloc((n_) * sizeof(double));
  dual_ = (double *) malloc((n_) * sizeof(double));
  status_ = (int *) malloc((n_ + m_) * sizeof(int));
}

void BpmpdOpt::updateAqp(const MatNd &A_qp) {
  qp_problem_.A_qp = A_qp;

  std::vector<int> qcolcnt_vec;
  std::vector<int> qcolidx_vec;
  std::vector<double> qcolnzs_vec;

  qn_ = 0;
  qnz_ = 0;
  for(int j = 0; j < qp_problem_.A_qp.cols(); j++) {
    int sum_nz_col = 0;
    int sum_nz_col_low_trian = 0;

    for (int i = 0; i < qp_problem_.A_qp.rows(); i++) {
      if (qp_problem_.A_qp(i,j) != 0) {
        sum_nz_col++;
      }	
      
      /*lower triangular*/
      if (j <= i) {
        if (qp_problem_.A_qp(i,j) != 0) {
          sum_nz_col_low_trian++;
          qcolnzs_vec.push_back(qp_problem_.A_qp(i,j));
          qcolidx_vec.push_back(i+1);
        }
      }
    }

    qnz_ += sum_nz_col_low_trian;

    qcolcnt_vec.push_back(sum_nz_col_low_trian);
    if(sum_nz_col != 0) {
      qn_++;
    }
  }
  qcolnzs_ = (double *) realloc(qcolnzs_, qcolnzs_vec.size() * sizeof(double));
  for (int i = 0; i < qcolnzs_vec.size(); i++) {
    qcolnzs_[i] = qcolnzs_vec[i];
  }

  qcolidx_ = (int *) realloc(qcolidx_, qcolidx_vec.size() * sizeof(int));
  for (int i = 0; i < qcolidx_vec.size(); i++) {
    qcolidx_[i] = qcolidx_vec[i];
  }

  qcolcnt_ = (int *) realloc(qcolcnt_, qcolcnt_vec.size() * sizeof(int));
  for (int i = 0; i < qcolcnt_vec.size(); i++) {
    qcolcnt_[i] = qcolcnt_vec[i];
  }
}

void BpmpdOpt::updateBqp(const VecNd &b_qp) {
  qp_problem_.b_qp = b_qp;
  for (int i = 0; i < n_; i++) {
    obj_[i] = qp_problem_.b_qp(i);
  }
}

void BpmpdOpt::setupQP( 	const MatNd &A_qp, const VecNd &b_qp, 
                const MatNd &A_eq, const VecNd &b_eq, 
                const MatNd &A_ieq, const VecNd &b_ieq ) 
{
  qp_problem_.A_qp = A_qp;
  qp_problem_.b_qp = b_qp;
  
  qp_problem_.A_eq = A_eq;
  qp_problem_.b_eq = b_eq;

  qp_problem_.A_ieq = A_ieq;
  qp_problem_.b_ieq = b_ieq;

  initializeBpmpdMatrices();
}

void BpmpdOpt::printReturnCode(){
  if (code_ < 0) {
        std::cout << "[BPMPD] not enough memory\n\n";
    }
    if (code_ == 1) {
        std::cout << "[BPMPD] solver stopped at feasible point (suboptimal solution)\n\n";
    }
    if (code_ == 2) {
        std::cout << "[BPMPD] optimal solution found\n\n";
    }
    if (code_ == 3) {
        std::cout << "problem dual infeasible\n\n";
    }
    if (code_ == 4) {
        std::cout << "problem primal infeasible\n\n";
    }
}
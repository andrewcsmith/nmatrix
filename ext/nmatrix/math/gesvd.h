/////////////////////////////////////////////////////////////////////
// = NMatrix
//
// A linear algebra library for scientific computation in Ruby.
// NMatrix is part of SciRuby.
//
// NMatrix was originally inspired by and derived from NArray, by
// Masahiro Tanaka: http://narray.rubyforge.org
//
// == Copyright Information
//
// SciRuby is Copyright (c) 2010 - 2014, Ruby Science Foundation
// NMatrix is Copyright (c) 2012 - 2014, John Woods and the Ruby Science Foundation
//
// Please see LICENSE.txt for additional copyright notices.
//
// == Contributing
//
// By contributing source code to SciRuby, you agree to be bound by
// our Contributor Agreement:
//
// * https://github.com/SciRuby/sciruby/wiki/Contributor-Agreement
//
// == gesvd.h
//
// Header file for interface with LAPACK's xGESVD functions.
//

#ifndef GESVD_H
# define GESVD_H

#ifndef HAVE_FRAMEWORK_ACCELERATE
extern "C" {
  void sgesvd_(char*, char*, int*, int*, float*, int*, float*, float*, int*, float*, int*, float*, int*, int*);
  void dgesvd_(char*, char*, int*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int*, int*);
  void cgesvd_(char*, char*, int*, int*, nm::Complex64*, int*, nm::Complex64*, nm::Complex64*, int*, nm::Complex64*, int*, nm::Complex64*, int*, float*, int*);
  void zgesvd_(char*, char*, int*, int*, nm::Complex128*, int*, nm::Complex128*, nm::Complex128*, int*, nm::Complex128*, int*, nm::Complex128*, int*, double*, int*);
}
#endif

namespace nm {
  namespace math {

    template <typename DType, typename CType>
    inline int gesvd(char jobu, char jobvt, int m, int n, DType* a, int lda, DType* s, DType* u, int ldu, DType* vt, int ldvt, DType* work, int lwork, CType* rwork) {
      rb_raise(rb_eNotImpError, "not yet implemented for non-BLAS dtypes");
      return -1;
    }

    template <>
    inline int gesvd(char jobu, char jobvt, int m, int n, float* a, int lda, float* s, float* u, int ldu, float* vt, int ldvt, float* work, int lwork, float* rwork) {
      int info;
      sgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);
      return info;
    }

    template <>
    inline int gesvd(char jobu, char jobvt, int m, int n, double* a, int lda, double* s, double* u, int ldu, double* vt, int ldvt, double* work, int lwork, double* rwork) {
      int info;
      dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);
      return info;
    }

    template <>
    inline int gesvd(char jobu, char jobvt, int m, int n, nm::Complex64* a, int lda, nm::Complex64* s, nm::Complex64* u, int ldu, nm::Complex64* vt, int ldvt, nm::Complex64* work, int lwork, float* rwork) {
      int info;
#if defined HAVE_FRAMEWORK_ACCELERATE
      __CLPK_complex* accelerate_A = (__CLPK_complex*) a;
      __CLPK_real* accelerate_s = (__CLPK_real*) s;
      __CLPK_complex* accelerate_u = (__CLPK_complex*) u;
      __CLPK_complex* accelerate_vt = (__CLPK_complex*) vt;
      __CLPK_complex* accelerate_work = (__CLPK_complex*) work;
      cgesvd_(&jobu, &jobvt, &m, &n, accelerate_A, &lda, accelerate_s, accelerate_u, &ldu, accelerate_vt, &ldvt, accelerate_work, &lwork, rwork, &info);
#else
      cgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, &info);
#endif
      return info;
    }

    template <>
    inline int gesvd(char jobu, char jobvt, int m, int n, nm::Complex128* a, int lda, nm::Complex128* s, nm::Complex128* u, int ldu, nm::Complex128* vt, int ldvt, nm::Complex128* work, int lwork, double* rwork) {
      int info;
#if defined HAVE_FRAMEWORK_ACCELERATE
      __CLPK_doublecomplex* accelerate_A = (__CLPK_doublecomplex*) a;
      __CLPK_doublereal* accelerate_s = (__CLPK_doublereal*) s;
      __CLPK_doublecomplex* accelerate_u = (__CLPK_doublecomplex*) u;
      __CLPK_doublecomplex* accelerate_vt = (__CLPK_doublecomplex*) vt;
      __CLPK_doublecomplex* accelerate_work = (__CLPK_doublecomplex*) work;
      zgesvd_(&jobu, &jobvt, &m, &n, accelerate_A, &lda, accelerate_s, accelerate_u, &ldu, accelerate_vt, &ldvt, accelerate_work, &lwork, rwork, &info);
#else
      zgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, &info);
#endif
      return info;
    }

  } // end of namespace math
} // end of namespace nm
#endif // GESVD_H

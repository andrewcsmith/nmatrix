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
// == ordering.h
//
// functions to convert from row-major to column-major ordering
//

#ifndef ORDERING_H
#define ORDERING_H

namespace nm { namespace math {

template <typename DType>
inline void convert_order(const int n, const int lda, DType* matrix) {
  return 0;
}

template <>
inline void convert_order(const int n, const int lda, __CLPK_real* matrix) {
  __CLPK_real buffer[lda * n];
  catlas_sset(lda * n, 0.0, buffer, 1);

  __CLPK_real identity[lda * n];
  catlas_sset(lda * n, 0.0, identity, 1);
  catlas_sset(lda, 1.0, identity, lda + 1);

  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, lda, n, n, 1.0,
      matrix, lda, identity, lda, 1.0, buffer, lda);

  cblas_scopy(lda * n, buffer, 1, matrix, 1);

  return;
}

template <>
inline void convert_order(const int n, const int lda, __CLPK_doublereal* matrix) {
  __CLPK_doublereal buffer[lda * n];
  catlas_dset(lda * n, 0.0, buffer, 1);

  __CLPK_doublereal identity[lda * n];
  catlas_dset(lda * n, 0.0, identity, 1);
  catlas_dset(lda, 1.0, identity, lda + 1);

  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, lda, n, n, 1.0,
      matrix, lda, identity, lda, 1.0, buffer, lda);

  cblas_dcopy(lda * n, buffer, 1, matrix, 1);

  return;
}

template <>
inline void convert_order(const int n, const int lda, __CLPK_complex* matrix) {
  __CLPK_complex complex_zero = {0.0, 0.0};
  __CLPK_complex complex_unit = {1.0, 0.0};

  __CLPK_complex buffer[lda * n];
  catlas_cset(lda * n, &complex_zero, buffer, 1);

  __CLPK_complex identity[lda * n];
  catlas_cset(lda * n, &complex_zero, identity, 1);
  catlas_cset(lda, &complex_unit, identity, lda + 1);

  cblas_cgemm(CblasRowMajor, CblasTrans, CblasNoTrans, lda, n, n, &complex_unit,
      matrix, lda, identity, lda, &complex_unit, buffer, lda);

  cblas_ccopy(lda * n, buffer, 1, matrix, 1);

  return;
}


template <>
inline void convert_order(const int n, const int lda, __CLPK_doublecomplex* matrix) {
  __CLPK_doublecomplex doublecomplex_zero = {0.0, 0.0};
  __CLPK_doublecomplex doublecomplex_unit = {1.0, 0.0};

  __CLPK_doublecomplex buffer[lda * n];
  catlas_zset(lda * n, &doublecomplex_zero, buffer, 1);

  __CLPK_doublecomplex identity[lda * n];
  catlas_zset(lda * n, &doublecomplex_zero, identity, 1);
  catlas_zset(lda, &doublecomplex_unit, identity, lda + 1);

  cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, lda, n, n, &doublecomplex_unit,
      matrix, lda, identity, lda, &doublecomplex_unit, buffer, lda);

  cblas_zcopy(lda * n, buffer, 1, matrix, 1);

  return;
}


} } // end nm::math

#endif // ORDERING_H

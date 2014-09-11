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
// == getri.h
//
// getri function in native C++.
//

/*
 *             Automatically Tuned Linear Algebra Software v3.8.4
 *                    (C) Copyright 1999 R. Clint Whaley
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions, and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *   3. The name of the ATLAS group or the names of its contributers may
 *      not be used to endorse or promote products derived from this
 *      software without specific written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE ATLAS GROUP OR ITS CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef GETRI_H
#define GETRI_H

namespace nm { namespace math {

template <typename DType>
inline int getri(const enum CBLAS_ORDER order, const int n, DType* a, const int lda, const int* ipiv) {
  rb_raise(rb_eNotImpError, "getri not yet implemented for non-BLAS dtypes");
  return 0;
}

#if defined (HAVE_CLAPACK_H) || defined (HAVE_ATLAS_CLAPACK_H)
template <>
inline int getri(const enum CBLAS_ORDER order, const int n, float* a, const int lda, const int* ipiv) {
#if defined HAVE_FRAMEWORK_ACCELERATE
  __CLPK_integer info;
  __CLPK_integer accelerate_N = (long int) n;
  __CLPK_integer accelerate_lda = (long int) lda;
  __CLPK_integer* accelerate_ipiv = (__CLPK_integer*) ipiv;
  __CLPK_integer lwork = n;
  __CLPK_real work[n];

  // Assign the pointer to the finished array
  __CLPK_real* accelerate_A = (__CLPK_real*) a;

  // If it is row-major, we need to transpose before and after
  if(order == CblasRowMajor) {
    convert_order(n, lda, accelerate_A);
    int return_value = sgetri_(&accelerate_N, accelerate_A, &accelerate_lda, 
        accelerate_ipiv, work, &lwork, &info);
    convert_order(n, lda, accelerate_A);
    return return_value;
  } else {
    return sgetri_(&accelerate_N, accelerate_A, &accelerate_lda,
        accelerate_ipiv, work, &lwork, &info);
  }
#else
  return clapack_sgetri(order, n, a, lda, ipiv);
#endif
}

template <>
inline int getri(const enum CBLAS_ORDER order, const int n, double* a, const int lda, const int* ipiv) {
#if defined HAVE_FRAMEWORK_ACCELERATE
  __CLPK_integer info;
  __CLPK_integer accelerate_N = (long int) n;
  __CLPK_integer accelerate_lda = (long int) lda;
  __CLPK_integer* accelerate_ipiv = (__CLPK_integer*) ipiv;
  __CLPK_integer lwork = n;
  __CLPK_doublereal work[n];

  // Assign the pointer to the finished array
  __CLPK_doublereal* accelerate_A = (__CLPK_doublereal*) a;

  // If it is row-major, we need to transpose before and after
  if(order == CblasRowMajor) {
    convert_order(n, lda, accelerate_A);
    int return_value = dgetri_(&accelerate_N, accelerate_A, &accelerate_lda,
        accelerate_ipiv, work, &lwork, &info);
    convert_order(n, lda, accelerate_A);
    return return_value;
  } else {
    return dgetri_(&accelerate_N, accelerate_A, &accelerate_lda,
        accelerate_ipiv, work, &lwork, &info);
  }
#else
  return clapack_dgetri(order, n, a, lda, ipiv);
#endif
}

template <>
inline int getri(const enum CBLAS_ORDER order, const int n, Complex64* a, const int lda, const int* ipiv) {
#if defined HAVE_FRAMEWORK_ACCELERATE
  __CLPK_integer info;
  __CLPK_integer accelerate_N = (long int) n;
  __CLPK_integer accelerate_lda = (long int) lda;
  __CLPK_integer* accelerate_ipiv = (__CLPK_integer*) ipiv;
  __CLPK_integer lwork = n;
  __CLPK_complex work[n];

  // Assign the pointer to the finished array
  __CLPK_complex* accelerate_A = (__CLPK_complex*) a;

  // If it is row-major, we need to transpose before and after
  if(order == CblasRowMajor) {
    convert_order(n, lda, accelerate_A);
    int return_value = cgetri_(&accelerate_N, accelerate_A, &accelerate_lda,
        accelerate_ipiv, work, &lwork, &info);
    convert_order(n, lda, accelerate_A);
    return return_value;
  } else {
    return cgetri_(&accelerate_N, accelerate_A, &accelerate_lda,
        accelerate_ipiv, work, &lwork, &info);
  }
#else
  return clapack_cgetri(order, n, reinterpret_cast<void*>(a), lda, ipiv);
#endif
}

template <>
inline int getri(const enum CBLAS_ORDER order, const int n, Complex128* a, const int lda, const int* ipiv) {
#if defined HAVE_FRAMEWORK_ACCELERATE
  __CLPK_integer info;
  __CLPK_integer accelerate_N = (long int) n;
  __CLPK_integer accelerate_lda = (long int) lda;
  __CLPK_integer* accelerate_ipiv = (__CLPK_integer*) ipiv;
  __CLPK_integer lwork = n;
  __CLPK_doublecomplex work[n];

  // Assign the pointer to the finished array
  __CLPK_doublecomplex* accelerate_A = (__CLPK_doublecomplex*) a;
  
  // If it is row-major, we need to transpose before and after
  if(order == CblasRowMajor) {
    convert_order(n, lda, accelerate_A);
    int return_value = zgetri_(&accelerate_N, accelerate_A, &accelerate_lda,
        accelerate_ipiv, work, &lwork, &info);
    convert_order(n, lda, accelerate_A);
    return return_value;
  } else {
    return zgetri_(&accelerate_N, accelerate_A, &accelerate_lda,
        accelerate_ipiv, work, &lwork, &info);
  }
#else
  return clapack_zgetri(order, n, reinterpret_cast<void*>(a), lda, ipiv);
#endif
}
#endif

/*
 * Function signature conversion for calling LAPACK's getri functions as directly as possible.
 *
 * For documentation: http://www.netlib.org/lapack/double/dgetri.f
 *
 * This function should normally go in math.cpp, but we need it to be available to nmatrix.cpp.
 */
template <typename DType>
inline int clapack_getri(const enum CBLAS_ORDER order, const int n, void* a, const int lda, const int* ipiv) {
  return getri<DType>(order, n, reinterpret_cast<DType*>(a), lda, ipiv);
}


} } // end nm::math

#endif // GETRI_H

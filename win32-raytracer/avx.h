#pragma once

#include "immintrin.h"

static const int XMM_REG_COUNT = 8;

using Vec  = __m256;
using Veci = __m256i;

Vec
vec_zero()
{
  return _mm256_setzero_ps();
}

Veci
veci_zero()
{
  return _mm256_setzero_si256();
}

Vec
vec_set1(const float v)
{
  return _mm256_set1_ps(v);
}

template <typename... Args>
Veci
veci_set(Args&&... args)
{
  return _mm256_set_epi32(args...);
}

Veci
veci_set_iota(int base)
{
  return veci_set(
    base + 7, base + 6, base + 5, base + 4, base + 3, base + 2, base + 1, base);
}

Vec
vec_load(const float* v)
{
  return _mm256_load_ps(v);
}

Vec
vec_sqrt(const Vec& a)
{
  return _mm256_sqrt_ps(a);
}

Vec
operator+(const Vec& a, const Vec& b)
{
  return _mm256_add_ps(a, b);
}

Vec
operator-(const Vec& a, const Vec& b)
{
  return _mm256_sub_ps(a, b);
}

Vec operator*(const Vec& a, const Vec& b)
{
  return _mm256_mul_ps(a, b);
}

Vec
operator/(const Vec& a, const Vec& b)
{
  return _mm256_div_ps(a, b);
}

Vec
operator>(const Vec& a, const Vec& b)
{
  return _mm256_cmp_ps(a, b, _CMP_GT_OQ);
}

Vec
operator>=(const Vec& a, const Vec& b)
{
  return _mm256_cmp_ps(a, b, _CMP_GE_OQ);
}

Vec
operator<(const Vec& a, const Vec& b)
{
  return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
}

Vec
operator<=(const Vec& a, const Vec& b)
{
  return _mm256_cmp_ps(a, b, _CMP_LE_OQ);
}

Vec
operator|(const Vec& a, const Vec& b)
{
  return _mm256_or_ps(a, b);
}

Vec operator&(const Vec& a, const Vec& b)
{
  return _mm256_and_ps(a, b);
}

void
conditionalAssign(Vec& dest, const Vec& src, const Vec& mask)
{
  // (~mask & dest) | (mask & src)

  // 1) Zero out values in dest, that are to be updated with new values
  const Vec maskedDest = _mm256_andnot_ps(mask, dest);

  // 2) Zero out values in src that will not be kept
  const Vec maskedSrc = _mm256_and_ps(mask, src);

  // 3) Combine the values that are to be retained
  dest = _mm256_or_ps(maskedDest, maskedSrc);
}

void
conditionalAssign(Veci& dest, const Veci& src, const Veci& mask)
{
  Vec dst = _mm256_castsi256_ps(dest);

  conditionalAssign(dst, _mm256_castsi256_ps(src), _mm256_castsi256_ps(mask));

  dest = _mm256_castps_si256(dst);

  // NB. The following code requires AVX2
  // const Veci maskedDest = _mm256_andnot_si256(mask, dest);
  // const Veci maskedSrc  = _mm256_and_si256(mask, src);
  // dest                  = _mm256_or_si256(maskedDest, maskedSrc);
}

int
vec_moveMask(const Vec& a)
{
  return _mm256_movemask_ps(a);
}

void
vec_store(float* dest, const Vec& source)
{
  _mm256_store_ps(dest, source);
}

void
veci_store(int* dest, const Veci& source)
{
  _mm256_store_si256((Veci*)dest, source);
}

Veci
to_Veci(const Vec& a)
{
  return _mm256_castps_si256(a);
}

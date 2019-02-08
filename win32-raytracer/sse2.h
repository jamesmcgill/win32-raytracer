#pragma once

#include "emmintrin.h"

static const int XMM_REG_COUNT = 4;

using Vec  = __m128;
using Veci = __m128i;

Vec
vec_zero()
{
  return _mm_setzero_ps();
}

Veci
veci_zero()
{
  return _mm_setzero_si128();
}

Vec
vec_set1(const float v)
{
  return _mm_set1_ps(v);
}

template <typename... Args>
Veci
veci_set(Args&&... args)
{
  return _mm_set_epi32(args...);
}

Veci
veci_set_iota(int base)
{
  return veci_set(
    base + 3, base + 2, base + 1, base);
}

Vec
vec_load(const float* v)
{
  return _mm_load_ps(v);
}

Vec
vec_sqrt(const Vec& a)
{
  return _mm_sqrt_ps(a);
}

Vec
operator+(const Vec& a, const Vec& b)
{
  return _mm_add_ps(a, b);
}

Vec
operator-(const Vec& a, const Vec& b)
{
  return _mm_sub_ps(a, b);
}

Vec operator*(const Vec& a, const Vec& b)
{
  return _mm_mul_ps(a, b);
}

Vec
operator/(const Vec& a, const Vec& b)
{
  return _mm_div_ps(a, b);
}

Vec
operator>(const Vec& a, const Vec& b)
{
  return _mm_cmpgt_ps(a, b);
}

Vec
operator>=(const Vec& a, const Vec& b)
{
  return _mm_cmpge_ps(a, b);
}

Vec
operator<(const Vec& a, const Vec& b)
{
  return _mm_cmplt_ps(a, b);
}

Vec
operator<=(const Vec& a, const Vec& b)
{
  return _mm_cmple_ps(a, b);
}

Vec
operator|(const Vec& a, const Vec& b)
{
  return _mm_or_ps(a, b);
}

Vec operator&(const Vec& a, const Vec& b)
{
  return _mm_and_ps(a, b);
}

void
conditionalAssign(Vec& dest, const Vec& src, const Vec& mask)
{
  // (~mask & dest) | (mask & src)

  // 1) Zero out values in dest, that are to be updated with new values
  const Vec maskedDest = _mm_andnot_ps(mask, dest);

  // 2) Zero out values in src that will not be kept
  const Vec maskedSrc = _mm_and_ps(mask, src);

  // 3) Combine the values that are to be retained
  dest = _mm_or_ps(maskedDest, maskedSrc);
}

void
conditionalAssign(Veci& dest, const Veci& src, const Veci& mask)
{
  const Veci maskedDest = _mm_andnot_si128(mask, dest);
  const Veci maskedSrc  = _mm_and_si128(mask, src);
  dest                  = _mm_or_si128(maskedDest, maskedSrc);
}

int
vec_moveMask(const Vec& a)
{
  return _mm_movemask_ps(a);
}

void
vec_store(float* dest, const Vec& source)
{
  _mm_store_ps(dest, source);
}

void
veci_store(int* dest, const Veci& source)
{
  _mm_store_si128((Veci*)dest, source);
}

Veci
to_Veci(const Vec& a)
{
  return _mm_castps_si128(a);
}

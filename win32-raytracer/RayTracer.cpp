#include "pch.h"
#include "RayTracer.h"

#include "emmintrin.h"

constexpr float EPSILON = 0.00001f;

__declspec(align(16)) static const unsigned int INT_MAX_VEC[4]
  = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};
__declspec(align(16)) static const __m128i INT_MAX_VEC_MM = _mm_load_si128(
  (__m128i*)INT_MAX_VEC);
__declspec(align(16)) static const __m128 F_MAX_VEC_MM = _mm_cvtepi32_ps(
  INT_MAX_VEC_MM);

__declspec(align(16)) static const float VEC_UNIT[4] = {1.0f, 1.0f, 1.0f, 1.0f};
__declspec(align(16)) static const __m128 VEC_UNIT_MM = _mm_load_ps(VEC_UNIT);

__declspec(align(16)) static const float VEC_HALF[4] = {0.5f, 0.5f, 0.5f, 0.5f};
__declspec(align(16)) static const __m128 VEC_HALF_MM = _mm_load_ps(VEC_HALF);

namespace ptr
{
//------------------------------------------------------------------------------
class ThreadContext
{
public:
  ThreadContext() { srand_sse(666); }

  // Modified from:
  // https://software.intel.com/en-us/articles/fast-random-number-generator-on-the-intel-pentiumr-4-processor/
  inline void rand_sse(float* result)
  {
    __declspec(align(16)) __m128i cur_seed_split;
    __declspec(align(16)) __m128i multiplier;
    __declspec(align(16)) __m128i adder;
    __declspec(align(16)) __m128i mod_mask;
    __declspec(align(16)) __m128i sra_mask;
    __declspec(align(16)) static const unsigned int mult[4]
      = {214013, 17405, 214013, 69069};
    __declspec(align(16)) static const unsigned int gadd[4]
      = {2531011, 10395331, 13737667, 1};
    __declspec(align(16)) static const unsigned int mask[4]
      = {0xFFFFFFFF, 0, 0xFFFFFFFF, 0};
    __declspec(align(16)) static const unsigned int masklo[4]
      = {0x00007FFF, 0x00007FFF, 0x00007FFF, 0x00007FFF};
    adder          = _mm_load_si128((__m128i*)gadd);
    multiplier     = _mm_load_si128((__m128i*)mult);
    mod_mask       = _mm_load_si128((__m128i*)mask);
    sra_mask       = _mm_load_si128((__m128i*)masklo);
    cur_seed_split = _mm_shuffle_epi32(cur_seed, _MM_SHUFFLE(2, 3, 0, 1));
    cur_seed       = _mm_mul_epu32(cur_seed, multiplier);
    multiplier     = _mm_shuffle_epi32(multiplier, _MM_SHUFFLE(2, 3, 0, 1));
    cur_seed_split = _mm_mul_epu32(cur_seed_split, multiplier);
    cur_seed       = _mm_and_si128(cur_seed, mod_mask);
    cur_seed_split = _mm_and_si128(cur_seed_split, mod_mask);
    cur_seed_split = _mm_shuffle_epi32(cur_seed_split, _MM_SHUFFLE(2, 3, 0, 1));
    cur_seed       = _mm_or_si128(cur_seed, cur_seed_split);
    cur_seed       = _mm_add_epi32(cur_seed, adder);

    // CUSTOM CODE for returning floats in range [0 -> 1)
    __declspec(align(16)) __m128 realConversion;
    realConversion = _mm_cvtepi32_ps(cur_seed);

    realConversion = _mm_div_ps(realConversion, F_MAX_VEC_MM);
    realConversion = _mm_add_ps(realConversion, VEC_UNIT_MM);
    realConversion = _mm_mul_ps(realConversion, VEC_HALF_MM);
    _mm_store_ps(result, realConversion);

    return;
  }

private:
  __declspec(align(16)) __m128i cur_seed;

  void srand_sse(unsigned int seed)
  {
    cur_seed = _mm_set_epi32(seed, seed + 1, seed, seed + 1);
  }
};

//------------------------------------------------------------------------------
enum class Material
{
  Lambertian,
  Metal,
  Dielectric,
};

struct LambertianMatProperties
{
  DirectX::SimpleMath::Color albedo;
};

struct MetalMatProperties
{
  DirectX::SimpleMath::Color albedo;
  float fuzz = 1.0f;
};

struct DielectricMatProperties
{
  float refractiveIndex = 1.0f;
};

using MaterialProperties = std::
  variant<LambertianMatProperties, MetalMatProperties, DielectricMatProperties>;

//------------------------------------------------------------------------------
struct HitRecord
{
  DirectX::SimpleMath::Vector3 hitPoint;
  DirectX::SimpleMath::Vector3 normal;
  float t;
  Material material;
  MaterialProperties matProperties;
};

//------------------------------------------------------------------------------
struct ScatterRecord
{
  DirectX::SimpleMath::Color attenuation;
  DirectX::SimpleMath::Ray ray;
};

//------------------------------------------------------------------------------
// [-1, 1] => [0, 1]
//------------------------------------------------------------------------------
float
quantize(float x)
{
  return 0.5f * (x + 1.0f);
}

//------------------------------------------------------------------------------
DirectX::SimpleMath::Vector3
reflect(
  const DirectX::SimpleMath::Vector3& in,
  const DirectX::SimpleMath::Vector3& normal)
{
  return in - (in.Dot(normal) * 2.0f) * normal;
}

//------------------------------------------------------------------------------
std::optional<DirectX::SimpleMath::Vector3>
refract(
  const DirectX::SimpleMath::Vector3& dir,
  const DirectX::SimpleMath::Vector3& normal,
  const float ni_over_nt)
{
  // TODO: Write your own
  // return DirectX::SimpleMath::Vector3::Refract(dir, normal, ni_over_nt);

  DirectX::SimpleMath::Vector3 normalisedDir = dir;
  normalisedDir.Normalize();

  float dt           = normalisedDir.Dot(normal);
  float discriminant = 2.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
  if (discriminant > 0.0f)
  {
    return ni_over_nt * (normalisedDir - normal * dt)
           - normal * sqrt(discriminant);
  }
  return std::nullopt;
}

//------------------------------------------------------------------------------
float
schlick(float cosTheta, float refractiveIndex)
{
  float r0 = (1.0f - refractiveIndex) / (1.0f + refractiveIndex);
  r0       = r0 * r0;
  return r0 + (1.0f - r0) * pow((1.0f - cosTheta), 5);
}

//------------------------------------------------------------------------------
DirectX::SimpleMath::Vector3
getRandomPointInUnitSphere(ThreadContext& ctx)
{
  using DirectX::SimpleMath::Vector3;
  Vector3 point;
  float r[4];
  do
  {
    ctx.rand_sse(r);
    point = 2.0f * Vector3(r[0], r[1], r[2]) - Vector3(1.f, 1.f, 1.f);
  } while (point.LengthSquared() >= 1.0f);

  return point;
}

//------------------------------------------------------------------------------
DirectX::SimpleMath::Vector3
getRandomPointOnUnitDisc(ThreadContext& ctx)
{
  using DirectX::SimpleMath::Vector3;
  Vector3 point;
  float r[4];
  do
  {
    ctx.rand_sse(r);
    point = 2.0f * Vector3(r[0], r[1], 0.0f) - Vector3(1.f, 1.f, 0.f);
  } while (point.Dot(point) >= 1.0f);

  return point;
}

//------------------------------------------------------------------------------
struct Camera
{
  using Vector3 = DirectX::SimpleMath::Vector3;

  Vector3 lower_left_corner;
  Vector3 horizontal;
  Vector3 vertical;
  Vector3 origin;

  Vector3 vLookAtDir;
  Vector3 vRightAxis;
  Vector3 vUpAxis;

  float lensRadius = 1.0f;
  ThreadContext& ctx;

  Camera(
    ThreadContext& _ctx,
    DirectX::SimpleMath::Vector3 lookFrom,
    DirectX::SimpleMath::Vector3 lookTo,
    DirectX::SimpleMath::Vector3 upDir,
    float verticalFovInDegrees,
    float aspectRatio,
    float aperture,
    float focusDist)
      : ctx(_ctx)
  {
    lensRadius             = aperture / 2.0f;
    const float theta      = DirectX::XMConvertToRadians(verticalFovInDegrees);
    const float halfHeight = tan(theta / 2.0f);
    const float halfWidth  = aspectRatio * halfHeight;

    vLookAtDir = lookTo - lookFrom;
    vLookAtDir.Normalize();

    vRightAxis = vLookAtDir.Cross(upDir);
    vRightAxis.Normalize();

    vUpAxis = vRightAxis.Cross(vLookAtDir);
    vUpAxis.Normalize();

    Vector3 vLeftEdge   = halfWidth * -vRightAxis;
    Vector3 vBottomEdge = halfHeight * -vUpAxis;

    origin            = lookFrom;
    lower_left_corner = origin + (vLookAtDir * focusDist)
                        + (vLeftEdge * focusDist) + (vBottomEdge * focusDist);
    horizontal = 2 * -vLeftEdge * focusDist;
    vertical   = 2 * -vBottomEdge * focusDist;
  }

  DirectX::SimpleMath::Ray getRay(float u, float v) const
  {
    Vector3 pointOnLens = lensRadius * getRandomPointOnUnitDisc(ctx);
    Vector3 offset      = vRightAxis * pointOnLens.x + vUpAxis * pointOnLens.y;

    return DirectX::SimpleMath::Ray(
      origin + offset,
      (lower_left_corner + (u * horizontal) + (v * vertical))
        - (origin + offset));
  }
};

//------------------------------------------------------------------------------
struct Spheres
{
  // Struct of Arrays layout (SoA)
  std::vector<float> _x;
  std::vector<float> _y;
  std::vector<float> _z;
  std::vector<float> _radius;
  std::vector<Material> _material;
  std::vector<MaterialProperties> _materialProperties;

  void add(
    float x,
    float y,
    float z,
    float radius,
    Material material,
    MaterialProperties matProperties)
  {
    _x.push_back(x);
    _y.push_back(y);
    _z.push_back(z);
    _radius.push_back(radius);
    _material.push_back(material);
    _materialProperties.push_back(matProperties);
  }

  void reserve(size_t count)
  {
    _x.reserve(count);
    _y.reserve(count);
    _z.reserve(count);
    _radius.reserve(count);
    _material.reserve(count);
  }

  size_t size() const { return _x.size(); }
};

//------------------------------------------------------------------------------
struct World
{
  Spheres spheres;

  bool empty() const { return spheres._x.empty(); }
};

//------------------------------------------------------------------------------
DirectX::SimpleMath::Color
getColor(
  ThreadContext& ctx,
  const DirectX::SimpleMath::Ray& ray,
  const World& world,
  int recurseDepth = 0)
{
  using DirectX::SimpleMath::Color;
  using DirectX::SimpleMath::Ray;
  using DirectX::SimpleMath::Vector3;

  if (recurseDepth > ptr::MAX_RECURSION)
  {
    return Color();
  }

  // Spheres test - find nearest object hit
  __m128 curTs          = _mm_set1_ps(std::numeric_limits<float>::max());
  __m128 curNormalX     = _mm_setzero_ps();
  __m128 curNormalY     = _mm_setzero_ps();
  __m128 curNormalZ     = _mm_setzero_ps();
  __m128 curHitPointX   = _mm_setzero_ps();
  __m128 curHitPointY   = _mm_setzero_ps();
  __m128 curHitPointZ   = _mm_setzero_ps();
  __m128 curIsHit       = _mm_setzero_ps();
  __m128i curSphereIdxs = _mm_setzero_si128();

  const float rayDirDotScalar = ray.direction.Dot(ray.direction);
  const __m128 rayDirDot      = _mm_set1_ps(rayDirDotScalar);

  const __m128 rayDirX = _mm_set1_ps(ray.direction.x);
  const __m128 rayDirY = _mm_set1_ps(ray.direction.y);
  const __m128 rayDirZ = _mm_set1_ps(ray.direction.z);

  const __m128 rayPosX = _mm_set1_ps(ray.position.x);
  const __m128 rayPosY = _mm_set1_ps(ray.position.y);
  const __m128 rayPosZ = _mm_set1_ps(ray.position.z);

  const __m128 zeros         = _mm_setzero_ps();
  const __m128 twos          = _mm_set1_ps(2.0f);
  const __m128 fours         = _mm_set1_ps(4.0f);
  const __m128 minThresholdT = _mm_set1_ps(0.001f);

  // TODO : handle non-multiples of 4 (i.e. if there are 5, only 4 will appear)
  for (size_t i = 0; i + 3 < world.spheres.size(); i += 4)
  {
    const int idx      = static_cast<int>(i);
    const __m128i idxs = _mm_set_epi32(idx + 3, idx + 2, idx + 1, idx);

    const __m128 posX = _mm_load_ps(&world.spheres._x[i]);
    const __m128 posY = _mm_load_ps(&world.spheres._y[i]);
    const __m128 posZ = _mm_load_ps(&world.spheres._z[i]);

    const __m128 rayStartX = _mm_sub_ps(rayPosX, posX);
    const __m128 rayStartY = _mm_sub_ps(rayPosY, posY);
    const __m128 rayStartZ = _mm_sub_ps(rayPosZ, posZ);

    // DOT PRODUCT ray.direction.Dot(rayStart)
    __m128 dx             = _mm_mul_ps(rayDirX, rayStartX);
    __m128 dy             = _mm_mul_ps(rayDirY, rayStartY);
    __m128 dz             = _mm_mul_ps(rayDirZ, rayStartZ);
    __m128 sum            = _mm_add_ps(dx, dy);
    __m128 rayDirDotStart = _mm_add_ps(sum, dz);

    __m128 radius   = _mm_load_ps(&world.spheres._radius[i]);
    __m128 radiusSq = _mm_mul_ps(radius, radius);

    // DOT PRODUCT rayStart.direction.Dot(rayStart)
    __m128 dx_           = _mm_mul_ps(rayStartX, rayStartX);
    __m128 dy_           = _mm_mul_ps(rayStartY, rayStartY);
    __m128 dz_           = _mm_mul_ps(rayStartZ, rayStartZ);
    __m128 sum_          = _mm_add_ps(dx_, dy_);
    __m128 startDotStart = _mm_add_ps(sum_, dz_);

    // discriminant  = b * b - 4.0f * a * c;
    __m128 a = rayDirDot;
    __m128 b = _mm_mul_ps(rayDirDotStart, twos);
    __m128 c = _mm_sub_ps(startDotStart, radiusSq);

    __m128 bSq          = _mm_mul_ps(b, b);
    __m128 ac           = _mm_mul_ps(a, c);
    __m128 fourAc       = _mm_mul_ps(ac, fours);
    __m128 discriminant = _mm_sub_ps(bSq, fourAc);

    // if (discriminant < 0.0f) continue
    // here: if ANY single register is greater than or equal to zero,
    // then continue
    __m128 isDiscrimInRange = _mm_cmpge_ps(discriminant, zeros);
    int isAnyDiscrimInRange = _mm_movemask_ps(isDiscrimInRange);
    if (isAnyDiscrimInRange == 0)
    {
      continue;
    }

    // t = (-b - sqrtDiscrim) / (2.0f * a);
    const __m128 bNeg             = _mm_sub_ps(zeros, b);
    const __m128 discrimSqrt      = _mm_sqrt_ps(discriminant);
    const __m128 bNegMinusDiscrim = _mm_sub_ps(bNeg, discrimSqrt);
    const __m128 twoA             = _mm_mul_ps(a, twos);
    __m128 t                      = _mm_div_ps(bNegMinusDiscrim, twoA);

    // hitPoint  = ray.position + (t * ray.direction);
    __m128 hitPointX = _mm_mul_ps(t, rayDirX);
    __m128 hitPointY = _mm_mul_ps(t, rayDirY);
    __m128 hitPointZ = _mm_mul_ps(t, rayDirZ);
    hitPointX        = _mm_add_ps(hitPointX, rayPosX);
    hitPointY        = _mm_add_ps(hitPointY, rayPosY);
    hitPointZ        = _mm_add_ps(hitPointZ, rayPosZ);

    // normal    = (ret.hitPoint - center) / radius;
    __m128 normalX = _mm_sub_ps(hitPointX, posX);
    __m128 normalY = _mm_sub_ps(hitPointY, posY);
    __m128 normalZ = _mm_sub_ps(hitPointZ, posZ);
    normalX        = _mm_div_ps(normalX, radius);
    normalY        = _mm_div_ps(normalY, radius);
    normalZ        = _mm_div_ps(normalZ, radius);

    // Filter out individual register results
    //---------------------------------------
    // 1) if (discriminant < 0.0f)
    // 2) if (t < 0.001f || t > nearestT)

    //------------------------
    // TODO: Draw back faces
    // if (t < 0.001f || t > nearestT)
    //{
    //  // Try the backface
    //  t = (-b + sqrtDiscrim) / (2.0f * a);
    //  if (t < 0.001f || t > nearestT)
    //  {
    //    return std::nullopt;
    //  }
    //}
    // t = (-b + sqrtDiscrim) / (2.0f * a);
    // const __m128 bNegPlusDiscrim = _mm_add_ps(bNeg, discrimSqrt);
    // const __m128 tBack                = _mm_div_ps(bNegPlusDiscrim, twoA);
    // t = tBack (when isTBackInRange and !isTInRange)
    // update isTInRange
    //------------------------

    const __m128 isTAboveMin   = _mm_cmpgt_ps(t, minThresholdT);
    const __m128 isTBelowMax   = _mm_cmplt_ps(t, curTs);
    const __m128 isTInRange    = _mm_and_ps(isTAboveMin, isTBelowMax);
    const __m128 isNewNearestT = _mm_and_ps(isDiscrimInRange, isTInRange);
    curIsHit                   = _mm_or_ps(isNewNearestT, curIsHit);

    // Update the currentTs in 3 steps
    // 1) Zero out values in currentTs, that are to be updated with new values
    const __m128 maskedTs = _mm_andnot_ps(isNewNearestT, curTs);
    // 2) Zero out values in newTs(t) that will not be kept
    const __m128 newTs = _mm_and_ps(isNewNearestT, t);
    // 3) Combine the values that are to be retained
    curTs = _mm_or_ps(maskedTs, newTs);

    // Repeat above process to update the current sphere indices
    const __m128i isNewT_si  = _mm_castps_si128(isNewNearestT);
    const __m128i maskedIdxs = _mm_andnot_si128(isNewT_si, curSphereIdxs);
    const __m128i newIdxs    = _mm_and_si128(isNewT_si, idxs);
    curSphereIdxs            = _mm_or_si128(maskedIdxs, newIdxs);

    // Repeat above process to update the current normals
    const __m128 maskedNormalX = _mm_andnot_ps(isNewNearestT, curNormalX);
    const __m128 maskedNormalY = _mm_andnot_ps(isNewNearestT, curNormalY);
    const __m128 maskedNormalZ = _mm_andnot_ps(isNewNearestT, curNormalZ);
    const __m128 newNormalX    = _mm_and_ps(isNewNearestT, normalX);
    const __m128 newNormalY    = _mm_and_ps(isNewNearestT, normalY);
    const __m128 newNormalZ    = _mm_and_ps(isNewNearestT, normalZ);
    curNormalX                 = _mm_or_ps(maskedNormalX, newNormalX);
    curNormalY                 = _mm_or_ps(maskedNormalY, newNormalY);
    curNormalZ                 = _mm_or_ps(maskedNormalZ, newNormalZ);

    // Repeat above process to update the current hitPoints
    const __m128 maskedHitX = _mm_andnot_ps(isNewNearestT, curHitPointX);
    const __m128 maskedHitY = _mm_andnot_ps(isNewNearestT, curHitPointY);
    const __m128 maskedHitZ = _mm_andnot_ps(isNewNearestT, curHitPointZ);
    const __m128 newHitX    = _mm_and_ps(isNewNearestT, hitPointX);
    const __m128 newHitY    = _mm_and_ps(isNewNearestT, hitPointY);
    const __m128 newHitZ    = _mm_and_ps(isNewNearestT, hitPointZ);
    curHitPointX            = _mm_or_ps(maskedHitX, newHitX);
    curHitPointY            = _mm_or_ps(maskedHitY, newHitY);
    curHitPointZ            = _mm_or_ps(maskedHitZ, newHitZ);
  }    // for world.spheres.size()

  __declspec(align(16)) float t_vec[4];
  __declspec(align(16)) float normalX[4];
  __declspec(align(16)) float normalY[4];
  __declspec(align(16)) float normalZ[4];
  __declspec(align(16)) float hitPointX[4];
  __declspec(align(16)) float hitPointY[4];
  __declspec(align(16)) float hitPointZ[4];
  __declspec(align(16)) int32_t idxs[4];

  _mm_store_ps(t_vec, curTs);
  _mm_store_si128((__m128i*)idxs, curSphereIdxs);
  _mm_store_ps(normalX, curNormalX);
  _mm_store_ps(normalY, curNormalY);
  _mm_store_ps(normalZ, curNormalZ);
  _mm_store_ps(hitPointX, curHitPointX);
  _mm_store_ps(hitPointY, curHitPointY);
  _mm_store_ps(hitPointZ, curHitPointZ);

  // Compare 4 register results to find first hitpoint
  int hitMask         = _mm_movemask_ps(curIsHit);
  int hitIndex        = -1;    // -1 means: no hit
  int32_t sphereIndex = -1;
  float curT          = std::numeric_limits<float>::max();
  for (int r = 0; r < 4; ++r)
  {
    if (hitMask & (0x1 << r))
    {
      if (t_vec[r] < curT)
      {
        curT        = t_vec[r];
        hitIndex    = r;
        sphereIndex = idxs[r];
        assert(sphereIndex != -1);
        assert(sphereIndex < (int32_t)world.spheres.size());
      }
    }
  }

  // Paint Material
  if (hitIndex != -1)
  {
    Vector3 hitPoint(
      hitPointX[hitIndex], hitPointY[hitIndex], hitPointZ[hitIndex]);
    Vector3 normal(normalX[hitIndex], normalY[hitIndex], normalZ[hitIndex]);

    ScatterRecord scatter;
    scatter.attenuation = Color(0.8f, 0.0f, 0.0f);

    size_t sphereIndex_sz = static_cast<int32_t>(sphereIndex);
    assert(sphereIndex_sz < world.spheres.size());
    const auto& matType = world.spheres._material[sphereIndex_sz];
    if (matType == Material::Lambertian)
    {
      const auto& mat = std::get<LambertianMatProperties>(
        world.spheres._materialProperties[sphereIndex_sz]);

      Vector3 reflectTo = hitPoint + normal + getRandomPointInUnitSphere(ctx);
      auto adjustedHitPoint = hitPoint + (EPSILON * normal);
      Vector3 reflectDir    = reflectTo - adjustedHitPoint;

      scatter.attenuation = mat.albedo;
      scatter.ray         = Ray(adjustedHitPoint, reflectDir);
      return scatter.attenuation
             * getColor(ctx, scatter.ray, world, ++recurseDepth);
    }
    else if (matType == Material::Metal)
    {
      const auto& mat = std::get<MetalMatProperties>(
        world.spheres._materialProperties[sphereIndex]);

      Vector3 reflectDir = reflect(ray.direction, normal)
                           + (mat.fuzz * getRandomPointInUnitSphere(ctx));
      if (reflectDir.Dot(normal) <= 0.0f)
      {
        return Color();
      }

      scatter.attenuation = mat.albedo;
      scatter.ray         = Ray(hitPoint + (EPSILON * normal), reflectDir);
      return scatter.attenuation
             * getColor(ctx, scatter.ray, world, ++recurseDepth);
    }
    else if (matType == Material::Dielectric)
    {
      const auto& mat = std::get<DielectricMatProperties>(
        world.spheres._materialProperties[sphereIndex]);

      scatter.attenuation = Color(1.0f, 1.0f, 1.0f);

      // To compare the ray direction with the normal. Both should be
      // facing away from the hit point.
      Vector3 dirToLight = -ray.direction;
      dirToLight.Normalize();
      float invRayDotNormal = dirToLight.Dot(normal);
      bool isEntering       = (invRayDotNormal > 0.0f);

      float ni_over_nt
        = (isEntering) ? 1.0f / mat.refractiveIndex : mat.refractiveIndex;
      Vector3 rayFacingNormal = (isEntering) ? normal : -normal;
      Vector3 offset          = (EPSILON * normal);
      Vector3 refractOffset   = (isEntering) ? -offset : offset;

      // Reflection
      float cosine             = dirToLight.Dot(rayFacingNormal);
      float reflectProbability = schlick(cosine, ni_over_nt);
      float r[4];
      ctx.rand_sse(r);
      constexpr float REFLECT_THRES = 0.05f;
      bool isReflected = (REFLECT_THRES + r[0] < reflectProbability);
      if (isReflected)
      {
        Vector3 reflectDir = reflect(ray.direction, normal);
        scatter.ray        = Ray(hitPoint - refractOffset, reflectDir);
        return scatter.attenuation
               * getColor(ctx, scatter.ray, world, ++recurseDepth);
      }

      // Refraction
      if (auto refracted = refract(-ray.direction, rayFacingNormal, ni_over_nt))
      {
        scatter.ray = Ray(hitPoint + refractOffset, *refracted);
      }
      else
      {
        // Fall-through case
        Vector3 reflectDir = reflect(ray.direction, rayFacingNormal);
        scatter.ray        = Ray(hitPoint - refractOffset, reflectDir);
      }

      return scatter.attenuation
             * getColor(ctx, scatter.ray, world, ++recurseDepth);
    }
  }
  else
  {
    // Background
    Vector3 unit_direction = ray.direction;
    unit_direction.Normalize();
    float t = quantize(unit_direction.y);
    return ((1.0f - t) * Color(1.0f, 1.0f, 1.0f))
           + (t * Color(0.5f, 0.7f, 1.0f));
  }

  return Color();
}

//------------------------------------------------------------------------------
World
getTestScene()
{
  using DirectX::SimpleMath::Color;
  using DirectX::SimpleMath::Vector3;

  World world;
  world.spheres.reserve(20);

  world.spheres.add(
    0.0f,
    -100.5f,
    0.0f,
    100.f,
    Material::Lambertian,
    LambertianMatProperties{Color(0.8f, 0.8f, 0.0f)});

  world.spheres.add(
    0.0f,
    0.0f,
    0.0f,
    -0.5f,
    Material::Lambertian,
    LambertianMatProperties{Color(0.1f, 0.2f, 0.5f)});

  world.spheres.add(
    1.0f,
    0.0f,
    0.0f,
    0.5f,
    Material::Metal,
    MetalMatProperties{Color(0.8f, 0.6f, 0.2f), 0.0f});

  world.spheres.add(
    -1.0f,
    0.0f,
    0.0f,
    -0.5f,
    Material::Dielectric,
    DielectricMatProperties{1.5f});

  world.spheres.add(
    -2.0f,
    0.0f,
    0.0f,
    0.5f,
    Material::Lambertian,
    LambertianMatProperties{Color(0.6f, 0.2f, 0.5f)});

  world.spheres.add(
    0.0f,
    0.0f,
    -1.0f,
    0.5f,
    Material::Lambertian,
    LambertianMatProperties{Color(0.3f, 0.7f, 0.5f)});

  return world;
}

//------------------------------------------------------------------------------
World
generateRandomScene()
{
  const int WORLD_LENGTH     = 22;
  const float RADIUS         = 0.2f;
  const float POS_RANDOMNESS = 0.9f;
  const float SPACING        = 1.0f;

  ThreadContext ctx;

  auto getMaterialType = [](float materialChoice) -> Material {
    if (materialChoice < 0.8f)
    {
      return Material::Lambertian;
    }
    else if (materialChoice < 0.95f)
    {
      return Material::Metal;
    }
    return Material::Dielectric;
  };

  using DirectX::SimpleMath::Color;
  using DirectX::SimpleMath::Vector3;

  World world;
  world.spheres.reserve(WORLD_LENGTH * WORLD_LENGTH + 20);

  world.spheres.add(
    0.0f,
    -1000.0f,
    0.0f,
    1000.f,
    Material::Lambertian,
    LambertianMatProperties{Color(0.5f, 0.5f, 0.5f)});

  world.spheres.add(
    0.0f,
    1.0f,
    0.0f,
    1.0f,
    Material::Dielectric,
    DielectricMatProperties{1.5f});

  world.spheres.add(
    -4.0f,
    1.0f,
    0.0f,
    1.0f,
    Material::Lambertian,
    LambertianMatProperties{Color(0.4f, 0.2f, 0.1f)});

  world.spheres.add(
    4.0f,
    1.0f,
    0.0f,
    1.0f,
    Material::Metal,
    MetalMatProperties{Color(0.7f, 0.6f, 0.5f), 0.0f});

  Color color;
  float fuzz;
  float r[4];
  const int HALF_SIDE = WORLD_LENGTH / 2;
  for (int a = -HALF_SIDE; a < HALF_SIDE; ++a)
  {
    for (int b = -HALF_SIDE; b < HALF_SIDE; ++b)
    {
      ctx.rand_sse(r);
      Vector3 center(
        a * SPACING + POS_RANDOMNESS * r[0],
        RADIUS,
        b * SPACING + POS_RANDOMNESS * r[1]);
      Material material = getMaterialType(r[2]);
      switch (material)
      {
        case Material::Lambertian:
          ctx.rand_sse(r);
          color = Color(r[0] * r[1], r[1] * r[2], r[2] * r[3]);

          world.spheres.add(
            center.x,
            center.y,
            center.z,
            RADIUS,
            Material::Lambertian,
            LambertianMatProperties{color});
          break;

        case Material::Metal:
          ctx.rand_sse(r);
          fuzz  = 0.5f * r[0];
          color = Color(
            0.5f * (1.0f + r[1]), 0.5f * (1.0f + r[2]), 0.5f * (1.0f + r[3]));

          world.spheres.add(
            center.x,
            center.y,
            center.z,
            RADIUS,
            Material::Metal,
            MetalMatProperties{color, fuzz});
          break;

        case Material::Dielectric:
          world.spheres.add(
            center.x,
            center.y,
            center.z,
            RADIUS,
            Material::Dielectric,
            DielectricMatProperties{1.5f});
          break;
      }
    }
  }

  return world;
}

//------------------------------------------------------------------------------
Image
generateImage(
  const World& world,
  const int imageWidth,
  const int imageHeight,
  const int startY,
  const int lastY,
  const int numSamples)
{
  ThreadContext ctx;

  using DirectX::SimpleMath::Vector3;
  const auto lookFrom     = Vector3(15.0f, 2.0f, 4.0f);
  const auto lookTo       = Vector3(0.0f, 1.0f, 0.0f);
  const auto upDir        = Vector3(0.0f, 1.0f, 0.0f);
  const float fov         = 20.0f;
  const float aspectRatio = static_cast<float>(imageWidth) / imageHeight;
  const float distToFocus = (lookTo - lookFrom).Length();
  const float aperture    = 0.1f;

  Camera camera(
    ctx, lookFrom, lookTo, upDir, fov, aspectRatio, aperture, distToFocus);

  using DirectX::SimpleMath::Color;
  const Color SAMPLE_COUNT(
    static_cast<float>(numSamples),
    static_cast<float>(numSamples),
    static_cast<float>(numSamples));

  Image image;
  image.width  = imageWidth;
  image.height = lastY - startY;
  image.buffer.resize(image.width * image.height);

  if (world.empty())
  {
    return image;
  }

  float r[4];
  for (int j = startY; j < lastY; ++j)
  {
    for (int i = 0; i < imageWidth; ++i)
    {
      Color color;
      for (int s = 0; s < numSamples; ++s)
      {
        ctx.rand_sse(r);
        const float u = float(i + r[0]) / imageWidth;
        const float v = float(imageHeight - j + r[1]) / imageHeight;
        color += getColor(ctx, camera.getRay(u, v), world);
      }
      color /= SAMPLE_COUNT;

      // Gamma Correction
      color = Color(sqrt(color.R()), sqrt(color.G()), sqrt(color.B()));

      auto& dest = image.buffer[(j - startY) * imageWidth + i];
      dest[0]    = u8(255.99f * color.R());
      dest[1]    = u8(255.99f * color.G());
      dest[2]    = u8(255.99f * color.B());
    }
  }
  return image;
}

//------------------------------------------------------------------------------
RenderResult
render(const int imageWidth, const int imageHeight, const int numSamples)
{
  RenderResult res;

  auto start = std::chrono::high_resolution_clock::now();

  auto world = generateRandomScene();

  std::vector<std::thread> threads(ptr::NUM_THREADS);
  res.imageParts.resize(ptr::NUM_THREADS);
  const int renderHeight = imageHeight / ptr::NUM_THREADS;

  int startY = 0;
  int endY   = 0;
  for (int i = 0; i < ptr::NUM_THREADS; ++i)
  {
    endY = (i == ptr::NUM_THREADS - 1) ? imageHeight : (endY + renderHeight);

    threads[i] = std::move(std::thread([=, &res]() {
      res.imageParts[i] = std::move(generateImage(
        world, imageWidth, imageHeight, startY, endY, numSamples));
    }));

    startY = endY;
  }

  for (int i = 0; i < NUM_THREADS; ++i)
  {
    threads[i].join();
  }

  res.renderDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::high_resolution_clock::now() - start);

  return res;
}

//------------------------------------------------------------------------------
};    // namespace ptr

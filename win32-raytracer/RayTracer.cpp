#include "pch.h"
#include "RayTracer.h"

#define AVX_ENABLED 1
#if AVX_ENABLED
#include "avx.h"
#else
#include "sse2.h"
#endif

#include "emmintrin.h"

constexpr float EPSILON = 0.00001f;

static const __m128i INT_MAX_VEC_MM = _mm_set1_epi32(INT_MAX);
static const __m128 F_MAX_VEC_MM    = _mm_cvtepi32_ps(INT_MAX_VEC_MM);
static const __m128 VEC_UNIT_MM     = _mm_set1_ps(1.0f);
static const __m128 VEC_HALF_MM     = _mm_set1_ps(0.5f);

//------------------------------------------------------------------------------
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
    __m128i adder      = _mm_setr_epi32(2531011, 10395331, 13737667, 1);
    __m128i multiplier = _mm_setr_epi32(214013, 17405, 214013, 69069);
    __m128i mod_mask   = _mm_setr_epi32(0xFFFFFFFF, 0, 0xFFFFFFFF, 0);
    __m128i sra_mask   = _mm_set1_epi32(0x00007FFF);
    __m128i cur_seed_split
      = _mm_shuffle_epi32(cur_seed, _MM_SHUFFLE(2, 3, 0, 1));

    cur_seed       = _mm_mul_epu32(cur_seed, multiplier);
    multiplier     = _mm_shuffle_epi32(multiplier, _MM_SHUFFLE(2, 3, 0, 1));
    cur_seed_split = _mm_mul_epu32(cur_seed_split, multiplier);
    cur_seed       = _mm_and_si128(cur_seed, mod_mask);
    cur_seed_split = _mm_and_si128(cur_seed_split, mod_mask);
    cur_seed_split = _mm_shuffle_epi32(cur_seed_split, _MM_SHUFFLE(2, 3, 0, 1));
    cur_seed       = _mm_or_si128(cur_seed, cur_seed_split);
    cur_seed       = _mm_add_epi32(cur_seed, adder);

    // CUSTOM CODE for returning floats in range [0 -> 1)
    __m128 realConversion = _mm_cvtepi32_ps(cur_seed);
    realConversion        = _mm_div_ps(realConversion, F_MAX_VEC_MM);
    realConversion        = _mm_add_ps(realConversion, VEC_UNIT_MM);
    realConversion        = _mm_mul_ps(realConversion, VEC_HALF_MM);

    _mm_store_ps(result, realConversion);

    return;
  }

private:
  __m128i cur_seed;

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
  Vec curTs          = vec_set1(std::numeric_limits<float>::max());
  Vec curNormalX     = vec_zero();
  Vec curNormalY     = vec_zero();
  Vec curNormalZ     = vec_zero();
  Vec curHitPointX   = vec_zero();
  Vec curHitPointY   = vec_zero();
  Vec curHitPointZ   = vec_zero();
  Vec curIsHit       = vec_zero();
  Veci curSphereIdxs = veci_zero();

  const float rayDirDotScalar = ray.direction.Dot(ray.direction);
  const Vec rayDirDot         = vec_set1(rayDirDotScalar);

  const Vec rayDirX = vec_set1(ray.direction.x);
  const Vec rayDirY = vec_set1(ray.direction.y);
  const Vec rayDirZ = vec_set1(ray.direction.z);

  const Vec rayPosX = vec_set1(ray.position.x);
  const Vec rayPosY = vec_set1(ray.position.y);
  const Vec rayPosZ = vec_set1(ray.position.z);

  const Vec zeros         = vec_zero();
  const Vec twos          = vec_set1(2.0f);
  const Vec fours         = vec_set1(4.0f);
  const Vec minThresholdT = vec_set1(0.001f);

  // TODO : handle non-multiples of 4 (i.e. if there are 5, only 4 will appear)
  for (size_t i = 0; i + (XMM_REG_COUNT - 1) < world.spheres.size();
       i += XMM_REG_COUNT)
  {
    const Veci idxs = veci_set_iota(static_cast<int>(i));

    const Vec posX = vec_load(&world.spheres._x[i]);
    const Vec posY = vec_load(&world.spheres._y[i]);
    const Vec posZ = vec_load(&world.spheres._z[i]);

    const Vec rayStartX = rayPosX - posX;
    const Vec rayStartY = rayPosY - posY;
    const Vec rayStartZ = rayPosZ - posZ;

    // DOT PRODUCT ray.direction.Dot(rayStart)
    Vec rayDirDotStart
      = rayDirX * rayStartX + rayDirY * rayStartY + rayDirZ * rayStartZ;

    Vec radius   = vec_load(&world.spheres._radius[i]);
    Vec radiusSq = radius * radius;

    // DOT PRODUCT rayStart.direction.Dot(rayStart)
    Vec startDotStart
      = rayStartX * rayStartX + rayStartY * rayStartY + rayStartZ * rayStartZ;

    // discriminant  = b * b - 4.0f * a * c;
    Vec a            = rayDirDot;
    Vec b            = rayDirDotStart * twos;
    Vec c            = startDotStart - radiusSq;
    Vec discriminant = b * b - fours * a * c;

    // if (discriminant < 0.0f) continue
    // here: if ANY single register is greater than or equal to zero,
    // then continue
    Vec isDiscrimInRange    = discriminant >= zeros;
    int isAnyDiscrimInRange = vec_moveMask(isDiscrimInRange);
    if (isAnyDiscrimInRange == 0)
    {
      continue;
    }

    // t = (-b - sqrtDiscrim) / (2.0f * a);
    const Vec bNeg        = zeros - b;
    const Vec discrimSqrt = vec_sqrt(discriminant);
    Vec t                 = (bNeg - discrimSqrt) / (a * twos);

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
    // const Vec bNegPlusDiscrim = _mm_add_ps(bNeg, discrimSqrt);
    // const Vec tBack                = _mm_div_ps(bNegPlusDiscrim, twoA);
    // t = tBack (when isTBackInRange and !isTInRange)
    // update isTInRange
    //------------------------

    const Vec isTAboveMin   = t > minThresholdT;
    const Vec isTBelowMax   = t < curTs;
    const Vec isTInRange    = isTAboveMin & isTBelowMax;
    const Vec isNewNearestT = isDiscrimInRange & isTInRange;
    int isAnyTNew           = vec_moveMask(isNewNearestT);
    if (isAnyTNew == 0)
    {
      continue;    // Early out, if no register has an interesting t
    }

    curIsHit = isNewNearestT | curIsHit;

    // hitPoint  = ray.position + (t * ray.direction);
    Vec hitPointX = rayPosX + (t * rayDirX);
    Vec hitPointY = rayPosY + (t * rayDirY);
    Vec hitPointZ = rayPosZ + (t * rayDirZ);

    // normal    = (ret.hitPoint - center) / radius;
    Vec normalX = (hitPointX - posX) / radius;
    Vec normalY = (hitPointY - posY) / radius;
    Vec normalZ = (hitPointZ - posZ) / radius;

    // Update the current nearest T
    conditionalAssign(curTs, t, isNewNearestT);

    // Update the current sphere indices
    conditionalAssign(curSphereIdxs, idxs, to_Veci(isNewNearestT));

    // Update the current normals
    conditionalAssign(curNormalX, normalX, isNewNearestT);
    conditionalAssign(curNormalY, normalY, isNewNearestT);
    conditionalAssign(curNormalZ, normalZ, isNewNearestT);

    // Repeat above process to update the current hitPoints
    conditionalAssign(curHitPointX, hitPointX, isNewNearestT);
    conditionalAssign(curHitPointY, hitPointY, isNewNearestT);
    conditionalAssign(curHitPointZ, hitPointZ, isNewNearestT);

  }    // for world.spheres.size()

  __declspec(align(16)) float t_vec[XMM_REG_COUNT];
  __declspec(align(16)) float normalX[XMM_REG_COUNT];
  __declspec(align(16)) float normalY[XMM_REG_COUNT];
  __declspec(align(16)) float normalZ[XMM_REG_COUNT];
  __declspec(align(16)) float hitPointX[XMM_REG_COUNT];
  __declspec(align(16)) float hitPointY[XMM_REG_COUNT];
  __declspec(align(16)) float hitPointZ[XMM_REG_COUNT];
  __declspec(align(16)) int32_t idxs[XMM_REG_COUNT];

  vec_store(t_vec, curTs);
  veci_store(idxs, curSphereIdxs);
  vec_store(normalX, curNormalX);
  vec_store(normalY, curNormalY);
  vec_store(normalZ, curNormalZ);
  vec_store(hitPointX, curHitPointX);
  vec_store(hitPointY, curHitPointY);
  vec_store(hitPointZ, curHitPointZ);

  // Compare all wide register results to find first hitpoint
  int hitMask         = vec_moveMask(curIsHit);
  int hitIndex        = -1;    // -1 means: no hit
  int32_t sphereIndex = -1;
  float curT          = std::numeric_limits<float>::max();
  for (int r = 0; r < XMM_REG_COUNT; ++r)
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

    static Color white(1.0f, 1.0f, 1.0f);
    static Color tint(0.5f, 0.7f, 1.0f);

    return ((1.0f - t) * white) + (t * tint);
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
  const int endY,
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
  image.height = endY - startY;
  image.buffer.resize(image.width * image.height);

  if (world.empty())
  {
    return image;
  }

  float r[4];
  for (int y = startY; y < endY; ++y)
  {
    for (int x = 0; x < imageWidth; ++x)
    {
      Color color;
      for (int s = 0; s < numSamples; ++s)
      {
        ctx.rand_sse(r);
        const float u = float(x + r[0]) / imageWidth;
        const float v = float(imageHeight - y + r[1]) / imageHeight;
        color += getColor(ctx, camera.getRay(u, v), world);
      }
      color /= SAMPLE_COUNT;

      // Gamma Correction
      color = Color(sqrt(color.R()), sqrt(color.G()), sqrt(color.B()));

      auto& dest = image.buffer[(y - startY) * imageWidth + x];
      dest[0]    = u8(255.99f * color.R());
      dest[1]    = u8(255.99f * color.G());
      dest[2]    = u8(255.99f * color.B());
    }
  }

  return image;
}    // namespace ptr

//------------------------------------------------------------------------------
RenderResult
render(const int imageWidth, const int imageHeight, const int numSamples)
{
  RenderResult res;

  auto start = std::chrono::high_resolution_clock::now();

  auto world = generateRandomScene();

  std::vector<std::thread> threads(NUM_THREADS);

  // Interleave work done on threads, i.e all threads should work on small
  // sections of the same areas of the image. Rather than one thread working
  // on the top of the image (usually simple and completed early),
  // and one thread working on the bottom of the image (usually complex).
  // This balances more the work across threads and prevents one thread left
  // working alone, while the others are finished and doing nothing to help.
  const int blockSizeY   = 8;
  const int strideSizeY  = NUM_THREADS * blockSizeY;
  const size_t numBlocks = (size_t)(ceilf(imageHeight / (float)blockSizeY));
  res.imageParts.resize(numBlocks);

  for (int i = 0; i < NUM_THREADS; ++i)
  {
    threads[i] = std::move(std::thread([=, &res]() {
      int imageSlot          = i;
      const int threadStartY = blockSizeY * i;
      for (int y = threadStartY; y < imageHeight; y += strideSizeY)
      {
        const int endY
          = ((y + blockSizeY) < imageHeight) ? (y + blockSizeY) : imageHeight;

        res.imageParts[imageSlot] = std::move(
          generateImage(world, imageWidth, imageHeight, y, endY, numSamples));
        imageSlot += NUM_THREADS;
      }
    }));
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

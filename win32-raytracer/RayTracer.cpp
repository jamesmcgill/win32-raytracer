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
struct IMaterial;
struct HitRecord
{
  DirectX::SimpleMath::Vector3 hitPoint;
  DirectX::SimpleMath::Vector3 normal;
  float t;
  IMaterial* pMaterial;
};

//------------------------------------------------------------------------------
struct ScatterRecord
{
  DirectX::SimpleMath::Color attenuation;
  DirectX::SimpleMath::Ray ray;
};

//------------------------------------------------------------------------------
struct IMaterial
{
  virtual std::unique_ptr<IMaterial> clone() const = 0;

  virtual std::optional<ScatterRecord> scatter(
    ThreadContext& ctx,
    const DirectX::SimpleMath::Ray& ray,
    const HitRecord& rec) const = 0;

  virtual ~IMaterial() = default;
};

//------------------------------------------------------------------------------
struct IHitable
{
  virtual std::unique_ptr<IHitable> clone() const = 0;

  virtual std::optional<HitRecord>
  hit(const DirectX::SimpleMath::Ray& ray, float tMin, float tMax) const = 0;

  virtual ~IHitable() = default;
};
using World = std::vector<std::unique_ptr<IHitable>>;

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
struct LambertianMaterial : public IMaterial
{
  const DirectX::SimpleMath::Color albedo;

  LambertianMaterial(DirectX::SimpleMath::Color _albedo)
      : albedo(_albedo)
  {
  }

  std::unique_ptr<IMaterial> clone() const override
  {
    return std::make_unique<LambertianMaterial>(*this);
  }

  std::optional<ScatterRecord> scatter(
    ThreadContext& ctx,
    const DirectX::SimpleMath::Ray& ray,
    const HitRecord& rec) const override
  {
    UNREFERENCED_PARAMETER(ray);

    using DirectX::SimpleMath::Ray;
    using DirectX::SimpleMath::Vector3;

    Vector3 reflectTo
      = rec.hitPoint + rec.normal + getRandomPointInUnitSphere(ctx);
    auto adjustedHitPoint = rec.hitPoint + (EPSILON * rec.normal);
    Vector3 reflectDir    = reflectTo - adjustedHitPoint;

    ScatterRecord scatter;
    scatter.attenuation = albedo;
    scatter.ray         = Ray(adjustedHitPoint, reflectDir);
    return scatter;
  }
};

//------------------------------------------------------------------------------
struct MetalMaterial : public IMaterial
{
  const DirectX::SimpleMath::Color albedo;
  float fuzz = 1.0f;

  MetalMaterial(DirectX::SimpleMath::Color _albedo, float _fuzz = 1.0f)
      : albedo(_albedo)
  {
    if (_fuzz < 1.0f)
    {
      fuzz = _fuzz;
    }
  }

  std::unique_ptr<IMaterial> clone() const override
  {
    return std::make_unique<MetalMaterial>(*this);
  }

  std::optional<ScatterRecord> scatter(
    ThreadContext& ctx,
    const DirectX::SimpleMath::Ray& ray,
    const HitRecord& rec) const override
  {
    using DirectX::SimpleMath::Ray;
    using DirectX::SimpleMath::Vector3;

    Vector3 reflectDir = reflect(ray.direction, rec.normal)
                         + (fuzz * getRandomPointInUnitSphere(ctx));
    if (reflectDir.Dot(rec.normal) <= 0.0f)
    {
      return std::nullopt;
    }

    ScatterRecord scatter;
    scatter.attenuation = albedo;
    scatter.ray = Ray(rec.hitPoint + (EPSILON * rec.normal), reflectDir);
    return scatter;
  }
};

//------------------------------------------------------------------------------
struct DielectricMaterial : public IMaterial
{
  float refractiveIndex = 1.0f;

  DielectricMaterial(float _refractiveIndex)
      : refractiveIndex(_refractiveIndex)
  {
  }

  std::unique_ptr<IMaterial> clone() const override
  {
    return std::make_unique<DielectricMaterial>(*this);
  }

  std::optional<ScatterRecord> scatter(
    ThreadContext& ctx,
    const DirectX::SimpleMath::Ray& ray,
    const HitRecord& rec) const override
  {
    using DirectX::SimpleMath::Color;
    using DirectX::SimpleMath::Ray;
    using DirectX::SimpleMath::Vector3;

    ScatterRecord scatter;
    scatter.attenuation = Color(1.0f, 1.0f, 1.0f);

    // To compare the ray direction with the normal. Both should be
    // facing away from the hit point.
    Vector3 dirToLight = -ray.direction;
    dirToLight.Normalize();
    float invRayDotNormal = dirToLight.Dot(rec.normal);
    bool isEntering       = (invRayDotNormal > 0.0f);

    float ni_over_nt = (isEntering) ? 1.0f / refractiveIndex : refractiveIndex;
    Vector3 rayFacingNormal = (isEntering) ? rec.normal : -rec.normal;
    Vector3 offset          = (EPSILON * rec.normal);
    Vector3 refractOffset   = (isEntering) ? -offset : offset;

    // Reflection
    float cosine             = dirToLight.Dot(rayFacingNormal);
    float reflectProbability = schlick(cosine, ni_over_nt);
    float r[4];
    ctx.rand_sse(r);
    constexpr float REFLECT_THRES = 0.05f;
    bool isReflected              = (REFLECT_THRES + r[0] < reflectProbability);
    if (isReflected)
    {
      Vector3 reflectDir = reflect(ray.direction, rec.normal);
      scatter.ray        = Ray(rec.hitPoint - refractOffset, reflectDir);
      return scatter;
    }

    // Refraction
    if (auto refracted = refract(-ray.direction, rayFacingNormal, ni_over_nt))
    {
      scatter.ray = Ray(rec.hitPoint + refractOffset, *refracted);
    }
    else
    {
      // Fall-through case
      Vector3 reflectDir = reflect(ray.direction, rayFacingNormal);
      scatter.ray        = Ray(rec.hitPoint - refractOffset, reflectDir);
    }

    return scatter;
  }
};

//------------------------------------------------------------------------------
struct Sphere : public IHitable
{
  DirectX::SimpleMath::Vector3 center;
  float radius = 1.0f;
  std::unique_ptr<IMaterial> pMaterial;

  Sphere(
    DirectX::SimpleMath::Vector3 _center,
    float _radius,
    std::unique_ptr<IMaterial> _pMaterial)
      : center(_center)
      , radius(_radius)
      , pMaterial(std::move(_pMaterial))
  {
  }

  Sphere(const Sphere& rhs)
  {
    center    = rhs.center;
    radius    = rhs.radius;
    pMaterial = rhs.pMaterial->clone();
  };

  std::unique_ptr<IHitable> clone() const override
  {
    return std::make_unique<Sphere>(*this);
  }

  std::optional<HitRecord> hit(
    const DirectX::SimpleMath::Ray& ray, float tMin, float tMax) const override
  {
    using DirectX::SimpleMath::Vector3;
    Vector3 rayStart = ray.position - center;

    float a            = ray.direction.Dot(ray.direction);
    float b            = 2.0f * ray.direction.Dot(rayStart);
    float c            = rayStart.Dot(rayStart) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f)
    {
      return std::nullopt;
    }

    float sqrtDiscrim = sqrt(discriminant);
    float t           = (-b - sqrtDiscrim) / (2.0f * a);
    if (t < tMin || t > tMax)
    {
      // Try the backface
      t = (-b + sqrtDiscrim) / (2.0f * a);
      if (t < tMin || t > tMax)
      {
        return std::nullopt;
      }
    }

    HitRecord ret;
    ret.t         = t;
    ret.hitPoint  = ray.position + (ret.t * ray.direction);
    ret.normal    = (ret.hitPoint - center) / radius;
    ret.pMaterial = pMaterial.get();
    return ret;
  }
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

  // World test - find nearest object hit
  float nearestT = std::numeric_limits<float>::max();
  std::optional<HitRecord> hitRecord;
  for (const auto& entity : world)
  {
    if (auto optRecord = entity->hit(ray, 0.001f, nearestT))
    {
      nearestT  = optRecord->t;
      hitRecord = optRecord;
    }
  }

  // Paint Object Colour
  if (hitRecord)
  {
    const auto& rec = *hitRecord;
    assert(rec.pMaterial);
    if (auto scatter = rec.pMaterial->scatter(ctx, ray, rec))
    {
      return scatter->attenuation
             * getColor(ctx, scatter->ray, world, ++recurseDepth);
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

  std::vector<std::unique_ptr<IHitable>> world;
  world.push_back(std::make_unique<Sphere>(
    Vector3(0.0f, -100.5f, 0.0f),
    100.f,
    std::make_unique<LambertianMaterial>(Color(0.8f, 0.8f, 0.0f))));

  world.push_back(std::make_unique<Sphere>(
    Vector3(0.0f, 0.0f, 0.0f),
    0.5f,
    std::make_unique<LambertianMaterial>(Color(0.1f, 0.2f, 0.5f))));

  world.push_back(std::make_unique<Sphere>(
    Vector3(1.0f, 0.0f, 0.0f),
    0.5f,
    std::make_unique<MetalMaterial>(Color(0.8f, 0.6f, 0.2f), 0.0f)));

  world.push_back(std::make_unique<Sphere>(
    Vector3(-1.0f, 0.0f, 0.0f),
    -0.5f,
    std::make_unique<DielectricMaterial>(1.5f)));

  world.push_back(std::make_unique<Sphere>(
    Vector3(-2.0f, 0.0f, 0.0f),
    0.5f,
    std::make_unique<LambertianMaterial>(Color(0.6f, 0.2f, 0.5f))));

  world.push_back(std::make_unique<Sphere>(
    Vector3(0.0f, 0.0f, -1.0f),
    0.5f,
    std::make_unique<LambertianMaterial>(Color(0.3f, 0.7f, 0.5f))));

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

  enum class Mat
  {
    Diffuse,
    Metal,
    Glass
  };
  auto getMaterialType = [](float materialChoice) -> Mat {
    if (materialChoice < 0.8f)
    {
      return Mat::Diffuse;
    }
    else if (materialChoice < 0.95f)
    {
      return Mat::Metal;
    }
    return Mat::Glass;
  };

  using DirectX::SimpleMath::Color;
  using DirectX::SimpleMath::Vector3;

  std::vector<std::unique_ptr<IHitable>> world;
  world.push_back(std::make_unique<Sphere>(
    Vector3(0.0f, -1000.0f, 0.0f),
    1000.f,
    std::make_unique<LambertianMaterial>(Color(0.5f, 0.5f, 0.5f))));

  world.push_back(std::make_unique<Sphere>(
    Vector3(0.0f, 1.0f, 0.0f),
    1.0f,
    std::make_unique<DielectricMaterial>(1.5f)));

  world.push_back(std::make_unique<Sphere>(
    Vector3(-4.0f, 1.0f, 0.0f),
    1.0f,
    std::make_unique<LambertianMaterial>(Color(0.4f, 0.2f, 0.1f))));

  world.push_back(std::make_unique<Sphere>(
    Vector3(4.0f, 1.0f, 0.0f),
    1.0f,
    std::make_unique<MetalMaterial>(Color(0.7f, 0.6f, 0.5f), 0.0f)));

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
      Mat material = getMaterialType(r[2]);
      switch (material)
      {
        case Mat::Diffuse:
          ctx.rand_sse(r);
          color = Color(r[0] * r[1], r[1] * r[2], r[2] * r[3]);

          world.push_back(std::make_unique<Sphere>(
            center, RADIUS, std::make_unique<LambertianMaterial>(color)));
          break;

        case Mat::Metal:
          ctx.rand_sse(r);
          fuzz  = 0.5f * r[0];
          color = Color(
            0.5f * (1.0f + r[1]), 0.5f * (1.0f + r[2]), 0.5f * (1.0f + r[3]));

          world.push_back(std::make_unique<Sphere>(
            center, RADIUS, std::make_unique<MetalMaterial>(color, fuzz)));
          break;

        case Mat::Glass:
          world.push_back(std::make_unique<Sphere>(
            center, RADIUS, std::make_unique<DielectricMaterial>(1.5f)));
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

  auto cloneWorld = [](const World& world) -> World {
    World newWorld;
    newWorld.reserve(world.size());
    std::transform(
      world.begin(),
      world.end(),
      std::back_inserter(newWorld),
      [](const auto& p) { return p->clone(); });
    return newWorld;
  };
  auto world = generateRandomScene();

  std::thread threads[ptr::NUM_THREADS];
  res.imageParts.resize(ptr::NUM_THREADS);
  const int renderHeight = imageHeight / ptr::NUM_THREADS;

  int startY = 0;
  int endY   = 0;
  for (int i = 0; i < ptr::NUM_THREADS; ++i)
  {
    endY = (i == ptr::NUM_THREADS - 1) ? imageHeight : (endY + renderHeight);

    auto newWorld = cloneWorld(world);
    threads[i]    = std::thread([=, &res, w{std::move(newWorld)}]() {
      res.imageParts[i] = std::move(
        generateImage(w, imageWidth, imageHeight, startY, endY, numSamples));
    });

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
};    // namespace ray

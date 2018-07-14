#include "pch.h"
#include "RayTracer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma warning(push)
#pragma warning(disable : 4996)
#include "stb_image_write.h"
#pragma warning(pop)

using namespace ray;
//------------------------------------------------------------------------------
struct HitRecord
{
  DirectX::SimpleMath::Vector3 hitPoint;
  DirectX::SimpleMath::Vector3 normal;
  float t;
};

//------------------------------------------------------------------------------
struct IHitable
{
  virtual std::optional<HitRecord>
  hit(const DirectX::SimpleMath::Ray& ray, float tMin, float tMax) = 0;

  virtual ~IHitable() = default;
};

//------------------------------------------------------------------------------
struct Sphere : public IHitable
{
  Sphere() = default;
  Sphere(DirectX::SimpleMath::Vector3 _center, float _radius)
      : center(_center)
      , radius(_radius)
  {
  }

  DirectX::SimpleMath::Vector3 center;
  float radius = 1.0f;

  std::optional<HitRecord>
  hit(const DirectX::SimpleMath::Ray& ray, float tMin, float tMax)
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
    ret.t        = t;
    ret.hitPoint = ray.position + (ret.t * ray.direction);
    ret.normal   = (ret.hitPoint - center) / radius;
    return ret;
  }
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
DirectX::SimpleMath::Color
color(
  const DirectX::SimpleMath::Ray& ray,
  const std::vector<std::unique_ptr<IHitable>>& world)
{
  using DirectX::SimpleMath::Color;
  using DirectX::SimpleMath::Vector3;

  // World test - find nearest object hit
  float nearestT = std::numeric_limits<float>::max();
  std::optional<HitRecord> hitRecord;
  for (const auto& entity : world)
  {
    if (auto optRecord = entity->hit(ray, 0.0f, nearestT))
    {
      nearestT  = optRecord->t;
      hitRecord = optRecord;
    }
  }

  // Paint Object Colour
  if (hitRecord)
  {
    auto& rec = *hitRecord;
    return Color(
      quantize(rec.normal.x), quantize(rec.normal.y), quantize(rec.normal.z));
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
}

//------------------------------------------------------------------------------
RayTracer::RayTracer() {}

//------------------------------------------------------------------------------
Image
RayTracer::generateImage() const
{
  const size_t nX = ray::IMAGE_WIDTH;
  const size_t nY = ray::IMAGE_HEIGHT;

  Image image;
  image.width  = nX;
  image.height = nY;
  image.buffer.resize(nX * nY);

  using DirectX::SimpleMath::Vector3;
  const float HALF_X = 4.0f;
  const float HALF_Y = 2.5f;
  Vector3 lower_left_corner(-HALF_X, -HALF_Y, -1.0f);
  Vector3 horizontal(2 * HALF_X, 0.0f, 0.0f);
  Vector3 vertical(0.0f, 2 * HALF_Y, 0.0f);
  Vector3 origin(0.0f, 0.0f, 0.0f);

  std::vector<std::unique_ptr<IHitable>> world;
  world.push_back(std::make_unique<Sphere>(Vector3(0.0f, 0.0f, -1.0f), 0.5f));
  world.push_back(std::make_unique<Sphere>(Vector3(0.0f, -100.5f, -1.0f), 100.f));

  for (int j = 0; j < nY; ++j)
  {
    for (int i = 0; i < nX; ++i)
    {
      const float u = float(i) / float(nX);
      const float v = float(nY - j) / float(nY);

      using DirectX::SimpleMath::Ray;
      Ray ray(origin, lower_left_corner + u * horizontal + v * vertical);

      using DirectX::SimpleMath::Color;
      Color col  = color(ray, world);
      auto& dest = image.buffer[j * nX + i];
      dest[0]    = u8(255.99 * col.R());
      dest[1]    = u8(255.99 * col.G());
      dest[2]    = u8(255.99 * col.B());
    }
  }
  return image;
}

//------------------------------------------------------------------------------
bool
RayTracer::saveImage(const Image& image, const std::wstring& fileName) const
{
  CHAR strFile[MAX_PATH];
  WideCharToMultiByte(
    CP_ACP,
    WC_NO_BEST_FIT_CHARS,
    fileName.c_str(),
    -1,
    strFile,
    MAX_PATH,
    nullptr,
    FALSE);

  return (
    stbi_write_bmp(strFile, image.width, image.height, 3, image.buffer.data())
    != 0);
}

//------------------------------------------------------------------------------

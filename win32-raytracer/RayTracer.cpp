#include "pch.h"
#include "RayTracer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma warning(push)
#pragma warning(disable : 4996)
#include "stb_image_write.h"
#pragma warning(pop)

using namespace ray;

//------------------------------------------------------------------------------
struct Sphere
{
  DirectX::SimpleMath::Vector3 center;
  float radius = 1.0f;
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

  for (int j = 0; j < nY; ++j)
  {
    for (int i = 0; i < nX; ++i)
    {
      const float u = float(i) / float(nX);
      const float v = float(nY - j) / float(nY);

      using DirectX::SimpleMath::Ray;
      Ray r(origin, lower_left_corner + u * horizontal + v * vertical);

      using DirectX::SimpleMath::Color;
      Color col  = color(r);
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
std::optional<float>
hit_sphere(
  const DirectX::SimpleMath::Vector3& center,
  float radius,
  const DirectX::SimpleMath::Ray& r)
{
  using DirectX::SimpleMath::Vector3;
  Vector3 rayStart = r.position - center;

  float a            = r.direction.Dot(r.direction);
  float b            = 2.0f * r.direction.Dot(rayStart);
  float c            = rayStart.Dot(rayStart) - radius * radius;
  float discriminant = b * b - 4.0f * a * c;

  if (discriminant < 0.0f)
  {
    return {};
  }

  return (-b - sqrtf(discriminant)) / (2.0f * a);
}

//------------------------------------------------------------------------------
DirectX::SimpleMath::Color
RayTracer::color(const DirectX::SimpleMath::Ray& r) const
{
  using DirectX::SimpleMath::Color;
  using DirectX::SimpleMath::Vector3;

  // Sphere
  Sphere sphere = {{0.0f, 0.0f, -1.0f}, 0.5f};
  if (auto t = hit_sphere(sphere.center, sphere.radius, r))
  {
    Vector3 hitPoint = r.position + (*t * r.direction);
    Vector3 normal   = hitPoint - sphere.center;
    normal.Normalize();

    return Color(quantize(normal.x), quantize(normal.y), quantize(normal.z));
  }

  // Background
  Vector3 unit_direction = r.direction;
  unit_direction.Normalize();
  float t = quantize(unit_direction.y);
  return ((1.0f - t) * Color(1.0f, 1.0f, 1.0f)) + (t * Color(0.5f, 0.7f, 1.0f));
}

//------------------------------------------------------------------------------

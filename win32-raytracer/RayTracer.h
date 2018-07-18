#pragma once

#include "pch.h"

namespace ray
{
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
  virtual std::optional<ScatterRecord>
  scatter(const DirectX::SimpleMath::Ray& ray, const HitRecord& rec) const = 0;

  virtual ~IMaterial() = default;
};

//------------------------------------------------------------------------------
struct IHitable
{
  virtual std::optional<HitRecord>
  hit(const DirectX::SimpleMath::Ray& ray, float tMin, float tMax) const = 0;

  virtual ~IHitable() = default;
};

//------------------------------------------------------------------------------
class RayTracer
{
public:
  RayTracer();

  using World = std::vector<std::unique_ptr<IHitable>>;

  World getTestScene() const;
  World generateRandomScene() const;

  Image generateImage(const World& world) const;
  bool saveImage(const Image& image, const std::wstring& fileName) const;
};

//------------------------------------------------------------------------------
};

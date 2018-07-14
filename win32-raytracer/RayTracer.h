#pragma once

#include "pch.h"

namespace ray
{
//------------------------------------------------------------------------------
class RayTracer
{
public:
  RayTracer();

  Image generateImage() const;
  bool saveImage(const Image& image, const std::wstring& fileName) const;
};

//------------------------------------------------------------------------------
};

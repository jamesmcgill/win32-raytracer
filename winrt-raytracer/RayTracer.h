#pragma once

#include "pch.h"

namespace ray
{
//------------------------------------------------------------------------------
class RayTracer
{
public:
  RayTracer();

  Image generateImage();
  bool saveImage(const Image& image, const std::string& fileName);
};

//------------------------------------------------------------------------------
};

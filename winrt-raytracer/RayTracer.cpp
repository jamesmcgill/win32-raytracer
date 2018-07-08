#include "pch.h"
#include "RayTracer.h"

using namespace ray;

//------------------------------------------------------------------------------
RayTracer::RayTracer() {}

//------------------------------------------------------------------------------
Image
RayTracer::generateImage()
{
  const size_t nX = 800;
  const size_t nY = 500;

  Image image;
  image.width  = nX;
  image.height = nY;
  image.buffer.resize(nX * nY);

  for (int j = 0; j < nY; ++j)
  {
    for (int i = 0; i < nX; ++i)
    {
      float r = float(i) / float(nX);
      float g = float(nY - j) / float(nY);
      float b = 0.2f;

      auto& dest = image.buffer[j * nX + i];
      dest[0]    = u8(255.99 * r);
      dest[1]    = u8(255.99 * g);
      dest[2]    = u8(255.99 * b);
    }
  }
  return image;
}

//------------------------------------------------------------------------------
bool
RayTracer::saveImage(const Image& image, const std::string& fileName)
{
  std::ofstream fileOut(fileName);
  if (!fileOut.is_open())
  {
    return false;
  }

  fileOut << "P3\n" << image.width << " " << image.height << "\n255\n";
  for (auto& px : image.buffer)
  {
    fileOut << (int)px[0] << " " << (int)px[1] << " " << (int)px[2] << "\n";
  }
  return true;
}
//------------------------------------------------------------------------------

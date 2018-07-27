#pragma once

#include "pch.h"

//------------------------------------------------------------------------------
namespace ptr
{
struct RenderResult
{
  std::chrono::milliseconds renderDuration;
  bool isError = false;
  std::vector<Image> imageParts;
};

//------------------------------------------------------------------------------
RenderResult
render(const int imageWidth, const int imageHeight, const int numSamples);

//------------------------------------------------------------------------------
template <typename Func>
std::thread
asyncRender(
  const int imageWidth,
  const int imageHeight,
  const int numSamples,
  Func onCompleteCallback)
{
  return std::thread(
    [imageWidth, imageHeight, numSamples, onCompleteCallback]() {
      RenderResult result = render(imageWidth, imageHeight, numSamples);
      onCompleteCallback(result);
    });
}

//------------------------------------------------------------------------------
};

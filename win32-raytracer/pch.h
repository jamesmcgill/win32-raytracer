//
// pch.h
// Header for standard system include files.
//

#pragma once

// Off by default warnings - Taken from DirectXTK pch.h
#pragma warning( \
  disable : 4619 4616 4061 4265 4365 4571 4582 4623 4625 4626 4628 4668 4710 4711 4746 4774 4820 4987 5026 5027 5031 5032 5039 5045)
// C4619/4616 #pragma warning warnings
// C4061 enumerator 'X' in switch of enum 'X' is not explicitly handled by a
// case label C4265 class has virtual functions, but destructor is not virtual
// C4365 signed/unsigned mismatch
// C4571 behavior change
// C4571 constructor is not implicitly called(issue with optional with POD type)
// C4623 default constructor was implicitly defined as deleted
// C4625 copy constructor was implicitly defined as deleted
// C4626 assignment operator was implicitly defined as deleted
// C4628 digraphs not supported
// C4668 not defined as a preprocessor macro
// C4710 function not inlined
// C4711 selected for automatic inline expansion
// C4746 volatile access of '<expression>' is subject to /volatile:<iso|ms>
// setting C4774 format string expected in argument 3 is not a string literal
// C4820 padding added after data member
// C4987 nonstandard extension used
// C5026 move constructor was implicitly defined as deleted
// C5027 move assignment operator was implicitly defined as deleted
// C5031/5032 push/pop mismatches in windows headers
// C5039 pointer or reference to potentially throwing function passed to extern
// C function under - EHc C5045 Spectre mitigation warning

#include <WinSDKVer.h>
#define _WIN32_WINNT 0x0601
#include <SDKDDKVer.h>

// Use the C++ standard templated min/max
#define NOMINMAX

// DirectX apps don't need GDI
#define NODRAWTEXT
#define NOGDI
#define NOBITMAP

// Include <mcx.h> if you need this
#define NOMCX

// Include <winsvc.h> if you need this
#define NOSERVICE

// WinHelp is deprecated
#define NOHELP

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#include <inttypes.h>

#include <wrl/client.h>

#include <d3d11_1.h>

#if defined(NTDDI_WIN10_RS2)
#include <dxgi1_6.h>
#else
#include <dxgi1_5.h>
#endif

#include <DirectXMath.h>
#include <DirectXColors.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <fstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <optional>
#include <limits>
#include <stdio.h>
#include <random>
#include <numeric>

#ifdef _DEBUG
#include <dxgidebug.h>
#endif

// DirectXTK
#include "CommonStates.h"
#include "DDSTextureLoader.h"
#include "DirectXHelpers.h"
#include "Effects.h"
#include "GamePad.h"
#include "GeometricPrimitive.h"
#include "GraphicsMemory.h"
#include "Keyboard.h"
#include "Model.h"
#include "Mouse.h"
#include "PostProcess.h"
#include "PrimitiveBatch.h"
#include "ScreenGrab.h"
#include "SimpleMath.h"
#include "SpriteBatch.h"
#include "SpriteFont.h"
#include "VertexTypes.h"
#include "WICTextureLoader.h"

#include "imgui-1.62/imgui.h"
#include "imgui-1.62/imgui_impl_dx11.h"
#include "imgui-1.62/imgui_impl_win32.h"

namespace DX
{
// Helper class for COM exceptions
class com_exception : public std::exception
{
public:
  com_exception(HRESULT hr)
      : result(hr)
  {
  }

  virtual const char* what() const override
  {
    static char s_str[64] = {};
    sprintf_s(
      s_str, "Failure with HRESULT of %08X", static_cast<unsigned int>(result));
    return s_str;
  }

private:
  HRESULT result;
};

// Helper utility converts D3D API failures into exceptions.
inline void
ThrowIfFailed(HRESULT hr)
{
  if (FAILED(hr))
  {
    throw com_exception(hr);
  }
}
}    // namespace DX

//------------------------------------------------------------------------------
using u8 = uint8_t;

namespace ptr
{
using Pixel       = std::array<u8, 3>;
using ImageBuffer = std::vector<Pixel>;
struct Image
{
  int width  = 0;
  int height = 0;
  ImageBuffer buffer;
};

constexpr int DEFAULT_IMAGE_WIDTH  = 800;
constexpr int DEFAULT_IMAGE_HEIGHT = 600;
constexpr int DEFAULT_NUM_SAMPLES  = 100;
constexpr int MAX_RECURSION        = 10;
constexpr int DEFAULT_NUM_THREADS  = 14;

extern int IMAGE_WIDTH;
extern int IMAGE_HEIGHT;
extern int NUM_SAMPLES;
extern int NUM_THREADS;

static const wchar_t* IMAGE_FILENAME = L"out.bmp";
};    // namespace ptr
//------------------------------------------------------------------------------

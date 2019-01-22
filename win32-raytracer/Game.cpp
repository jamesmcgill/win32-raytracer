//
// Game.cpp
//

#include "pch.h"
#include "Game.h"
#include "RayTracer.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_IMPLEMENTATION
#pragma warning(push)
#pragma warning(disable : 4100)    // Unreferenced parameter
#include "stb_image.h"
#pragma warning(pop)

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

extern void ExitGame();

using namespace DirectX;

using Microsoft::WRL::ComPtr;

//------------------------------------------------------------------------------
bool
saveImage(const ptr::Image& image, const std::wstring& fileName)
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
Game::Game() noexcept(false)
{
  m_deviceResources = std::make_unique<DX::DeviceResources>();
  m_deviceResources->RegisterDeviceNotify(this);
}

//------------------------------------------------------------------------------
// Initialize the Direct3D resources required to run.
//------------------------------------------------------------------------------
void
Game::Initialize(HWND window, int width, int height)
{
  m_hwnd = window;
  m_deviceResources->SetWindow(window, width, height);

  m_deviceResources->CreateDeviceResources();
  CreateDeviceDependentResources();

  m_deviceResources->CreateWindowSizeDependentResources();
  CreateWindowSizeDependentResources();

  // TODO: Change the timer settings if you want something other than the
  // default variable timestep mode. e.g. for 60 FPS fixed timestep update
  // logic, call:
  /*
  m_timer.SetFixedTimeStep(true);
  m_timer.SetTargetElapsedSeconds(1.0 / 60);
  */

  ImGui::CreateContext();
  ImGui_ImplWin32_Init(window);
  ImGui_ImplDX11_Init(
    m_deviceResources->GetD3DDevice(),
    m_deviceResources->GetD3DDeviceContext());

  // TODO: Fill optional settings of the io structure later.
  // TODO: Load fonts if you don't want to use the default font.

  auto onRenderComplete = [this](const ptr::RenderResult& result) {
    m_renderDuration = result.renderDuration;

    if (result.imageParts.empty())
    {
      m_isError     = true;
      m_isRendering = false;
      return;
    }

    // Stich image pieces together
    ptr::Image image;
    image.width  = ptr::IMAGE_WIDTH;
    image.height = 0;
    for (const auto& i : result.imageParts)
    {
      image.buffer.insert(image.buffer.end(), i.buffer.begin(), i.buffer.end());
      image.height += i.height;
    }

    if (!saveImage(image, ptr::IMAGE_FILENAME))
    {
      m_isError = true;
    }

    m_isRendering = false;    // Always set last
  };

  // Start Rendering
  m_isRendering  = true;
  m_renderThread = ptr::asyncRender(
    ptr::IMAGE_WIDTH, ptr::IMAGE_HEIGHT, ptr::NUM_SAMPLES, onRenderComplete);
}

//------------------------------------------------------------------------------
void
Game::ShutDown()
{
  if (m_renderThread.joinable())
  {
    m_renderThread.join();
  }

  ImGui_ImplDX11_Shutdown();
  ImGui_ImplWin32_Shutdown();
  ImGui::DestroyContext();
}

//------------------------------------------------------------------------------

#pragma region Frame Update

//------------------------------------------------------------------------------
// Executes the basic game loop.
//------------------------------------------------------------------------------
void
Game::Tick()
{
  m_timer.Tick([&]() { Update(m_timer); });

  Render();
}

//------------------------------------------------------------------------------
// Updates the world.
//------------------------------------------------------------------------------
void
Game::Update(DX::StepTimer const& timer)
{
  UNREFERENCED_PARAMETER(timer);
  // float elapsedTime = float(timer.GetElapsedSeconds());

  // TODO: Add your game logic here.
}
#pragma endregion

#pragma region Frame Render

//------------------------------------------------------------------------------
bool
Game::CreateTexture()
{
  ComPtr<ID3D11Resource> resource;
  DX::ThrowIfFailed(CreateWICTextureFromFile(
    m_deviceResources->GetD3DDevice(),
    ptr::IMAGE_FILENAME,
    resource.GetAddressOf(),
    m_texture.ReleaseAndGetAddressOf()));

  ComPtr<ID3D11Texture2D> cat;
  DX::ThrowIfFailed(resource.As(&cat));

  CD3D11_TEXTURE2D_DESC catDesc;
  cat->GetDesc(&catDesc);

  m_origin.x = float(catDesc.Width / 2);
  m_origin.y = float(catDesc.Height / 2);

  return true;
}

//------------------------------------------------------------------------------
void
write_perf_results(uint64_t durationMs)
{
  std::ofstream file("perf.txt");
  file << durationMs;
}

//------------------------------------------------------------------------------
// Draws the scene.
//------------------------------------------------------------------------------
void
Game::Render()
{
  // Don't try to render anything before the first Update.
  if (m_timer.GetFrameCount() == 0)
  {
    return;
  }

  Clear();

  m_deviceResources->PIXBeginEvent(L"Render");

  // Call NewFrame(), after this point you can use ImGui::* functions anytime
  // (So you want to try calling Newframe() as early as you can in your
  // mainloop to be able to use imgui everywhere)
  ImGui_ImplDX11_NewFrame();
  ImGui_ImplWin32_NewFrame();
  ImGui::NewFrame();

  if (m_isRendering)
  {
    ImGui::Text("Reticulating splines...");
  }
  else
  {
    auto duration = m_renderDuration.load();
    if (ptr::IS_PERF_TEST)
    {
      write_perf_results(duration.count());
      ExitGame();
      return;
    }

    if (m_isError)
    {
      ImGui::Text("Error saving file!");
    }
    else
    {
      ImGui::Text("Done!");
      if (!m_isTextureCreated)
      {
        m_isTextureCreated = true;
        CreateTexture();
      }
      m_spriteBatch->Begin();

      m_spriteBatch->Draw(
        m_texture.Get(), m_screenPos, nullptr, Colors::White, 0.f, m_origin);

      m_spriteBatch->End();
    }

    ImGui::Text("Render duration: %ld ms", duration);
  }

  // auto context = m_deviceResources->GetD3DDeviceContext();

  // TODO: Add your rendering code here.
  // context;

  // Render imgui, swap buffers
  // (You want to try calling EndFrame/Render as late as you can, to be able
  // to use imgui in your own game rendering code)
  ImGui::EndFrame();
  ImGui::Render();
  ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

  m_deviceResources->PIXEndEvent();

  // Show the new frame.
  m_deviceResources->Present();
}

//------------------------------------------------------------------------------
// Helper method to clear the back buffers.
//------------------------------------------------------------------------------
void
Game::Clear()
{
  m_deviceResources->PIXBeginEvent(L"Clear");

  // Clear the views.
  auto context      = m_deviceResources->GetD3DDeviceContext();
  auto renderTarget = m_deviceResources->GetRenderTargetView();
  auto depthStencil = m_deviceResources->GetDepthStencilView();

  context->ClearRenderTargetView(renderTarget, Colors::CornflowerBlue);
  context->ClearDepthStencilView(
    depthStencil, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
  context->OMSetRenderTargets(1, &renderTarget, depthStencil);

  // Set the viewport.
  auto viewport = m_deviceResources->GetScreenViewport();
  context->RSSetViewports(1, &viewport);

  m_deviceResources->PIXEndEvent();
}
#pragma endregion

#pragma region Message Handlers
//------------------------------------------------------------------------------
// Message handlers
//------------------------------------------------------------------------------
void
Game::OnActivated()
{
  // TODO: Game is becoming active window.
}

//------------------------------------------------------------------------------
void
Game::OnDeactivated()
{
  // TODO: Game is becoming background window.
}

//------------------------------------------------------------------------------
void
Game::OnSuspending()
{
  // TODO: Game is being power-suspended (or minimized).
}

//------------------------------------------------------------------------------
void
Game::OnResuming()
{
  m_timer.ResetElapsedTime();

  // TODO: Game is being power-resumed (or returning from minimize).
}

//------------------------------------------------------------------------------
void
Game::OnWindowMoved()
{
  auto r = m_deviceResources->GetOutputSize();
  m_deviceResources->WindowSizeChanged(r.right, r.bottom);
}

//------------------------------------------------------------------------------
void
Game::OnWindowSizeChanged(int width, int height)
{
  if (!m_deviceResources->WindowSizeChanged(width, height))
    return;

  CreateWindowSizeDependentResources();
}
#pragma endregion

#pragma region Direct3D Resources
//------------------------------------------------------------------------------
// These are the resources that depend on the device.
//------------------------------------------------------------------------------
void
Game::CreateDeviceDependentResources()
{
  ImGui_ImplDX11_CreateDeviceObjects();

  m_spriteBatch
    = std::make_unique<SpriteBatch>(m_deviceResources->GetD3DDeviceContext());
}

//------------------------------------------------------------------------------
// Allocate all memory resources that change on a window SizeChanged event.
//------------------------------------------------------------------------------
void
Game::CreateWindowSizeDependentResources()
{
  RECT outputSize = m_deviceResources->GetOutputSize();
  float width     = static_cast<float>(outputSize.right - outputSize.left);
  float height    = static_cast<float>(outputSize.bottom - outputSize.top);

  m_screenPos.x = width / 2.f;
  m_screenPos.y = height / 2.f;
}

//------------------------------------------------------------------------------
void
Game::OnDeviceLost()
{
  m_spriteBatch.reset();
  m_texture.Reset();
  ImGui_ImplDX11_InvalidateDeviceObjects();
}

//------------------------------------------------------------------------------
void
Game::OnDeviceRestored()
{
  CreateDeviceDependentResources();

  CreateWindowSizeDependentResources();
}

//------------------------------------------------------------------------------
#pragma endregion

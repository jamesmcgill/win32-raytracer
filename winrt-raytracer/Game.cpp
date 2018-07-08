//
// Game.cpp
//

#include "pch.h"
#include "Game.h"
#include "RayTracer.h"

extern void ExitGame();

using namespace DirectX;

using Microsoft::WRL::ComPtr;

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

  // Start Rendering
  auto render = [&]() {
    auto start = std::chrono::high_resolution_clock::now();

    ray::RayTracer rt;
    auto image       = rt.generateImage();
    m_renderDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start);

    start = std::chrono::high_resolution_clock::now();
    if (!rt.saveImage(image, "out.ppm"))
    {
      m_isError = true;
    }
    m_saveDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start);
    m_isDone = true;

  };

  m_isDone       = false;
  m_renderThread = std::thread(render);
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

  if (m_isDone)
  {
    if (m_isError)
    {
      ImGui::Text("Error saving file!");
    }
    else
    {
      ImGui::Text("Done!");
    }
    ImGui::Text("Render duration: %ld ms", m_renderDuration.load());
    ImGui::Text("File Save duration: %ld ms", m_saveDuration.load());
  }
  else
  {
    ImGui::Text("Reticulating splines...");
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

//------------------------------------------------------------------------------
// Properties
//------------------------------------------------------------------------------
void
Game::GetDefaultSize(int& width, int& height) const
{
  // TODO: Change to desired default window size (note minimum size is 320x200).
  width  = 800;
  height = 600;
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
}

//------------------------------------------------------------------------------
// Allocate all memory resources that change on a window SizeChanged event.
//------------------------------------------------------------------------------
void
Game::CreateWindowSizeDependentResources()
{
  // TODO: Initialize windows-size dependent objects here.
}

//------------------------------------------------------------------------------
void
Game::OnDeviceLost()
{
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

//
// Game.h
//

#pragma once

#include "DeviceResources.h"
#include "StepTimer.h"

// A basic game implementation that creates a D3D11 device and
// provides a game loop.
class Game : public DX::IDeviceNotify
{
public:
  Game() noexcept(false);
  virtual ~Game() {}

  Game(const Game&) = delete;
  Game& operator=(const Game&) = delete;

  // Initialization and management
  void Initialize(HWND window, int width, int height);
  void ShutDown();

  // Basic game loop
  void Tick();

  // IDeviceNotify
  virtual void OnDeviceLost() override;
  virtual void OnDeviceRestored() override;

  // Messages
  void OnActivated();
  void OnDeactivated();
  void OnSuspending();
  void OnResuming();
  void OnWindowMoved();
  void OnWindowSizeChanged(int width, int height);

private:
  bool CreateTexture();

  void Update(DX::StepTimer const& timer);
  void Render();

  void Clear();

  void CreateDeviceDependentResources();
  void CreateWindowSizeDependentResources();

  // Device resources.
  std::unique_ptr<DX::DeviceResources> m_deviceResources;

  // Rendering loop timer.
  DX::StepTimer m_timer;

  HWND m_hwnd = nullptr;

  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_texture;
  bool m_isTextureCreated = false;

  std::atomic<bool> m_isRendering = true;
  std::atomic<bool> m_isError     = false;
  std::atomic<std::chrono::milliseconds> m_renderDuration;
  std::thread m_renderThread;

  std::unique_ptr<DirectX::SpriteBatch> m_spriteBatch;
  DirectX::SimpleMath::Vector2 m_screenPos;
  DirectX::SimpleMath::Vector2 m_origin;
};

//
// Main.cpp
//

#include "pch.h"
#include "Game.h"

using namespace DirectX;

namespace
{
std::unique_ptr<Game> g_game;
};

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(
  HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Indicates to hybrid graphics systems to prefer the discrete part by default
extern "C" {
__declspec(dllexport) DWORD NvOptimusEnablement                = 0x00000001;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

int ptr::IMAGE_WIDTH = ptr::DEFAULT_IMAGE_WIDTH;
int ptr::IMAGE_HEIGHT = ptr::DEFAULT_IMAGE_HEIGHT;
int ptr::NUM_SAMPLES  = ptr::DEFAULT_NUM_SAMPLES;
int ptr::NUM_THREADS  = ptr::DEFAULT_NUM_THREADS;

//------------------------------------------------------------------------------
// Entry point
//------------------------------------------------------------------------------
int WINAPI
wWinMain(
  _In_ HINSTANCE hInstance,
  _In_opt_ HINSTANCE hPrevInstance,
  _In_ LPWSTR lpCmdLine,
  _In_ int nCmdShow)
{
  UNREFERENCED_PARAMETER(hPrevInstance);
  UNREFERENCED_PARAMETER(lpCmdLine);

  if (!XMVerifyCPUSupport())
    return 1;

  HRESULT hr = CoInitializeEx(nullptr, COINITBASE_MULTITHREADED);
  if (FAILED(hr))
    return 1;

  g_game = std::make_unique<Game>();

  // Register class and create window
  {
    // Register class
    WNDCLASSEX wcex;
    wcex.cbSize        = sizeof(WNDCLASSEX);
    wcex.style         = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc   = WndProc;
    wcex.cbClsExtra    = 0;
    wcex.cbWndExtra    = 0;
    wcex.hInstance     = hInstance;
    wcex.hIcon         = LoadIcon(hInstance, L"IDI_ICON");
    wcex.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszMenuName  = nullptr;
    wcex.lpszClassName = L"win32_raytracerWindowClass";
    wcex.hIconSm       = LoadIcon(wcex.hInstance, L"IDI_ICON");
    if (!RegisterClassEx(&wcex))
      return 1;


    // Parse Command Line (width, height, samples, num threads)
    int nArgs;
    LPWSTR* szArglist = CommandLineToArgvW(GetCommandLineW(), &nArgs);
    if (NULL != szArglist)
    {
      int value;
      if (nArgs > 2)
      {
         value = _wtoi(szArglist[1]);
        if (value != 0 && value != INT_MAX && value != INT_MIN)
        {
          ptr::IMAGE_WIDTH = value;
        }

        value = _wtoi(szArglist[2]);
        if (value != 0 && value != INT_MAX && value != INT_MIN)
        {
          ptr::IMAGE_HEIGHT = value;
        }
      }

      if (nArgs > 3)
      {
        value = _wtoi(szArglist[3]);
        if (value != 0 && value != INT_MAX && value != INT_MIN)
        {
          ptr::NUM_SAMPLES = value;
        }
      }

      if (nArgs > 4)
      {
        value = _wtoi(szArglist[4]);
        if (value != 0 && value != INT_MAX && value != INT_MIN)
        {
          ptr::NUM_THREADS = value;
        }
      }

    }
    LocalFree(szArglist);

  // Create window
    RECT rc;
    rc.top    = 0;
    rc.left   = 0;
    rc.right  = static_cast<LONG>(ptr::IMAGE_WIDTH);
    rc.bottom = static_cast<LONG>(ptr::IMAGE_HEIGHT);

    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

    HWND hwnd = CreateWindowEx(
      0,
      L"win32_raytracerWindowClass",
      L"Raytracer",
      WS_OVERLAPPEDWINDOW,
      CW_USEDEFAULT,
      CW_USEDEFAULT,
      rc.right - rc.left,
      rc.bottom - rc.top,
      nullptr,
      nullptr,
      hInstance,
      nullptr);
    // TODO: Change to CreateWindowEx(WS_EX_TOPMOST,
    // L"win32_raytracerWindowClass", L"win32-raytracer", WS_POPUP, to default
    // to fullscreen.

    if (!hwnd)
      return 1;

    ShowWindow(hwnd, nCmdShow);
    // TODO: Change nCmdShow to SW_SHOWMAXIMIZED to default to fullscreen.

    SetWindowLongPtr(
      hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(g_game.get()));

    GetClientRect(hwnd, &rc);

    g_game->Initialize(hwnd, rc.right - rc.left, rc.bottom - rc.top);
  }

  // Main message loop
  MSG msg = {};
  while (WM_QUIT != msg.message)
  {
    if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
    {
      TranslateMessage(&msg);
      DispatchMessage(&msg);
    }
    else
    {
      g_game->Tick();
    }
  }

  g_game->ShutDown();
  g_game.reset();

  CoUninitialize();

  return (int)msg.wParam;
}

//------------------------------------------------------------------------------
// Windows procedure
//------------------------------------------------------------------------------
LRESULT CALLBACK
WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
  if (ImGui_ImplWin32_WndProcHandler(hWnd, message, wParam, lParam))
  {
    return true;
  }

  PAINTSTRUCT ps;
  HDC hdc;

  static bool s_in_sizemove = false;
  static bool s_in_suspend  = false;
  static bool s_minimized   = false;
  static bool s_fullscreen  = false;
  // TODO: Set s_fullscreen to true if defaulting to fullscreen.

  auto game = reinterpret_cast<Game*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));

  switch (message)
  {
    case WM_PAINT:
      if (s_in_sizemove && game)
      {
        game->Tick();
      }
      else
      {
        hdc = BeginPaint(hWnd, &ps);
        EndPaint(hWnd, &ps);
      }
      break;

    case WM_MOVE:
      if (game)
      {
        game->OnWindowMoved();
      }
      break;

    case WM_SIZE:
      if (wParam == SIZE_MINIMIZED)
      {
        if (!s_minimized)
        {
          s_minimized = true;
          if (!s_in_suspend && game)
            game->OnSuspending();
          s_in_suspend = true;
        }
      }
      else if (s_minimized)
      {
        s_minimized = false;
        if (s_in_suspend && game)
          game->OnResuming();
        s_in_suspend = false;
      }
      else if (!s_in_sizemove && game)
      {
        game->OnWindowSizeChanged(LOWORD(lParam), HIWORD(lParam));
      }
      break;

    case WM_ENTERSIZEMOVE:
      s_in_sizemove = true;
      break;

    case WM_EXITSIZEMOVE:
      s_in_sizemove = false;
      if (game)
      {
        RECT rc;
        GetClientRect(hWnd, &rc);

        game->OnWindowSizeChanged(rc.right - rc.left, rc.bottom - rc.top);
      }
      break;

    case WM_GETMINMAXINFO:
    {
      auto info              = reinterpret_cast<MINMAXINFO*>(lParam);
      info->ptMinTrackSize.x = 320;
      info->ptMinTrackSize.y = 200;
    }
    break;

    case WM_ACTIVATEAPP:
      if (game)
      {
        if (wParam)
        {
          game->OnActivated();
        }
        else
        {
          game->OnDeactivated();
        }
      }
      break;

    case WM_POWERBROADCAST:
      switch (wParam)
      {
        case PBT_APMQUERYSUSPEND:
          if (!s_in_suspend && game)
            game->OnSuspending();
          s_in_suspend = true;
          return TRUE;

        case PBT_APMRESUMESUSPEND:
          if (!s_minimized)
          {
            if (s_in_suspend && game)
              game->OnResuming();
            s_in_suspend = false;
          }
          return TRUE;
      }
      break;

    case WM_DESTROY:
      PostQuitMessage(0);
      break;

    case WM_SYSKEYDOWN:
      if (wParam == VK_RETURN && (lParam & 0x60000000) == 0x20000000)
      {
        // Implements the classic ALT+ENTER fullscreen toggle
        if (s_fullscreen)
        {
          SetWindowLongPtr(hWnd, GWL_STYLE, WS_OVERLAPPEDWINDOW);
          SetWindowLongPtr(hWnd, GWL_EXSTYLE, 0);
          ShowWindow(hWnd, SW_SHOWNORMAL);

          SetWindowPos(
            hWnd,
            HWND_TOP,
            0,
            0,
            ptr::IMAGE_WIDTH,
            ptr::IMAGE_HEIGHT,
            SWP_NOMOVE | SWP_NOZORDER | SWP_FRAMECHANGED);
        }
        else
        {
          SetWindowLongPtr(hWnd, GWL_STYLE, 0);
          SetWindowLongPtr(hWnd, GWL_EXSTYLE, WS_EX_TOPMOST);

          SetWindowPos(
            hWnd,
            HWND_TOP,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);

          ShowWindow(hWnd, SW_SHOWMAXIMIZED);
        }

        s_fullscreen = !s_fullscreen;
      }
      break;

    case WM_MENUCHAR:
      // A menu is active and the user presses a key that does not correspond
      // to any mnemonic or accelerator key. Ignore so we don't produce an error
      // beep.
      return MAKELRESULT(0, MNC_CLOSE);
  }

  return DefWindowProc(hWnd, message, wParam, lParam);
}

//------------------------------------------------------------------------------
// Exit helper
//------------------------------------------------------------------------------
void
ExitGame()
{
  PostQuitMessage(0);
}

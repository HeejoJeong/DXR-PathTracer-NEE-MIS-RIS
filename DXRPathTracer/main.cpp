#include "pch.h"
#include "DXRPathTracer.h"
#include "D3D12Screen.h"
#include "SceneLoader.h"
#include "Input.h"
#include "timer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

HWND createWindow(const char* winTitle, uint width, uint height);
void exportIMG(uint frame, const TracedResult& trResult);


IGRTTracer* tracer;
IGRTScreen* screen;
uint width  = 1200;
uint height = 900;
bool minimized = false;


int main()
{
	HWND hwnd = createWindow("Integrated GPU Path Tracer", width, height);
	ShowWindow(hwnd, SW_SHOW);
	InputEngine input(hwnd);

	tracer = new DXRPathTracer(width, height);
	screen = new D3D12Screen(hwnd, width, height);

	SceneLoader sceneLoader;

	Scene* scene = sceneLoader.push_hyperionTestScene();
	//Scene* scene = sceneLoader.push_RIStestScene();

	tracer->setupScene(scene);
	
	double fps, old_fps = 0;
	uint cur_frame = 0;
	while (IsWindow(hwnd))
	{
		input.update();
		
		if (!minimized)
		{
			tracer->update(input);
			TracedResult trResult = tracer->shootRays();
			screen->display(trResult);

			//exportIMG(cur_frame++, trResult);
			//if (cur_frame == 512 )
			//	PostMessage(hwnd, WM_CLOSE, 0, 0);

		}

		MSG msg;
		while(PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}

		fps = updateFPS(1.0);
		if (fps != old_fps)
		{
			printf("FPS: %f\n", fps);
			old_fps = fps;
		}
	}

	return 0;
}

LRESULT CALLBACK msgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

HWND createWindow(const char* winTitle, uint width, uint height)
{
	WNDCLASSA wc = {};
	wc.lpfnWndProc = msgProc;
	wc.hInstance = GetModuleHandle(nullptr);
	wc.lpszClassName = "anything";
	wc.style = CS_HREDRAW | CS_VREDRAW;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	RegisterClass(&wc);

	RECT r{ 0, 0, (LONG)width, (LONG)height };
	AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, false);

	HWND hWnd = CreateWindowA(
		wc.lpszClassName,
		winTitle,
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		r.right - r.left,
		r.bottom - r.top,
		nullptr,
		nullptr,
		wc.hInstance,
		nullptr);

	return hWnd;
}


void exportIMG(uint frame, const TracedResult& trResult) {

	if (frame != 512 - 1)
		return;

	printf("EXPORT at 512 frame \n");

	struct float4 { float x, y, z, w; };

	float4* pixels = reinterpret_cast<float4*>(trResult.data);

	uint nanInfCount = 0;
	uint firstX = 0, firstY = 0, firstIdx = 0;
	float4 firstVal = {};

	for (uint y = 0; y < trResult.height; ++y)
	{
		for (uint x = 0; x < trResult.width; ++x)
		{
			uint idx = y * trResult.width + x;
			const float4& p = pixels[idx];

			if (!std::isfinite(p.x) ||
				!std::isfinite(p.y) ||
				!std::isfinite(p.z) ||
				!std::isfinite(p.w))
			{
				if (nanInfCount == 0)
				{
					firstX = x;
					firstY = y;
					firstIdx = idx;
					firstVal = p;
				}
				++nanInfCount;

				printf("Nan/Inf: (%u,%u)\n", x, y);
			}
		}
	}

	if (nanInfCount == 0)
	{
		printf("No NaN / INF found in tracer output.\n");
	}
	else
	{
		printf(
			"NaN/INF found: %u pixels. First at (%u,%u) idx=%u : (%f %f %f %f)\n",
			nanInfCount,
			firstX, firstY, firstIdx,
			firstVal.x, firstVal.y, firstVal.z, firstVal.w
		);
		throw Error("NaN/INF found.");
	}

	bool allZero = true;

	for (int i = 0; i < trResult.width * trResult.height; ++i)
	{
		if (pixels[i].x != 0.0f ||
			pixels[i].y != 0.0f ||
			pixels[i].z != 0.0f)
		{
			allZero = false;
			break;
		}
	}

	if (allZero)
		printf("All pixels are zero.\n");


	const int width = trResult.width;
	const int height = trResult.height;

	float exposure = 1.0f;
	float gamma = 2.2f;

	Array<unsigned char> ldr(width * height * 4);
	
	//auto toByte = [&](float v)
	//	{
	//		v = min(max(v, 0.0f), 1.0f);
	//		return static_cast<unsigned char>(min(v, 1.0f) * 255.0f + 0.5f);
	//	};

	auto toByte = [&](float v)
		{
			v *= exposure;
			v = v / (1.0f + v);
			v = powf(v, 1.0f / gamma);
			v = min(max(v, 0.0f), 1.0f);
			return static_cast<unsigned char>(v * 255.0f + 0.5f);
		};

	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			int i = y * width + x;
			int o = i * 4;

			const float4& p = pixels[i];

			ldr[o + 0] = toByte(p.x);
			ldr[o + 1] = toByte(p.y);
			ldr[o + 2] = toByte(p.z);
			ldr[o + 3] = 255;
		}
	}

	const char* filename = "res.png";
	if (frame == 512 * 8 - 1 )
		filename = "res2.png";

	stbi_write_png(filename, width, height, 4, ldr.data(), width * 4);
}

LRESULT CALLBACK msgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_SIZE:
		if (tracer && screen)
		{
			uint width = (uint) LOWORD(lParam);
			uint height = (uint) HIWORD(lParam);
			if (width == 0 || height == 0)
			{
				minimized = true;
				return 0;
			}
			else if (minimized)
			{
				minimized = false;
			}

			tracer->onSizeChanged(width, height);
			screen->onSizeChanged(width, height);
		}
		return 0;

	/*case WM_PAINT:
		if(tracer && screenOutFormat)
		{
			TracedResult trResult = tracer->shootRays();
			screen->display(trResult);
		}
		return 0;

	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;*/
	}

	return DefWindowProc(hWnd, message, wParam, lParam);
}

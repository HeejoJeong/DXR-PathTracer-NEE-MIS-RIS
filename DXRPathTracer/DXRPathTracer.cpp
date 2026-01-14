#include "pch.h"
#include "DXRPathTracer.h"
#include "Camera.h"
#include "Scene.h"


namespace DescriptorID {
	enum {
		// First RootParameter
		outUAV = 0,

		// Third RootParameter
		sceneObjectBuff = 1,
		vertexBuff = 2,
		tridexBuff = 3,
		materialBuff = 4,
		cdfBuff = 5,
		transformBuff = 6,

		staticLightBuff = 7,

		// Not used since we use RootPointer instead of RootTable
		accelerationStructure = 10,

		maxDesciptors = 32
	};
}

namespace RootParamID {
	enum {
		tableForOutBuffer = 0,
		pointerForAccelerationStructure = 1,
		tableForGeometryInputs = 2,
		pointerForGlobalConstants = 3,
		numParams = 4
	};
}

namespace HitGroupParamID {
	enum {
		constantsForObject = 0,
		numParams = 1
	};
}

DXRPathTracer::~DXRPathTracer()
{
	SAFE_RELEASE(mCmdQueue);
	SAFE_RELEASE(mCmdList);
	SAFE_RELEASE(mCmdAllocator);
	SAFE_RELEASE(mDevice);
}

DXRPathTracer::DXRPathTracer(uint width, uint height)
	: IGRTTracer(width, height)
{
	initD3D12();
	
	mSrvUavHeap.create(DescriptorID::maxDesciptors);

	declareRootSignatures();

	buildRaytracingPipeline();

	initializeApplication();

	//mFence.waitCommandQueue(mCmdQueue);
}

void DXRPathTracer::initD3D12()
{
	mDevice = (ID3D12Device5*) createDX12Device(getRTXAdapter());

	D3D12_COMMAND_QUEUE_DESC cqDesc = {};
	cqDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

	ThrowFailedHR(mDevice->CreateCommandQueue(&cqDesc, IID_PPV_ARGS(&mCmdQueue)));

	ThrowFailedHR(mDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&mCmdAllocator)));

	ThrowFailedHR(mDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, mCmdAllocator, nullptr, IID_PPV_ARGS(&mCmdList)));
	
	ThrowFailedHR(mCmdList->Close());
	ThrowFailedHR(mCmdAllocator->Reset());
	ThrowFailedHR(mCmdList->Reset(mCmdAllocator, nullptr));

	mFence.create(mDevice);
}

void DXRPathTracer::declareRootSignatures()
{
	assert(mSrvUavHeap.get() != nullptr);

	// Global(usual) Root Signature
	mGlobalRS.resize(RootParamID::numParams);
	mGlobalRS[RootParamID::tableForOutBuffer] 
		= new RootTable("u0", mSrvUavHeap[DescriptorID::outUAV].getGpuHandle());
	mGlobalRS[RootParamID::pointerForAccelerationStructure] 
		= new RootPointer("(100) t0");					// It will be bound to mAccelerationStructure that is not initialized yet.
	mGlobalRS[RootParamID::tableForGeometryInputs] 
		= new RootTable("(0) t0-t6", mSrvUavHeap[DescriptorID::sceneObjectBuff].getGpuHandle());
	mGlobalRS[RootParamID::pointerForGlobalConstants] 
		= new RootPointer("b0");						// It will be bound to mGlobalConstantsBuffer that is not initialized yet.
	mGlobalRS.build();

	// Local Root Sinatures
	mHitGroupRS.resize(HitGroupParamID::numParams);
	mHitGroupRS[HitGroupParamID::constantsForObject] 
		= new RootConstants<ObjectContants>("b1");		// Every local root signature's parameter is bound to its values via shader table.
	mHitGroupRS.build(D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE);
}

void DXRPathTracer::buildRaytracingPipeline()
{
	//dxrLib.load(L"DXRShader1.cso");
	//mNumRayTypes = 1;
	//mRtPipeline.setDXRLib(&dxrLib);
	//mRtPipeline.setGlobalRootSignature(&mGlobalRS);
	//mRtPipeline.addHitGroup(HitGroup(L"hitGp", L"closestHit", nullptr));
	//mRtPipeline.addHitGroup(HitGroup(L"hitGpGlass", L"closestHitGlass", nullptr));
	//mRtPipeline.addLocalRootSignature(LocalRootSignature(&mHitGroupRS, { L"hitGp", L"hitGpGlass" }));// L"hitGpShadow"
	//mRtPipeline.setMaxPayloadSize(sizeof(float) * 20);
	//mRtPipeline.setMaxRayDepth(2);
	//mRtPipeline.build();

	dxrLib.load(L"DXRShader_nee.cso");
	mNumRayTypes = 2;
	mRtPipeline.setDXRLib(&dxrLib);
	mRtPipeline.setGlobalRootSignature(&mGlobalRS);
	mRtPipeline.addHitGroup(HitGroup(L"hitGp", L"closestHit", nullptr));
	mRtPipeline.addHitGroup(HitGroup(L"hitGpGlass", L"closestHitGlass", nullptr));
	mRtPipeline.addHitGroup(HitGroup(L"hitGpShadow", L"closestHitShadow", nullptr));
	mRtPipeline.addLocalRootSignature(LocalRootSignature(&mHitGroupRS, { L"hitGp", L"hitGpGlass" }));// L"hitGpShadow"
	mRtPipeline.setMaxPayloadSize(sizeof(float) *20);
	mRtPipeline.setMaxRayDepth(2);
	mRtPipeline.build();
}

void DXRPathTracer::initializeApplication()
{
	camera.setFovY(60.0f);
	camera.setScreenSize((float) tracerOutW, (float) tracerOutH);
	camera.initOrbit(float3(0.0f, 1.5f, 0.0f), 10.0f, 0.0f, 0.0f);

	mGlobalConstants.rayTmin = 0.001f;  // 1mm
	mGlobalConstants.accumulatedFrames = 0;
	mGlobalConstants.numSamplesPerFrame = 32;
	mGlobalConstants.maxPathLength = 5;
	mGlobalConstants.backgroundLight = float3(.0f);

	mGlobalConstantsBuffer.create(sizeof(GloabalContants));
	* (RootPointer*) mGlobalRS[RootParamID::pointerForGlobalConstants] 
		= mGlobalConstantsBuffer.getGpuAddress();

	uint64 maxBufferSize = _bpp(tracerOutFormat) * 1920 *1080;
	mReadBackBuffer.create(maxBufferSize);

	uint64 bufferSize = _bpp(tracerOutFormat) * tracerOutW * tracerOutH;
	mTracerOutBuffer.create(bufferSize);

	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
	{
		uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		uavDesc.Format = tracerOutFormat;
		uavDesc.Buffer.NumElements = tracerOutW * tracerOutH;
	}
	mSrvUavHeap[DescriptorID::outUAV].assignUAV(mTracerOutBuffer, &uavDesc);
}

void DXRPathTracer::onSizeChanged(uint width, uint height)
{
	if (width == tracerOutW && height == tracerOutH)
		return;

	width = width ? width : 1;
	height = height ? height : 1;

	tracerOutW = width;
	tracerOutH = height;

	camera.setScreenSize((float) tracerOutW, (float) tracerOutH);

	UINT64 bufferSize = _bpp(tracerOutFormat) * tracerOutW * tracerOutH;
	mTracerOutBuffer.destroy();
	mTracerOutBuffer.create(bufferSize);

	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
	{
		uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		uavDesc.Format = tracerOutFormat;
		uavDesc.Buffer.NumElements = tracerOutW * tracerOutH;
	}
	mSrvUavHeap[DescriptorID::outUAV].assignUAV(mTracerOutBuffer, &uavDesc);
}

void DXRPathTracer::update(const InputEngine& input)
{
	camera.update(input);

	if (camera.notifyChanged())
	{
		mGlobalConstants.cameraPos = camera.getCameraPos();
		mGlobalConstants.cameraX = camera.getCameraX();	
		mGlobalConstants.cameraY = camera.getCameraY();
		mGlobalConstants.cameraZ = camera.getCameraZ();
		mGlobalConstants.cameraAspect = camera.getCameraAspect();
		mGlobalConstants.accumulatedFrames = 0;
	}
	else
		mGlobalConstants.accumulatedFrames++;

	
	* (GloabalContants*) mGlobalConstantsBuffer.map() = mGlobalConstants;
}

TracedResult DXRPathTracer::shootRays()
{
	mReadBackBuffer.unmap();
	
	mRtPipeline.bind(mCmdList);
	mSrvUavHeap.bind(mCmdList);
	mGlobalRS.bindCompute(mCmdList);

	D3D12_DISPATCH_RAYS_DESC desc = {};
	{
		desc.Width = tracerOutW;
		desc.Height = tracerOutH;
		desc.Depth = 1;
		desc.RayGenerationShaderRecord = mShaderTable.getRecord(0);
		desc.MissShaderTable = mShaderTable.getSubTable(1, 2);
		desc.HitGroupTable = mShaderTable.getSubTable(3, scene->numObjects() * mNumRayTypes);
	}
	mCmdList->DispatchRays(&desc);

	mReadBackBuffer.readback(mCmdList, mTracerOutBuffer);
	
	ThrowFailedHR(mCmdList->Close());
	ID3D12CommandList* cmdLists[] = { mCmdList };
	mCmdQueue->ExecuteCommandLists(1, cmdLists);
	mFence.waitCommandQueue(mCmdQueue);
	ThrowFailedHR(mCmdAllocator->Reset());
	ThrowFailedHR(mCmdList->Reset(mCmdAllocator, nullptr));

	TracedResult result;
	result.data = mReadBackBuffer.map();
	result.width = tracerOutW;
	result.height = tracerOutH;
	result.pixelSize = _bpp(tracerOutFormat);

	return result;
}

void DXRPathTracer::setupScene(const Scene* scene)
{
	uint numObjs = scene->numObjects();

	const Array<Vertex> vtxArr = scene->getVertexArray();
	const Array<Tridex> tdxArr = scene->getTridexArray();
	const Array<Transform> trmArr = scene->getTransformArray();
	const Array<float> cdfArr = scene->getCdfArray();
	const Array<Material> mtlArr = scene->getMaterialArray();
	const Array<StaticEmissiveTriangle> staticLightArr = scene->getStaticLightArray();				

	//assert(cdfArr.size() == 0 || cdfArr.size() == tdxArr.size());
	if (cdfArr.size() != 0 && cdfArr.size() != tdxArr.size())
		printf("WARNING: cdfArr is defined for emissive triangles. This framework originally assumes per-triangle CDF.\n");

	uint64 vtxBuffSize = vtxArr.size() * sizeof(Vertex);
	uint64 tdxBuffSize = tdxArr.size() * sizeof(Tridex);
	uint64 trmBuffSize = trmArr.size() * sizeof(Transform);
	uint64 cdfBuffSize = cdfArr.size() * sizeof(float);
	uint64 mtlBuffSize = mtlArr.size() * sizeof(Material);
	uint64 objBuffSize = numObjs * sizeof(GPUSceneObject);
	uint64 staticLightBuffSize = staticLightArr.size() * sizeof(StaticEmissiveTriangle);

	UploadBuffer uploader(vtxBuffSize + tdxBuffSize + trmBuffSize + cdfBuffSize + mtlBuffSize + objBuffSize + staticLightBuffSize);
	uint64 uploaderOffset = 0;

	auto initBuffer = [&](DefaultBuffer& buff, uint64 buffSize, void* srcData) {
		if(buffSize == 0)
			return;
		buff.create(buffSize); 
		memcpy((uint8*) uploader.map() + uploaderOffset, srcData, buffSize);
		buff.uploadData(mCmdList, uploader, uploaderOffset);
		uploaderOffset += buffSize;
	};

	initBuffer(mVertexBuffer,	 vtxBuffSize, (void*) vtxArr.data());
	initBuffer(mTridexBuffer,	 tdxBuffSize, (void*) tdxArr.data());
	initBuffer(mTransformBuffer, trmBuffSize, (void*) trmArr.data());
	initBuffer(mCdfBuffer,		 cdfBuffSize, (void*) cdfArr.data());
	initBuffer(mMaterialBuffer,	 mtlBuffSize, (void*) mtlArr.data());
	initBuffer(mStaticLightBuffer,	 staticLightBuffSize, (void*) staticLightArr.data());

	mSceneObjectBuffer.create(objBuffSize);
	GPUSceneObject* copyDst = (GPUSceneObject*) ((uint8*) uploader.map() + uploaderOffset);
	for (uint objIdx = 0; objIdx < numObjs; ++objIdx)
	{
		const SceneObject& obj = scene->getObject(objIdx);

		GPUSceneObject gpuObj = {};
		gpuObj.vertexOffset = obj.vertexOffset;
		gpuObj.tridexOffset = obj.tridexOffset;
		gpuObj.numTridices = obj.numTridices;
		gpuObj.cdfOffset = obj.cdfOffset;
		//gpuObj.objectArea = obj.meshArea * obj.scale * obj.scale;
		gpuObj.twoSided = obj.twoSided;
		gpuObj.materialIdx = obj.materialIdx;
		gpuObj.backMaterialIdx = obj.backMaterialIdx;
		//gpuObj.material = obj.material;
		//gpuObj.emittance = obj.lightColor * obj.lightIntensity;
		gpuObj.modelMatrix = obj.modelMatrix;
		
		copyDst[objIdx] = gpuObj;
	}
	mSceneObjectBuffer.uploadData(mCmdList, uploader, uploaderOffset);

	ThrowFailedHR(mCmdList->Close());
	ID3D12CommandList* cmdLists[] = { mCmdList };
	mCmdQueue->ExecuteCommandLists(1, cmdLists);
	mFence.waitCommandQueue(mCmdQueue);
	ThrowFailedHR(mCmdAllocator->Reset());
	ThrowFailedHR(mCmdList->Reset(mCmdAllocator, nullptr));

	this->scene = const_cast<Scene*>(scene);

	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	{
		srvDesc.Format = DXGI_FORMAT_UNKNOWN;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Buffer.StructureByteStride = sizeof(GPUSceneObject);
		srvDesc.Buffer.NumElements = numObjs;
	}
	mSrvUavHeap[DescriptorID::sceneObjectBuff].assignSRV(mSceneObjectBuffer, &srvDesc);

	{
		srvDesc.Format = DXGI_FORMAT_UNKNOWN;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Buffer.StructureByteStride = sizeof(Vertex);
		srvDesc.Buffer.NumElements = vtxArr.size();
	}
	mSrvUavHeap[DescriptorID::vertexBuff].assignSRV(mVertexBuffer, &srvDesc);

	{
		srvDesc.Format = DXGI_FORMAT_R32G32B32_UINT;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Buffer.StructureByteStride = 0;
		srvDesc.Buffer.NumElements = tdxArr.size();
	}
	mSrvUavHeap[DescriptorID::tridexBuff].assignSRV(mTridexBuffer, &srvDesc);

	{
		srvDesc.Format = DXGI_FORMAT_UNKNOWN;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Buffer.StructureByteStride = sizeof(Material);
		srvDesc.Buffer.NumElements = mtlArr.size();
	}
	mSrvUavHeap[DescriptorID::materialBuff].assignSRV(mMaterialBuffer, &srvDesc);
	
	{
		srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Buffer.StructureByteStride = 0;
		srvDesc.Buffer.NumElements = cdfArr.size();
	}
	mSrvUavHeap[DescriptorID::cdfBuff].assignSRV(mCdfBuffer, &srvDesc);

	{
		srvDesc.Format = DXGI_FORMAT_UNKNOWN;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Buffer.StructureByteStride = sizeof(StaticEmissiveTriangle);
		srvDesc.Buffer.NumElements = staticLightArr.size();
	}
	mSrvUavHeap[DescriptorID::staticLightBuff].assignSRV(mStaticLightBuffer, &srvDesc);

	mGlobalConstants.numEmissiveTriangles = cdfArr.size();

	setupShaderTable();

	buildAccelerationStructure();
	
	* (RootPointer*) mGlobalRS[RootParamID::pointerForAccelerationStructure]
		= mAccelerationStructure.getGpuAddress();


}

void DXRPathTracer::setupShaderTable()
{
	ShaderIdentifier* rayGenID = mRtPipeline.getIdentifier(L"rayGen");
	ShaderIdentifier* missRayID = mRtPipeline.getIdentifier(L"missRay");
	ShaderIdentifier* missShadowID = mRtPipeline.getIdentifier(L"missShadow");
	ShaderIdentifier* hitGpID = mRtPipeline.getIdentifier(L"hitGp");
	ShaderIdentifier* hitGpGlassID = mRtPipeline.getIdentifier(L"hitGpGlass");

	ShaderIdentifier* hitGpshadowID = nullptr;
	if (mNumRayTypes == 2) // use shadow ray
		hitGpshadowID = mRtPipeline.getIdentifier(L"hitGpShadow");

	uint numObjs = scene->numObjects();
	
	mShaderTable.create(recordSize, mNumRayTypes * numObjs + 3);	// radiance & shadow ray 

	HitGroupRecord* table = (HitGroupRecord*) mShaderTable.map();
	table[0].shaderIdentifier = *rayGenID;
	table[1].shaderIdentifier = *missRayID;
	table[2].shaderIdentifier = *missShadowID;

	auto& mtlArr = scene->getMaterialArray();

	for (uint i = 0; i < numObjs; ++i)
	{
		// we can further simplify this code...
		if (mNumRayTypes == 1) {
			if (mtlArr[scene->getObject(i).materialIdx].type == Glass)
				table[3 + i].shaderIdentifier = *hitGpGlassID;
			else
				table[3 + i].shaderIdentifier = *hitGpID;
			table[3 + i].objConsts.objectIdx = i;
		}
		else if (mNumRayTypes == 2) {
			if (mtlArr[scene->getObject(i).materialIdx].type == Glass)
				table[3 + 2 * i].shaderIdentifier = *hitGpGlassID;
			else
				table[3 + 2 * i].shaderIdentifier = *hitGpID;

			table[3 + 2 * i + 1].shaderIdentifier = *hitGpshadowID;

			table[3 + 2 * i].objConsts.objectIdx = i;
			//table[3 + 2 * i + 1].objConsts.objectIdx = i;
		}
		else
			assert(mNumRayTypes == 1 || mNumRayTypes == 2);

	}

	mShaderTable.uploadData(mCmdList);
}

void DXRPathTracer::buildAccelerationStructure()
{
	uint numObjs = scene->numObjects();
	Array<GPUMesh> gpuMeshArr(numObjs);
	Array<dxTransform> transformArr(numObjs);

	D3D12_GPU_VIRTUAL_ADDRESS vtxAddr = mVertexBuffer.getGpuAddress();
	D3D12_GPU_VIRTUAL_ADDRESS tdxAddr = mTridexBuffer.getGpuAddress();
	for (uint objIdx = 0; objIdx < numObjs; ++objIdx)
	{
		const SceneObject& obj = scene->getObject(objIdx);
		
		gpuMeshArr[objIdx].numVertices = obj.numVertices;
		gpuMeshArr[objIdx].vertexBufferVA = vtxAddr + obj.vertexOffset * sizeof(Vertex);
		gpuMeshArr[objIdx].numTridices = obj.numTridices;
		gpuMeshArr[objIdx].tridexBufferVA = tdxAddr + obj.tridexOffset * sizeof(Tridex);
	
		transformArr[objIdx] = obj.modelMatrix;
	}

	// TODO: Assume each BLAS contains only one D3D12_RAYTRACING_GEOMETRY_DESC.
	// Supporting sub-materials or multiple geometries requires handling of InstanceContributionToHitGroupIndex.
	mAccelerationStructure.build(mCmdList, gpuMeshArr, transformArr, 
		sizeof(Vertex), mNumRayTypes, buildMode, buildFlags);
	

	ThrowFailedHR(mCmdList->Close());
	ID3D12CommandList* cmdLists[] = { mCmdList };
	mCmdQueue->ExecuteCommandLists(1, cmdLists);
	mFence.waitCommandQueue(mCmdQueue);
	ThrowFailedHR(mCmdAllocator->Reset());
	ThrowFailedHR(mCmdList->Reset(mCmdAllocator, nullptr));

	//mAccelerationStructure.flush();
}

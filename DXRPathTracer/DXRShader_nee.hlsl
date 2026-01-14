//#pragma pack_matrix( row_major )    // It does not work!
#include "sampling.hlsli"

RaytracingAccelerationStructure scene : register(t0, space100);
RWBuffer<float4> tracerOutBuffer : register(u0);

struct Vertex
{
	float3 position;
	float3 normal;
	float2 texcoord;
};

static const int Lambertian = 0;
static const int Metal = 1;
static const int Plastic = 2;
static const int Glass = 3;
struct Material 
{
	float3 emittance;
	uint type;
	float3 albedo;
	float roughness;
	float reflectivity;
	float transmittivity;
};

struct GPUSceneObject
{
	uint vertexOffset;
	uint tridexOffset;
	uint cdfOffset;
	uint numTridices;
	//float objectArea;
	uint twoSided;
	uint materialIdx;
	uint backMaterialIdx;
	//Material material;
	//float3 emittance;
	row_major float4x4 modelMatrix;
};

struct StaticEmissiveTriangle {
	float3 emittance;
	uint objIdx;
	uint triIdx;
	float area;
};

StructuredBuffer<GPUSceneObject> objectBuffer			: register(t0);
StructuredBuffer<Vertex> vertexBuffer					: register(t1);
Buffer<uint3> tridexBuffer								: register(t2);				//ByteAddressBuffer IndexBuffer : register(t2);
StructuredBuffer<Material> materialBuffer				: register(t3);
Buffer<float> cdfBuff									: register(t4);
StructuredBuffer<StaticEmissiveTriangle> staticLightBuffer		: register(t6);


cbuffer GLOBAL_CONSTANTS : register(b0)
{
	float3 backgroundLight;
	float3 cameraPos;
	float3 cameraX;
	float3 cameraY;
	float3 cameraZ;
	float2 cameraAspect;
	float rayTmin;
	float rayTmax;
	uint accumulatedFrames;
	uint numSamplesPerFrame;
	uint maxPathLength;
	uint numEmissiveTriangles;
}

cbuffer OBJECT_CONSTANTS : register(b1)
{
	uint objIdx;
};

struct RayPayload
{
	float3 radiance;
	float3 attenuation;		
	float3 hitPos;		
	float3 bounceDir;
	//uint terminateRay;		
	uint rayDepth;
	uint seed;	
};

struct ShadowPayload
{
	uint occluded;
};

RayDesc Ray(in float3 origin, in float3 direction, in float tMin, in float tMax)
{
	RayDesc ray;
	ray.Origin = origin;
    ray.Direction = direction;
	ray.TMin = tMin;
    ray.TMax = tMax;
	return ray;
}

void computeNormal(out float3 normal, out float3 faceNormal, in BuiltInTriangleIntersectionAttributes attr)
{
	GPUSceneObject obj = objectBuffer[objIdx];

	uint3 tridex = tridexBuffer[obj.tridexOffset + PrimitiveIndex()];
	Vertex vtx0 = vertexBuffer[obj.vertexOffset + tridex.x];
	Vertex vtx1 = vertexBuffer[obj.vertexOffset + tridex.y];
	Vertex vtx2 = vertexBuffer[obj.vertexOffset + tridex.z];
	
	float t0 = 1.0f - attr.barycentrics.x - attr.barycentrics.y;
	float t1 = attr.barycentrics.x;
	float t2 = attr.barycentrics.y;
	
	float3x3 transform = (float3x3) obj.modelMatrix;
	
	faceNormal =  normalize( mul(transform, 
		 cross(vtx1.position - vtx0.position, vtx2.position - vtx0.position)
	) );
	normal =  normalize( mul(transform, 
		t0 * vtx0.normal + t1 * vtx1.normal + t2 * vtx2.normal 
	) );
}

void computeNormal(out float3 normal, in BuiltInTriangleIntersectionAttributes attr)
{
	GPUSceneObject obj = objectBuffer[objIdx];

	uint3 tridex = tridexBuffer[obj.tridexOffset + PrimitiveIndex()];
	Vertex vtx0 = vertexBuffer[obj.vertexOffset + tridex.x];
	Vertex vtx1 = vertexBuffer[obj.vertexOffset + tridex.y];
	Vertex vtx2 = vertexBuffer[obj.vertexOffset + tridex.z];
	
	float t0 = 1.0f - attr.barycentrics.x - attr.barycentrics.y;
	float t1 = attr.barycentrics.x;
	float t2 = attr.barycentrics.y;
	
	float3x3 transform = (float3x3) obj.modelMatrix;
	
	normal =  normalize( mul(transform, 
		t0 * vtx0.normal + t1 * vtx1.normal + t2 * vtx2.normal 
	) );
}

float3 tracePath(in float3 startPos, in float3 startDir, inout uint seed)
{
	float3 radiance = 0.0f;
	float3 attenuation = 1.0f;

	RayDesc ray = Ray(startPos, startDir, rayTmin, rayTmax);
	RayPayload prd;
	prd.seed = seed;
	prd.rayDepth = 0;
	//prd.terminateRay = false;

	while(prd.rayDepth < maxPathLength)
	{
		
		TraceRay(scene, 0, ~0, 0, 2, 0, ray, prd);
		
		//if(!prd.rayDepth)
		//	radiance += prd.radiance;
		//attenuation *= prd.attenuation;

		radiance += attenuation * prd.radiance;
		attenuation *= prd.attenuation;

		/*if(prd.terminateRay)
			break;*/
	
		ray.Origin = prd.hitPos;
		ray.Direction = prd.bounceDir;
		++prd.rayDepth;
	}
	
	seed = prd.seed;

	return radiance;
}

[shader("raygeneration")]
void rayGen()
{
	uint2 launchIdx = DispatchRaysIndex().xy;
	uint2 launchDim = DispatchRaysDimensions().xy;
	uint bufferOffset = launchDim.x * launchIdx.y + launchIdx.x;
	
	uint seed = getNewSeed(bufferOffset, accumulatedFrames, 8);

	float3 newRadiance = 0.0f;
	for (uint i = 0; i < numSamplesPerFrame; ++i)
	{
		float2 screenCoord = float2(launchIdx) + float2(rnd(seed), rnd(seed));
		float2 ndc = screenCoord / float2(launchDim) * 2.f - 1.f;	
		float3 rayDir = normalize(ndc.x*cameraAspect.x*cameraX + ndc.y*cameraAspect.y*cameraY + cameraZ);

		newRadiance += tracePath(cameraPos, rayDir, seed);
	}
	newRadiance *= 1.0f / float(numSamplesPerFrame);

	float3 avrRadiance;
	if(accumulatedFrames == 0)
		avrRadiance = newRadiance;
	else
		avrRadiance = lerp( tracerOutBuffer[bufferOffset].xyz, newRadiance, 1.f / (accumulatedFrames + 1.0f) );
		
	tracerOutBuffer[bufferOffset] = float4(avrRadiance, 1.0f);
}

void samplingBRDF(out float3 sampleDir, out float sampleProb, out float3 brdfCos, 
	in float3 surfaceNormal, in float3 baseDir, in uint materialIdx, inout uint seed)
{
	Material mtl = materialBuffer[materialIdx];

	float3 brdfEval;
	float3 albedo = mtl.albedo;	
	uint reflectType = mtl.type;

	float3 I, O = baseDir, N = surfaceNormal, H;
	float ON = dot(O, N), IN, HN, OH;
	float alpha2 = mtl.roughness * mtl.roughness;

	if (reflectType == Lambertian)
	{
		I = sample_hemisphere_cos(seed);
		IN = I.z;
		I = applyRotationMappingZToN(N, I);
		
		sampleProb = InvPi * IN;
		brdfEval = InvPi * albedo;
	}

	else if (reflectType == Metal)
	{
		H = sample_hemisphere_TrowbridgeReitzCos(alpha2, seed);
		HN = H.z;
		H = applyRotationMappingZToN(N, H);
		OH = dot(O, H);

		I = 2 * OH * H - O;
		IN = dot(I, N);

		if (IN < 0)
		{
			brdfEval = 0;
			sampleProb = 0;		// sampleProb = D*HN / (4*abs(OH));  if allowing sample negative hemisphere
		}
		else
		{
			float D = TrowbridgeReitz(HN*HN, alpha2);
			float G = Smith_TrowbridgeReitz(I, O, H, N, alpha2);
			float3 F = albedo + (1 - albedo) * pow(max(0, 1-OH), 5);
			brdfEval = ((D * G) / (4 * IN * ON)) * F;
			sampleProb = D*HN / (4*OH);		// IN > 0 imply OH > 0
		}
	}

	else if (reflectType == Plastic)
	{
		float r = mtl.reflectivity;
		
		if (rnd(seed) < r)
		{
			H = sample_hemisphere_TrowbridgeReitzCos(alpha2, seed);
			HN = H.z;
			H = applyRotationMappingZToN(N, H);
			OH = dot(O, H);

			I = 2 * OH * H - O;
			IN = dot(I, N);
		}
		else
		{
			I = sample_hemisphere_cos(seed);
			IN = I.z;
			I = applyRotationMappingZToN(N, I);

			H = O + I;
			H = (1/length(H)) * H;
			HN = dot(H, N);
			OH = dot(O, H);
		}

		if (IN < 0)
		{
			brdfEval = 0;
			sampleProb = 0;		//sampleProb = r * (D*HN / (4*abs(OH)));  if allowing sample negative hemisphere
		}
		else
		{
			float D = TrowbridgeReitz(HN*HN, alpha2);
			float G = Smith_TrowbridgeReitz(I, O, H, N, alpha2);
			float3 spec = ((D * G) / (4 * IN * ON));
			brdfEval = r * spec + (1 - r) * InvPi * albedo;
			sampleProb = r * (D*HN / (4*OH)) + (1 - r) * (InvPi * IN);
		}
	}

	sampleDir = I;
	brdfCos = brdfEval * IN;
}


float3 evalBRDF(in float3 shadowRayDir, in float3 surfaceNormal, in float3 baseDir, in uint materialIdx) {
	Material mtl = materialBuffer[materialIdx];

	float3 albedo = mtl.albedo;
	uint reflectType = mtl.type;

	if (reflectType == Lambertian)
		return InvPi * albedo;

	float3 brdfEval;

	float3 I = shadowRayDir, O = baseDir, N = surfaceNormal;
	float3 H = normalize(I + O);

	float ON = dot(O, N), IN = dot(I, N), HN = dot(N, H), OH = dot(O, H);

	float alpha2 = mtl.roughness * mtl.roughness;

	if (reflectType == Metal)
	{
		if (IN < 0)
			brdfEval = 0;
		else
		{
			float D = TrowbridgeReitz(HN * HN, alpha2);
			float G = Smith_TrowbridgeReitz(I, O, H, N, alpha2);
			float3 F = albedo + (1 - albedo) * pow(max(0, 1 - OH), 5);
			brdfEval = ((D * G) / (4 * IN * ON)) * F;
		}
	}
	else if (reflectType == Plastic)
	{
		float r = mtl.reflectivity;
		if (IN < 0)
			brdfEval = 0;
		else
		{
			float D = TrowbridgeReitz(HN * HN, alpha2);
			float G = Smith_TrowbridgeReitz(I, O, H, N, alpha2);
			float3 spec = ((D * G) / (4 * IN * ON));
			brdfEval = r * spec + (1 - r) * InvPi * albedo;
		}
	}

	return brdfEval;
}

void selectLight(out uint cidx, out float lightSelectionProb, inout float xi) {

	GPUSceneObject obj = objectBuffer[objIdx];

	uint numEle = numEmissiveTriangles;
	uint left = 0;
	uint right = numEle - 1;

	if (obj.cdfOffset == uint(-1))							// The shadow ray starts from non-emissive surface.
	{
		float totalPower = cdfBuff[numEle - 1];
		float u = xi * totalPower;

		while (left < right)
		{
			uint mid = (left + right) >> 1;
			if (u <= cdfBuff[mid])
				right = mid;
			else
				left = mid + 1;
		}

		float cdfPrev = (left == 0) ? 0.0f : cdfBuff[left - 1];
		float range = cdfBuff[left] - cdfPrev;
		lightSelectionProb = range / totalPower;

		xi = (u - cdfPrev) / range;
		cidx = left;

	}
	else {													// The shadow ray starts from emissive surface.
		uint excluded = obj.cdfOffset + PrimitiveIndex();	//exclude self.

		float cdfPrevExcluded = (excluded > 0) ? cdfBuff[excluded - 1] : 0.0f;
		float rangeExcluded = cdfBuff[excluded] - cdfPrevExcluded;

		float totalPower = cdfBuff[numEle - 1] - rangeExcluded;

		float u = xi * totalPower;

		if (u >= cdfPrevExcluded)
			u += rangeExcluded;
		
		//if (u < cdfPrev)
		//	u = u;
		//else
		//	u = u + wExcluded;

		while (left < right){
			uint mid = (left + right) >> 1;

			if (u <= cdfBuff[mid])
				right = mid;
			else
				left = mid + 1;
		}

		float cdfPrev = (left == 0) ? 0.0f : cdfBuff[left - 1];
		float range = cdfBuff[left] - cdfPrev;
		lightSelectionProb = range / totalPower;

		xi = (u - cdfPrev) / range;
		cidx = left;
	}

	return;
}

void sampleLightPoint(out float3 samplePoint, out float3 lightNormal, in float xi1, in float xi2, in StaticEmissiveTriangle light) {


	GPUSceneObject obj = objectBuffer[light.objIdx];
	uint vtxOffset = obj.vertexOffset;
	uint tdxOffset = obj.tridexOffset;
	uint tdxLocal = light.triIdx;

	uint3 tridex = tridexBuffer[tdxOffset + tdxLocal];

	Vertex vtx0 = vertexBuffer[vtxOffset + tridex.x];
	Vertex vtx1 = vertexBuffer[vtxOffset + tridex.y];
	Vertex vtx2 = vertexBuffer[vtxOffset + tridex.z];

	float xi1sqrt = sqrt(xi1);

	float b0 = 1.0f - xi1sqrt, b1 = xi1sqrt * (1.0f - xi2), b2 = xi1sqrt * xi2;

	float3 p0 = vtx0.position, p1 = vtx1.position, p2 = vtx2.position;

	float3 wp0 = mul(obj.modelMatrix, float4(p0, 1.0)).xyz;
	float3 wp1 = mul(obj.modelMatrix, float4(p1, 1.0)).xyz;
	float3 wp2 = mul(obj.modelMatrix, float4(p2, 1.0)).xyz;

	samplePoint = b0 * wp0 + b1 * wp1 + b2 * wp2;
	lightNormal = normalize(cross(wp1 - wp0, wp2 - wp0));

}

float3 evalDirectLight(in float3 surfaceNormal, in float3 baseDir, in uint materialIdx, inout RayPayload payload)
{
	float xi1 = rnd(payload.seed), xi2 = rnd(payload.seed);

	uint cidx;
	float lightSelectionProb;
	selectLight(cidx, lightSelectionProb, xi1);


	StaticEmissiveTriangle light = staticLightBuffer[cidx];
	float3 Le = light.emittance;

	float3 lightPoint,lightNormal;
	sampleLightPoint(lightPoint, lightNormal, xi1, xi2, light);

	// NOTE: In NEE, lightNormal must be the geometric normal.
	// The cos term (dot(n_L, -wi)) is part of the area -> solid-angle Jacobian,
	// so it must be defined w.r.t. the geometric surface, not the shading normal.

	float3 shadowRayOrigin = payload.hitPos;
	float3 shadowRayDir = normalize(lightPoint - shadowRayOrigin);

	float cos1 = dot(surfaceNormal, shadowRayDir);
	float cos2 = dot(lightNormal, -shadowRayDir);

	if (cos1 <= 0.0f || cos2 <= 0.0f)
		return 0.0f;

	float dist = distance(lightPoint, shadowRayOrigin);
	float tmax = dist - rayTmin;

	ShadowPayload sprd;
	RayDesc sray = Ray(shadowRayOrigin, shadowRayDir, rayTmin, tmax);
	TraceRay(scene, 0, ~0, 1, 2, 1, sray, sprd);

	if (sprd.occluded)
		return 0.0f;

	float3 brdf = evalBRDF(shadowRayDir, surfaceNormal, baseDir, materialIdx);

	float areaSampleProb = lightSelectionProb / light.area;

	if (lightSelectionProb <= 0.0f)
		return 0.0f;

	return brdf * Le * cos1 * cos2 / (dist * dist * areaSampleProb);	// visibility = 1;
}




/*
1. Closed manifold assumption(except for emitting source): we can only consider the shading normal N, 
   i.e ignoring the face nomal fN since dot(E, fN)<0 never occur.
2. The hit point should be considerd as being transparent in case dot(E, N)<0, but not yet implemented. 
3. In case dot(R, fN)<0, where R is sampled reflected ray, we do not terminate ray,
   but do terminate when dot(R, N)<0 in which it force monte calro estimation to zero.
4. Note that in case 2 and 3 above, the next closest hit point might be in dot(E, fN)<0 && dot(E, N)>0, 
   but this is rare so we ignore the codition dot(E, fN)<0 and only check dot(E, N)<0.
5. In results, we do not need the face nomal fN which take a little time to compute.
*/
[shader("closesthit")]
void closestHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
	GPUSceneObject obj = objectBuffer[objIdx];

	float3 N, fN, E = - WorldRayDirection();
	computeNormal(N, fN, attr);	
	float EN = dot(E, N), EfN = dot(E, fN);

	payload.radiance = 0.0f;
	payload.attenuation = 1.0f;
	payload.hitPos = WorldRayOrigin() - RayTCurrent() * E;

	uint mtlIdx = obj.materialIdx;

	if (obj.twoSided && EfN < 0)
	{
		mtlIdx = obj.backMaterialIdx;
		N = -N;
		EN = -EN;
	}
	
	if(EN < 0)
	{
		payload.bounceDir = WorldRayDirection();
		--payload.rayDepth;
		return;
	}
	
	Material mtl = materialBuffer[mtlIdx];

	//if (any(mtl.emittance))																
	if (any(mtl.emittance) && payload.rayDepth == 0)													// changed 1				
	{
		payload.radiance += mtl.emittance;
	}

	float3 sampleDir, brdfCos;
	float sampleProb;
	samplingBRDF(sampleDir, sampleProb, brdfCos, N, E, mtlIdx, payload.seed);
	
	if(dot(sampleDir, N) <= 0)
		payload.rayDepth = maxPathLength;
	//payload.terminateRay = dot(sampleDir, N) <= 0.0f
	payload.attenuation = brdfCos / sampleProb;
	payload.bounceDir = sampleDir;

	payload.radiance += evalDirectLight(N, E, mtlIdx, payload);

}

[shader("closesthit")]
void closestHitGlass(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
	GPUSceneObject obj = objectBuffer[objIdx];

	float3 N, fN, E = - WorldRayDirection();
	computeNormal(N, fN, attr);	
	float EN = dot(E, N), EfN = dot(E, fN);

	payload.radiance = 0.0f;
	payload.attenuation = 1.0f;
	payload.hitPos = WorldRayOrigin() - RayTCurrent() * E;

	if(EN * EfN < 0)
	{
		payload.bounceDir = WorldRayDirection();
		--payload.rayDepth;
		return;
	}

	Material mtl = materialBuffer[obj.materialIdx];

	if (any(mtl.emittance) && EN > 0)
	{
		payload.radiance += mtl.emittance;
	}

	float3 sampleDir;
	float sampleProb, Fresnel;

	float T0 = mtl.transmittivity;
	float n = sqrt(1 - T0);
	n = (1+n) / (1-n);			// n <- refractive index of glass
	
	float R, g, x, y;
	float c1, gg;

	if (EN > 0)
	{
		n = 1 / n;				// n <- relative index of air-to-glass (n_air/n_glass)
		c1 = EN;
	}
	else
	{
		n = n;					// n <- relative index of glass-to-air (n_glass/n_air)
		c1 = -EN;
	}

	gg = 1/(n*n) - 1 + c1*c1;	// gg == (c2/n)^2
	if (gg < 0)
	{
		R = 1;
	}
	else
	{
		g = sqrt(gg);
		x = (c1*(g + c1) - 1) / (c1*(g - c1) + 1);
		y = (g - c1) / (g + c1);
		R = 0.5 * y*y * (1 + x * x);
	}

	//if (rnd(payload.seed) < 0.5)
	if (rnd(payload.seed) < R)
	{
		sampleProb = R;
		Fresnel = R;
		sampleDir = 2 * EN * N - E;
	}
	else
	{
		sampleProb = 1 - R;
		Fresnel = 1 - R;

		if (gg < 0)
		{
			payload.rayDepth = maxPathLength;
		}
		else
		{
			//if (EN > 0)			// Additional bounce only for the case of air-to-glass transmission
			//	--payload.rayDepth;
			float ON = -sign(EN) * n * g;
			sampleDir = (ON + n * EN)*N - n * E;
		}
	}

	if (EN > 0)			// Additional bounce for the case of air-to-glass.
		--payload.rayDepth;

	//sampleProb = 0.5;
	payload.attenuation = Fresnel / sampleProb;
	payload.bounceDir = sampleDir;
}

[shader("closesthit")]
void closestHitShadow(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
	// if the hit position is same as sampled light point occluded = false
	payload.occluded = true;
}

[shader("miss")]
void missRay(inout RayPayload payload)
{
	//payload.radiance = 0.1;
	payload.radiance = backgroundLight;
	payload.rayDepth = maxPathLength;
}

[shader("miss")]
void missShadow(inout ShadowPayload payload)
{
	payload.occluded = false;
}

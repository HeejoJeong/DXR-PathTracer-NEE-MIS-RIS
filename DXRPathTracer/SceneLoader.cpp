#include "pch.h"
#include "SceneLoader.h"
#include "generateMesh.h"
#include "loadMesh.h"


void SceneLoader::initializeGeometryFromMeshes(Scene* scene, const Array<Mesh*>& meshes)
{
	scene->clear();

	uint numObjs = meshes.size();
	Array<Vertex>& vtxArr = scene->vtxArr;
	Array<Tridex>& tdxArr = scene->tdxArr;
	Array<SceneObject>& objArr = scene->objArr;

	objArr.resize(numObjs);

	uint totVertices = 0;
	uint totTridices = 0;
	
	for (uint i = 0; i < numObjs; ++i)
	{
		uint nowVertices = meshes[i]->vtxArr.size();
		uint nowTridices = meshes[i]->tdxArr.size();

		objArr[i].vertexOffset = totVertices;
		objArr[i].tridexOffset = totTridices;
		objArr[i].numVertices = nowVertices;
		objArr[i].numTridices = nowTridices;

		totVertices += nowVertices;
		totTridices += nowTridices;
	}

	vtxArr.resize(totVertices);
	tdxArr.resize(totTridices);

	for (uint i = 0; i < numObjs; ++i)
	{
		memcpy(&vtxArr[objArr[i].vertexOffset], &meshes[i]->vtxArr[0], sizeof(Vertex) * objArr[i].numVertices);
		memcpy(&tdxArr[objArr[i].tridexOffset], &meshes[i]->tdxArr[0], sizeof(Tridex) * objArr[i].numTridices);
	}
}

void SceneLoader::computeModelMatrices(Scene* scene)
{
	for (auto& obj : scene->objArr)
	{
		obj.modelMatrix = composeMatrix(obj.translation, obj.rotation, obj.scale);
	}
}

void SceneLoader::setupStaticLightBuffer(Scene* scene)
{
	// setup cdfArr and staticLightArr 
	Array<uint> emissiveObjIdx;

	Array<Material>& mtlArr = scene->mtlArr;
	Array<SceneObject>& objArr = scene->objArr;

	uint numEmissiveTriangles = 0;
	uint numObjs = objArr.size();

	for (int i = 0; i < numObjs; i++) {
		//printf("%d \n", objArr[i].numTridices);
		float3 emit = mtlArr[objArr[i].materialIdx].emittance;
		if (emit.x > 0 || emit.y > 0 || emit.z > 0) {
			objArr[i].cdfOffset = numEmissiveTriangles;
			//printf("%d \n", scene->objArr[i].numTridices);
			numEmissiveTriangles += objArr[i].numTridices;
			emissiveObjIdx.push_back(i);
		}
	}

	if (numEmissiveTriangles == 0)
		throw Error("No Emissive Surface.");

	Array<float>& cdfArr = scene->cdfArr;
	Array<StaticEmissiveTriangle>& staticLightArr = scene->staticLightArr;
	cdfArr.resize(numEmissiveTriangles);
	staticLightArr.resize(numEmissiveTriangles);


	int faceId = 0;
	for (int i = 0; i < emissiveObjIdx.size(); i++) {
		uint oIdx = emissiveObjIdx[i];
		const SceneObject& emissiveObj = objArr[oIdx];

		for (int tri = 0; tri < emissiveObj.numTridices; tri++) {

			float3 emittance = mtlArr[emissiveObj.materialIdx].emittance;
			float luminance = 0.2126f * emittance.x + 0.7152f * emittance.y + 0.0722f * emittance.z;

			Tridex tdx = scene->tdxArr[emissiveObj.tridexOffset + tri];
			int vOffset = emissiveObj.vertexOffset;

			float3 p0 = scene->vtxArr[vOffset + tdx[0]].position;
			float3 p1 = scene->vtxArr[vOffset + tdx[1]].position;
			float3 p2 = scene->vtxArr[vOffset + tdx[2]].position;

			float area = 0.5f * length(cross(p1 - p0, p2 - p0)) * emissiveObj.scale * emissiveObj.scale; // TODO: non-uniform scale.

			float power = luminance * area;
			cdfArr[faceId] = (faceId > 0) ? cdfArr[faceId - 1] + power : power;

			staticLightArr[faceId].objIdx = oIdx;
			staticLightArr[faceId].triIdx = tri;
			staticLightArr[faceId].emittance = emittance;
			staticLightArr[faceId].area = area;

			faceId++;
		}
	}

	if (faceId != numEmissiveTriangles)
		throw Error("mismatch the number of emissive triangles");

	int numCdfLoss = 0;
	for (int tri = 1; tri < cdfArr.size(); tri++) {
		float prob = (cdfArr[tri] - cdfArr[tri - 1]) / cdfArr[cdfArr.size()-1];
		if (prob < 1e-9)
			numCdfLoss++;
	}

	if (numCdfLoss  > float(numEmissiveTriangles) * 0.1f) {
		printf("WARNING: FP precision loss may occur (%d / %d triangles). Sampling Light Uniformly \n", numCdfLoss, numEmissiveTriangles);
		for (int tri = 0; tri < cdfArr.size(); tri++)
			cdfArr[tri] = (tri > 0) ? cdfArr[tri - 1] + 1.0f : 1.0f;
	}

	//Build Alias Tables
	Array<float>& probArr = scene->probArr;
	Array<uint>& aliasArr = scene->aliasArr;

	probArr.resize(numEmissiveTriangles);
	aliasArr.resize(numEmissiveTriangles);

	Array<float> lightSelectionProb;
	lightSelectionProb.resize(numEmissiveTriangles);

	Array<uint> smaller;
	Array<uint> larger;

	for (uint i = 0; i < numEmissiveTriangles; i++) {
		lightSelectionProb[i] = (i > 0) ? (cdfArr[i] - cdfArr[i - 1]) / cdfArr[numEmissiveTriangles - 1] : cdfArr[0] / cdfArr[numEmissiveTriangles - 1];
		lightSelectionProb[i] *= numEmissiveTriangles;
	
		if (lightSelectionProb[i] < 1.0f)
			smaller.push_back(i);
		else
			larger.push_back(i);
	}

	while (smaller.size() && larger.size()) {
		uint s_size = smaller.size();
		uint l_size = larger.size();

		// pop
		uint less = smaller[s_size - 1];
		uint more = larger[l_size - 1];
		smaller.resize(s_size - 1);
		larger.resize(l_size - 1);

		probArr[less] = lightSelectionProb[less];
		aliasArr[less] = more;

		lightSelectionProb[more] = lightSelectionProb[more] + lightSelectionProb[less] - 1.0f;

		if (lightSelectionProb[more] < 1.0f)
			smaller.push_back(more);
		else
			larger.push_back(more);
	}

	for (uint i = 0; i<smaller.size(); i++){
		probArr[smaller[i]] = 1.0f;
		aliasArr[smaller[i]] = smaller[i];
	}

	for (uint i = 0; i < larger.size(); i++){
		probArr[larger[i]] = 1.0f;
		aliasArr[larger[i]] = larger[i];
	}

}


Scene* SceneLoader::push_testScene1()
{
	Scene* scene = new Scene;
	sceneArr.push_back(scene);

	Mesh ground = generateRectangleMesh(float3(0.0f), float3(20.f, 0.f, 20.f), FaceDir::up);
	Mesh box = generateCubeMesh(float3(0.0f, 1.0f, 0.0f), float3(5.f, 2.f, 3.f)); 
	Mesh quadLight = generateRectangleMesh(float3(0.0f), float3(3.0f, 0.f, 3.0f), FaceDir::down);
	
	initializeGeometryFromMeshes(scene, { &ground, &box, &quadLight });

	Array<Material>& mtlArr = scene->mtlArr;
	mtlArr.resize(3);

	mtlArr[0].albedo = float3(0.7f, 0.3f, 0.4f);
	
	mtlArr[1].type = Plastic;
	mtlArr[1].albedo = float3(0.1f);
	mtlArr[1].reflectivity = 0.01f;
	mtlArr[1].roughness = 0.2f;
	
	mtlArr[2].emittance = float3(200.0f);

	//scene->objArr[0].twoSided = true;
	scene->objArr[0].materialIdx = 0;
	scene->objArr[0].backMaterialIdx = 0;
	scene->objArr[1].materialIdx = 1;
	scene->objArr[2].materialIdx = 2;

	scene->objArr[0].translation = float3(0.0f);
	scene->objArr[1].translation = float3(0.0f, 0.5f, 0.0f);
	scene->objArr[2].translation = float3(-20.0f, 17.f, 0.0f);
	
	computeModelMatrices(scene);

	setupStaticLightBuffer(scene);

	return scene;
}

Scene* SceneLoader::push_RIStestScene()
{

	Scene* scene = new Scene;
	sceneArr.push_back(scene);

	Mesh groundM = generateRectangleMesh(float3(0.0f, -0.0f, 0.0f), float3(1000.0f, 0.0f, 1000.0f), FaceDir::up);
	Mesh groundM2 = generateRectangleMesh(float3(0.0f, -0.0f, 0.0f), float3(1000.0f, 0.0f, 1000.0f), FaceDir::down);
	Mesh sphereM = generateSphereMesh(float3(0, 1, 0), 1.0f,8U,16U);
	Mesh puzzleM = loadMeshFromOBJFile("../data/mesh/burrPuzzle.obj", true);

	initializeGeometryFromMeshes(scene, { &groundM, &groundM2, &sphereM, &puzzleM });

	enum SceneObjectId { ground1, ground2, numObjs };
	enum MaterialId { groundMtl1, groundMtl2, numMtls };

	uint num_spheres = 30000;
	uint num_lights = 300;

	auto wanghash = [](uint& seed) -> uint {
		seed = (seed ^ 61u) ^ (seed >> 16);
		seed *= 9u;
		seed = seed ^ (seed >> 4);
		seed *= 0x27d4eb2du;
		seed = seed ^ (seed >> 15);
		return seed;
		};
	auto rnd01 = [&](uint& seed) -> float {
		return (wanghash(seed) & 0x00FFFFFFu) / float(0x01000000u);
		};


	Array<SceneObject> objArr(numObjs + num_spheres, scene->objArr[2]);
	objArr[ground1] = scene->objArr[0];
	objArr[ground2] = scene->objArr[1];

	objArr[ground1].translation = float3(0.0, -20.0, 0.0);
	objArr[ground2].translation = float3(0.0f, 20.0, 0.0f);
	objArr[ground1].scale = 1.0f;
	objArr[ground2].scale = 1.0f;

	const float xStart = -950.0f;
	const float xEnd = 950.0f;
	const float yMin = -18.0f;   
	const float yMax = 18.0f;
	const float zMin = -950.0;
	const float zMax = 950.0f;

	float xStep = 0.0f;
	if (num_spheres > 1) xStep = (xEnd - xStart) / float(num_spheres - 1);

	uint seed = 1337u;
	bool intChanged = false;

	for (uint i = 0; i < num_spheres; i++)
	{
		uint objIdx = numObjs + i;
		objArr[objIdx] = scene->objArr[2];

		float x = xStart + xStep * float(i);
		float y = yMin + (yMax - yMin) * rnd01(seed);
		float z = zMin + (zMax - zMin) * rnd01(seed);

		objArr[objIdx].translation = float3(x, y, z);

		objArr[objIdx].scale = 0.75f + 0.5f * rnd01(seed); 
	}
	objArr.swap(scene->objArr);


	Array<Material>& mtlArr = scene->mtlArr;
	mtlArr.resize(numMtls + num_spheres);

	mtlArr[groundMtl1].albedo = float3(0.1f, 0.1f, 0.1f);
	mtlArr[groundMtl2].albedo = float3(0.1f, 0.1f, 0.1f);
	mtlArr[groundMtl1].emittance = float3(0, 0, 0);
	mtlArr[groundMtl2].emittance = float3(0, 0, 0);
	mtlArr[groundMtl1].type = Lambertian;
	mtlArr[groundMtl2].type = Lambertian;

	for (uint i = 0; i < num_spheres; i++)
	{
		uint mtlIdx = numMtls + i;

		mtlArr[mtlIdx].albedo = float3(rnd01(seed), rnd01(seed), rnd01(seed));

		if (i % (num_spheres / num_lights) == 0){

			float e1 = 100.0f + 100.0f * rnd01(seed);
			float e2 = 100.0f + 100.0f * rnd01(seed);
			float e3 = 100.0f + 100.0f * rnd01(seed);
			mtlArr[mtlIdx].emittance = float3(e1, e2, e3);

		}

		float r = rnd01(seed);
		if (r < 0.333f)
			mtlArr[mtlIdx].type = Lambertian;
		else if (r < 0.666f)
			mtlArr[mtlIdx].type = Metal;
		else
			mtlArr[mtlIdx].type = Plastic;

		mtlArr[mtlIdx].roughness = 0.003f + 0.10f * rnd01(seed);
		mtlArr[mtlIdx].reflectivity = 0.003f + 0.50f * rnd01(seed);
		mtlArr[mtlIdx].transmittivity = 0.0f;

	}

	for (uint i = 0; i < numMtls + num_spheres; ++i)
		scene->objArr[i].materialIdx = i;

	computeModelMatrices(scene);
	setupStaticLightBuffer(scene);

	return scene;
}

Scene* SceneLoader::push_hyperionTestScene()
{
	Scene* scene = new Scene;
	sceneArr.push_back(scene);

	Mesh groundM = generateRectangleMesh(float3(0.0, -0.4, 0.0), float3(40.0, 0.0, 40.0), FaceDir::up);
	Mesh tableM = generateBoxMesh(float3(-5.0, -0.38, -4.0), float3(5.0, -0.01, 3.0));
	Mesh sphereM = generateSphereMesh(float3(0, 1, 0), 1.0f);
	Mesh ringM = loadMeshFromOBJFile("../data/mesh/ring.obj", true);
	Mesh golfBallM = loadMeshFromOBJFile("../data/mesh/golfball.obj", true);
	Mesh puzzleM = loadMeshFromOBJFile("../data/mesh/burrPuzzle.obj", true);
	initializeGeometryFromMeshes(scene, { &groundM, &tableM, &sphereM, &ringM, &golfBallM, &puzzleM });

	enum SceneObjectId {
		ground, table, light, glass, metal, pingpong, bouncy, orange, wood, golfball, marble1,
		marble2, ring1, ring2, ring3, numObjs
	};
	enum MaterialId {
		groundMtl, tableMtl, lightMtl, glassMtl, metalMtl, pingpongMtl, bouncyMtl, orangeMtl, woodMtl, golfballMtl, marble1Mtl,
		marble2Mtl, ringMtl, numMtls
	};

	Array<SceneObject> objArr(numObjs, scene->objArr[2]);
	objArr[ground] = scene->objArr[0];
	objArr[table] = scene->objArr[1];
	objArr[ring1] = scene->objArr[3];
	objArr[ring2] = scene->objArr[3];
	objArr[ring3] = scene->objArr[3];
	objArr[golfball] = scene->objArr[4];
	objArr[wood] = scene->objArr[5];

	objArr[light].scale = 2.0f;
	objArr[light].translation = float3(-20, 17, 0);

	objArr[ground].translation = float3(0.0, -0.04, 0.0);
	objArr[table].translation = float3(0.0, -0.02, 0.0);
	objArr[glass].translation = float3(3.5, 0.0, 0.0);
	objArr[metal].translation = float3(-3.5, 0.0, 0.0);
	objArr[pingpong].translation = float3(-1.5, 0.0, 1.1);
	objArr[bouncy].translation = float3(-2.0, 0.0, -1.1);
	objArr[orange].translation = float3(2.0, 0.0, -1.1);
	objArr[marble1].translation = float3(-0.5, 0.0, 2.0);
	objArr[marble2].translation = float3(0.5, 0.0, 2.0);
	objArr[ring1].translation = float3(0.0, -0.02, 0.0);
	objArr[ring2].translation = float3(0.6, -0.02, 0.3);
	objArr[ring3].translation = float3(-1.3, -0.02, -0.3);

	objArr[ground].scale = 1.0f;
	objArr[table].scale = 1.0f;
	objArr[glass].scale = 0.55f;
	objArr[metal].scale = 0.6f;
	objArr[pingpong].scale = 0.45f;
	objArr[bouncy].scale = 0.25f;
	objArr[orange].scale = 0.5f;
	objArr[marble1].scale = 0.1f;
	objArr[marble2].scale = 0.15f;
	objArr[ring1].scale = 0.005f;
	objArr[ring2].scale = 0.005f;
	objArr[ring3].scale = 0.005f;

	objArr[golfball].translation = float3(-12.3, -13.1, -140.0);
	objArr[golfball].scale = 0.25f;

	// burrPuzzle.obj
	objArr[wood].translation = float3(-0.2, 1.0, -2.3);
	objArr[wood].rotation = getRotationAsQuternion({ 0,1,0 }, 30.0f);
	objArr[wood].scale = 20.0f;

	objArr.swap(scene->objArr);


	Array<Material>& mtlArr = scene->mtlArr;
	mtlArr.resize(numMtls);
	mtlArr[groundMtl].albedo = float3(0.75, 0.6585, 0.5582);
	mtlArr[tableMtl].albedo = float3(0.87, 0.7785, 0.6782);
	mtlArr[lightMtl].albedo = float3(0);
	mtlArr[glassMtl].albedo = float3(0);
	mtlArr[metalMtl].albedo = float3(0.3);
	//mtlArr[pingpongMtl].albedo = float3(0.93, 0.89, 0.85);
	mtlArr[pingpongMtl].albedo = float3(0.4, 0.2, 0.2);
	mtlArr[bouncyMtl].albedo = float3(0.9828262, 0.180144, 0.0780565);
	mtlArr[orangeMtl].albedo = float3(0.7175, 0.17, 0.005);
	mtlArr[woodMtl].albedo = float3(0.3992, 0.21951971, 0.10871);
	mtlArr[golfballMtl].albedo = float3(0.9, 0.87, 0.95);
	mtlArr[marble1Mtl].albedo = float3(0.276, 0.344, 0.2233);
	mtlArr[marble2Mtl].albedo = float3(0.2549, 0.3537, 0.11926);
	mtlArr[ringMtl].albedo = float3(0.95, 0.93, 0.88);

	mtlArr[lightMtl].emittance = float3(200.0f);
	mtlArr[bouncyMtl].emittance = 10.0f * float3(0.9828262, 0.180144, 0.0780565);
	mtlArr[marble1Mtl].emittance = 10.0f * float3(0.276, 0.344, 0.2233);
	mtlArr[marble2Mtl].emittance = 10.0f * float3(0.2549, 0.3537, 0.11926);

	//mtlArr[woodMtl].emittance = 5.0f * float3(0.3992, 0.21951971, 0.10871);
	//mtlArr[woodMtl].albedo = float3(1);

	//scene->objArr[marble2 ].scale = 0.1f;
	//scene->objArr[marble2 ].translation = float3(3.2, 0.7, 0.0);
	//mtlArr[marble2Mtl ].emittance = float3(100.0f);

	mtlArr[pingpongMtl].type = Metal;
	mtlArr[pingpongMtl].roughness = 0.2f;

	mtlArr[metalMtl].type = Metal;
	mtlArr[metalMtl].roughness = 0.003f;
	mtlArr[ringMtl].type = Metal;
	mtlArr[ringMtl].roughness = 0.02f;

	mtlArr[orangeMtl].type = Plastic;
	mtlArr[orangeMtl].roughness = 0.01f;
	mtlArr[orangeMtl].reflectivity = 0.1f;

	mtlArr[woodMtl].type = Plastic;
	mtlArr[woodMtl].roughness = 0.3f;
	mtlArr[woodMtl].reflectivity = 0.1f;

	mtlArr[golfballMtl].type = Plastic;
	mtlArr[golfballMtl].reflectivity = 0.1f;
	mtlArr[golfballMtl].roughness = 0.05f;

	mtlArr[glassMtl].type = Glass;
	mtlArr[glassMtl].transmittivity = 0.96f;


	for (uint i = 0; i < ringMtl; ++i)
		scene->objArr[i].materialIdx = i;
	scene->objArr[ring1].materialIdx = ringMtl;
	scene->objArr[ring2].materialIdx = ringMtl;
	scene->objArr[ring3].materialIdx = ringMtl;

	computeModelMatrices(scene);

	setupStaticLightBuffer(scene);

	return scene;
}

Scene* SceneLoader::push_testScene2()
{

	Scene* scene = new Scene;
	sceneArr.push_back(scene);

	Mesh groundM = generateRectangleMesh(float3(0.0, -0.4, 0.0), float3(40.0, 0.0, 40.0), FaceDir::up);
	Mesh tableM = generateBoxMesh(float3(-5.0, -0.38, -4.0), float3(5.0, -0.01, 3.0));
	Mesh sphereM = generateSphereMesh(float3(0, 1, 0), 1.0f);
	Mesh puzzleM = loadMeshFromOBJFile("../data/mesh/burrPuzzle.obj", true);
	initializeGeometryFromMeshes(scene, { &groundM, &tableM, &sphereM, &puzzleM });

	enum SceneObjectId {
		ground, table, light, wood, numObjs
	};
	enum MaterialId {
		groundMtl, tableMtl, lightMtl, woodMtl, numMtls
	};

	Array<SceneObject> objArr(numObjs, scene->objArr[2]);
	objArr[ground] = scene->objArr[0];
	objArr[table] = scene->objArr[1];
	objArr[wood] = scene->objArr[3];

	objArr[light].scale = 1.0f;
	//objArr[light].translation = float3(0, 1, 0);
	objArr[light].translation = float3(0, 1000, 0);


	objArr[ground].translation = float3(0.0, -0.04, 0.0);
	objArr[table].translation = float3(0.0, -0.02, 0.0);

	objArr[ground].scale = 1.0f;
	objArr[table].scale = 1.0f;

	// burrPuzzle.obj
	objArr[wood].translation = float3(-0.2, 1.0, -2.3);
	//objArr[wood].translation = float3(-0.2, 10000.0, -2.3);
	objArr[wood].rotation = getRotationAsQuternion({ 0,1,0 }, 30.0f);
	objArr[wood].scale = 20.0f;

	objArr.swap(scene->objArr);


	Array<Material>& mtlArr = scene->mtlArr;
	mtlArr.resize(numMtls);
	mtlArr[groundMtl].albedo = float3(0.75, 0.6585, 0.5582);
	mtlArr[tableMtl].albedo = float3(0.87, 0.7785, 0.6782);
	mtlArr[lightMtl].albedo = float3(0);
	mtlArr[woodMtl].albedo = float3(1);

	mtlArr[woodMtl].emittance = 5.0f * float3(0.3992, 0.21951971, 0.10871);
	//mtlArr[lightMtl].emittance = float3(20.0f);

	mtlArr[woodMtl].type = Plastic;
	mtlArr[woodMtl].roughness = 0.3f;
	mtlArr[woodMtl].reflectivity = 0.1f;

	for (uint i = 0; i < numMtls; ++i)
		scene->objArr[i].materialIdx = i;

	computeModelMatrices(scene);

	setupStaticLightBuffer(scene);

	return scene;
}


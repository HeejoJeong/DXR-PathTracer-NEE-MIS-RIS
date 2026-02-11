DXR PathTracer
==============
A DXR-based path tracer extended with Next Event Estimation (NEE), 
Multiple Importance Sampling (MIS), and Resampled Importance Sampling (RIS).

This implementation builds upon the original DXR PathTracer: https://github.com/meta-plane/DXR-PathTracer

The path tracing mode can be selected by modifying `PTType`
inside `DXRPathTracer::buildRaytracingPipeline()`:

- `0` — Baseline BRDF sampling
- `1` — NEE (Next Event Estimation)
- `2` — MIS (Multiple Importance Sampling)
- `3` — RIS (Resampled Importance Sampling, 32 proposals)


Results
------

Disney Hyperion's table test scene (https://www.disneyanimation.com/technology/innovations/hyperion)

16k spp

| **Brute Force** | **NEE** | **MIS** | **RIS** |
|:---------------:|:------:|:------:|:------:|
| <img src="images/hyperion_bf_16kspp.png" width="100%"> | <img src="images/hyperion_nee_16kspp.png" width="100%"> | <img src="images/hyperion_mis_16kspp.png" width="100%"> | <img src="images/hyperion_ris_16kspp.png" width="100%"> |


RIS Test scene
  
1024 spp

| **Brute Force** | **NEE** | **MIS** | **RIS** |
|:---------------:|:------:|:------:|:------:|
| <img src="images/ristest_bf_1024spp.png" width="100%"> | <img src="images/ristest_nee_1024spp.png" width="100%"> | <img src="images/ristest_mis_1024spp.png" width="100%"> | <img src="images/ristest_ris_1024spp.png" width="100%"> |


16k spp

| **Brute Force** | **NEE** | **MIS** | **RIS** |
|:---------------:|:------:|:------:|:------:|
| <img src="images/ristest_bf_16kspp.png" width="100%"> | <img src="images/ristest_nee_16kspp.png" width="100%"> | <img src="images/ristest_mis_16kspp.png" width="100%"> | <img src="images/ristest_ris_16kspp.png" width="100%"> |



// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/umesh/mc/UMeshCuBQLSampler.h"

namespace BARNEY_NS {
  namespace native {

    UMeshCuBQLSampler::UMeshCuBQLSampler(UMeshField *mesh)
      : mesh(mesh),
        devices(mesh->devices)
    {
      perLogical.resize(devices->numLogical);
    }

    UMeshCuBQLSampler::PLD *UMeshCuBQLSampler::getPLD(Device *device) 
    {
      assert(device);
      assert(device->contextRank() >= 0);
      assert(device->contextRank() < perLogical.size());
      return &perLogical[device->contextRank()];
    }
  
    UMeshCuBQLSampler::DD UMeshCuBQLSampler::getDD(Device *device)
    {
      DD dd;
      (UMeshField::DD &)dd = mesh->getDD(device);
      dd.bvh = getPLD(device)->bvh;
      return dd;
    }

    void UMeshCuBQLSampler::build()
    {
      int numCells = mesh->numCells;
      for (auto device : *devices) {
        PLD *pld = getPLD(device);
        if (pld->bvh.nodes != 0) {
          /* BVH already built! */
          continue;
        }

        // std::cout << "------------------------------------------" << std::endl;
        // std::cout << "building UMeshCuBQL BVH!" << std::endl;
        // std::cout << "------------------------------------------" << std::endl;
      
        SetActiveGPU forDuration(device);
      
        box3f *primBounds
          = (box3f*)device->rtc->allocMem(numCells*sizeof(box3f));
        range1f *valueRanges
          = (range1f*)device->rtc->allocMem(numCells*sizeof(range1f));
        mesh->computeElementBBs(device,
                                primBounds,valueRanges);
        device->rtc->sync();

#if BARNEY_RTC_CPU
        cuBQL::cpu::spatialMedian(pld->bvh,
                                  (const cuBQL::box_t<float,3>*)primBounds,
                                  numCells,
                                  cuBQL::BuildConfig());
#elif defined(__HIPCC__)
        // The CPU builder consumes and produces host memory. GPU
        // traversal must instead receive device-resident BVH arrays.
        std::vector<cuBQL::box3f> hostBounds(numCells);
        device->rtc->copy(hostBounds.data(), primBounds,
                         numCells * sizeof(cuBQL::box3f));
        bvh_t hostBVH = {0,0,0,0};
        cuBQL::cpu::spatialMedian(hostBVH, hostBounds.data(), numCells,
                                 cuBQL::BuildConfig());
        pld->bvh = hostBVH;
        pld->bvh.nodes = static_cast<node_t *>(device->rtc->allocMem(
            hostBVH.numNodes * sizeof(node_t)));
        pld->bvh.primIDs = static_cast<uint32_t *>(device->rtc->allocMem(
            hostBVH.numPrims * sizeof(uint32_t)));
        device->rtc->copy(pld->bvh.nodes, hostBVH.nodes,
                         hostBVH.numNodes * sizeof(node_t));
        device->rtc->copy(pld->bvh.primIDs, hostBVH.primIDs,
                         hostBVH.numPrims * sizeof(uint32_t));
        delete[] hostBVH.nodes;
        delete[] hostBVH.primIDs;
#else
        /*! iw - make sure to have cubql use regular device memory, not
          async mallocs; else we may allocate all memory on the first
          gpu */
        cuBQL::DeviceMemoryResource memResource;
        cuBQL::gpuBuilder(pld->bvh,
                          (const cuBQL::box_t<float,3>*)primBounds,
                          numCells,
                          cuBQL::BuildConfig(),
                          0,
                          memResource);
#endif
        device->rtc->sync();
        device->rtc->freeMem(primBounds);
        device->rtc->freeMem(valueRanges);
      }
    }

  }
} // ::barney

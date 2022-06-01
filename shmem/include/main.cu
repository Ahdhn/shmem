#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "gtest/gtest.h"

#include "CUDASharedMemory.cuh"
#include "helper.h"

__global__ void exec_kernel(int* d_success, size_t max_shmem)
{
    CUDASharedMemory::Allocator shmem_allocator;

    char* char_ptr = shmem_allocator.alloc(2);

    uint32_t* uint32_ptr = shmem_allocator.alloc<uint32_t>(1);


    // this should be zero if uin32_ptr is aligned to 32 bit
    d_success[0] += reinterpret_cast<size_t>(uint32_ptr) & 2;

    if (shmem_allocator.get_max_size_bytes() != max_shmem) {
        d_success[0]++;
    }
}

TEST(Test, exe)
{
    int*   success;
    size_t shmem_bytes = 10;

    CUDA_ERROR(cudaMallocManaged((void**)&success, sizeof(int)));

    exec_kernel<<<1, 1, shmem_bytes, NULL>>>(success, shmem_bytes);
    auto err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
    EXPECT_EQ(success[0], 0);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

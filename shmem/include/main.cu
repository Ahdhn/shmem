#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "gtest/gtest.h"

#include "CUDASharedMemory.cuh"

__global__ void exec_kernel()
{
    CUDASharedMemory::Allocator shmem_allocator;
    shmem_allocator.alloc(2);
}

TEST(Test, exe)
{
    exec_kernel<<<1, 1, 10, NULL>>>();
    auto err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

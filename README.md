# shmem [![Windows](https://github.com/Ahdhn/shmem/actions/workflows/Windows.yml/badge.svg)](https://github.com/Ahdhn/shmem/actions/workflows/Windows.yml) [![Ubuntu](https://github.com/Ahdhn/shmem/actions/workflows/Ubuntu.yml/badge.svg)](https://github.com/Ahdhn/shmem/actions/workflows/Ubuntu.yml)

## CUDA Shared Memory Allocator 
Very simple CUDA shared memory allocator that could be useful if shared memory is used for storing data of different types and sizes. 

## How to use 
The user should make sure that enough shared memory is passed during the kernel. An example kernel is shown below
```C++
__global__ void shmem_kernel()
{
    CUDASharedMemory::Allocator shmem_allocator;

    // get a pointer in shared memory of type char with allocation size of 2
    // bytes
    char* char_ptr = shmem_allocator.alloc(2);

    // get a pointer in shared memory (after char_ptr) of type uint32_t to store
    // a single uint32_t i.e., 4 bytes
    uint32_t* uint32_ptr = shmem_allocator.alloc<uint32_t>(1);
}
```

## Build 
```
mkdir build
cd build 
cmake ..
```

Depending on the system, this will generate either a `.sln` project on Windows or a `make` file for a Linux system. 

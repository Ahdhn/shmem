#pragma once

#include <cuda_runtime.h>
namespace CUDASharedMemory {

extern __shared__ char SHMEM_START[];

/**
 * @brief Shared memory allocator that should make it easy to allocate different
 * segments of the shared memory of different types.
 */
struct Allocator
{
    __device__ Allocator() : m_ptr(nullptr)
    {
        m_ptr = SHMEM_START;
    }

    /**
     * @brief Allocate num_bytes and return a pointer to the start of the
     * allocation. The return pointer is aligned to bytes_alignment.
     * This function could be called by all threads if Allocator is in the
     * register. If Allocator is declared as __shared__, only one thread per
     * block should call this function.
     * @param num_bytes to allocate
     * @param byte_alignment alignment size
     */
    __device__ __forceinline__ char* alloc(size_t num_bytes,
                                           size_t byte_alignment = 8)
    {
        m_ptr = static_cast<char*>(m_ptr) + num_bytes;

        assert(get_allocated_size_bytes() <= get_max_size_bytes());

        return m_ptr;
    }

    /**
     * @brief return the maximum allocation size which is the same as the number
     * of bytes passed during the kernel launch
     */
    __device__ __forceinline__ uint32_t get_max_size_bytes()
    {
        uint32_t ret;
        asm("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
        return ret;
    }

    /**
     * @brief return the number of bytes that has been allocated
     */
    __device__ __forceinline__ uint32_t get_allocated_size_bytes()
    {
        return m_ptr - SHMEM_START;
    }

   private:
    char* m_ptr;
};


}  // namespace CUDASharedMemory
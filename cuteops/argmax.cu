#include <cuda_runtime.h>

/**
 * @brief returns the indices of the maximum value of all elements in the input tensor
 * 
 * @tparam T tensor data type
 * @param data input tensor
 * @param output output index tensor
 * @param n total elements number of output index tensor
 * @param stride stride of reduce dimension
 * @param dim_len reduce dimension length
 */
template<typename T>
__global__ void argmax_kernel(T* data, int* output, int n, int stride, int dim_len) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n) {
        T max_val = data[x * stride];
        int idx = 0;
        for(int i = 1; i< dim_len; ++i) {
            if (data[(x + i) * stride] > max_val) {
                max_val = data[(x + i) * stride];
                idx = i;
            }
        }
        output[x] = idx;
    }
}

/**
 * @brief returns the indices of the maximum value of all elements in the input tensor
 * 
 * @tparam T tensor data type
 * @param data 
 * @param output 
 * @param shapes 
 * @param ndim 
 * @param dim 
 */
template<typename T>
void argmax(T* data, T* output, int* shapes, int ndim, int dim) {
    int ele_num = 1; // total output tensor element number
    for(int i = 0; i < ndim; ++i) {
        ele_num *= shapes[i];
    }
    ele_num /= shapes[dim];
    
    int stride = 1;
    for(int i = dim; i < ndim; ++i) {
        stride *= shapes[i];
    }

    int BLOCK_SIZE = 1024;
    argmax_kernel<T><<<ceil(ele_num / BLOCK_SIZE), BLOCK_SIZE>>>(data, output, ele_num, stride, shapes[dim]);
}
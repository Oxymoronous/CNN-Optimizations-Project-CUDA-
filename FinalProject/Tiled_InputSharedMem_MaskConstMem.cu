#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define MAXNUM_THREADS 1024
#define TILE_WIDTH 8

__constant__ float constMem[16*4*7*7+5];

__global__ void conv_forward_kernel(float* output, const float* input, const float*  mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define constmask_4d(i3, i2, i1, i0) constMem[(i3) * (Channel * K * K) + (i2) * ( K * K) + (i1*K)  + i0]
    
    //int W_grid = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int W_grid = (Width_out - 1)/TILE_WIDTH + 1;
    int H_grid = ceil(Height_out/(1.0*TILE_WIDTH));

    int n, m, h0, w0, h, w, h_base, w_base;
    int X_tile_width = TILE_WIDTH + K - 1;
    extern __shared__ float shmem[];

    n = blockIdx.x;		//b_out
    m = blockIdx.y;		//m_out
    h0 = threadIdx.y;		
    w0 = threadIdx.x;
    h_base = (blockIdx.z / W_grid) * TILE_WIDTH; //tile_leftup_h
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH;//tileleftup_w
    h = h0 + h_base;    //h_out
    w = w0 + w_base;    //w_out

    float acc = 0.0f;    
    int c, p, q, i, j;
    for(c = 0; c < Channel; c++){

	//loading shared memory with input buffer values
    	for(i = h0; i < X_tile_width; i+= TILE_WIDTH){
            for(j = w0; j < X_tile_width; j+= TILE_WIDTH){
                if (i+h_base < Height && j+w_base < Width)
                    shmem[i * X_tile_width + j] = in_4d(n, c, i+h_base, j+w_base);
            }
	}
	
	    __syncthreads();
	
	    //calculating partial output
        for(p=0; p<K; p++){
            for(q=0; q<K; q++){
                if (h<Height_out && w < Width_out){
                    acc += (shmem[(h0+p)*X_tile_width+(w0+q)] *constmask_4d(m, c, p, q)); 
                }
            }
        }
	    __syncthreads();

    }

    //if (h <Height_out && w< Width_out){
    if (h < Height_out && w < Width_out){
	out_4d(n, m, h, w) = acc;
    }
    
    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    int height_out = Height - K + 1;
    int width_out = Width - K + 1;
    int output_elem_count = Batch * Map_out * height_out * width_out;
    int input_elem_count = Batch * Channel * Height * Width;
    int mask_elem_count = Map_out * Channel * K * K;

    printf("Mask:%d Channel:%d K:%d  \n", Map_out, Channel, K);
    cudaMalloc((void**)device_output_ptr, output_elem_count*sizeof(float));
    cudaMalloc((void**)device_input_ptr, input_elem_count*sizeof(float));
    //cudaMalloc((void**)device_mask_ptr, mask_elem_count*sizeof(float));

    //Streams
    //cudaMalloc((void**)my_input_ptr1, SEGMENT_SIZE*Map_out*height_out*width_out*sizeof(float));
    //cudaMalloc((void**)my_input_ptr2, SEGMENT_SIZE*Map_out*height_out*width_out*sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, input_elem_count*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(*device_mask_ptr, host_mask, mask_elem_count*sizeof(float), cudaMemcpyHostToDevice);
    
    //Constant memory
    cudaMemcpyToSymbol(constMem, host_mask, mask_elem_count*sizeof(float));

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    //these calculations are basically only for dimension use in the following block
    int height_out = Height - K + 1;
    int width_out = Width - K + 1;
    int w_grid = ceil(width_out /(1.0*(TILE_WIDTH)));  //number of horizontal tiles per output feature map
    int h_grid = ceil(height_out /(1.0*(TILE_WIDTH))); //number of vertical tiles per output feature map
    int z = w_grid * h_grid;
   
    //setting dimensions and call kernel
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    //dim3 gridDim(Map_out, z, Batch); //checkpoint2 version

	
    dim3 gridDim(Batch, Map_out, z);  //number of samples in the batch
                                      //number of output feature maps
                                      //Z = location of output tile inside the output feature map

    size_t shmem_size = (TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1)*sizeof(float);
    conv_forward_kernel<<<gridDim, blockDim, shmem_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K); 
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int height_out = Height - K + 1;
    int width_out = Width - K + 1;
    int output_elem_num = Batch * Map_out * height_out * width_out;
    cudaMemcpy(host_output, device_output, output_elem_num * sizeof(float), cudaMemcpyDeviceToHost);    
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    //cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}




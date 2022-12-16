// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE	 128

//@@ insert code here
__global__ void scan(unsigned int *input, float *output, int len, int width, int height) {
  //using Brent-Kung scan kernel
  //else section_size = len;       //phase 2, then section size is just the length of the S array
  //creating a shared memory for each block that is double the block size
  //each thread loads two elements into the shared memory
  __shared__ float sharedmem[2*BLOCK_SIZE];
  int block_idx = blockIdx.x;
  int thread_idx = threadIdx.x;
  int i = 2 * block_idx * blockDim.x + thread_idx;

  if (i < len) sharedmem[thread_idx] = input[i];
  else sharedmem[thread_idx] = 0.0;

  if (i + blockDim.x < len) sharedmem[thread_idx + blockDim.x] = input[i + blockDim.x];
  else sharedmem[thread_idx + blockDim.x] = 0.0;

  //each shared memory for a single block is the scan subsection
  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2){
    __syncthreads();
    int index = (thread_idx + 1) * 2 * stride - 1;
    if ((index < 2*BLOCK_SIZE)){
      sharedmem[index] += sharedmem[index - stride];
    }
  }

  //this recycles the usable value from scans that are already closed (no longer needs to calculate)
  for (int stride = ceil(BLOCK_SIZE/(2.0)); stride > 0; stride /= 2){
    __syncthreads();
    int index = (thread_idx + 1) * 2 *stride - 1;
    if (index + stride < BLOCK_SIZE*2) {
      sharedmem[index + stride] += sharedmem[index];
    }
  }
  //at this stage, the shared memory already contains the complete scans for the entire section
  
  __syncthreads();
  if (i < len) output[i] = sharedmem[thread_idx]/(width*height*1.0);
  else output[i] = 0.0;
  if (i + blockDim.x < len) output[i + blockDim.x] = sharedmem[thread_idx + blockDim.x]/(width*height*1.0);
  else output[i] = 0.0;
  __syncthreads();
}
//Kernel to cast image from float to unsigned char
__global__ void float_to_unsigned(float* wanttoconvert, unsigned char* converted, int imagewidth, int imageheight, int imagechannels){
	#define out_3d(i3, i2, i1) converted[ (i3 * imageheight *imagewidth) + (i2 * imagewidth) + i1]
	#define in_3d(i3, i2, i1) wanttoconvert[ (i3 * imageheight * imagewidth) + (i2*imagewidth) + i1]
	
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tz = blockIdx.z;

	if (tx < imagewidth && ty < imageheight && tz < imagechannels){
		out_3d(tz, ty, tx) = (unsigned char) (255 * in_3d(tz, ty, tx));		
	}
	
	#undef out_3d
	#undef in_3d
}

//CUDA kernel to cast image from unsigned char to float
__global__ void unsigned_to_float(unsigned char* wanttoconvert, float* converted, int imagewidth, int imageheight, int imagechannels){
	#define out_3dd(i3, i2, i1) converted[ (i3 * imageheight * imagewidth) + ( i2* imagewidth) + i1]
	#define in_3dd(i3, i2, i1) wanttoconvert[ (i3*imageheight*imagewidth) + (i2*imagewidth) + i1]
	
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
        int tz = blockIdx.z;

	if (tx < imagewidth && ty <imageheight && tz < imagechannels){
		out_3dd(tz, ty, tx) = (float) (in_3dd(tz, ty, tx) /(1.0* 255));
	}
	
	#undef out_3dd
	#undef in_3dd
}


//CUDA kernel for converting RGB image into grayscale
__global__ void rgbtograyscale(unsigned char* grayimage, unsigned char* rgbimage, int width, int height, int channels){
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col < width && row < height){
		int grayoffset = row * width + col;
		int rgboffset = grayoffset * channels;
		unsigned char r = rgbimage[rgboffset];
		unsigned char g = rgbimage[rgboffset + 1];
		unsigned char b = rgbimage[rgboffset + 2];

		grayimage[grayoffset] =(unsigned char)(0.21*r + 0.71*g + 0.07*b);
	}
}

//CUDA kernel to perform histogramming
__global__ void histo_kernel(unsigned char* grayscaleimage, int size, unsigned int* histo){
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	//size is imageheight * imagewidth
	while (tx < size){
		atomicAdd( &(histo[grayscaleimage[tx]]), 1);
		tx += stride;
	}
}

//CUDA kernel to compute the cdf by parallel scan method
//perform all the additions first, then perform the multiplication at the end
//output is the cdf
__global__ void kogge_stone_scan_kernel(unsigned int* histo, float* output, int width, int height){
	__shared__ unsigned int XY[BLOCK_SIZE*2];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < HISTOGRAM_LENGTH){
		XY[threadIdx.x] = histo[i];
	}

	for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
                __syncthreads();
		if (threadIdx.x >= stride) {
			unsigned int tmp = XY[threadIdx.x - stride];
                        __syncthreads();
			XY[threadIdx.x] += tmp;
                        __syncthreads();
		}
                __syncthreads();
	}
        if (i < HISTOGRAM_LENGTH)
        //output[i] = XY[threadIdx.x]*1.0;
	output[i] = (XY[threadIdx.x] / (width * height*1.0));
}

//CUDA kernel to implement histogram equalization function
__global__ void correct(unsigned char* image, float* cdf, int imagewidth, int imageheight, int imagechannels){
	#define clamp(x, start, end) min(max(x, start), end)
	#define correct_color(val) clamp(255* (cdf[val]-cdf[0])/(1.0 - cdf[0]), 0.0, 255.0)

	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;
	int tz = blockIdx.z;

	if(tx < imagewidth && ty < imageheight && tz < imagechannels){
		image[ (tz*imagewidth*imageheight) + (ty*imagewidth) + tx] = (unsigned char)(correct_color(image[ (tz*imagewidth*imageheight) + (ty*imagewidth) + tx]));
	}
	
	#undef correct_color
	#undef clamp
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float* deviceimage1;
  unsigned char* deviceimage_converted;
  unsigned char* deviceimage_gray;
  unsigned int* histogram1;	
  float* cdf1;
  float* deviceimage_converted2;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  int imagesize = imageChannels * imageWidth * imageHeight;
  cudaMalloc((void**)&deviceimage1, imagesize * sizeof(float));
  cudaMalloc((void**)&deviceimage_converted, imagesize * sizeof(unsigned char));
  cudaMalloc((void**)&deviceimage_gray, imageWidth * imageHeight * sizeof(unsigned char));	//grayscale image does not have the RGB channels
  cudaMalloc((void**)&histogram1, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset((void *)histogram1,0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void**)&cdf1, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void**)&deviceimage_converted2, imagesize * sizeof(float));

  cudaMemcpy(deviceimage1, hostInputImageData, imagesize*sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 griddim = dim3(ceil(imageWidth/(32*1.0)), ceil(imageHeight/(32*1.0)), 3);
  dim3 blockdim = dim3(32, 32, 1);

  float_to_unsigned<<<griddim, blockdim>>>(deviceimage1, deviceimage_converted, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  griddim = dim3(ceil(imageWidth/(32*1.0)), ceil(imageHeight/(32*1.0)), 1);
  rgbtograyscale<<<griddim, blockdim>>>(deviceimage_gray, deviceimage_converted, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  
  griddim = dim3(1, 1, 1);
  blockdim = dim3(32, 1, 1);
  histo_kernel<<<griddim, blockdim>>>(deviceimage_gray, imageWidth*imageHeight, histogram1);
  cudaDeviceSynchronize();
  
  
  unsigned int* debug0 = (unsigned int*) malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));
  //float* debug00 = (float*) malloc(HISTOGRAM_LENGTH * sizeof(float));
  cudaMemcpy(debug0, histogram1, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  /*
  for(int m = 0; m<HISTOGRAM_LENGTH; m++){
    wbLog(TRACE, "DEBUG HHH ", m , " ", debug0[m]);	
  }
  */  
  //cudaDeviceSynchronize();
  //cudaMemcpy(cdf1, debug00, HISTOGRAM_LENGTH*sizeof(float), cudaMemcpyHostToDevice);
  griddim = dim3(1, 1, 1);
  blockdim = dim3(128, 1, 1);
  //kogge_stone_scan_kernel<<<griddim, blockdim>>>(histogram1, cdf1, imageWidth, imageHeight);  
  scan<<<griddim, blockdim>>>(histogram1, cdf1, HISTOGRAM_LENGTH, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  float* debug = (float*)malloc(HISTOGRAM_LENGTH * sizeof(float));
  cudaMemcpy(debug, cdf1, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);
  /*
  for (int p = 0; p<256; p++){  
    wbLog(TRACE, "Debug CDF : ",p, " ", debug[p]);
  } 
  */
  wbLog(TRACE, imageWidth*imageHeight);

  griddim = dim3(ceil(imageWidth/(32*1.0)), ceil(imageHeight/(32*1.0)), 3);
  blockdim = dim3(32, 32, 1);
  correct<<<griddim, blockdim>>>(deviceimage_converted, cdf1, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  unsigned_to_float<<<griddim, blockdim>>>(deviceimage_converted, deviceimage_converted2, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceimage_converted2, imagesize*sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceimage1);
  cudaFree(deviceimage_converted);
  cudaFree(deviceimage_gray);
  cudaFree(histogram1);
  cudaFree(cdf1);
  cudaFree(deviceimage_converted2);
  free(debug);
  free(debug0);
  //free(debug00);
  return 0;
}

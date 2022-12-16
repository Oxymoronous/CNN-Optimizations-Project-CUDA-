#include <wb.h>

#define block_size 1024 //@@ you can change this
#define section_size 2*block_size
#define wbcheck(stmt)                                                     \
  do {                                                                    \
    cudaerror_t err = stmt;                                               \
    if (err != cudasuccess) {                                             \
      wblog(error, "failed to run stmt ", #stmt);                         \
      wblog(error, "got cuda error ...  ", cudageterrorstring(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void kernel3(float* s, float* y, int len){
  int i = 2 * blockidx.x * blockdim.x + threadidx.x;
  if (i < len && blockidx.x > 0){
	//checking whether we are within the bounds
  	y[i] += s[blockidx.x-1];
  } 

  if (i + blockdim.x < len && blockidx.x > 0){
    y[i + blockdim.x] += s[blockidx.x-1];
  }
}

__global__ void scan(float *input, float *output, int len, float* s, int phase_boolean) {
  //using brent-kung scan kernel
  int section_size;

  section_size = section_size; //phase 1, section size is the share memory size
  //else section_size = len;       //phase 2, then section size is just the length of the s array
  //creating a shared memory for each block that is double the block size
  //each thread loads two elements into the shared memory
  __shared__ float sharedmem[section_size];
  int block_idx = blockidx.x;
  int thread_idx = threadidx.x;
  int i = 2 * block_idx * blockdim.x + thread_idx;

  if (i < len) sharedmem[thread_idx] = input[i];
  else sharedmem[thread_idx] = 0;

  if (i + blockdim.x < len) sharedmem[thread_idx + blockdim.x] = input[i + blockdim.x];
  else sharedmem[thread_idx + blockdim.x] = 0;

  //each shared memory for a single block is the scan subsection
  for (unsigned int stride = 1; stride <= blockdim.x; stride *= 2){
    __syncthreads();
    int index = (thread_idx + 1) * 2 * stride - 1;
    if (index < section_size){
      sharedmem[index] += sharedmem[index - stride];
    }
  }

  //this recycles the usable value from scans that are already closed (no longer needs to calculate)
  for (int stride = section_size/4; stride > 0; stride /= 2){
    __syncthreads();
    int index = (thread_idx + 1) * 2 *stride - 1;
    if (index + stride < section_size) {
      sharedmem[index + stride] += sharedmem[index];
    }
  }
  //at this stage, the shared memory already contains the complete scans for the entire section
  
  __syncthreads();
  if (i < len) output[i] = sharedmem[thread_idx];
  if (i + blockdim.x < len) output[i + blockdim.x] = sharedmem[thread_idx + blockdim.x];
  __syncthreads();

  if (phase_boolean == 1){
	//we only extract the last element from the section in phase 1
  	if (thread_idx == blockdim.x - 1){
    		s[block_idx] = sharedmem[section_size-1];
  	}
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  wbTime_start(Compute, "Performing CUDA computation");
  
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  float* S;
  float* out_debug;
  float* S2_debug;
  float* S_debug;
  FILE* f = fopen("debug.txt", "a");
  FILE* f2 = fopen("auxliiary.txt", "a");
  int phase = 1;
  int S_input_len = ceil(numElements / (1.0*SECTION_SIZE)); //consider any arbitrary input length
							//there can be sections with "not completely filled up" cells
 
  S_debug = (float*)malloc(S_input_len * sizeof(float));
  S2_debug = (float*)malloc(S_input_len * sizeof(float));
  out_debug = (float*)malloc(numElements*sizeof(float));


  dim3 phase1_grid(ceil(numElements/(1.0*SECTION_SIZE)), 1, 1);
  dim3 phase1_block(BLOCK_SIZE, 1, 1);
  cudaMalloc((void**)&S, S_input_len * sizeof(float));
  scan<<<phase1_grid, phase1_block>>>(deviceInput, deviceOutput, numElements, S, phase);
  cudaDeviceSynchronize();
  wbTime_start(Copy, "Debug copy1 start");
  cudaMemcpy(S_debug, S, S_input_len, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Debug copy1 start");
  //------------------------ end of Phase 1, starting phase 2
  phase += 1;
  dim3 phase2_grid(1, 1, 1);		//because we are only using 1 element from each shared memory
					//shared memory : input size = 1 : 1
					//shared memory : block      = 2 : 1
					//8 inputs = 8 shared memory = 4 blocks

  int phase2_calc = ceil(S_input_len/2.0); 
  dim3 phase2_block(phase2_calc, 1, 1);  
  scan<<<phase2_grid, phase2_block>>>(S, S, S_input_len, S, phase);	//the array S is passed in as a dummy here
  									//because of the phase flag variable, array S is safe and not affected
  cudaDeviceSynchronize();
  
  wbTime_start(Copy, "Debug copy2start");
  cudaMemcpy(S2_debug, S, S_input_len, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Debug copy2start");
  for (int i=0; i<S_input_len; i++){
	fprintf(f2, "%f ", S_debug[i]);
  }
  fprintf(f2, "-----------------------------------------------------------------------\n");


  for (int i=0; i<S_input_len; i++){
    fprintf(f2, "%f ", S2_debug[i]);
  }  
  //printf("------------------------------------------------------------------------------------\n");
  
  //--------------------phase 3----------------------------------
  phase += 1;
  dim3 phase3_grid(ceil(numElements/(SECTION_SIZE*1.0)), 1, 1);
  dim3 phase3_block(BLOCK_SIZE, 1, 1);		//using section_size here because we only process 1 output in the kernel
  kernel3<<<phase3_grid, phase3_block>>>(S, deviceOutput, numElements);
  cudaDeviceSynchronize();
  
  wbTime_start(Copy, "Debug 3 start");
  cudaMemcpy(out_debug, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Debug 3 start");
  fprintf(f, "Debug Output\n");
  for (int i=0; i<numElements; i++){
	fprintf(f, "%f\n", out_debug[i]);
  } 

  fprintf(f, "\n");
  fclose(f);

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(S);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);
  free(S_debug);
  free(S2_debug);
  free(out_debug);
  return 0;
}





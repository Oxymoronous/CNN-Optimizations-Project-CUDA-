
#include <wb.h>
#define TILE_WIDTH 16
#define BLOCK_WIDTH 16
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {

	//matrix multiply requires that numAColumns == numBRows
	__shared__ float subtileM[TILE_WIDTH][TILE_WIDTH];
	__shared__ float subtileN[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
        int tx = threadIdx.x;
	int by = blockIdx.y;
	int ty = threadIdx.y;
	int prow = by * TILE_WIDTH + ty;	//y index of grid relates to first matrix => relates to rows of output matrix
	int pcol = bx * TILE_WIDTH + tx;	//x index relates to second matrix        => relates to columns of output matrix

	float Pvalue = 0;
	float conversionforceil = 1.0 * numAColumns;
	//looping through each tile for both A and B (A and B will have same number of tiles 
	//because tiles = A columns = B Rows = the same thing)
	for(int i=0; i<ceil(conversionforceil/TILE_WIDTH); ++i){//looping through each tile
		//condition is checking 1. whether we are in range for the matrix A rows (y-direction)
		//			2. whether each column that we try to access from matrix A columns (x-direction) 
		if ( (prow < numARows) && ( i*TILE_WIDTH+tx < numAColumns) ){
                        //determinant = max(ARows, ACols) 
			subtileM[ty][tx] = A[prow * numAColumns + i*TILE_WIDTH + tx];
			//prow independent of threadIdx.x, 
		}
		else{
			subtileM[ty][tx] = 0;
		}
		//notice that the second ifs always both check for numAColumns!!!
		if ( (pcol < numBColumns) && (i*TILE_WIDTH+ty < numAColumns) ){
			//determinant = max(BCols, ACols)
			subtileN[ty][tx] = B[(i*TILE_WIDTH+ty)*numBColumns + pcol];
		}
		else{
			subtileN[ty][tx] = 0;
		}
		__syncthreads();//waiting for all threads to finish loading 1 tile
		for(int j=0; j<TILE_WIDTH; j++){
		  Pvalue += subtileM[ty][j] * subtileN[j][tx];
		}
		__syncthreads();
	}

	if ((prow < numCRows) && (pcol<numCColumns)){
		C[prow * numCColumns + pcol] = Pvalue;
	}
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  int Celem = numCRows * numCColumns;
  int Belem = numBRows * numBColumns;
  int Aelem = numARows * numAColumns;
  hostC = (float*)malloc(Celem*sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**)&deviceA, Aelem*sizeof(float));
  cudaMalloc((void**)&deviceB, Belem*sizeof(float));
  cudaMalloc((void**)&deviceC, Celem*sizeof(float));
 
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, Aelem*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, Belem*sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil((numBColumns*1.0)/BLOCK_WIDTH), ceil((numARows*1.0)/BLOCK_WIDTH), 1);
  dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, Celem*sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  
  wbTime_stop(GPU, "Freeing GPU Memory");
  
  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}

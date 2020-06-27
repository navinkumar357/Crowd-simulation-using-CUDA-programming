
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_testkernel.h"
#include <stdio.h>
#include <math.h>
#include "ped_model.h"
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t computeNextPositionWithCuda(const int *x, const int *y, const float *destinationX, const float *destinationY, unsigned int size, int *desiredX, int *desiredY);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void computeNextPositionKernel(int *desiredX, int *desiredY, int *x, int *y, float *destinationX, float *destinationY)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	float diffX = destinationX[i] - x[i];
	float diffY = destinationY[i] - y[i];
	float len = sqrt(diffX * diffX + diffY * diffY);

	float desiredXFloat = round(x[i] + diffX / len);
	float desiredYFloat = round(y[i] + diffY / len);

	desiredX[i] = int(desiredXFloat);
	desiredY[i] = int(desiredYFloat);
}

__global__ void fadeHeatmapKernel(int *heatmap, int size)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int row = 0; row < size; row++) {
		heatmap[row * size + col] = (int)round(heatmap[row * size + col] * 0.8);
	}
}

__global__ void updateHeatmapKernel(int *heatmap, int size, int *desiredX, int *desiredY, int agents)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx > agents) return;
	if (desiredX[idx] < 0 || desiredX[idx] >= size || desiredY[idx] < 0 || desiredY[idx] >= size)
	{
		return;
	}

	atomicAdd(heatmap + desiredY[idx] * size + desiredX[idx], 40);
	atomicMin(heatmap + desiredY[idx] * size + desiredX[idx], 255);
}

/*__global__ void scaleHeatmapKernel(int *heatmap, int *scaled_heatmap, int size, int cell_size)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int start = threadId % size;
	int step = threadId / size;
	start += step * size * cell_size;
	int value = heatmap[threadId];
	for (int cellY = 0; cellY < cell_size; cellY++)
	{
		for (int cellX = 0; cellX < cell_size; cellX++)
		{
			int index = start * cell_size + cellX;
			index += size * cell_size * cellY;
			scaled_heatmap[index] = value;
		}
	}
}*/

__global__ void blurHeatmapKernel(int *heatmap, int *blurred_heatmap, int size, int cell_size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int row = blockIdx.x;
	int j = 0;
	// Block size 
	// num of rows per block = 5 + 4 = 9
	int thread_id = threadIdx.x;
	__shared__ int hm_s[SIZE * 2];
	// store two rows of the heatmap inide shared memory

	for (int i = 0; i < 2; i += 1) {
		hm_s[thread_id + i * SIZE] = heatmap[thread_id + (row + i) * SIZE] < 255 ? heatmap[thread_id + (row + i) * SIZE] : 255;
		heatmap[thread_id + (row + i) * SIZE] = hm_s[thread_id + i * SIZE];
	}
	// synchronize threads
	__syncthreads();
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
	{ 4, 16, 26, 16, 4 },
	{ 7, 26, 41, 26, 7 },
	{ 4, 16, 26, 16, 4 },
	{ 1, 4, 7, 4, 1 }
	};
	#define WEIGHTSUM 273
	// every thread calculates 25 values of the scaled heatmap
	// first row 
	if ((thread_id != 0) & (thread_id != 1023)) {
		for (int i = 0; i < 5; i++) {
			int sum = 0;
			int index;
			for (int k = -2; k < 3; k++) {
				for (int l = -2; l < 3; l++) {
					index = (int)((thread_id * 5 + i + k) / 5);
					sum += hm_s[index] * w[2 + k][2 + l];
				}
			}
			int value = sum / WEIGHTSUM;
			blurred_heatmap[thread_id * 5 + i + (row * 5 * SCALED_SIZE)] = 0x00FF0000 | value << 24;
		}
		for (int j = 2; j < 6; j++) {
			for (int i = 0; i < 5; i++) {
				int index;
				int sum = 0;
				for (int k = -2; k < 3; k++) {
					for (int l = -2; l < 3; l++) {
						index = (int)((thread_id * 5 + i + k) / 5) + SIZE * (int)((j + l) / 4);
						sum += hm_s[index] * w[2 + k][2 + l];
					}
				}
				int value = sum / WEIGHTSUM;
				blurred_heatmap[thread_id * 5 + i + (((row * 5) + j - 1)* SCALED_SIZE)] = 0x00FF0000 | value << 24;
			}
		}
	}

}

void cuda_setupHeatmap(int *heatmap, int *scaled_heatmap, int *blurred_heatmap)
{
	const int size = 1024;
	const int cell_size = 5;
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	cudaStatus = cudaHostAlloc(&heatmap, size * size * sizeof(int), cudaHostAllocDefault);
	if (cudaStatus != cudaSuccess)
		printf("Error allocating pinned host memory\n");
	cudaStatus = cudaHostAlloc(&scaled_heatmap, size * cell_size * size * cell_size * sizeof(int), cudaHostAllocDefault);
	if (cudaStatus != cudaSuccess)
		printf("Error allocating pinned host memory\n");
	cudaStatus = cudaHostAlloc(&blurred_heatmap, size * cell_size * size * cell_size * sizeof(int), cudaHostAllocDefault);
	if (cudaStatus != cudaSuccess)
		printf("Error allocating pinned host memory\n");
}

cudaError_t createHeatmapWithCuda(Ped::Model *model, int *heatmap, int *scaled_heatmap, int *blurred_heatmap, const int size, const int cell_size, int *desiredX, int *desiredY, const int agents)
{
	int *dev_heatmap = 0;
	int *dev_scaled_heatmap = 0;
	int *dev_blurred_heatmap = 0;
	int *dev_desiredX = 0;
	int *dev_desiredY = 0;
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);

	cudaStatus = cudaMalloc(&dev_heatmap, size * size * sizeof(int));
	cudaStatus = cudaMalloc(&dev_scaled_heatmap, size * cell_size * size * cell_size * sizeof(int));
	cudaStatus = cudaMalloc(&dev_blurred_heatmap, size * cell_size * size * cell_size * sizeof(int));
	cudaStatus = cudaMalloc(&dev_desiredX, size * sizeof(int));
	cudaStatus = cudaMalloc(&dev_desiredY, size * sizeof(int));

	cudaStatus = cudaMemcpyAsync(dev_heatmap, heatmap, size * size * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpyAsync(dev_scaled_heatmap, scaled_heatmap, size * cell_size * size * cell_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpyAsync(dev_blurred_heatmap, blurred_heatmap, size * cell_size * size * cell_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpyAsync(dev_desiredX, desiredX, size * sizeof(int), cudaMemcpyHostToDevice); // Not pinned.
	cudaStatus = cudaMemcpyAsync(dev_desiredY, desiredY, size * sizeof(int), cudaMemcpyHostToDevice); // Not pinned.
	
	dim3 grid(1024, 5, 5);
	dim3 block(1024, 1, 1);


	float fade_time;
	cudaEvent_t start_fade_event, stop_fade_event;
	cudaEventCreate(&start_fade_event);
	cudaEventCreate(&stop_fade_event);
	cudaEventRecord(start_fade_event, 0);

	fadeHeatmapKernel << < 1, 1024 >> > (dev_heatmap, size);
	
	cudaEventRecord(stop_fade_event, 0);
	cudaEventSynchronize(stop_fade_event);
	cudaEventElapsedTime(&fade_time, start_fade_event, stop_fade_event);
	cudaEventDestroy(start_fade_event);
	cudaEventDestroy(stop_fade_event);

	float update_hm_time;
	cudaEvent_t start_update_event, stop_update_event;
	cudaEventCreate(&start_update_event);
	cudaEventCreate(&stop_update_event);
	cudaEventRecord(start_update_event, 0);

	updateHeatmapKernel << < 1, 1024 >> > (dev_heatmap, size, dev_desiredX, dev_desiredY, agents);
	
	cudaEventRecord(stop_update_event, 0);
	cudaEventSynchronize(stop_update_event);
	cudaEventElapsedTime(&update_hm_time, start_update_event, stop_update_event);
	cudaEventDestroy(start_update_event);
	cudaEventDestroy(stop_update_event);

	float blur_time;
	cudaEvent_t start_blur_event, stop_blur_event;
	cudaEventCreate(&start_blur_event);
	cudaEventCreate(&stop_blur_event);
	cudaEventRecord(start_blur_event, 0);

	blurHeatmapKernel << < 1024, 1024>> > (dev_heatmap, dev_blurred_heatmap, size, cell_size);

	cudaEventRecord(stop_blur_event, 0);
	cudaEventSynchronize(stop_blur_event);
	cudaEventElapsedTime(&blur_time, start_blur_event, stop_blur_event);
	cudaEventDestroy(start_blur_event);
	cudaEventDestroy(stop_update_event);

	float collision_time;
	cudaEvent_t start_collision_event, stop_collision_event;
	cudaEventCreate(&start_collision_event);
	cudaEventCreate(&stop_collision_event);
	cudaEventRecord(start_collision_event, 0);

	model->collision_detection_regions();
	
	cudaEventRecord(stop_collision_event, 0);
	cudaEventSynchronize(stop_collision_event);
	cudaEventElapsedTime(&collision_time, start_collision_event, stop_collision_event);
	cudaEventDestroy(start_collision_event);
	cudaEventDestroy(stop_collision_event);
	
	
	printf("Fade event:  %0.6f ms\n", fade_time);
	printf("Update heatmap:  %0.6f ms\n", update_hm_time);
	printf("Blur heatmap:  %0.6f ms\n", blur_time);
	printf("Collision Detection:  %0.6f ms\n", collision_time);

	cudaStatus = cudaMemcpyAsync(heatmap, dev_heatmap, size * size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpyAsync(scaled_heatmap, dev_scaled_heatmap, size * cell_size * size * cell_size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpyAsync(blurred_heatmap, dev_blurred_heatmap, size * cell_size * size * cell_size * sizeof(int), cudaMemcpyDeviceToHost);

	//cudaStatus = cudaDeviceSynchronize();
	cudaDeviceSynchronize();
	cudaFree(dev_heatmap);
	cudaFree(dev_scaled_heatmap);
	cudaFree(dev_blurred_heatmap);
	cudaFree(dev_desiredX);
	cudaFree(dev_desiredY);

	/*cudaFreeHost(heatmap);
	cudaFreeHost(scaled_heatmap);
	cudaFreeHost(blurred_heatmap);
	cudaFreeHost(desiredX);
	cudaFreeHost(desiredY);*/

	return cudaStatus;
}

void cuda_updateHeatmap(Ped::Model *model, int *heatmap, int *scaled_heatmap, int *blurred_heatmap, int size, int cell_size, int *desiredX, int *desiredY, const int agents)
{
	cudaError_t cudaStatus = createHeatmapWithCuda(model, heatmap, scaled_heatmap, blurred_heatmap, size, cell_size, desiredX, desiredY, agents);
}

int cuda_test()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

Tuple cuda_tick(const int *x, const int *y, const float *destinationX, const float *destinationY, int *desiredX, int *desiredY, const int size1)
{
	cudaError_t cudaStatus = computeNextPositionWithCuda(x, y, destinationX, destinationY, size1, desiredX, desiredY);

	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "computeNextPositionWithCuda failed!");
	//	//return 1;
	//}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceReset failed!");
	//	//return 1;
	//}

	Tuple r = { desiredX, desiredY };
	return r;
}

cudaError_t computeNextPositionWithCuda(const int *x, const int *y, const float *destinationX, const float *destinationY, unsigned int size, int *desiredX, int *desiredY)
{
	int *dev_x = 0;
	int *dev_y = 0;
	float *dev_destinationX = 0;
	float *dev_destinationY = 0;
	int *dev_desiredX = 0;
	int *dev_desiredY = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	/*if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	fprintf(stderr, "%s.\n", cudaGetErrorString(cudaGetLastError()));
	goto Error;
	}*/

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_x, size * sizeof(int));
	/*if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMalloc failed!");
	goto Error;
	}*/

	cudaStatus = cudaMalloc((void**)&dev_y, size * sizeof(int));
	/*if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMalloc failed!");
	goto Error;
	}*/

	cudaStatus = cudaMalloc((void**)&dev_destinationX, size * sizeof(float));
	/*if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMalloc failed!");
	goto Error;
	}*/

	cudaStatus = cudaMalloc((void**)&dev_destinationY, size * sizeof(float));
	/*if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMalloc failed!");
	goto Error;
	}
	*/
	cudaStatus = cudaMalloc((void**)&dev_desiredX, size * sizeof(int));
	/*if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMalloc failed!");
	goto Error;
	}
	*/
	cudaStatus = cudaMalloc((void**)&dev_desiredY, size * sizeof(int));
	/*if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMalloc failed!");
	goto Error;
	}
	*/
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_x, x, size * sizeof(int), cudaMemcpyHostToDevice);
	/*if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed!");
	goto Error;
	}
	*/
	cudaStatus = cudaMemcpy(dev_y, y, size * sizeof(int), cudaMemcpyHostToDevice);
	/*if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed!");
	goto Error;
	}*/

	cudaStatus = cudaMemcpy(dev_destinationX, destinationX, size * sizeof(float), cudaMemcpyHostToDevice);
	/*if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed!");
	goto Error;
	}*/

	cudaStatus = cudaMemcpy(dev_destinationY, destinationY, size * sizeof(float), cudaMemcpyHostToDevice);
	/*if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed!");
	goto Error;
	}
	*/
	// Launch a kernel on the GPU with one thread for each element.
	int blocks, width;
	if (size > 1024) {
		blocks = 4;
		width = size / 4;
	}
	else {
		blocks = 1;
		width = size;
	}
	computeNextPositionKernel << <blocks, width >> >(dev_desiredX, dev_desiredY, dev_x, dev_y, dev_destinationX, dev_destinationY);

	// Check for any errors launching the kernel
	//cudaStatus = cudaGetLastError();
	/*if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "computeNextPositionKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	goto Error;
	}
	else
	{
	fprintf(stderr, "Cuda launch succeeded! \n");
	}*/

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	// cudaStatus = cudaDeviceSynchronize();
	/*if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	goto Error;
	}*/

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(desiredX, dev_desiredX, size * sizeof(int), cudaMemcpyDeviceToHost);
	//if (cudaStatus != cudaSuccess) {
	//		fprintf(stderr, "cudaMemcpy failed!");
	//		goto Error;
	//	}

	cudaStatus = cudaMemcpy(desiredY, dev_desiredY, size * sizeof(int), cudaMemcpyDeviceToHost);
	/*if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed!");
	goto Error;
	}*/

	//Error:
	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_destinationX);
	cudaFree(dev_destinationY);
	cudaFree(dev_desiredX);
	cudaFree(dev_desiredY);
	//if (cudaStatus != 0) {
	//	fprintf(stderr, "Cuda does not seem to be working properly.\n"); // This is not a good thing
	//}
	//else {
	//	//fprintf(stderr, "Cuda functionality test succeeded.\n"); // This is a good thing
	//}

	return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		fprintf(stderr, "%s.\n", cudaGetErrorString(cudaGetLastError()));
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	else
	{
		//fprintf(stderr, "Cuda launch succeeded! \n");
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	if (cudaStatus != 0) {
		fprintf(stderr, "Cuda does not seem to be working properly.\n"); // This is not a good thing
	}
	else {
		fprintf(stderr, "Cuda functionality test succeeded.\n"); // This is a good thing
	}

	return cudaStatus;
}
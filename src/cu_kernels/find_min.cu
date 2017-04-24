/*
 * find_min.cu
 *
 *  Created on: 19 nov. 2015
 *      Author: tkostas
 */

#include "find_min.cuh"
#include "stdio.h"

void cpu_max(unsigned char *values, unsigned int size, unsigned char gpu_max);



void __global__ find_min_k(unsigned char *input, unsigned char *output, unsigned int nb_elt)
{
	extern __shared__ unsigned char sdatac[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	sdatac[tid] = 0;

	if (i  < nb_elt)
	{
		if (i + blockDim.x > nb_elt - 1)
			sdatac[tid] = input[i * 3];
		else if (i < nb_elt)
			sdatac[tid] = fminf(input[i * 3 + blockDim.x * 3], input[i * 3]);
	}
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
			sdatac[tid] = fminf(sdatac[tid + s], sdatac[tid]);

		__syncthreads();
	}

	if (tid == 0)
		output[blockIdx.x * 3] = sdatac[0];
}

void __global__ find_max_k(unsigned char *input, unsigned char *output, unsigned int nb_elt)
{
	extern __shared__ unsigned char sdatac[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	sdatac[tid] = 0;

	if (i  < nb_elt)
	{
		if (i + blockDim.x > nb_elt - 1)
			sdatac[tid] = input[i * 3];
		else if (i < nb_elt)
			sdatac[tid] = fmaxf(input[i * 3 + blockDim.x * 3], input[i * 3]);
	}
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
			sdatac[tid] = fmaxf(sdatac[tid + s], sdatac[tid]);

		__syncthreads();
	}

	if (tid == 0)
		output[blockIdx.x * 3] = sdatac[0];
}


unsigned char find_min(unsigned int pixel_size, unsigned char *input, unsigned char *output, unsigned char *temp_input) // min value is located in output[0]
{

	unsigned int threads;
	unsigned int block;
	get_threads_blocks(&threads, &block, pixel_size);

	if (block % 2 == 0)
		block = block / 2;
	else
		block = block / 2 + 1;

	bool run_loop = true;
	bool first_kernel_run = true;

	while (run_loop)
	{
		if (block == 1)
			run_loop = false;
		if (first_kernel_run)
			find_min_k <<<block, threads, threads * sizeof(unsigned char) >>>(input, output, pixel_size);
		else
			find_min_k <<<block, threads, threads * sizeof(unsigned char) >>>(temp_input, output, pixel_size);
		first_kernel_run = false;
		cudaDeviceSynchronize();
		cudaMemcpy(temp_input, output, block * sizeof(unsigned char) * 3, cudaMemcpyDeviceToDevice);
		pixel_size = block;
		get_threads_blocks(&threads, &block, pixel_size);
		if (block % 2 == 0)
			block = block / 2;
		else
			block = block / 2 + 1;
	}
	unsigned char min_found;
	printf("ERROR += %s \n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(&min_found, output, sizeof(unsigned char), cudaMemcpyDeviceToHost);
	//cpu_max(input,1000000 , min_found);
	return min_found;
}


unsigned char find_max(unsigned int pixel_size, unsigned char *input, unsigned char *output, unsigned char *temp_input) // min value is located in output[0]
{

	unsigned int threads;
	unsigned int block;
	get_threads_blocks(&threads, &block, pixel_size);

	if (block % 2 == 0)
		block = block / 2;
	else
		block = block / 2 + 1;

	bool run_loop = true;
	bool first_kernel_run = true;

	while (run_loop)
	{
		if (block == 1)
			run_loop = false;
		if (first_kernel_run)
			find_max_k <<<block, threads, threads * sizeof(unsigned char) >>>(input, output, pixel_size);
		else
			find_max_k <<<block, threads, threads * sizeof(unsigned char) >>>(temp_input, output, pixel_size);
		first_kernel_run = false;
		cudaDeviceSynchronize();
		cudaMemcpy(temp_input, output, block * sizeof(unsigned char) * 3, cudaMemcpyDeviceToDevice);
		pixel_size = block;
		get_threads_blocks(&threads, &block, pixel_size);
		if (block % 2 == 0)
			block = block / 2;
		else
			block = block / 2 + 1;
	}
	unsigned char max_found;
	printf("ERROR += %s \n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(&max_found, output, sizeof(unsigned char), cudaMemcpyDeviceToHost);
	//cpu_max(input,1000000 , min_found);
	return max_found;
}
/////////////////////////DEBUG FUNCTIONS/////////////////////////////////////////////////////


void cpu_max(unsigned char *values, unsigned int size, unsigned char gpu_max)
{
	unsigned char *values_cpu = (unsigned char*) malloc(sizeof(unsigned char) * size * 3);
	cudaMemcpy(values_cpu, values, size * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	unsigned char max = 0;
	for (int i = 0; i < size; i++)
	{

		if (max <  values_cpu[i * 3])
		{
			max = values_cpu[i * 3];
			printf("cpu val = %i \n", values_cpu[i * 3] );
		}
	}

	if (gpu_max != max)
	{
	printf("cpu_max = %i %i\n", max, gpu_max);
	getchar();
	}
	free(values_cpu);
}

void make_random(float **vector, unsigned int vec_size)
{
	float *v1_gpu_x;
	cudaMalloc(&v1_gpu_x, sizeof(float)* vec_size);
	float *v1_cpu_x = (float*)malloc(sizeof(float)* vec_size);
	for (unsigned int i = 0; i < vec_size; i++)
	{
		v1_cpu_x[i] = (float)(vec_size - i);
	}
	//cpu_min(v1_cpu_x, vec_size);
	cudaMemcpy(v1_gpu_x, v1_cpu_x, sizeof(float)* vec_size, cudaMemcpyHostToDevice);
	free(v1_cpu_x);
	*vector = v1_gpu_x;
}

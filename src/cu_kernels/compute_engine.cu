/*
 * compute_engine.cu
 *
 *  Created on: Dec 18, 2016
 *      Author: tkostas
 */

#include "../hardware_limits.hh"
#include <stdio.h>
#include <sys/time.h>
#include "device_functions.h"
#include "../tools.hh"

__global__ void find_perfect_night_k(unsigned char *gpu_nights, unsigned int data_size, unsigned int night_size, unsigned int *score)
{

	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__  int shem[1];

	while (tid < data_size)
	{
			if (threadIdx.x == 0)
			{
			//unsigned int night_number = tid / night_size;
			shem[0] = (int) gpu_nights[tid + night_size - 1];
			}
			// make all threads initialize shared score memory


			__syncthreads(); // ensure that the score has been written into shared memory by thirst thread

			unsigned int sleep_phase = (int) gpu_nights[tid];
			if(sleep_phase != 0) //not awake
			atomicAdd(score + threadIdx.x + night_size * (sleep_phase - 1), shem[0]);
			tid += blockDim.x * gridDim.x;
	}






}

void make_night(unsigned int *score, unsigned int night_size)
{
	unsigned char *night = (unsigned char*) malloc(sizeof(unsigned char) * night_size);
	unsigned int max_minute_score = 0;
	unsigned char best_state;
	for (int i = 0; i < night_size - 1; i++)
	{
		max_minute_score = 0;
		best_state = 0;
		for (unsigned int j = 0; j < 5; j++)
		if ( max_minute_score < score[i + j * night_size])
		{
			max_minute_score = score[i + j * night_size];
			best_state = j + 1;
		}
		night[i] = best_state;
	}
	save_night_to_text("best_night", night_size, night, night_size);
}



void find_perfect_night(unsigned char *cpu_nights, unsigned int nb_nights, unsigned int night_size)
{
	struct timeval t1, t2;
	gettimeofday(&t1, 0);

	unsigned char *gpu_nights;
	unsigned int *score;
	unsigned int *score_cpu;
	cudaMalloc((void **)&score, 5 * night_size * sizeof(int)); // one vector per possibility
	cudaMemset(score, 0, 5 * night_size * sizeof(int));
	cudaMalloc((void **)&gpu_nights,nb_nights * night_size);
	cudaMemcpy(gpu_nights,cpu_nights, nb_nights * night_size, cudaMemcpyHostToDevice);

	unsigned int data_size = nb_nights * night_size;
	unsigned int nb_blocks = data_size / night_size + 1;
	printf("nights processed %i \n", nb_nights);



	find_perfect_night_k<<< nb_blocks, 531>>>(gpu_nights, data_size, night_size, score);
	cudaDeviceSynchronize();

	gettimeofday(&t2, 0);



	cudaMallocHost((void **)&score_cpu, 5 * night_size * sizeof(int));

	cudaMemcpy(score_cpu, score, 5 * night_size * sizeof(int), cudaMemcpyDeviceToHost);
	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

	printf("Time to process:  %3.1f ms \n", time);
	save_night_to_text("score", 5 * night_size, score_cpu, night_size);
	make_night(score_cpu, night_size);

}










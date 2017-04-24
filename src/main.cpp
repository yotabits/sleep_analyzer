/*
 * main.cpp
 *
 *  Created on: Dec 18, 2016
 *      Author: tkostas
 */
#include "cu_kernels/julia.cuh"
#include "cu_kernels/compute_engine.cuh"
#include <cstdio>
#include <string.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "video_saver.hh"
#include <random>
#include "tools.hh"

void save_to_raw(void *gpu_img, unsigned char *cpu_img, char *filename, unsigned int img_byte_size)
{
	if (cudaMemcpy(cpu_img, gpu_img, img_byte_size, cudaMemcpyDeviceToHost) != cudaSuccess)
		printf("file copy failed\n");
	add_picture(cpu_img, 2000);

/*
	FILE *pFile = fopen(filename,"a+b");

	unsigned int element_written = fwrite(cpu_img, img_byte_size, 1, pFile);
	if (element_written ==  1)
	{
		fclose(pFile);
	}
*/
}


unsigned char *cpu_gen_nights(unsigned int nb_nights)
{
	// we could have use bitset to represent night and save memory space but it won't fit further cuda computations
	unsigned short sleep_duration = 531; // 530 minutes of sleep + 1 space for ranking
	unsigned char *nights = NULL; //= (unsigned char *) malloc(sizeof(unsigned char) * nb_nights * sleep_duration);

	// we use cudamalloc host instead of malloc for host to device copy performance
	cudaMallocHost((void **)&nights, sizeof(unsigned char) * nb_nights * sleep_duration);

	memset(nights,0,sizeof(unsigned char) * nb_nights * sleep_duration);
	std::default_random_engine generator(time(0));
	std::uniform_real_distribution<> distribution(-0.20, 0.20);
	std::uniform_int_distribution<> wake_up(0,100);

	unsigned int phase_1_sleep_time = 7; // max ->
	unsigned int phase_2_sleep_time = 10; // max ->
	unsigned int phase_3_sleep_time = 15; // max ->
	unsigned int phase_4_sleep_time = 30; // max ->
	unsigned int phase_REM_sleep_time = 15; //

	unsigned int cycle_max_duration = phase_2_sleep_time + (unsigned int)round((0.20 * phase_2_sleep_time))
									 + phase_3_sleep_time + (unsigned int)round((0.20 * phase_3_sleep_time))
									 + phase_4_sleep_time + (unsigned int)round((0.20 * phase_4_sleep_time))
									 + phase_3_sleep_time + (unsigned int)round((0.20 * phase_3_sleep_time))
									 + phase_2_sleep_time + (unsigned int)round((0.20 * phase_2_sleep_time))
									 + phase_REM_sleep_time + (unsigned int)round((0.20 * phase_REM_sleep_time))
									 + phase_1_sleep_time + (unsigned int)round((0.20 * phase_1_sleep_time));

	unsigned int wake_up_cycle_max_duration = phase_2_sleep_time + (unsigned int)round((0.20 * phase_2_sleep_time))
											 + phase_REM_sleep_time + (unsigned int)round((0.20 * phase_REM_sleep_time));


	unsigned int phase_1_sleep_time_random;
	unsigned int phase_2_sleep_time_random;
	unsigned int phase_3_sleep_time_random;
	unsigned int phase_4_sleep_time_random;
	unsigned int phase_REM_sleep_time_random;

	unsigned int j;
	unsigned int k;
	unsigned int night_start_index;
	unsigned int position_in_night = 0;

	for (unsigned int subject = 0; subject < nb_nights; subject++)
	{
		// we will generate pseudo random sleep time respecting
		//different sleep phasis according to articles read on the net eg : 1, 2, 3, 4, REM(5)

		//fall in sleep phases generation
	    // according to articles read on the net we can consider during all the sleep
		// cycles of a night phases time stays slighlty the same So we will make them differ randomly from (-15% to 15%)
	    // for every cycle of this night
		// we will also give a rank between one and 10 to this sleep,
		// the rank is arbitrary
	    // According to articles we read :
	    // 1 then 2-3-4-3-2-REM-sometimes 1-2-3-4-3-2-REM-sometimes 1[...] REM-wake up to finish
	    //the following code has been written naively for understanding purpose
	    // we could have used memset(ptr, value, size) for faster execution and avoid all this awful loops which
	    //brings complexity up to N

		position_in_night = 0;
		night_start_index = subject * sleep_duration;

	    phase_1_sleep_time_random = phase_1_sleep_time + (unsigned int)round((distribution(generator) * phase_1_sleep_time));
		for (j = 0 ; j < phase_1_sleep_time_random; j++)
			nights[j + night_start_index] = 1;

		position_in_night = j;

	while(position_in_night < sleep_duration -1 - (wake_up_cycle_max_duration + cycle_max_duration))
	{
		phase_2_sleep_time_random = phase_2_sleep_time + (unsigned int)round((distribution(generator) * phase_2_sleep_time));
		for (k = position_in_night ; k < phase_2_sleep_time_random + position_in_night; k++)
			nights[k + night_start_index] = 2;



		phase_3_sleep_time_random = phase_3_sleep_time + (unsigned int)round((distribution(generator) * phase_3_sleep_time));
		for (j = k ; j < phase_3_sleep_time_random + k; j++)
			nights[j + night_start_index] = 3;


		phase_4_sleep_time_random = phase_4_sleep_time + (unsigned int)round((distribution(generator) * phase_4_sleep_time));
		for (k = j ; k < phase_4_sleep_time_random + j; k++)
			nights[k + night_start_index] = 4;



		phase_3_sleep_time_random = phase_3_sleep_time + (unsigned int)round((distribution(generator) * phase_3_sleep_time));
		for (j = k ; j < phase_3_sleep_time_random + k; j++)
			nights[j + night_start_index] = 3;



		phase_2_sleep_time_random = phase_2_sleep_time + (unsigned int)round((distribution(generator) * phase_2_sleep_time));
		for (k = j ; k < phase_2_sleep_time_random + j; k++)
			nights[k + night_start_index] = 2;



		phase_REM_sleep_time_random = phase_REM_sleep_time + (unsigned int)round((distribution(generator) * phase_REM_sleep_time));
		for (j = k ; j < phase_REM_sleep_time_random + k; j++)
		  nights[j + night_start_index] = 5;



		position_in_night = j;
		// will we have a small wake up time ? 20% chance
		// if we do phase 1 should happen again !
		if (wake_up(generator) < 20)
		{
			phase_1_sleep_time_random = phase_1_sleep_time + (unsigned int)round((distribution(generator) * phase_1_sleep_time));
			for (k = j ; k < phase_1_sleep_time_random + j; k++)
				nights[k + night_start_index] = 1;
			position_in_night = k;
		}
	}
	// We will admit for demonstration purpose that the wake up phasis is composed of !deepsleep(phase 3/3)
	//so it will be Phase2 then REM to finish.
	phase_2_sleep_time_random = phase_2_sleep_time + (unsigned int)round((distribution(generator) * phase_2_sleep_time));
	for (k = position_in_night ; k < phase_2_sleep_time_random + position_in_night; k++)
		nights[k + night_start_index] = 2;

	phase_REM_sleep_time_random = phase_REM_sleep_time + (unsigned int)round((distribution(generator) * phase_REM_sleep_time));
	for (j = k ; j < phase_REM_sleep_time_random + k; j++)
		nights[j + night_start_index] = 5;

	//give a mark to this night
	nights[subject * sleep_duration + sleep_duration - 1] = (subject) % 10;


	}
return nights;
}

int main(void)
{
	std::default_random_engine generator(time(0));
	std::uniform_real_distribution<> distribution(-0.10, 0.10);
	std::uniform_int_distribution<> wake_up(0,100);
	unsigned char *nights = cpu_gen_nights(20000000);
	printf("night generation finished\n");
	getchar();
	find_perfect_night(nights, 20000000, 531);
	return 0;
}




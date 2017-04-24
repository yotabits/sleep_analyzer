#include "../hardware_limits.hh"
#include <stdio.h>
#include "find_min.cuh"

#define MAX_U8 255
#define TRESHOLD 200
unsigned char *min_tmp_input;
unsigned char *min_output;
void rescale_escape(unsigned char *img, unsigned int nb_pixels, unsigned int color_range);


struct cuComplex {
	float r;
	float i;
	__device__ cuComplex( float a, float b ) : r(a), i(b)  {}

	__device__ float magnitude2(void)
	{
		return r * r + i * i;
	}
	__device__ float abs(void)
	{
		return sqrtf(r * r + i * i);
	}
	__device__ cuComplex operator*(const cuComplex& a)
	{
		return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a)
	{
		return cuComplex(r + a.r, i + a.i);
	}
	__device__ cuComplex operator-(const cuComplex& a)
	{
		return cuComplex(r - a.r, i - a.i);
	}
	__device__ cuComplex neg(void)
	{
		return cuComplex(-r, -i);
	}
};

__device__ float julia(int x, int y, int size_x, float r_value)
{
	const float scale = 1.5 ;
	float jx = scale * (float)( size_x / 2 - x) / (size_x / 2);
	float jy = scale * (float)( size_x / 2 - y) / (size_x / 2);
	cuComplex c(r_value, 0.156);
	cuComplex a(jx, jy);
	float smooth_color = expf(-a.abs());
	int i = 0;
	for (i = 0; i < TRESHOLD; i++)
	{
		a = a *a  + c;
		smooth_color += expf(-a.abs());
		if (a.magnitude2() > 2000)
		{
			//return i;
			return (smooth_color/TRESHOLD);
		}
	}
	//return i;
	return (smooth_color/TRESHOLD);
}

__device__ void hsl_to_rgb(unsigned char *r, unsigned char *g, unsigned char *b, unsigned char escape)
{
	float h = float(escape);
	float s = 1.0f;
	float l = 0.5f;
	float c = l * s;//(1 - fabsf(2 * l - 1)) * s;
	float x = c * (1 - fabsf(fmodf(h / 60, 2) - 1));
	float m = l - c;
	float r_ = 0;
	float g_ = 0;
	float b_ = 0;
	if(h == TRESHOLD)
	{
		*r = 0;
		*g = 0;
		*b = 0;
		return;
	}
	if (h < 60)
	{
		r_ = c;
		g_ = x;
	}
	else if (h < 120)
	{
		r_ = x;
		g_ = c;
	}
	else if (h < 180)
	{
		g_ = c;
		b_ = x;
	}
	else if (h < 240)
	{
		g_ = x;
		b_ = c;
	}
	else if (h < 300)
	{
		r_ = x;
		b_ = c;
	}
	else
	{
		r_ = c;
		b_ = x;
	}
	*r = (r_ + m) * 255;
	*g = (g_ + m) * 255;
	*b = (b_ + m) * 255;
}

__global__ void make_image_k(unsigned char *img, unsigned int nb_pixels, unsigned int size_x, float r_value)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = tid / size_x;
	unsigned int x = tid - (y * size_x);

	if (tid < nb_pixels)
	{
		unsigned char r = 0;
		unsigned char g = 0;
		unsigned char b = 0;
		hsl_to_rgb(&r, &g, &b, 0.95f + 950.f * julia(x, y, size_x, r_value));

		img[tid * 3] = b;
		img[tid * 3 + 1] = g;
		img[tid * 3 + 2] = r;

	}

}

__global__ void make_coloring_k(unsigned char *img, unsigned int nb_pixels)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < nb_pixels)
	{
		unsigned char r = 0;
		unsigned char g = 0;
		unsigned char b = 0;
		hsl_to_rgb(&r, &g, &b, img[tid * 3]);
		img[tid * 3] = b;
		img[tid * 3 + 1] = g;
		img[tid * 3 + 2] = r;
	}
}

__global__ void rescale_escape_k(unsigned char *img, unsigned int nb_pixels, float max, float min, float range)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < nb_pixels)
	{
		img[tid * 3] = (((float)(img[tid * 3]) - min) * range)/ (max - min);
	}
}


void make_image(unsigned char *img, unsigned int nb_pixels, unsigned int size_x, float r_value)
{
	unsigned int threads;
	unsigned int blocks;
	get_threads_blocks(&threads, &blocks, nb_pixels);
	make_image_k<<<blocks, threads>>>(img, nb_pixels, size_x, r_value);
	cudaDeviceSynchronize();
	//rescale_escape(img, nb_pixels, 360);
	//cudaDeviceSynchronize();
	//make_coloring_k<<<blocks, threads>>>(img, nb_pixels);
	//cudaDeviceSynchronize();

	printf("block %i \n", blocks);
	printf("thread %i", threads);
}

void init_gpu_mem(unsigned int nb_pixels)
{
	unsigned int byte_size = nb_pixels * 3 * sizeof(unsigned char);
	cudaMalloc(&min_output, byte_size);
	cudaMalloc(&min_tmp_input, byte_size);
	cudaMemset(min_output, 0, byte_size);
	cudaMemset(min_tmp_input, 0, byte_size);
}

void rescale_escape(unsigned char *img, unsigned int nb_pixels, unsigned int color_range)
{
	unsigned char min = find_min(nb_pixels, img, min_output, min_tmp_input);
	unsigned char max = find_max(nb_pixels, img, min_output, min_tmp_input);
	unsigned int threads;
	unsigned int blocks;
	get_threads_blocks(&threads, &blocks, nb_pixels);
	rescale_escape_k<<<blocks, threads>>>(img, nb_pixels, (float)max, (float)min, color_range);
	printf("gpu_min found = %i \n", min);
	printf("gpu_maxfound = %i \n", max);
}

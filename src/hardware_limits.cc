// Copyrigth (c) 2015
// Mines-Paristech
// 60, boulevard Saint-Michel
// 75271 Paris cedex 06 (FRANCE). All rights reserved.
// This file is a part of GLCU library.
// This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
// WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
//
// Author(s) : T.Kostas, O.Stab 
//

//
// File : hardware_limits.cc
// Object :
// Modification(s) : 
//
/*
 * hardware_limits.cc
 *
 *  Created on: 10 sept. 2015
 *      Author: tkostas
 */




#include "hardware_limits.hh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

unsigned int get_max_threads_1d()
{
  static int max_threads_per_block_1d;

  static bool initialized = false;

  if (!initialized)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    max_threads_per_block_1d = prop.maxThreadsPerBlock;
    initialized = true;
  }

  return max_threads_per_block_1d;
}

unsigned int get_max_threads_2d()
{
  static int max_threads_per_block_2d;
  static bool initialized = false;

  if (!initialized)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    max_threads_per_block_2d = static_cast<unsigned int>(sqrt(prop.maxThreadsPerBlock));
    initialized = true;
  }

  return max_threads_per_block_2d;
}

unsigned int get_max_blocks()
{
  static int max_blocks;
  static bool initialized = false;

  if (!initialized)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    max_blocks = prop.maxGridSize[0];
    initialized = true;
  }

  return max_blocks;
}

void get_threads_blocks(unsigned int *threads, unsigned int *blocks, unsigned int data_size )
{
	*threads = get_max_threads_1d();
	*blocks = (data_size + *threads - 1) / *threads;
}







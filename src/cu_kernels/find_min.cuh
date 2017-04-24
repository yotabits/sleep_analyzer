/*
 * find_min.cuh
 *
 *  Created on: 19 nov. 2015
 *      Author: tkostas
 */

#ifndef FIND_MIN_CUH_
#define FIND_MIN_CUH_

#include "../hardware_limits.hh"
unsigned char find_min(unsigned int pixel_size, unsigned char *input, unsigned char *output, unsigned char *temp_input);
unsigned char find_max(unsigned int pixel_size, unsigned char *input, unsigned char *output, unsigned char *temp_input) ;
// min value is located in output[0]

#endif /* FIND_MIN_CUH_ */

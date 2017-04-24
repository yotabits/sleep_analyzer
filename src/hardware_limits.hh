/*
 * hardware_limits.hh
 *
 *  Created on: 10 sept. 2015
 *      Author: tkostas
 */

#ifndef HARDWARE_LIMITS_HH_
#define HARDWARE_LIMITS_HH_

unsigned int get_max_threads_1d();
unsigned int get_max_threads_2d();
unsigned int get_max_blocks();
void get_threads_blocks(unsigned int *threads,unsigned int *blocks, unsigned int data_size );


#endif /* !HARDWARE_LIMITS_HH */

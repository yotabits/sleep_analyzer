/*
 * tools.cc
 *
 *  Created on: Dec 18, 2016
 *      Author: tkostas
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cuda.h>
using namespace std;

// error handling should be added
void save_night_to_text(char *filename, unsigned int byte_numbers, unsigned char *data, unsigned int night_lenght)
{
	FILE *pFile = fopen(filename,"w+");
	char str[10];
	for (unsigned int i = 0; i < byte_numbers; i++)
	{
		if(data[i] == 5)
			sprintf(str, "0.5 ");
		else
			sprintf(str, "%d ", data[i]);
		fwrite(str, sizeof(unsigned char),strlen(str), pFile);

			if ((i + 1) % (night_lenght) == 0 && i != 0)
			{
				fwrite("\n", sizeof(unsigned char),1, pFile);
			}
	}
    fclose(pFile);
}

void save_night_to_text(char *filename, unsigned int byte_numbers, unsigned int *data, unsigned int night_lenght)
{
	FILE *pFile = fopen(filename,"w+");
	char str[30];
	for (unsigned int i = 0; i < byte_numbers; i++)
	{
		sprintf(str, "%d,", data[i]);
		fwrite(str, sizeof(char), strlen(str), pFile);

			if ((i + 1) % (night_lenght) == 0 && i != 0)
			{
				fwrite("\n", sizeof(unsigned char),1, pFile);
			}
	}
    fclose(pFile);
}


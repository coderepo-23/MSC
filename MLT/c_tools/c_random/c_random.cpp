/**
 * generate random number
 * dll:
 * g++ -shared -o c_random.dll -fPIC c_random.cpp
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <memory.h>
#include <time.h>
#include "c_random.h"

void c_srand(int seed)
{
    if (seed < 0)
    {
        time_t tt1;
        time(&tt1);
        srand((unsigned)tt1);
    }
    else
    {
        srand(seed);
    }
}

int c_rand(void)
{
    return rand();
}

#define drand48() (rand()*1.0/RAND_MAX)

int main(int argc,char** argv)
{
    // if (argc < 2)
    // {
    //     exit(1);
    // }

    // int seed = atoi(argv[1]);
    // time_t tt1;

    // printf("%d\n", seed);

    // if (seed == -1)
    // {
    //     time(&tt1);
    //     srand((unsigned)tt1);
    // }
    // else
    // {
    //     srand(seed);
    // }

    srand(100);
    // printf("%d\n", rand());

    printf("%f\n", drand48());

    return 0;
}
/**
 * float calculation
 * dll:
 * g++ -shared -o c_float.dll -fPIC c_float.cpp
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <memory.h>
#include <time.h>
#include "c_float.h"

float c_pow(double a, double b)
{
    return pow(a, b);
}

double c_div(double a, double b)
{
    return a / b;
}

unsigned short c_divUshort(double a, double b)
{
    return (unsigned short)(a / b);
}

double c_mul(double a, double b)
{
    return a * b;
}

int main(int argc,char** argv)
{

    return 0;
}
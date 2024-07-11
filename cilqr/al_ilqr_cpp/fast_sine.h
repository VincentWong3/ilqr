#ifndef FAST_SINE_H_
#define FAST_SINE_H_


#define TABLE_SIZE 16385
#define TWO_PI 6.283185307179586476925286766559
#define PI 3.141592653589793238462643383279
#define HALF_PI 1.5707963267948966192313216916398


double fast_sin(double angle_rad);
double fast_cos(double angle_rad);
double fast_tan(double angle_rad);

extern const float SIN_TABLE[TABLE_SIZE];


#endif
#include <cuda_runtime.h>
#include "DeviceUtilities.h"

__device__ ll device_modpow(ll base, ll exp, ll modulus)
{
	base %= modulus;
	ll result = 1;
	while (exp > 0)
	{
		if (exp & 1)
		{
			result = device_mulmod(result, base, modulus);
		}
		base = device_mulmod(base, base, modulus);
		exp >>= 1;
	}
	return result;
}

__device__ ll device_mulmod(ll a, ll b, ll m)
{
	ll res = 0;
	ll temp_b;

	/* Only needed if b may be >= m */
	if (b >= m)
	{
		if (m > ULLONG_MAX / 2u)
		{
			b -= m;
		}
		else
		{
			b %= m;
		}
	}

	while (a != 0)
	{
		if (a & 1)
		{ 
			/* Add b to res, modulo m, without overflow */
			if (b >= m - res) /* Equiv to if (res + b >= m), without overflow */
			{
				res -= m;
			}
			res += b;
		}
		a >>= 1;

		/* Double b, modulo m */
		temp_b = b;
		if (b >= m - b)       /* Equiv to if (2 * b >= m), without overflow */
		{
			temp_b -= m;
		}
		b += temp_b;
	}
	return res;
}

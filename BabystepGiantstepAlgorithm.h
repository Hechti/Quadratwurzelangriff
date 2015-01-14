#ifndef BABYSTEP_GIANTSTEP_ALGORITHM_H_
#define BABYSTEP_GIANTSTEP_ALGORITHM_H_

void babystepGiantstepAlgorithm(const InfInt& n, const InfInt& g, const InfInt& a, InfInt &result);
void babystepGiantstepAlgorithmCUDA(const InfInt& n, const InfInt& g, const InfInt& a, InfInt &result);
__global__ void babyStep(const InfInt *n, const InfInt *m, const InfInt *g, InfInt *mapBabyStep);

#endif

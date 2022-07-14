//******************************************
// operators.f90
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#ifndef OPERATORS_H
#define OPERATORS_H

#include "data.h"

namespace operators
{

void diffusion(data::Field &u, data::Field &s);
void exchange(data::Field const& U);
void exchange_rdma(data::Field const& U);

} // namespace operators

#endif // OPERATORS_H

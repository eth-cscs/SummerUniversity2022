//******************************************
// operators
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
//
// implements
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#include <mpi.h>

#include "cuda_helpers.h"
#include "data.h"
#include "operators.h"
#include "stats.h"

namespace operators {

// POD type holding information for device
struct DiffusionParams {
    int nx;
    int ny;
    double alpha;
    double dxs;
    double *x_old;
    double *bndN;
    double *bndE;
    double *bndS;
    double *bndW;
};

// This function will copy a 1D strip from a 2D field to a 1D buffer.
// It is used to copy the values from along the edge of a field to
// a flat buffer for sending to MPI neighbors.
void pack_buffer(data::Field const& from, data::Field &buffer, int startx, int starty, int stride) {
    int nx = from.xdim();
    int ny = from.ydim();
    int pos = startx + starty*nx;
    auto status = cublasDcopy(
        cublas_handle(), buffer.length(),
        from.device_data() + pos, stride,
        buffer.device_data(),    1
    );
    if(status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "error : cublas copy for boundary condition" << std::endl;
        exit(-1);
    }
}

// Exchange that performs MPI send/recv from/to host memory, and copies
// results from and to the GPU.
void exchange_rdma(data::Field const& U) {
    using data::domain;

    using data::bndE;
    using data::bndW;
    using data::bndN;
    using data::bndS;

    using data::buffE;
    using data::buffW;
    using data::buffN;
    using data::buffS;

    MPI_Status statuses[8];
    int requests[8];
    int num_requests = 0;

    int nx = domain.nx;
    int ny = domain.ny;

    // TODO: Make sure that device has completed its work before communicating
    if(domain.neighbour_north>=0) {
        MPI_Irecv(bndN.device_data(), nx, MPI_DOUBLE, domain.neighbour_north, domain.neighbour_north,
            domain.comm_cart, requests+num_requests);
        num_requests++;

        pack_buffer(U, buffN, 0, ny-1, 1);

        MPI_Isend(buffN.device_data(), nx, MPI_DOUBLE, domain.neighbour_north, domain.rank,
            domain.comm_cart, requests+num_requests);
        num_requests++;
    }
    if(domain.neighbour_south>=0) {
        MPI_Irecv(bndS.device_data(), nx, MPI_DOUBLE, domain.neighbour_south, domain.neighbour_south,
            domain.comm_cart, requests+num_requests);
        num_requests++;

        pack_buffer(U, buffS, 0, 0, 1);

        MPI_Isend(buffS.device_data(), nx, MPI_DOUBLE, domain.neighbour_south, domain.rank,
            domain.comm_cart, requests+num_requests);
        num_requests++;
    }
    if(domain.neighbour_east>=0) {
        MPI_Irecv(bndE.device_data(), ny, MPI_DOUBLE, domain.neighbour_east, domain.neighbour_east,
            domain.comm_cart, requests+num_requests);
        num_requests++;

        pack_buffer(U, buffE, nx-1, 0, nx);

        MPI_Isend(buffE.device_data(), ny, MPI_DOUBLE, domain.neighbour_east, domain.rank,
            domain.comm_cart, requests+num_requests);
        num_requests++;
    }
    if(domain.neighbour_west>=0) {
        MPI_Irecv(bndW.device_data(), ny, MPI_DOUBLE, domain.neighbour_west, domain.neighbour_west,
            domain.comm_cart, requests+num_requests);
        num_requests++;

        pack_buffer(U, buffW, 0, 0, nx);

        MPI_Isend(buffW.device_data(), ny, MPI_DOUBLE, domain.neighbour_west, domain.rank,
            domain.comm_cart, requests+num_requests);
        num_requests++;
    }

    MPI_Waitall(num_requests, requests, statuses);
}

// Exchange that performs MPI send/recv from/to host memory, and copies
// results from and to the GPU.
void exchange(data::Field const& U) {
    using data::domain;

    using data::bndE;
    using data::bndW;
    using data::bndN;
    using data::bndS;

    using data::buffE;
    using data::buffW;
    using data::buffN;
    using data::buffS;

    MPI_Status statuses[8];
    int requests[8];
    int num_requests = 0;

    int nx = domain.nx;
    int ny = domain.ny;

    // TODO: Make sure that device has completed its work before communicating
    if(domain.neighbour_north>=0) {
        MPI_Irecv(bndN.host_data(), nx, MPI_DOUBLE, domain.neighbour_north, domain.neighbour_north,
            domain.comm_cart, requests+num_requests);
        num_requests++;

        pack_buffer(U, buffN, 0, ny-1, 1);
        buffN.update_host();

        MPI_Isend(buffN.host_data(), nx, MPI_DOUBLE, domain.neighbour_north, domain.rank,
            domain.comm_cart, requests+num_requests);
        num_requests++;
    }
    if(domain.neighbour_south>=0) {
        MPI_Irecv(bndS.host_data(), nx, MPI_DOUBLE, domain.neighbour_south, domain.neighbour_south,
            domain.comm_cart, requests+num_requests);
        num_requests++;

        pack_buffer(U, buffS, 0, 0, 1);
        buffS.update_host();

        MPI_Isend(buffS.host_data(), nx, MPI_DOUBLE, domain.neighbour_south, domain.rank,
            domain.comm_cart, requests+num_requests);
        num_requests++;
    }
    if(domain.neighbour_east>=0) {
        MPI_Irecv(bndE.host_data(), ny, MPI_DOUBLE, domain.neighbour_east, domain.neighbour_east,
            domain.comm_cart, requests+num_requests);
        num_requests++;

        pack_buffer(U, buffE, nx-1, 0, nx);
        buffE.update_host();

        MPI_Isend(buffE.host_data(), ny, MPI_DOUBLE, domain.neighbour_east, domain.rank,
            domain.comm_cart, requests+num_requests);
        num_requests++;
    }
    if(domain.neighbour_west>=0) {
        MPI_Irecv(bndW.host_data(), ny, MPI_DOUBLE, domain.neighbour_west, domain.neighbour_west,
            domain.comm_cart, requests+num_requests);
        num_requests++;

        pack_buffer(U, buffW, 0, 0, nx);
        buffW.update_host();

        MPI_Isend(buffW.host_data(), ny, MPI_DOUBLE, domain.neighbour_west, domain.rank,
            domain.comm_cart, requests+num_requests);
        num_requests++;
    }

    MPI_Waitall(num_requests, requests, statuses);
    if(domain.neighbour_north>=0) {
        bndN.update_device();
    }
    if(domain.neighbour_south>=0) {
        bndS.update_device();
    }
    if(domain.neighbour_east>=0) {
        bndE.update_device();
    }
    if(domain.neighbour_west>=0) {
        bndW.update_device();
    }
}

// const qualifier issues a "cannot determine bounds for array" error in PGI
// void diffusion(const data::Field &U, data::Field &S)
void diffusion(data::Field &U, data::Field &S)
{
    using data::options;
    using data::domain;

    using data::bndE;
    using data::bndW;
    using data::bndN;
    using data::bndS;

    using data::buffE;
    using data::buffW;
    using data::buffN;
    using data::buffS;

    using data::x_old;

    double dxs = 1000. * (options.dx * options.dx);
    double alpha = options.alpha;
    int nx = domain.nx;
    int ny = domain.ny;
    int iend  = nx - 1;
    int jend  = ny - 1;

    exchange(U);

    // TODO: Offload the following loops to the GPU
    for (int j=1; j < jend; j++) {
        for (int i=1; i < iend; i++) {
            S(i,j) = -(4. + alpha) * U(i,j)               // central point
                                    + U(i-1,j) + U(i+1,j) // east and west
                                    + U(i,j-1) + U(i,j+1) // north and south
                                    + alpha * x_old(i,j)
                                    + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // the east boundary
    {
        int i = nx - 1;
        for (int j = 1; j < jend; j++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i,j-1) + U(i,j+1)
                        + alpha*x_old(i,j) + bndE[j]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // the west boundary
    {
        int i = 0;
        for (int j = 1; j < jend; j++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i+1,j) + U(i,j-1) + U(i,j+1)
                        + alpha * x_old(i,j) + bndW[j]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // the north boundary (plus NE and NW corners)
    {
        int j = ny - 1;

        {
            int i = 0; // NW corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i+1,j) + U(i,j-1)
                        + alpha * x_old(i,j) + bndW[j] + bndN[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        // north boundary
        for (int i = 1; i < iend; i++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i+1,j) + U(i,j-1)
                        + alpha*x_old(i,j) + bndN[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        {
            int i = nx-1; // NE corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i,j-1)
                        + alpha * x_old(i,j) + bndE[j] + bndN[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // the south boundary
    {
        int j = 0;

        {
            int i = 0; // SW corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i+1,j) + U(i,j+1)
                        + alpha * x_old(i,j) + bndW[j] + bndS[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        // south boundary
        for (int i = 1; i < iend; i++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i+1,j) + U(i,j+1)
                        + alpha * x_old(i,j) + bndS[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        {
            int i = nx - 1; // SE corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i,j+1)
                        + alpha * x_old(i,j) + bndE[j] + bndS[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }
}

} // namespace operators

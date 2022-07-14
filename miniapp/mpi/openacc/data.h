#pragma once

#include <iostream>
#include <cassert>
#include <openacc.h>

namespace data
{

// define some helper types that can be used to pass simulation
// data around without haveing to pass individual parameters
struct Discretization
{
    int nx;       // x dimension
    int ny;       // y dimension
    int nt;       // number of time steps
    double dt;    // time step size
    double dx;    // distance between grid points
    double alpha; // dx^2/(D*dt)
};

struct SubDomain
{
    // initialize a subdomain
    void init(int, int, Discretization&);

    // print subdomain information
    void print();

    // i and j dimensions of the global decomposition
    int ndomx;
    int ndomy;

    // the i and j index of this sub-domain
    int domx;
    int domy;

    // the i and j bounding box of this sub-domain
    int startx;
    int starty;
    int endx;
    int endy;

    // the rank of neighbouring domains
    int neighbour_north;
    int neighbour_east;
    int neighbour_south;
    int neighbour_west;

    // mpi info
    int size;
    int rank;
    MPI_Comm comm_cart;

    // x and y dimension in grid points of the sub-domain
    int nx;
    int ny;

    // total number of grid points
    int N;
};

// thin wrapper around a pointer that can be accessed as either a 2D or 1D array
// Field has dimension xdim * ydim in 2D, or length=xdim*ydim in 1D
class Field {
    public:
    // default constructor
    Field()
    :   xdim_(0),
        ydim_(0),
        ptr_(nullptr)
    {};

    // constructor
    Field(int xdim, int ydim)
    :   xdim_(xdim),
        ydim_(ydim),
        ptr_(nullptr)
    {
        init(xdim, ydim);
    };

    // destructor
    ~Field() {
        free();
    }

    void init(int xdim, int ydim) {
        #ifdef DEBUG
        assert(xdim>0 && ydim>0);
        #endif

        free();
        allocate(xdim, ydim);
        fill(0.);
    }

    double*       host_data()         { return ptr_; }
    const double* host_data()   const { return ptr_; }

    double*       device_data()       { return (double *) acc_deviceptr(ptr_); }
    const double* device_data() const { return (double *) acc_deviceptr(ptr_); }

    // access via (i,j) pair
    // TODO: This will be called from the device
    inline double&       operator() (int i, int j)        {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_ && j>=0 && j<ydim_);
        #endif
        return ptr_[i+j*xdim_];
    }
    // TODO: This will be called from the device
    inline double const& operator() (int i, int j) const  {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_ && j>=0 && j<ydim_);
        #endif
        return ptr_[i+j*xdim_];
    }

    // access as a 1D field
    // TODO: This will be called from the device
    inline double      & operator[] (int i) {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_*ydim_);
        #endif
        return ptr_[i];
    }
    // TODO: This will be called from the device
    inline double const& operator[] (int i) const {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_*ydim_);
        #endif
        return ptr_[i];
    }

    int xdim()   const { return xdim_; }
    int ydim()   const { return ydim_; }
    int length() const { return xdim_*ydim_; }

    /////////////////////////////////////////////////
    // helpers for coordinating host-device transfers
    /////////////////////////////////////////////////
    void update_host() {
        // We want host updates to be synchronous, so that the data is really
        // updated after we exit this function
        // TODO: update the host copy of ptr_
    }

    void update_device() {
        // TODO: update the device copy of ptr_
    }

    private:

    void allocate(int xdim, int ydim) {
        xdim_ = xdim;
        ydim_ = ydim;
        ptr_ = new double[xdim*ydim];
        // TODO: Copy the whole object to the device
        //       Pay attention to the copy order
    }

    // set to a constant value
    void fill(double val) {
        // initialize the host and device copy at the same time
        // TODO: Offload this loop to the GPU
        for(int i=0; i<xdim_*ydim_; ++i)
            ptr_[i] = val;

        #pragma omp parallel for
        for(int i=0; i<xdim_*ydim_; ++i)
            ptr_[i] = val;
    }

    void free() {
        if (ptr_) {
            // TODO: Delete this object's copy from the GPU
            delete[] ptr_;
        }

        ptr_ = nullptr;
    }

    double* ptr_;
    int xdim_;
    int ydim_;
};

// fields that hold the solution
extern Field x_new; // 2d
extern Field x_old; // 2d

// fields that hold the boundary values
extern Field bndN; // 1d
extern Field bndE; // 1d
extern Field bndS; // 1d
extern Field bndW; // 1d

// buffers used in boundary exchange
extern Field buffN;
extern Field buffE;
extern Field buffS;
extern Field buffW;

extern Discretization options;
extern SubDomain      domain;

} // namespace data

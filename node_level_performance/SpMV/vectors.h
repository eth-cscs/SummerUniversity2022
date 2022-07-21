#pragma once

#include "spmv.h"

#include <complex>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <sstream>
#include <type_traits>
#include <vector>

// TODO: rename float_type to value_type

template<typename VT, typename IT>
class BaseVector
{
public:

    BaseVector() = default;

    explicit BaseVector(const IT n_rows)
    : n_rows(n_rows)
    {}

    BaseVector(const BaseVector &) = delete;
    BaseVector(BaseVector && rhs)
        : n_rows(rhs.n_rows), values_(rhs.values_)
    {
        rhs.n_rows  = -1;
        rhs.values_ = nullptr;
    }

    BaseVector & operator=(const BaseVector &) = delete;
    BaseVector & operator=(BaseVector && rhs)
    {
        n_rows       = rhs.n_rows;
        values_      = rhs.values_;

        rhs.n_rows  = -1;
        rhs.values_ = nullptr;

        return *this;
    }

    virtual ~BaseVector() = default;

    IT n_rows{};

    VT * data()             { return values_; }
    const VT * data() const { return values_; }

protected:
    void data(VT * values) { values_ = values; }

private:

    VT * values_{};
};


template<typename VT, typename IT>
class Vector : public BaseVector<VT, IT>
{
public:

    Vector() = default;

    explicit Vector(const IT n_rows)
    : BaseVector<VT, IT>(n_rows)
    {
        this->data(allocate(this->n_rows));
    }

    Vector(const Vector &) = delete;
    Vector(Vector && rhs) = default;

    Vector & operator=(const Vector &) = delete;
    Vector & operator=(Vector && rhs)
    {
        if (this->data())
            free(this->data());

        BaseVector<VT, IT>::operator=(std::move(rhs));

        return *this;
    }

    virtual ~Vector()
    {
        if (this->data())
            free(this->data());
    }

    inline       VT & operator()(IT r)       { return this->data()[r]; }
    inline const VT & operator()(IT r) const { return this->data()[r]; }

    inline       VT & operator[](IT r)       { return this->data()[r]; }
    inline const VT & operator[](IT r) const { return this->data()[r]; }


    operator std::vector<VT> () const
    {
        std::vector<VT> lhs(this->n_rows);

        for (ST i = 0; i < this->n_rows; ++i) {
            lhs[i] = (*this)[i];
        }

        return lhs;
    }

private:

    static VT*
    allocate(IT n_els)
    {
        VT * memory{};

        size_t n_bytes_to_alloc = sizeof(VT) * n_els;

        int err = posix_memalign(
                    (void **)&memory,
                    DEFAULT_ALIGNMENT,
                    n_bytes_to_alloc);

        if (err != 0) {
            fprintf(stderr,
                    "ERROR: posix_memalign(%lu b) failed: %d - %s\n",
                    n_bytes_to_alloc, err, strerror(err));
            exit(EXIT_FAILURE);
        }

        return memory;
    }
};

template<typename VT, typename IT>
class VectorGpu : public BaseVector<VT, IT>
{
public:

    explicit VectorGpu(const Vector<VT, IT> & dv)
    : BaseVector<VT, IT>(dv.n_rows)
    {
#ifdef __NVCC__
        size_t n_bytes_to_alloc = sizeof(VT) * this->n_rows;

        double t_start = get_time();

        VT * memory;
        assert_gpu(cudaMalloc(&memory, n_bytes_to_alloc));
        this->data(memory);

        assert_gpu(cudaMemcpy(this->data(), dv.data(), n_bytes_to_alloc, cudaMemcpyHostToDevice));

        double duration = get_time() - t_start;

        log("VectorGpu: copy host -> device:  %9e MB   %9e MB/s   %e s\n",
            n_bytes_to_alloc / 1e6, n_bytes_to_alloc / 1e6 / duration, duration);
#endif
    }

    VectorGpu(const VectorGpu &) = delete;
    VectorGpu(VectorGpu && rhs) = default;

    VectorGpu & operator=(const VectorGpu &) = delete;
    VectorGpu & operator=(VectorGpu && rhs)
    {
        if (this->data())
            assert_gpu(cudaFree(this->data()));

        BaseVector<VT, IT>::operator=(std::move(rhs));

        return *this;
    }

    ~VectorGpu()
    {
        if (this->data())
            assert_gpu(cudaFree(this->data()));
    }

    Vector<VT, IT>
    copy_from_device()
    {
        Vector<VT, IT> dv(this->n_rows);
#ifdef __NVCC__
        double t_start = get_time();

        assert_gpu(cudaMemcpy(dv.data(),
                              this->data(),
                              sizeof(VT) * this->n_rows,
                              cudaMemcpyDeviceToHost));

        double duration = get_time() - t_start;

        size_t n_bytes_to_alloc = sizeof(VT) * this->n_rows;
        log("VectorGpu: copy device -> host:  %9e MB   %9e MB/s   %e s\n",
            n_bytes_to_alloc / 1e6, n_bytes_to_alloc / 1e6 / duration, duration);
#endif
        return dv;
    }
};


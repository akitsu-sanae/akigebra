/*============================================================================
  Copyright (C) 2016 akitsu sanae
  https://github.com/akitsu-sanae/akigebra
  Distributed under the Boost Software License, Version 1.0. (See accompanying
  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
============================================================================*/

#ifndef AKIGEBRA_MATRIX_HPP
#define AKIGEBRA_MATRIX_HPP

#include <cfloat>
#include <cstddef>
#include <cmath>
#include <array>
#include <tuple>
#include <complex>
#include <iostream>

namespace akigebra {

template<typename, std::size_t, std::size_t H> struct matrix;

template<typename T, std::size_t W, std::size_t H>
matrix<T, W, H> conj(matrix<T, W, H> const&);

template<typename T, std::size_t W, std::size_t H>
struct matrix {
    using value_type = T;
    static constexpr std::size_t Width = W;
    static constexpr std::size_t Height = H;
    using this_type = matrix<value_type, Width, Height>;

    struct invalid_index_exception {};

    value_type const& at(int x, int y) const {
        if (x < 0 || x > Width)
            throw invalid_index_exception{};
        if (y < 0 || y > Height)
            throw invalid_index_exception{};
        return elems[y][x];
    }
    value_type& at(int x, int y) {
        if (x < 0 || x > Width)
            throw invalid_index_exception{};
        if (y < 0 || y > Height)
            throw invalid_index_exception{};

        return elems[y][x];
    }

    this_type operator+(this_type const& rhs) const {
        this_type result = *this;
        for (int y=0; y<Height; y++) {
            for (int x=0; x<Width; x++) {
                result.at(x, y) += rhs.at(x, y);
            }
        }
        return result;
    }

    template<std::size_t I>
    matrix<value_type, I, Height> operator*(matrix<value_type, I, Width> const& rhs) const {
        matrix<value_type, I, Height> result = {};
        for (int y=0; y<Height; y++) {
            for (int x=0; x<I; x++) {
                for (int i=0; i<Width; i++)
                    result.at(x, y) += this->at(i, y) * rhs.at(x, i);
            }
        }
        return result;
    }
    matrix<value_type, Width, Height> operator*(value_type const& rhs) const {
        matrix<value_type, Width, Height> result = *this;
        for (int y=0; y<Height; y++) {
            for (int x=0; x<Width; x++) {
                result.at(x, y) *= rhs;
            }
        }
        return result;
    }
    matrix<value_type, Width, Height> operator/(value_type const& rhs) const {
        matrix<value_type, Width, Height> result = *this;
        for (int y=0; y<Height; y++) {
            for (int x=0; x<Width; x++) {
                result.at(x, y) /= rhs;
            }
        }
        return result;
    }

    template<typename F>
    this_type map(F&& f) const {
        this_type result = *this;
        for (int y=0; y<Height; y++) {
            for (int x=0; x<Width; x++)
                result.at(x, y) = f(at(x, y));
        }
        return result;
    }

    matrix<value_type, Height, Width> transpose() const {
        matrix<value_type, Height, Width> result;
        for (int y=0; y<Height; y++) {
            for (int x=0; x<Width; x++) {
                result.at(y, x) = this->at(x, y);
            }
        }
        return result;
    }

    this_type adjoint() const {
        return conj(*this).transpose();
    }

    struct not_squared_exception {};
    struct not_symmetry_exception {};
    struct not_orthogonal_exception {};

    static bool is_squared() {
        return Width == Height;
    }
    bool is_symmetry() const {
        for (int y=0; y<Height; y++) {
            for (int x=y+1; x<Width; x++) {
                if (at(x, y) != at(y, x))
                    return false;
            }
        }
        return true;
    }
    bool is_orthogonal() const {
        // A * (A^t) == I
        auto tmp = (*this) * this->transpose();
        return tmp == matrix<value_type, Width, Height>::identity();
    }
    bool is_hermitian() const {
        return *this == this->adjoint();
    }
    bool is_unitary() const {
        return this->adjoint() * *this == this_type::identity();
    }
    bool is_regular() const {
        return *this * this->adjoint() == this->adjoint() * *this;
    }

    static matrix<value_type, Width, Height> identity() {
        if (!is_squared())
            throw not_squared_exception{};
        matrix<value_type, Width, Height> result = {{}};
        for (int i=0; i<Width; i++)
            result.at(i, i) = static_cast<value_type>(1.0);
        return result;
    }

    matrix<value_type, Width-1, Height-1> child(int x, int y) const {
        if (!is_squared())
            throw not_squared_exception{};
        matrix<value_type, Width-1, Height-1> result;
        int ty = 0;
        for (int y_=0; y_<Height; y_++) {
            if (y_ == y)
                continue;
            int tx = 0;
            for (int x_=0; x_<Width; x_++) {
                if (x_ == x)
                    continue;
                result.at(tx, ty) = at(x_, y_);
                tx++;
            }
            ty++;
        }
        return result;
    }

    value_type determinant() const;
    value_type trace() const {
        if (!is_squared())
            throw not_squared_exception{};
        auto result = static_cast<value_type>(0);
        for (int i=0; i<Width; i++)
            result += at(i, i);
        return result;
    }

    std::tuple<this_type, this_type> lu_decompose() const {
        auto result = lu_decompose_impl(0, *this);
        this_type L = this_type::identity();
        this_type U = {{}};
        for (int i=0; i<Width; i++) {
            for (int j=0; j<Height; j++) {
                if (i < j)
                    L.at(i, j) = result.at(i, j);
                else
                    U.at(i, j) = result.at(i, j);
            }
        }
        return std::make_tuple(L, U);
    }

    this_type lu_decompose_impl(int n, this_type buf) const {
        if (!is_squared())
            throw not_squared_exception{};
        if (n == Width)
            return buf;
        for (int j=n+1; j<Height; j++)
            buf.at(n, j) /= buf.at(n, n);
        for (int i=n+1; i<Width; i++) {
            for (int j=n+1; j<Height; j++)
                buf.at(i, j) -= buf.at(n, j)*buf.at(j, n);
        }
        return lu_decompose_impl(n+1, buf);
    }

    value_type elems[Height][Width];
};

namespace detail {

template<typename T, std::size_t W, std::size_t H>
struct determinant_impl {
    matrix<T, W, H> const& data;
    using value_type = typename matrix<T, W, H>::value_type;
    value_type calc() const {
        if (!data.is_squared())
            throw typename matrix<T, W, H>::not_squared_exception{};
        auto result = static_cast<value_type>(0);
        int k = 1;
        for (int i=0; i<W; i++) {
            result += k*data.at(0, i) * data.child(0, i).determinant();
            k *= -1;
        }
        return result;
    }
};

template<typename T>
struct determinant_impl<T, 1, 1> {
    matrix<T, 1, 1> const& data;
    T calc() const {
        return data.at(0, 0);
    }
};

}

template<typename T, std::size_t W, std::size_t H>
inline T matrix<T, W, H>::determinant() const {
    return detail::determinant_impl<T, W, H>{*this}.calc();
}

template<typename T, std::size_t W, std::size_t H>
inline matrix<T, W, H>
conj(matrix<T, W, H> const& input) {
    return input.map([](T const& e) {
            return std::conj(e);
            });
}

template<typename T, std::size_t W, std::size_t H>
inline static bool operator==(matrix<T, W, H> const& lhs, matrix<T, W, H> const& rhs) {
    for (std::size_t y=0; y<H; y++) {
        for (std::size_t x=0; x<W; x++) {
            if (std::abs(lhs.at(x, y) - rhs.at(x, y)) > DBL_EPSILON) {
                return false;
            }
        }
    }
    return true;
}

template<typename T, std::size_t W, std::size_t H>
inline static std::ostream& operator<<(std::ostream& os, matrix<T, W, H> const& m) {
    os << W << " x " << H << " matrix" << std::endl;
    for (std::size_t y=0; y<H; y++) {
        std::cout << "    ";
        for (std::size_t x=0; x<W; x++) {
            std::cout << m.at(x, y) << ", ";
        }
        std::cout << std::endl;
    }
    return os;
}

}

#endif


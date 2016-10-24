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
#include <vector>
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

    value_type const& at(std::size_t x, std::size_t y) const {
        if (x < 0 || x > Width)
            throw invalid_index_exception{};
        if (y < 0 || y > Height)
            throw invalid_index_exception{};
        return elems[y][x];
    }
    value_type& at(std::size_t x, std::size_t y) {
        if (x < 0 || x > Width)
            throw invalid_index_exception{};
        if (y < 0 || y > Height)
            throw invalid_index_exception{};

        return elems[y][x];
    }

    this_type operator+(this_type const& rhs) const {
        this_type result = *this;
        for (std::size_t y=0; y<Height; y++) {
            for (std::size_t x=0; x<Width; x++) {
                result.at(x, y) += rhs.at(x, y);
            }
        }
        return result;
    }
    this_type operator-(this_type const& rhs) const {
        this_type result = *this;
        for (std::size_t y=0; y<Height; y++) {
            for (std::size_t x=0; x<Width; x++) {
                result.at(x, y) -= rhs.at(x, y);
            }
        }
        return result;
    }

    template<std::size_t I>
    matrix<value_type, I, Height> operator*(matrix<value_type, I, Width> const& rhs) const {
        matrix<value_type, I, Height> result = {};
        for (std::size_t y=0; y<Height; y++) {
            for (std::size_t x=0; x<I; x++) {
                for (std::size_t i=0; i<Width; i++)
                    result.at(x, y) += this->at(i, y) * rhs.at(x, i);
            }
        }
        return result;
    }
    matrix<value_type, Width, Height> operator*(value_type const& rhs) const {
        matrix<value_type, Width, Height> result = *this;
        for (std::size_t y=0; y<Height; y++) {
            for (std::size_t x=0; x<Width; x++) {
                result.at(x, y) *= rhs;
            }
        }
        return result;
    }
    matrix<value_type, Width, Height> operator/(value_type const& rhs) const {
        matrix<value_type, Width, Height> result = *this;
        for (std::size_t y=0; y<Height; y++) {
            for (std::size_t x=0; x<Width; x++) {
                result.at(x, y) /= rhs;
            }
        }
        return result;
    }

    this_type& operator=(this_type const& rhs) {
        for (std::size_t x=0; x<Width; x++) {
            for (std::size_t y=0; y<Height; y++)
                at(x, y) = rhs.at(x, y);
        }
        return *this;
    }

    this_type& operator+=(this_type const& rhs) {
        *this = *this + rhs;
        return *this;
    }
    this_type& operator-=(this_type const& rhs) {
        *this = *this - rhs;
        return *this;
    }
    this_type& operator*=(value_type const& rhs) {
        *this = *this * rhs;
        return *this;
    }
    this_type& operator/=(value_type const& rhs) {
        *this = *this / rhs;
        return *this;
    }

    template<typename F>
    this_type map(F&& f) const {
        this_type result = *this;
        for (std::size_t y=0; y<Height; y++) {
            for (std::size_t x=0; x<Width; x++)
                result.at(x, y) = f(at(x, y));
        }
        return result;
    }

    matrix<value_type, Height, Width> transpose() const {
        matrix<value_type, Height, Width> result;
        for (std::size_t y=0; y<Height; y++) {
            for (std::size_t x=0; x<Width; x++) {
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
    struct not_regular_exception {};

    static bool is_squared() {
        return Width == Height;
    }
    bool is_symmetry() const {
        for (std::size_t y=0; y<Height; y++) {
            for (std::size_t x=y+1; x<Width; x++) {
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

    // (A^*) * A == A * (A^*)
    bool is_normal() const {
        return *this * this->adjoint() == this->adjoint() * *this;
    }
    bool is_regular() const {
        if (!is_squared())
            return false;
        if (determinant() == static_cast<value_type>(0))
            return false;
        return true;
    }
    bool is_upper_triangular() const {
        for (std::size_t i=0; i<Width; i++) {
            for (std::size_t j=i+1; j<Height; j++) {
                if (at(i, j) != static_cast<value_type>(0))
                    return false;
            }
        }
        return true;
    }
    bool is_lower_triangular() const {
        for (std::size_t i=0; i<Width; i++) {
            for (std::size_t j=0; j<i; j++) {
                if (at(i, j) != static_cast<value_type>(0))
                    return false;
            }
        }
        return true;
    }
    bool is_triangular() const {
        return is_upper_triangular() || is_lower_triangular();
    }
    bool is_diagonal() const {
        if (!is_squared())
            throw not_squared_exception{};
        for (std::size_t x=0; x<Width; x++) {
            for (std::size_t y=0; y<Height; y++) {
                if (x == y)
                    continue;
                if (at(x, y) != static_cast<value_type>(0))
                    return false;
            }
        }
        return true;
    }

    static matrix<value_type, Width, Height> identity() {
        if (!is_squared())
            throw not_squared_exception{};
        matrix<value_type, Width, Height> result = {{}};
        for (std::size_t i=0; i<Width; i++)
            result.at(i, i) = static_cast<value_type>(1.0);
        return result;
    }

    value_type minor_det(std::size_t x, std::size_t y) const {
        return child(x, y).determinant();
    }
    value_type cofactor(std::size_t x, std::size_t y) const {
        if ((x+y)%2 ==0)
            return minor_det(x, y);
        else
            return -minor_det(x, y);
    }

    matrix<value_type, Width-1, Height-1> child(std::size_t x, std::size_t y) const {
        if (!is_squared())
            throw not_squared_exception{};
        matrix<value_type, Width-1, Height-1> result;
        std::size_t ty = 0;
        for (std::size_t y_=0; y_<Height; y_++) {
            if (y_ == y)
                continue;
            std::size_t tx = 0;
            for (std::size_t x_=0; x_<Width; x_++) {
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
        for (std::size_t i=0; i<Width; i++)
            result += at(i, i);
        return result;
    }

    this_type inverse() const {
        if (!is_regular())
            throw not_regular_exception{};
        this_type result = {};
        auto det = determinant();
        for (std::size_t x=0; x<Width; x++) {
            for (std::size_t y=0; y<Height; y++) {
                result.at(x, y) = cofactor(y, x) / det;
            }
        }
        return result;
    }

    std::tuple<this_type, this_type> lu_decompose() const {
        auto result = lu_decompose_impl(0, *this);
        this_type L = this_type::identity();
        this_type U = {{}};
        for (std::size_t i=0; i<Width; i++) {
            for (std::size_t j=0; j<Height; j++) {
                if (i < j)
                    L.at(i, j) = result.at(i, j);
                else
                    U.at(i, j) = result.at(i, j);
            }
        }
        return std::make_tuple(L, U);
    }

    this_type lu_decompose_impl(std::size_t n, this_type buf) const {
        if (!is_squared())
            throw not_squared_exception{};
        if (n == Width)
            return buf;
        for (std::size_t j=n+1; j<Height; j++)
            buf.at(n, j) /= buf.at(n, n);
        for (std::size_t i=n+1; i<Width; i++) {
            for (std::size_t j=n+1; j<Height; j++)
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
        std::size_t k = 1;
        for (std::size_t i=0; i<W; i++) {
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
inline static matrix<T, W, H> operator*(T const& lhs, matrix<T, W, H> const& rhs) {
    return rhs * lhs;
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

template<typename T, std::size_t H>
using vector = matrix<T, 1, H>;

template<typename T, std::size_t N>
static inline T norm(akigebra::vector<T, N> const& v) {
    auto result = static_cast<typename akigebra::vector<T, N>::value_type>(0);
    for (std::size_t i=0; i<N; i++)
        result += v.at(0, i) * v.at(0, i);
    return std::sqrt(result);
}

template<typename T, std::size_t N>
static inline std::vector<vector<T, N>>
orthogonalization(std::vector<vector<T, N>> const& input) {
    std::vector<vector<T, N>> result(input.size());
    for (std::size_t i=0; i<N; i++) {
        result.at(i) = input.at(i);
        for (std::size_t j=0; j<i; j++) {
            auto k = input.at(i).transpose() * result.at(j);
            result.at(i) -= result.at(j) * k;
        }
        result.at(i) /= norm(result.at(i));
    }
    return result;
}

}

#endif


/*============================================================================
  Copyright (C) 2016 akitsu sanae
  https://github.com/akitsu-sanae/akigebra
  Distributed under the Boost Software License, Version 1.0. (See accompanying
  file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
============================================================================*/

#include "../akigebra.hpp"

using namespace akigebra;

int main() {
    auto A = matrix<int, 3, 3>{{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    }};

    std::cout << A << std::endl;
    std::cout << matrix<int, 3, 3>::identity() << std::endl;
    std::cout << std::boolalpha;
    std::cout << (A == matrix<int, 3, 3>::identity()) << std::endl;

    auto B = matrix<int, 3, 3> {{
         0, -2, 0,
        -1,  3, 1,
         4,  2, 1
    }};
    std::cout << B << std::endl;
    std::cout << B.determinant() << std::endl;

    std::cout << matrix<int, 2, 2>{{
        3, 1,
        0, 1
    }}.determinant() << std::endl;

    auto C = matrix<int, 2, 2> {{
        1, 2, 3, 4
    }};
    matrix<int, 2, 2> CL, CU;
    std::tie(CL, CU) = C.lu_decompose();
    std::cout << C << " => LU decompose " << std::endl;
    std::cout << CL << " * " << CU << std::endl;

    using namespace std;
    auto D = matrix<std::complex<int>, 2, 2> {
        complex<int>{1, 2}, complex<int>{3, 2},
        complex<int>{-1, -2}, complex<int>{5, -4}
    };
    std::cout << "D" << std::endl;
    std::cout << D << std::endl;
    std::cout << "S's conj matrix" << std::endl;
    std::cout << conj(D) << std::endl;
    std::cout << "D's adjoint matrix" << std::endl;
    std::cout << D.adjoint() << std::endl;
}



#ifndef UTILS_H_F7USKNYL
#define UTILS_H_F7USKNYL

#include <iostream>
#include <limits>
#include <stdexcept>

#include "aa.h" /* libaffa */
#include "cusvna/aa_variable.cuh"

template <typename T>
bool close(const T& a, const T& b) {
    return abs(b - a) <= 10 * std::numeric_limits<T>::epsilon() * max(T(1), max(a, b));
}

bool operator==(const interval& lhs, const cusvna::Interval<double>& rhs) {
    return close(lhs.left(), rhs.a) && close(lhs.right(), rhs.b);
}

template <cusvna::type_id_t... Is>
bool operator==(const AAF& lhs, const cusvna::AA<double,Is...>& rhs) {
    return lhs.convert() == rhs.interval();
}

std::ostream& print_affa_as_cusvna(std::ostream& os, const AAF& a) {
    os << "AA(b=" << a.get_center() << ", x=[";
    for (unsigned i = 0; i < a.get_length(); ++i) {
        os << a.get_index(i) << ": " << a.get_coeff(i);
        if (i != a.get_length()-1) { os << ", "; }
    }
    os << "])";
    return os;
}

template <cusvna::type_id_t... Is>
void assert_equal(const cusvna::AA<double,Is...>& lhs, const AAF& rhs, 
        const std::string& msg = "") {
    if (rhs == lhs) { return; }
    if (msg == "") {
        std::cerr << "Affine forms are different:" << std::endl;
    }
    else {
        std::cerr << "Affine forms for '" << msg << "' are different:" << std::endl;
    }
    std::cerr << "   cusvna: " << lhs << std::endl;
    std::cerr << "  libaffa: ";
    print_affa_as_cusvna(std::cerr, rhs) << std::endl << std::endl;
    throw std::runtime_error( "AA mistmatch." );
}


void assert_equal(const cusvna::Interval<double>& lhs, const interval& rhs) {
    if (rhs == lhs) { return; }
    std::cerr << "Interval are different:" << std::endl;
    std::cerr << "   cusvna: " << lhs << std::endl;
    std::cerr << "  libaffa: [" << rhs.left() << ", " << rhs.right() << ']' << std::endl;
    throw std::runtime_error( "Interval mismatch." );
}

#endif /* end of include guard: UTILS_H_F7USKNYL */

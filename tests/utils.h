
#ifndef UTILS_H_F7USKNYL
#define UTILS_H_F7USKNYL

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "aa.h" /* libaffa */
#include "cusvna/aa.cuh"

/* cusvna bounds are rounded outward while libaffa bounds are not, allow a few ulps of difference */
template <typename T>
T tolerance(const T& a, const T& b) {
    return 64 * std::numeric_limits<T>::epsilon() * std::max(T(1), std::max(std::fabs(a), std::fabs(b)));
}

template <typename T>
bool close(const T& a, const T& b) {
    return std::fabs(b - a) <= tolerance(a, b);
}

bool operator==(const interval& lhs, const cusvna::Interval<double>& rhs) {
    return close(lhs.left(), rhs.a) && close(lhs.right(), rhs.b);
}

bool tighter(const interval& lhs, const cusvna::Interval<double>& rhs) {
    return (lhs.left() <= rhs.a + tolerance(lhs.left(), rhs.a)) && (lhs.right() >= rhs.b - tolerance(lhs.right(), rhs.b));
}

template <typename... Is>
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


void assert_equal(const cusvna::Interval<double>& lhs, const interval& rhs, bool raise=true) {
    if (rhs == lhs) { return; }
    std::cerr << "Interval are different:" << std::endl;
    std::cerr << "   cusvna: " << lhs << std::endl;
    std::cerr << "  libaffa: [" << rhs.left() << ", " << rhs.right() << ']' << std::endl;
    if (raise) {
        throw std::runtime_error( "Interval mismatch." );
    }
}

template <typename... Is>
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
    assert_equal(lhs.interval(), rhs.convert(), false);
    throw std::runtime_error( "AA mistmatch." );
}

template <typename... Is>
void assert_tighter(const cusvna::AA<double,Is...>& lhs, const AAF& rhs, 
        const std::string& msg = "") {
    if (tighter(rhs.convert(), lhs.interval())) { return; }
    if (msg == "") {
        std::cerr << "Cusvna does not return a tighter result:" << std::endl;
    }
    else {
        std::cerr << "Cusvna does not return a tighter result for '" << msg << "':" << std::endl;
    }
    std::cerr << "   cusvna: " << lhs << std::endl;
    std::cerr << "  libaffa: ";
    print_affa_as_cusvna(std::cerr, rhs) << std::endl << std::endl;
    assert_equal(lhs.interval(), rhs.convert(), false);
    throw std::runtime_error( "AA mistmatch." );
}

#endif /* end of include guard: UTILS_H_F7USKNYL */

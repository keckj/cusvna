
#ifndef UTILS_H_F7USKNYL
#define UTILS_H_F7USKNYL

#include "aa.h" /* libaffa */
#include "cusvna/aa.cuh"

std::ostream& print_affa_as_cusvna(std::ostream& os, const AAF& a) {
    os << "AA(b=" << a.get_center() << ", x=[";
    for (unsigned i = 0; i < a.get_length(); ++i) {
        os << a.get_index(i) << ": " << a.get_coeff(i);
        if (i != a.get_length()-1) { os << ", "; }
    }
    os << "])";
    return os;
}

template <typename G, unsigned N>
bool operator==(const AAF& lhs, const cusvna::AA<double,unsigned,G,N>& rhs) {
    bool eq = lhs.get_length() == rhs.n();
    eq &= lhs.get_center() == rhs.x0();
    for (int i = 0; i < rhs.n(); i++) {
        eq &= lhs.get_index(i) == rhs.idx(i);
        eq &= lhs.get_coeff(i) == rhs.x(i);
    }
    return eq;
}

template <typename G, unsigned N>
void assert_equal(const cusvna::AA<double,unsigned,G,N>& lhs, const AAF& rhs) {
    if (rhs == lhs) { return; }
    std::cerr << "Affine forms are different:" << std::endl;
    std::cerr << "   cusvna: " << lhs << std::endl;
    std::cerr << "  libaffa: ";
    print_affa_as_cusvna(std::cerr, rhs) << std::endl << std::endl;
    assert(false);
}

bool operator==(const interval& lhs, const cusvna::Interval<double>& rhs) {
    return (lhs.left() == rhs.a) && (lhs.right() == rhs.b);
}

void assert_equal(const cusvna::Interval<double>& lhs, const interval& rhs) {
    if (rhs == lhs) { return; }
    std::cerr << "Interval are different:" << std::endl;
    std::cerr << "   cusvna: " << lhs << std::endl;
    std::cerr << "  libaffa: [" << rhs.left() << ", " << rhs.right() << ']' << std::endl;
    assert(false);
}

#endif /* end of include guard: UTILS_H_F7USKNYL */

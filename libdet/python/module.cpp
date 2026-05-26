#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <libdet/hamiltonian.hpp>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using U64Array = nb::ndarray<
    const std::uint64_t,
    nb::numpy,
    nb::c_contig,
    nb::device::cpu
>;

using F64Vec = nb::ndarray<
    const double,
    nb::numpy,
    nb::shape<-1>,
    nb::c_contig,
    nb::device::cpu
>;

using F64Mat = nb::ndarray<
    const double,
    nb::numpy,
    nb::shape<-1, -1>,
    nb::c_contig,
    nb::device::cpu
>;

using I64Vec = nb::ndarray<
    const libdet::i64,
    nb::numpy,
    nb::shape<-1>,
    nb::c_contig,
    nb::device::cpu
>;

[[nodiscard]] libdet::DetBatchView to_det_view(const U64Array& a) {
    if (a.ndim() != 3 || a.shape(1) != 2 || a.shape(2) <= 0) {
        throw std::invalid_argument("determinants must have shape (N, 2, nword)");
    }

    return {
        a.data(),
        static_cast<std::size_t>(a.shape(0)),
        static_cast<libdet::u32>(a.shape(2)),
    };
}

[[nodiscard]] libdet::DetRef to_single_det(const U64Array& a) {
    const auto v = to_det_view(a);

    if (v.n_dets != 1) {
        throw std::invalid_argument("expected exactly one determinant");
    }

    return libdet::packed_det(v.data, v.nword);
}

[[nodiscard]] std::span<const double> as_f64(const F64Vec& a) {
    return {a.data(), static_cast<std::size_t>(a.shape(0))};
}

[[nodiscard]] std::span<const double> as_f64_matrix(const F64Mat& a) {
    return {
        a.data(),
        static_cast<std::size_t>(a.shape(0) * a.shape(1))
    };
}

[[nodiscard]] std::span<const libdet::i64> as_i64(const I64Vec& a) {
    return {a.data(), static_cast<std::size_t>(a.shape(0))};
}

[[nodiscard]] std::span<const double> optional_f64(nb::object obj) {
    if (obj.is_none()) return {};
    return as_f64(nb::cast<F64Vec>(obj));
}

template <class T>
auto own_1d(std::vector<T>&& values) {
    auto* heap = new std::vector<T>(std::move(values));

    nb::capsule owner(heap, [](void* p) noexcept {
        delete static_cast<std::vector<T>*>(p);
    });

    return nb::ndarray<T, nb::numpy, nb::shape<-1>>(
        heap->data(),
        {heap->size()},
        owner
    );
}

template <class T>
auto own_2d(std::vector<T>&& values, std::size_t nrow, std::size_t ncol) {
    auto* heap = new std::vector<T>(std::move(values));

    nb::capsule owner(heap, [](void* p) noexcept {
        delete static_cast<std::vector<T>*>(p);
    });

    return nb::ndarray<T, nb::numpy, nb::shape<-1, -1>>(
        heap->data(),
        {nrow, ncol},
        owner
    );
}

template <class T>
auto view_1d(std::span<const T> values) {
    return nb::ndarray<
        const T,
        nb::numpy,
        nb::shape<-1>,
        nb::c_contig,
        nb::device::cpu
    >(
        values.data(),
        {values.size()}
    );
}

auto view_dets(std::span<const std::uint64_t> words, std::size_t ndet, libdet::u32 nword) {
    return nb::ndarray<
        const std::uint64_t,
        nb::numpy,
        nb::c_contig,
        nb::device::cpu
    >(
        words.data(),
        {ndet, std::size_t{2}, static_cast<std::size_t>(nword)}
    );
}

} // anonymous namespace

NB_MODULE(_libdet_cpp, m) {
    m.doc() = "libdet determinant-driven Hamiltonian primitives";

    nb::class_<libdet::Determinants>(m, "Determinants")
        .def_prop_ro("dets", [](const libdet::Determinants& x) {
            return view_dets(x.det_words(), x.n_dets(), x.nword());
        }, nb::rv_policy::reference_internal);

    nb::class_<libdet::Edges>(m, "Edges")
        .def_prop_ro("n_rows", &libdet::Edges::n_rows)
        .def_prop_ro("n_cols", &libdet::Edges::n_cols)
        .def_prop_ro("row_dets", [](const libdet::Edges& x) {
            return view_dets(x.row_words(), x.n_rows(), x.nword());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("col_dets", [](const libdet::Edges& x) {
            return view_dets(x.col_words(), x.n_cols(), x.nword());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("diags", [](const libdet::Edges& x) {
            return view_1d(x.diags());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("row_ptr", [](const libdet::Edges& x) {
            return view_1d(x.row_ptr());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("col", [](const libdet::Edges& x) {
            return view_1d(x.col());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("h", [](const libdet::Edges& x) {
            return view_1d(x.h());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("row_weight", [](const libdet::Edges& x) {
            return view_1d(x.row_weight());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("row_nnz", [](const libdet::Edges& x) {
            return view_1d(x.row_nnz());
        }, nb::rv_policy::reference_internal);

    nb::class_<libdet::Degrees>(m, "Degrees")
        .def_prop_ro("n_rows", &libdet::Degrees::n_rows)
        .def_prop_ro("row_nnz", [](const libdet::Degrees& x) {
            return view_1d(x.row_nnz());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("row_weight", [](const libdet::Degrees& x) {
            return view_1d(x.row_weight());
        }, nb::rv_policy::reference_internal);

    nb::class_<libdet::Matrix>(m, "Matrix")
        .def_prop_ro("shape", [](const libdet::Matrix& x) {
            return nb::make_tuple(x.n_bra(), x.n_ket());
        })
        .def_prop_ro("diags", [](const libdet::Matrix& x) {
            return view_1d(x.diags());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("row_ptr", [](const libdet::Matrix& x) {
            return view_1d(x.row_ptr());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("col", [](const libdet::Matrix& x) {
            return view_1d(x.col());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("h", [](const libdet::Matrix& x) {
            return view_1d(x.h());
        }, nb::rv_policy::reference_internal);

    nb::class_<libdet::Projection>(m, "Projection")
        .def_prop_ro("bras", [](const libdet::Projection& x) {
            return view_dets(x.bra_words(), x.n_bras(), x.nword());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("hpsi", [](const libdet::Projection& x) {
            return view_1d(x.hpsi());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("diags", [](const libdet::Projection& x) {
            return view_1d(x.diags());
        }, nb::rv_policy::reference_internal);

    nb::class_<libdet::EdgeSamples>(m, "EdgeSamples")
        .def_prop_ro("row_nnz", [](const libdet::EdgeSamples& x) {
            return view_1d(x.row_nnz());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("row_weight", [](const libdet::EdgeSamples& x) {
            return view_1d(x.row_weight());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("rows", [](const libdet::EdgeSamples& x) {
            return view_1d(x.rows());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("dets", [](const libdet::EdgeSamples& x) {
            return view_dets(x.det_words(), x.n_samples(), x.nword());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("h", [](const libdet::EdgeSamples& x) {
            return view_1d(x.h());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("pgen", [](const libdet::EdgeSamples& x) {
            return view_1d(x.pgen());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("counts", [](const libdet::EdgeSamples& x) {
            return view_1d(x.counts());
        }, nb::rv_policy::reference_internal);

    nb::class_<libdet::ShellSamples>(m, "ShellSamples")
        .def_prop_ro("rep_ptr", [](const libdet::ShellSamples& x) {
            return view_1d(x.rep_ptr());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("dets", [](const libdet::ShellSamples& x) {
            return view_dets(x.det_words(), x.n_samples(), x.nword());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("diags", [](const libdet::ShellSamples& x) {
            return view_1d(x.diags());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("hpsi_strong", [](const libdet::ShellSamples& x) {
            return view_1d(x.hpsi_strong());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("hpsi_a", [](const libdet::ShellSamples& x) {
            return view_1d(x.hpsi_a());
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("hpsi_b", [](const libdet::ShellSamples& x) {
            return view_1d(x.hpsi_b());
        }, nb::rv_policy::reference_internal);

    nb::class_<libdet::Hamiltonian>(m, "Hamiltonian")
        .def_static("rhf", [](const F64Mat& h1, const F64Vec& eri, double ecore) {
            if (h1.shape(0) != h1.shape(1)) {
                throw std::invalid_argument("Hamiltonian.rhf: h1 must be square");
            }

            const int norb = static_cast<int>(h1.shape(0));

            return libdet::Hamiltonian::make(
                std::span<const double>(
                    h1.data(),
                    static_cast<std::size_t>(h1.shape(0) * h1.shape(1))
                ),
                norb,
                as_f64(eri),
                ecore
            );
        }, "h1"_a.noconvert(), "eri"_a.noconvert(), "ecore"_a = 0.0)

        .def_prop_ro("norb", &libdet::Hamiltonian::norb)
        .def_prop_ro("nword", &libdet::Hamiltonian::nword)

        .def("hij", [](const libdet::Hamiltonian& ham, const U64Array& bra, const U64Array& ket) {
            return ham.hij(to_single_det(bra), to_single_det(ket));
        }, "bra"_a.noconvert(), "ket"_a.noconvert())

        .def("diags", [](const libdet::Hamiltonian& ham, const U64Array& dets) {
            return own_1d(ham.diags(to_det_view(dets)));
        }, "dets"_a.noconvert())

        .def("expand", [](const libdet::Hamiltonian& ham,
                          const U64Array& kets,
                          double eps,
                          nb::object coeffs_obj,
                          nb::object exclude_obj) {
            const auto ket_view = to_det_view(kets);
            const auto coeff_view = optional_f64(coeffs_obj);

            if (!exclude_obj.is_none()) {
                const U64Array exclude_arr = nb::cast<U64Array>(exclude_obj);
                const auto exclude_view = to_det_view(exclude_arr);
                return ham.expand(ket_view, eps, coeff_view, &exclude_view);
            }

            return ham.expand(ket_view, eps, coeff_view, nullptr);
        }, "kets"_a.noconvert(),
           "eps"_a,
           "coeffs"_a = nb::none(),
           "exclude"_a = nb::none())

        .def("project", [](const libdet::Hamiltonian& ham,
                           const U64Array& bras,
                           const U64Array& kets,
                           const F64Vec& coeffs,
                           double eps) {
            return ham.project(
                to_det_view(bras),
                to_det_view(kets),
                as_f64(coeffs),
                eps
            );
        }, "bras"_a.noconvert(),
           "kets"_a.noconvert(),
           "coeffs"_a.noconvert(),
           "eps"_a = 0.0)

        .def("edges", [](const libdet::Hamiltonian& ham,
                         const U64Array& dets,
                         double eps) {
            return ham.edges(to_det_view(dets), eps);
        }, "dets"_a.noconvert(), "eps"_a)

        .def("degrees", [](const libdet::Hamiltonian& ham,
                           const U64Array& dets,
                           double eps) {
            return ham.degrees(to_det_view(dets), eps);
        }, "dets"_a.noconvert(), "eps"_a)

        .def("matrix", [](const libdet::Hamiltonian& ham,
                          const U64Array& bras,
                          const U64Array& kets) {
            return ham.matrix(to_det_view(bras), to_det_view(kets));
        }, "bras"_a.noconvert(), "kets"_a.noconvert())

        .def("matvec", [](const libdet::Hamiltonian& ham,
                          const U64Array& bras,
                          const U64Array& kets,
                          const F64Vec& x) {
            return own_1d(
                ham.matvec(
                    to_det_view(bras),
                    to_det_view(kets),
                    as_f64(x)
                )
            );
        }, "bras"_a.noconvert(),
           "kets"_a.noconvert(),
           "x"_a.noconvert())

        .def("matmat", [](const libdet::Hamiltonian& ham,
                          const U64Array& bras,
                          const U64Array& kets,
                          const F64Mat& x) {
            const auto bra_view = to_det_view(bras);
            const std::size_t nrhs = static_cast<std::size_t>(x.shape(1));

            auto out = ham.matmat(
                bra_view,
                to_det_view(kets),
                as_f64_matrix(x),
                nrhs
            );

            return own_2d(std::move(out), bra_view.n_dets, nrhs);
        }, "bras"_a.noconvert(),
           "kets"_a.noconvert(),
           "x"_a.noconvert())

        .def("sample_edges", [](const libdet::Hamiltonian& ham,
                                const U64Array& dets,
                                nb::object counts_obj,
                                double eps1,
                                double eps2,
                                std::uint64_t seed) {
            const auto det_view = to_det_view(dets);

            if (counts_obj.is_none()) {
                return ham.sample_edges(det_view, {}, eps1, eps2, seed);
            }

            const I64Vec counts = nb::cast<I64Vec>(counts_obj);

            return ham.sample_edges(
                det_view,
                as_i64(counts),
                eps1,
                eps2,
                seed
            );
        }, "dets"_a.noconvert(),
           "counts"_a = nb::none(),
           "eps1"_a = 1.0e-6,
           "eps2"_a = 0.0,
           "seed"_a = std::uint64_t{0})

        .def("sample_shell", [](const libdet::Hamiltonian& ham,
                                const U64Array& kets,
                                const F64Vec& coeffs,
                                double eps1,
                                double eps2,
                                const I64Vec& counts,
                                nb::object exclude_obj,
                                libdet::i64 n_rep,
                                std::uint64_t seed) {
            const auto ket_view = to_det_view(kets);

            if (!exclude_obj.is_none()) {
                const U64Array exclude_arr = nb::cast<U64Array>(exclude_obj);
                const auto exclude_view = to_det_view(exclude_arr);

                return ham.sample_shell(
                    ket_view,
                    as_f64(coeffs),
                    eps1,
                    eps2,
                    as_i64(counts),
                    &exclude_view,
                    n_rep,
                    seed
                );
            }

            return ham.sample_shell(
                ket_view,
                as_f64(coeffs),
                eps1,
                eps2,
                as_i64(counts),
                nullptr,
                n_rep,
                seed
            );
        }, "kets"_a.noconvert(),
           "coeffs"_a.noconvert(),
           "eps1"_a,
           "eps2"_a,
           "counts"_a.noconvert(),
           "exclude"_a = nb::none(),
           "n_rep"_a = libdet::i64{1},
           "seed"_a = std::uint64_t{0});
}
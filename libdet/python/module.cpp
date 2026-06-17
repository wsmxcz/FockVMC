#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
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

using I64Array = nb::ndarray<
    const libdet::i64,
    nb::numpy,
    nb::c_contig,
    nb::device::cpu
>;

[[nodiscard]] libdet::DetBatchView to_det_view(const U64Array& dets) {
    if (dets.ndim() != 3 || dets.shape(1) != 2 || dets.shape(2) <= 0) {
        throw std::invalid_argument("determinants must have shape (N, 2, nword)");
    }

    return {
        dets.data(),
        static_cast<std::size_t>(dets.shape(0)),
        static_cast<libdet::u32>(dets.shape(2)),
    };
}

[[nodiscard]] libdet::DetRef to_single_det(const U64Array& dets) {
    const auto view = to_det_view(dets);

    if (view.n_dets != 1) {
        throw std::invalid_argument("expected exactly one determinant");
    }

    return libdet::packed_det(view.data, view.nword);
}

[[nodiscard]] std::span<const double> as_f64(const F64Vec& values) {
    return {values.data(), static_cast<std::size_t>(values.shape(0))};
}

[[nodiscard]] std::span<const double> as_f64_matrix(const F64Mat& values) {
    return {
        values.data(),
        static_cast<std::size_t>(values.shape(0) * values.shape(1))
    };
}

[[nodiscard]] std::span<const libdet::i64> as_i64(const I64Vec& values) {
    return {values.data(), static_cast<std::size_t>(values.shape(0))};
}

[[nodiscard]] std::span<const double> optional_f64(nb::object obj) {
    if (obj.is_none()) return {};
    return as_f64(nb::cast<F64Vec>(obj));
}

template <class T>
auto own_1d(std::vector<T>&& values) {
    auto* heap = new std::vector<T>(std::move(values));

    nb::capsule owner(heap, [](void* ptr) noexcept {
        delete static_cast<std::vector<T>*>(ptr);
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

    nb::capsule owner(heap, [](void* ptr) noexcept {
        delete static_cast<std::vector<T>*>(ptr);
    });

    return nb::ndarray<T, nb::numpy, nb::shape<-1, -1>>(
        heap->data(),
        {nrow, ncol},
        owner
    );
}

auto own_dets(
    std::vector<std::uint64_t>&& words,
    std::size_t n_dets,
    libdet::u32 nword
) {
    auto* heap = new std::vector<std::uint64_t>(std::move(words));

    nb::capsule owner(heap, [](void* ptr) noexcept {
        delete static_cast<std::vector<std::uint64_t>*>(ptr);
    });

    return nb::ndarray<
        std::uint64_t,
        nb::numpy,
        nb::shape<-1, 2, -1>
    >(
        heap->data(),
        {n_dets, std::size_t{2}, static_cast<std::size_t>(nword)},
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

template <class T>
auto view_1d(const std::vector<T>& values) {
    return view_1d(std::span<const T>(values));
}

auto view_dets(
    std::span<const std::uint64_t> words,
    std::size_t n_dets,
    libdet::u32 nword
) {
    return nb::ndarray<
        const std::uint64_t,
        nb::numpy,
        nb::c_contig,
        nb::device::cpu
    >(
        words.data(),
        {n_dets, std::size_t{2}, static_cast<std::size_t>(nword)}
    );
}

auto view_dets(
    const std::vector<std::uint64_t>& words,
    std::size_t n_dets,
    libdet::u32 nword
) {
    return view_dets(std::span<const std::uint64_t>(words), n_dets, nword);
}

} // anonymous namespace

NB_MODULE(_libdet_cpp, m) {
    m.doc() = "Determinant Hamiltonian primitives";

    nb::class_<libdet::Conns>(m, "Conns")
        .def_prop_ro("n_kets", [](const libdet::Conns& out) {
            return out.n_kets;
        })
        .def_prop_ro("x", [](const libdet::Conns& out) {
            return view_dets(
                out.det_words,
                out.det_words.size() / libdet::det_size(out.nword),
                out.nword
            );
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("diag", [](const libdet::Conns& out) {
            return view_1d(out.diag);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("ket_ptr", [](const libdet::Conns& out) {
            return view_1d(out.ket_ptr);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("bra_idx", [](const libdet::Conns& out) {
            return view_1d(out.bra_idx);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("h", [](const libdet::Conns& out) {
            return view_1d(out.h);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("weight", [](const libdet::Conns& out) {
            return view_1d(out.weight);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("sample_ket_ptr", [](const libdet::Conns& out) {
            return view_1d(out.sample_ket_ptr);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("sample_bra_idx", [](const libdet::Conns& out) {
            return view_1d(out.sample_bra_idx);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("sample_h", [](const libdet::Conns& out) {
            return view_1d(out.sample_h);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("sample_count", [](const libdet::Conns& out) {
            return view_1d(out.sample_count);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("sample_weight", [](const libdet::Conns& out) {
            return view_1d(out.sample_weight);
        }, nb::rv_policy::reference_internal);

    nb::class_<libdet::Projection>(m, "Projection")
        .def_prop_ro("bra", [](const libdet::Projection& out) {
            return view_dets(out.bra_words, out.hpsi.size(), out.nword);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("hpsi", [](const libdet::Projection& out) {
            return view_1d(out.hpsi);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("diag", [](const libdet::Projection& out) {
            return view_1d(out.diags);
        }, nb::rv_policy::reference_internal);

    nb::class_<libdet::ConnSamples>(m, "ConnSamples")
        .def_prop_ro("n_kets", [](const libdet::ConnSamples& out) {
            return out.n_kets;
        })
        .def_prop_ro("n_streams", [](const libdet::ConnSamples& out) {
            return out.n_streams;
        })
        .def_prop_ro("x", [](const libdet::ConnSamples& out) {
            return view_dets(
                out.det_words,
                out.det_words.size() / libdet::det_size(out.nword),
                out.nword
            );
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("ket_ptr", [](const libdet::ConnSamples& out) {
            return view_1d(out.ket_ptr);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("bra_idx", [](const libdet::ConnSamples& out) {
            return view_1d(out.bra_idx);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("h", [](const libdet::ConnSamples& out) {
            return view_1d(out.h);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("count", [](const libdet::ConnSamples& out) {
            return view_1d(out.count);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("weight", [](const libdet::ConnSamples& out) {
            return view_1d(out.weight);
        }, nb::rv_policy::reference_internal);

    nb::class_<libdet::ProjectSamples>(m, "ProjectSamples")
        .def_prop_ro("rep_ptr", [](const libdet::ProjectSamples& out) {
            return view_1d(out.rep_ptr);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("bra", [](const libdet::ProjectSamples& out) {
            return view_dets(out.bra_words, out.hpsi_a.size(), out.nword);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("diag", [](const libdet::ProjectSamples& out) {
            return view_1d(out.diags);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("hpsi_strong", [](const libdet::ProjectSamples& out) {
            return view_1d(out.hpsi_strong);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("hpsi_a", [](const libdet::ProjectSamples& out) {
            return view_1d(out.hpsi_a);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("hpsi_b", [](const libdet::ProjectSamples& out) {
            return view_1d(out.hpsi_b);
        }, nb::rv_policy::reference_internal);

    nb::class_<libdet::Hamiltonian>(m, "Hamiltonian")
        .def_static("rhf", [](const F64Mat& h1, const F64Vec& eri, double ecore) {
            if (h1.shape(0) != h1.shape(1)) {
                throw std::invalid_argument("Hamiltonian.rhf: h1 must be square");
            }

            const int norb = static_cast<int>(h1.shape(0));
            const auto h1_view = std::span<const double>(
                h1.data(),
                static_cast<std::size_t>(h1.shape(0) * h1.shape(1))
            );
            const auto eri_view = as_f64(eri);

            return [&]() {
                nb::gil_scoped_release release;
                return libdet::Hamiltonian::make(h1_view, norb, eri_view, ecore);
            }();
        }, "h1"_a.noconvert(), "eri"_a.noconvert(), "ecore"_a = 0.0)

        .def_static("guga", [](const F64Mat& h1,
                               const F64Vec& eri,
                               int n_alpha,
                               int n_beta,
                               double ecore) {
            if (h1.shape(0) != h1.shape(1)) {
                throw std::invalid_argument("Hamiltonian.guga: h1 must be square");
            }

            const int norb = static_cast<int>(h1.shape(0));
            const auto h1_view = std::span<const double>(
                h1.data(),
                static_cast<std::size_t>(h1.shape(0) * h1.shape(1))
            );
            const auto eri_view = as_f64(eri);

            return [&]() {
                nb::gil_scoped_release release;
                return libdet::Hamiltonian::guga(
                    h1_view,
                    norb,
                    eri_view,
                    n_alpha,
                    n_beta,
                    ecore
                );
            }();
        }, "h1"_a.noconvert(),
           "eri"_a.noconvert(),
           "n_alpha"_a,
           "n_beta"_a,
           "ecore"_a = 0.0)

        .def_prop_ro("norb", &libdet::Hamiltonian::norb)
        .def_prop_ro("nword", &libdet::Hamiltonian::nword)

        .def("hij", [](const libdet::Hamiltonian& ham,
                       const U64Array& bra,
                       const U64Array& ket) {
            const auto bra_ref = to_single_det(bra);
            const auto ket_ref = to_single_det(ket);

            nb::gil_scoped_release release;
            return ham.hij(bra_ref, ket_ref);
        }, "bra"_a.noconvert(), "ket"_a.noconvert())

        .def("diag", [](const libdet::Hamiltonian& ham, const U64Array& dets) {
            const auto det_view = to_det_view(dets);
            std::vector<double> out;

            {
                nb::gil_scoped_release release;
                out = ham.diags(det_view);
            }

            return own_1d(std::move(out));
        }, "dets"_a.noconvert())

        .def("expand", [](const libdet::Hamiltonian& ham,
                          const U64Array& kets,
                          double eps,
                          nb::object coeffs_obj,
                          nb::object exclude_obj) {
            const auto ket_view = to_det_view(kets);
            const auto coeff_view = optional_f64(coeffs_obj);
            std::vector<std::uint64_t> out;

            if (!exclude_obj.is_none()) {
                const U64Array exclude_arr = nb::cast<U64Array>(exclude_obj);
                const auto exclude_view = to_det_view(exclude_arr);

                {
                    nb::gil_scoped_release release;
                    out = ham.expand(ket_view, eps, coeff_view, &exclude_view);
                }

                const libdet::u32 nword = ham.nword();
                const std::size_t n_dets =
                    out.size() / libdet::det_size(nword);
                return own_dets(std::move(out), n_dets, nword);
            }

            {
                nb::gil_scoped_release release;
                out = ham.expand(ket_view, eps, coeff_view, nullptr);
            }

            const libdet::u32 nword = ham.nword();
            const std::size_t n_dets =
                out.size() / libdet::det_size(nword);
            return own_dets(std::move(out), n_dets, nword);
        }, "kets"_a.noconvert(),
           "eps"_a,
           "coeffs"_a = nb::none(),
           "exclude"_a = nb::none())

        .def("project", [](const libdet::Hamiltonian& ham,
                           nb::object bras_obj,
                           const U64Array& kets,
                           const F64Vec& coeffs,
                           double eps,
                           nb::object exclude_obj) {
            const auto ket_view = to_det_view(kets);
            const auto coeff_view = as_f64(coeffs);
            libdet::Projection out;

            if (bras_obj.is_none()) {
                if (!exclude_obj.is_none()) {
                    const U64Array exclude_arr = nb::cast<U64Array>(exclude_obj);
                    const auto exclude_view = to_det_view(exclude_arr);

                    {
                        nb::gil_scoped_release release;
                        out = ham.project(ket_view, coeff_view, eps, &exclude_view);
                    }

                    return out;
                }

                {
                    nb::gil_scoped_release release;
                    out = ham.project(ket_view, coeff_view, eps, nullptr);
                }

                return out;
            }

            if (!exclude_obj.is_none()) {
                throw std::invalid_argument("project: exclude is only valid when bras is None");
            }

            const U64Array bras_arr = nb::cast<U64Array>(bras_obj);
            const auto bra_view = to_det_view(bras_arr);

            {
                nb::gil_scoped_release release;
                out = ham.project(bra_view, ket_view, coeff_view, eps);
            }

            return out;
        }, "bras"_a.none(),
           "kets"_a.noconvert(),
           "coeffs"_a.noconvert(),
           "eps"_a = 0.0,
           "exclude"_a = nb::none())

        .def("conns", [](const libdet::Hamiltonian& ham,
                         const U64Array& kets,
                         double eps,
                         libdet::i64 sample,
                         double sample_eps,
                         std::uint64_t seed) {
            const auto ket_view = to_det_view(kets);
            libdet::Conns out;

            {
                nb::gil_scoped_release release;
                out = ham.conns(ket_view, eps, sample, sample_eps, seed);
            }

            return out;
        }, "kets"_a.noconvert(),
           "eps"_a,
           "sample"_a = libdet::i64{0},
           "sample_eps"_a = 0.0,
           "seed"_a = std::uint64_t{0})

        .def("degrees", [](const libdet::Hamiltonian& ham,
                           const U64Array& kets,
                           double eps) {
            const auto ket_view = to_det_view(kets);
            std::pair<std::vector<double>, std::vector<libdet::i64>> out;

            {
                nb::gil_scoped_release release;
                out = ham.degrees(ket_view, eps);
            }

            return nb::make_tuple(
                own_1d(std::move(out.first)),
                own_1d(std::move(out.second))
            );
        }, "kets"_a.noconvert(), "eps"_a)

        .def("matrix", [](const libdet::Hamiltonian& ham,
                          const U64Array& bras,
                          const U64Array& kets) {
            const auto bra_view = to_det_view(bras);
            const auto ket_view = to_det_view(kets);
            libdet::Matrix out;

            {
                nb::gil_scoped_release release;
                out = ham.matrix(bra_view, ket_view);
            }

            return nb::make_tuple(
                own_1d(std::move(out.indptr)),
                own_1d(std::move(out.indices)),
                own_1d(std::move(out.data)),
                nb::make_tuple(out.n_bra, out.n_ket)
            );
        }, "bras"_a.noconvert(), "kets"_a.noconvert())

        .def("matvec", [](const libdet::Hamiltonian& ham,
                          const U64Array& bras,
                          const U64Array& kets,
                          const F64Vec& x) {
            const auto bra_view = to_det_view(bras);
            const auto ket_view = to_det_view(kets);
            const auto x_view = as_f64(x);
            std::vector<double> out;

            {
                nb::gil_scoped_release release;
                out = ham.matvec(bra_view, ket_view, x_view);
            }

            return own_1d(std::move(out));
        }, "bras"_a.noconvert(),
           "kets"_a.noconvert(),
           "x"_a.noconvert())

        .def("matmat", [](const libdet::Hamiltonian& ham,
                          const U64Array& bras,
                          const U64Array& kets,
                          const F64Mat& x) {
            const auto bra_view = to_det_view(bras);
            const auto ket_view = to_det_view(kets);
            const auto x_view = as_f64_matrix(x);
            const std::size_t nrhs = static_cast<std::size_t>(x.shape(1));
            std::vector<double> out;

            {
                nb::gil_scoped_release release;
                out = ham.matmat(bra_view, ket_view, x_view, nrhs);
            }

            return own_2d(std::move(out), bra_view.n_dets, nrhs);
        }, "bras"_a.noconvert(),
           "kets"_a.noconvert(),
           "x"_a.noconvert())

        .def("sample_conns", [](const libdet::Hamiltonian& ham,
                                const U64Array& kets,
                                const I64Array& counts,
                                double eps1,
                                double eps2,
                                std::uint64_t seed) {
            const auto ket_view = to_det_view(kets);
            if (counts.ndim() != 1 && counts.ndim() != 2) {
                throw std::invalid_argument(
                    "counts must have shape (N,) or (S, N)"
                );
            }

            const std::size_t n_streams = counts.ndim() == 1
                ? 1u
                : static_cast<std::size_t>(counts.shape(0));
            const std::size_t n_kets = static_cast<std::size_t>(
                counts.shape(counts.ndim() - 1)
            );
            if (n_kets != ket_view.n_dets) {
                throw std::invalid_argument(
                    "counts last dimension must match kets"
                );
            }

            const std::span<const libdet::i64> count_view{
                counts.data(),
                n_streams * n_kets
            };
            libdet::ConnSamples out;
            {
                nb::gil_scoped_release release;
                out = ham.sample_conns(
                    ket_view,
                    count_view,
                    n_streams,
                    eps1,
                    eps2,
                    seed
                );
            }
            return out;
        }, "kets"_a.noconvert(),
           "counts"_a.noconvert(),
           "eps1"_a = 1.0e-6,
           "eps2"_a = 0.0,
           "seed"_a = std::uint64_t{0})

        .def("sample_project", [](const libdet::Hamiltonian& ham,
                                  const U64Array& kets,
                                  const F64Vec& coeffs,
                                  double eps1,
                                  double eps2,
                                  const I64Vec& counts,
                                  nb::object exclude_obj,
                                  libdet::i64 n_rep,
                                  std::uint64_t seed) {
            const auto ket_view = to_det_view(kets);
            const auto coeff_view = as_f64(coeffs);
            const auto count_view = as_i64(counts);
            libdet::ProjectSamples out;

            if (!exclude_obj.is_none()) {
                const U64Array exclude_arr = nb::cast<U64Array>(exclude_obj);
                const auto exclude_view = to_det_view(exclude_arr);

                {
                    nb::gil_scoped_release release;
                    out = ham.sample_project(
                        ket_view,
                        coeff_view,
                        eps1,
                        eps2,
                        count_view,
                        &exclude_view,
                        n_rep,
                        seed
                    );
                }

                return out;
            }

            {
                nb::gil_scoped_release release;
                out = ham.sample_project(
                    ket_view,
                    coeff_view,
                    eps1,
                    eps2,
                    count_view,
                    nullptr,
                    n_rep,
                    seed
                );
            }

            return out;
        }, "kets"_a.noconvert(),
           "coeffs"_a.noconvert(),
           "eps1"_a,
           "eps2"_a,
           "counts"_a.noconvert(),
           "exclude"_a = nb::none(),
           "n_rep"_a = libdet::i64{1},
           "seed"_a = std::uint64_t{0});

}

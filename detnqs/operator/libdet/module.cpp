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

using I64Array = nb::ndarray<
    const libdet::i64,
    nb::numpy,
    nb::c_contig,
    nb::device::cpu
>;

[[nodiscard]] libdet::StateBatchView to_state_view(const U64Array& dets) {
    if (dets.ndim() != 3 || dets.shape(1) != 2 || dets.shape(2) <= 0) {
        throw std::invalid_argument("determinants must have shape (N, 2, nword)");
    }

    return {
        dets.data(),
        static_cast<std::size_t>(dets.shape(0)),
        static_cast<libdet::u32>(dets.shape(2)),
    };
}

[[nodiscard]] libdet::StateRef to_single_state(const U64Array& dets) {
    const auto view = to_state_view(dets);
    if (view.n_states != 1) {
        throw std::invalid_argument("expected exactly one determinant");
    }
    return view[0];
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

template <class T>
auto view_2d(const std::vector<T>& values, std::size_t nrow, std::size_t ncol) {
    return nb::ndarray<
        const T,
        nb::numpy,
        nb::shape<-1, -1>,
        nb::c_contig,
        nb::device::cpu
    >(
        values.data(),
        {nrow, ncol}
    );
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

} // namespace

NB_MODULE(libdet, m) {
    m.doc() = "Electronic Hamiltonian primitives";

    nb::class_<libdet::Conns>(m, "Conns")
        .def_prop_ro("n_kets", [](const libdet::Conns& out) {
            return out.n_kets;
        })
        .def_prop_ro("n_streams", [](const libdet::Conns& out) {
            return out.n_streams;
        })
        .def_prop_ro("x", [](const libdet::Conns& out) {
            return view_dets(
                out.bra_words,
                out.bra_words.size() / libdet::word_pair_size(out.nword),
                out.nword
            );
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("diag", [](const libdet::Conns& out) {
            return view_1d(out.diag);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("ptr", [](const libdet::Conns& out) {
            return view_1d(out.ptr);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("bra", [](const libdet::Conns& out) {
            return view_1d(out.bra);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("h", [](const libdet::Conns& out) {
            return view_1d(out.h);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("weight", [](const libdet::Conns& out) {
            return view_1d(out.weight);
        }, nb::rv_policy::reference_internal);

    nb::class_<libdet::Projections>(m, "Projections")
        .def_prop_ro("n_streams", [](const libdet::Projections& out) {
            return out.n_streams;
        })
        .def_prop_ro("bra", [](const libdet::Projections& out) {
            return view_dets(
                out.bra_words,
                out.bra_words.size() / libdet::word_pair_size(out.nword),
                out.nword
            );
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("hpsi", [](const libdet::Projections& out) {
            const std::size_t n_bra =
                out.bra_words.size() / libdet::word_pair_size(out.nword);
            return view_2d(out.hpsi, out.n_streams, n_bra);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("diag", [](const libdet::Projections& out) {
            return view_1d(out.diags);
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

    nb::class_<libdet::Hamiltonian>(m, "Hamiltonian")
        .def_static("det", [](const F64Mat& h1, const F64Vec& eri, double ecore) {
            if (h1.shape(0) != h1.shape(1)) {
                throw std::invalid_argument("Hamiltonian.det: h1 must be square");
            }

            const int norb = static_cast<int>(h1.shape(0));
            const auto h1_view = std::span<const double>(
                h1.data(),
                static_cast<std::size_t>(h1.shape(0) * h1.shape(1))
            );
            const auto eri_view = as_f64(eri);

            return [&]() {
                nb::gil_scoped_release release;
                return libdet::Hamiltonian::det(h1_view, norb, eri_view, ecore);
            }();
        }, "h1"_a.noconvert(), "eri"_a.noconvert(), "ecore"_a = 0.0)

        .def_static("spin", [](const F64Mat& h1,
                               const F64Vec& eri,
                               int n_alpha,
                               int n_beta,
                               double ecore) {
            if (h1.shape(0) != h1.shape(1)) {
                throw std::invalid_argument("Hamiltonian.spin: h1 must be square");
            }

            const int norb = static_cast<int>(h1.shape(0));
            const auto h1_view = std::span<const double>(
                h1.data(),
                static_cast<std::size_t>(h1.shape(0) * h1.shape(1))
            );
            const auto eri_view = as_f64(eri);

            return [&]() {
                nb::gil_scoped_release release;
                return libdet::Hamiltonian::spin(
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
            const auto bra_ref = to_single_state(bra);
            const auto ket_ref = to_single_state(ket);

            nb::gil_scoped_release release;
            return ham.hij(bra_ref, ket_ref);
        }, "bra"_a.noconvert(), "ket"_a.noconvert())

        .def("diag", [](const libdet::Hamiltonian& ham, const U64Array& x) {
            const auto x_view = to_state_view(x);
            std::vector<double> out;

            {
                nb::gil_scoped_release release;
                out = ham.diags(x_view);
            }

            return own_1d(std::move(out));
        }, "x"_a.noconvert())

        .def("expand", [](const libdet::Hamiltonian& ham,
                          const U64Array& kets,
                          double eps,
                          nb::object scale_obj,
                          nb::object exclude_obj) {
            const auto ket_view = to_state_view(kets);
            const auto scale_view = optional_f64(scale_obj);
            std::vector<std::uint64_t> out;

            if (!exclude_obj.is_none()) {
                const U64Array exclude_arr = nb::cast<U64Array>(exclude_obj);
                const auto exclude_view = to_state_view(exclude_arr);

                {
                    nb::gil_scoped_release release;
                    out = ham.expand(ket_view, eps, scale_view, &exclude_view);
                }
            } else {
                nb::gil_scoped_release release;
                out = ham.expand(ket_view, eps, scale_view, nullptr);
            }

            const libdet::u32 nword = ham.nword();
            const std::size_t n_dets = out.size() / libdet::word_pair_size(nword);
            return own_dets(std::move(out), n_dets, nword);
        }, "kets"_a.noconvert(),
           "eps"_a,
           "scale"_a = nb::none(),
           "exclude"_a = nb::none())

        .def("project", [](const libdet::Hamiltonian& ham,
                           nb::object bras_obj,
                           const U64Array& kets,
                           const F64Vec& scale,
                           double eps,
                           nb::object exclude_obj) {
            const auto ket_view = to_state_view(kets);
            const auto scale_view = as_f64(scale);
            libdet::Projection out;

            if (bras_obj.is_none()) {
                if (!exclude_obj.is_none()) {
                    const U64Array exclude_arr = nb::cast<U64Array>(exclude_obj);
                    const auto exclude_view = to_state_view(exclude_arr);

                    {
                        nb::gil_scoped_release release;
                        out = ham.project(ket_view, scale_view, eps, &exclude_view);
                    }

                    return out;
                }

                {
                    nb::gil_scoped_release release;
                    out = ham.project(ket_view, scale_view, eps, nullptr);
                }

                return out;
            }

            if (!exclude_obj.is_none()) {
                throw std::invalid_argument(
                    "project: exclude is only valid when bras is None"
                );
            }

            const U64Array bras_arr = nb::cast<U64Array>(bras_obj);
            const auto bra_view = to_state_view(bras_arr);

            {
                nb::gil_scoped_release release;
                out = ham.project(bra_view, ket_view, scale_view, eps);
            }

            return out;
        }, "bras"_a.none(),
           "kets"_a.noconvert(),
           "scale"_a.noconvert(),
           "eps"_a = 0.0,
           "exclude"_a = nb::none())

        .def("conn", [](const libdet::Hamiltonian& ham,
                        const U64Array& kets,
                        double eps,
                        nb::object include_obj) {
            const auto ket_view = to_state_view(kets);
            libdet::Conns out;

            if (!include_obj.is_none()) {
                const U64Array include_arr = nb::cast<U64Array>(include_obj);
                const auto include_view = to_state_view(include_arr);

                {
                    nb::gil_scoped_release release;
                    out = ham.conn(ket_view, eps, &include_view);
                }

                return out;
            }

            {
                nb::gil_scoped_release release;
                out = ham.conn(ket_view, eps, nullptr);
            }

            return out;
        }, "kets"_a.noconvert(),
           "eps"_a = 0.0,
           "include"_a = nb::none())

        .def("sample_conn", [](const libdet::Hamiltonian& ham,
                               const U64Array& kets,
                               const I64Array& counts,
                               double eps1,
                               double eps2,
                               std::uint64_t seed,
                               bool bra_weight,
                               nb::object include_obj) {
            const auto ket_view = to_state_view(kets);

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

            if (n_kets != ket_view.n_states) {
                throw std::invalid_argument(
                    "counts last dimension must match kets"
                );
            }

            const std::span<const libdet::i64> count_view{
                counts.data(),
                n_streams * n_kets
            };

            libdet::Conns out;
            if (!include_obj.is_none()) {
                const U64Array include_arr = nb::cast<U64Array>(include_obj);
                const auto include_view = to_state_view(include_arr);

                {
                    nb::gil_scoped_release release;
                    out = ham.sample_conn(
                        ket_view,
                        count_view,
                        n_streams,
                        eps1,
                        eps2,
                        seed,
                        bra_weight,
                        &include_view
                    );
                }

                return out;
            }

            {
                nb::gil_scoped_release release;
                out = ham.sample_conn(
                    ket_view,
                    count_view,
                    n_streams,
                    eps1,
                    eps2,
                    seed,
                    bra_weight,
                    nullptr
                );
            }

            return out;
        }, "kets"_a.noconvert(),
           "counts"_a.noconvert(),
           "eps1"_a,
           "eps2"_a = 0.0,
           "seed"_a = std::uint64_t{0},
           "bra_weight"_a = false,
           "include"_a = nb::none())

        .def("sample_project", [](const libdet::Hamiltonian& ham,
                                  const U64Array& kets,
                                  const F64Vec& scale,
                                  const I64Array& counts,
                                  double eps1,
                                  double eps2,
                                  nb::object exclude_obj,
                                  std::uint64_t seed) {
            const auto ket_view = to_state_view(kets);
            const auto scale_view = as_f64(scale);

            if (scale_view.size() != ket_view.n_states) {
                throw std::invalid_argument(
                    "scale length must match number of kets"
                );
            }
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

            if (n_kets != ket_view.n_states) {
                throw std::invalid_argument(
                    "counts last dimension must match kets"
                );
            }

            const std::span<const libdet::i64> count_view{
                counts.data(),
                n_streams * n_kets
            };

            libdet::Projections out;
            if (!exclude_obj.is_none()) {
                const U64Array exclude_arr = nb::cast<U64Array>(exclude_obj);
                const auto exclude_view = to_state_view(exclude_arr);

                {
                    nb::gil_scoped_release release;
                    out = ham.sample_project(
                        ket_view,
                        scale_view,
                        count_view,
                        n_streams,
                        eps1,
                        eps2,
                        &exclude_view,
                        seed
                    );
                }

                return out;
            }

            {
                nb::gil_scoped_release release;
                out = ham.sample_project(
                    ket_view,
                    scale_view,
                    count_view,
                    n_streams,
                    eps1,
                    eps2,
                    nullptr,
                    seed
                );
            }

            return out;
        }, "kets"_a.noconvert(),
           "scale"_a.noconvert(),
           "counts"_a.noconvert(),
           "eps1"_a,
           "eps2"_a = 0.0,
           "exclude"_a = nb::none(),
           "seed"_a = std::uint64_t{0})

        .def("matrix", [](const libdet::Hamiltonian& ham,
                          const U64Array& bras,
                          const U64Array& kets) {
            const auto bra_view = to_state_view(bras);
            const auto ket_view = to_state_view(kets);
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
            const auto bra_view = to_state_view(bras);
            const auto ket_view = to_state_view(kets);
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
            const auto bra_view = to_state_view(bras);
            const auto ket_view = to_state_view(kets);
            const auto x_view = as_f64_matrix(x);
            const std::size_t nrhs = static_cast<std::size_t>(x.shape(1));
            std::vector<double> out;

            {
                nb::gil_scoped_release release;
                out = ham.matmat(bra_view, ket_view, x_view, nrhs);
            }

            return own_2d(std::move(out), bra_view.n_states, nrhs);
        }, "bras"_a.noconvert(),
           "kets"_a.noconvert(),
           "x"_a.noconvert());
}

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <libdet/rhf/hamiltonian.hpp>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using StateArray = nb::ndarray<
    const std::uint64_t,
    nb::numpy,
    nb::shape<-1, 2, -1>,
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

[[nodiscard]] libdet::rhf::DetBatchView states(const StateArray& x) noexcept {
    return {
        x.data(),
        static_cast<std::size_t>(x.shape(0)),
        static_cast<libdet::u32>(x.shape(2))
    };
}

[[nodiscard]] libdet::rhf::DetRef state(const StateArray& x) {
    const auto view = states(x);
    if (view.n_dets != 1) throw std::invalid_argument("expected exactly one state");
    return view[0];
}

[[nodiscard]] std::span<const double> f64(const F64Vec& x) noexcept {
    return {x.data(), static_cast<std::size_t>(x.shape(0))};
}

[[nodiscard]] std::span<const double> f64(const F64Mat& x) noexcept {
    return {x.data(), static_cast<std::size_t>(x.shape(0) * x.shape(1))};
}

template <class F>
decltype(auto) no_gil(F&& f) {
    nb::gil_scoped_release release;
    return std::forward<F>(f)();
}

struct OptionalStates {
    explicit OptionalStates(nb::object obj) {
        if (!obj.is_none()) {
            array.emplace(nb::cast<StateArray>(obj));
            view.emplace(states(*array));
        }
    }

    [[nodiscard]] const libdet::rhf::DetBatchView* ptr() const noexcept {
        return view ? &*view : nullptr;
    }

    std::optional<StateArray> array;
    std::optional<libdet::rhf::DetBatchView> view;
};

struct Counts {
    std::span<const libdet::i64> data;
    std::size_t n_stream = 0;
};


[[nodiscard]] Counts counts(const I64Array& x, std::size_t n_ket) {
    if (x.ndim() != 1 && x.ndim() != 2) {
        throw std::invalid_argument("counts must have shape (N,) or (S, N)");
    }

    const std::size_t n_stream =
        x.ndim() == 1 ? 1u : static_cast<std::size_t>(x.shape(0));
    const std::size_t last = static_cast<std::size_t>(x.shape(x.ndim() - 1));

    if (last != n_ket) throw std::invalid_argument("counts last dimension must match kets");
    return {{x.data(), n_stream * n_ket}, n_stream};
}

template <class T>
auto own(std::vector<T>&& x) {
    auto* heap = new std::vector<T>(std::move(x));
    nb::capsule owner(heap, [](void* p) noexcept {
        delete static_cast<std::vector<T>*>(p);
    });
    return nb::ndarray<T, nb::numpy, nb::shape<-1>>(heap->data(), {heap->size()}, owner);
}

template <class T>
auto own(std::vector<T>&& x, std::size_t nrow, std::size_t ncol) {
    auto* heap = new std::vector<T>(std::move(x));
    nb::capsule owner(heap, [](void* p) noexcept {
        delete static_cast<std::vector<T>*>(p);
    });
    return nb::ndarray<T, nb::numpy, nb::shape<-1, -1>>(
        heap->data(), {nrow, ncol}, owner
    );
}

auto own_states(std::vector<std::uint64_t>&& x, libdet::u32 nword) {
    const std::size_t n_state = x.size() / libdet::word_pair_size(nword);
    auto* heap = new std::vector<std::uint64_t>(std::move(x));
    nb::capsule owner(heap, [](void* p) noexcept {
        delete static_cast<std::vector<std::uint64_t>*>(p);
    });
    return nb::ndarray<std::uint64_t, nb::numpy, nb::shape<-1, 2, -1>>(
        heap->data(), {n_state, std::size_t{2}, static_cast<std::size_t>(nword)}, owner
    );
}

template <class T>
auto view(std::span<const T> x) {
    return nb::ndarray<const T, nb::numpy, nb::shape<-1>, nb::c_contig, nb::device::cpu>(
        x.data(), {x.size()}
    );
}

template <class T>
auto view(const std::vector<T>& x) {
    return view(std::span<const T>(x));
}

template <class T>
auto view(const std::vector<T>& x, std::size_t nrow, std::size_t ncol) {
    return nb::ndarray<const T, nb::numpy, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>(
        x.data(), {nrow, ncol}
    );
}

auto view_states(std::span<const std::uint64_t> x, std::size_t n_state, libdet::u32 nword) {
    return nb::ndarray<const std::uint64_t, nb::numpy, nb::c_contig, nb::device::cpu>(
        x.data(), {n_state, std::size_t{2}, static_cast<std::size_t>(nword)}
    );
}

auto view_states(const std::vector<std::uint64_t>& x, std::size_t n_state, libdet::u32 nword) {
    return view_states(std::span<const std::uint64_t>(x), n_state, nword);
}

auto view_states(const std::vector<std::uint64_t>& x, libdet::u32 nword) {
    return view_states(x, x.size() / libdet::word_pair_size(nword), nword);
}

} // namespace

NB_MODULE(libdet, m) {
    m.doc() = "Electronic Hamiltonian primitives";

    nb::class_<libdet::Conns>(m, "Conns")
        .def_prop_ro("n_kets", [](const libdet::Conns& x) { return x.n_kets; })
        .def_prop_ro("n_streams", [](const libdet::Conns& x) { return x.n_streams; })
        .def_prop_ro("bra", [](const libdet::Conns& x) {
            return view_states(x.bra, x.nword);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("diag", [](const libdet::Conns& x) { return view(x.diag); }, nb::rv_policy::reference_internal)
        .def_prop_ro("ptr", [](const libdet::Conns& x) { return view(x.ptr); }, nb::rv_policy::reference_internal)
        .def_prop_ro("h", [](const libdet::Conns& x) { return view(x.h); }, nb::rv_policy::reference_internal)
        .def_prop_ro("degree", [](const libdet::Conns& x) { return view(x.degree); }, nb::rv_policy::reference_internal);


    nb::class_<libdet::LocalConn>(m, "LocalConn")
        .def_prop_ro("n_kets", [](const libdet::LocalConn& x) { return x.n_kets; })
        .def_prop_ro("bra", [](const libdet::LocalConn& x) {
            return view_states(x.bra, x.nword);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("diag", [](const libdet::LocalConn& x) { return view(x.diag); }, nb::rv_policy::reference_internal)
        .def_prop_ro("strong_ptr", [](const libdet::LocalConn& x) { return view(x.strong_ptr); }, nb::rv_policy::reference_internal)
        .def_prop_ro("strong_h", [](const libdet::LocalConn& x) { return view(x.strong_h); }, nb::rv_policy::reference_internal)
        .def_prop_ro("strong_degree", [](const libdet::LocalConn& x) { return view(x.strong_degree); }, nb::rv_policy::reference_internal)
        .def_prop_ro("weak_ptr", [](const libdet::LocalConn& x) { return view(x.weak_ptr); }, nb::rv_policy::reference_internal)
        .def_prop_ro("weak_h", [](const libdet::LocalConn& x) { return view(x.weak_h); }, nb::rv_policy::reference_internal)
        .def_prop_ro("weak_count", [](const libdet::LocalConn& x) { return view(x.weak_count); }, nb::rv_policy::reference_internal)
        .def_prop_ro("weak_degree", [](const libdet::LocalConn& x) { return view(x.weak_degree); }, nb::rv_policy::reference_internal);

    nb::class_<libdet::Projection>(m, "Projection")
        .def_prop_ro("bra", [](const libdet::Projection& x) {
            return view_states(x.bra, x.hpsi.size(), x.nword);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("hpsi", [](const libdet::Projection& x) { return view(x.hpsi); }, nb::rv_policy::reference_internal)
        .def_prop_ro("diag", [](const libdet::Projection& x) { return view(x.diag); }, nb::rv_policy::reference_internal);

    nb::class_<libdet::Projections>(m, "Projections")
        .def_prop_ro("n_streams", [](const libdet::Projections& x) { return x.n_streams; })
        .def_prop_ro("bra", [](const libdet::Projections& x) {
            return view_states(x.bra, x.nword);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("hpsi", [](const libdet::Projections& x) {
            const std::size_t n_bra = x.bra.size() / libdet::word_pair_size(x.nword);
            return view(x.hpsi, x.n_streams, n_bra);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("diag", [](const libdet::Projections& x) { return view(x.diag); }, nb::rv_policy::reference_internal);

    nb::class_<libdet::rhf::Hamiltonian>(m, "Hamiltonian")
        .def("__init__", [](libdet::rhf::Hamiltonian* self, const F64Mat& h1, const F64Vec& eri, double ecore) {
            if (h1.shape(0) != h1.shape(1)) throw std::invalid_argument("h1 must be square");
            const int norb = static_cast<int>(h1.shape(0));
            const auto h1v = f64(h1);
            const auto eriv = f64(eri);
            no_gil([&] {
                new (self) libdet::rhf::Hamiltonian(h1v, norb, eriv, ecore);
            });
        }, "h1"_a.noconvert(), "eri"_a.noconvert(), "ecore"_a = 0.0)

        .def("hij", [](const libdet::rhf::Hamiltonian& ham, const StateArray& bra, const StateArray& ket) {
            const auto b = state(bra);
            const auto k = state(ket);
            return no_gil([&] { return ham.hij(b, k); });
        }, "bra"_a.noconvert(), "ket"_a.noconvert())

        .def("diag", [](const libdet::rhf::Hamiltonian& ham, const StateArray& x) {
            const auto xv = states(x);
            return own(no_gil([&] { return ham.diag(xv); }));
        }, "x"_a.noconvert())

        .def("expand", [](
            const libdet::rhf::Hamiltonian& ham,
            const StateArray& kets,
            double eps,
            nb::object scale,
            nb::object exclude
        ) {
            const auto kv = states(kets);
            std::optional<F64Vec> scale_array;
            std::span<const double> scale_view;
            if (!scale.is_none()) {
                scale_array.emplace(nb::cast<F64Vec>(scale));
                scale_view = f64(*scale_array);
            }
            const OptionalStates exclude_v(exclude);
            return own_states(
                no_gil([&] { return ham.expand(kv, eps, scale_view, exclude_v.ptr()); }),
                kv.nword
            );
        }, "kets"_a.noconvert(), "eps"_a, "scale"_a = nb::none(), "exclude"_a = nb::none())

        .def("project", [](
            const libdet::rhf::Hamiltonian& ham,
            nb::object bras,
            const StateArray& kets,
            const F64Vec& scale,
            double eps,
            nb::object exclude
        ) {
            const auto kv = states(kets);
            const auto sv = f64(scale);
            if (bras.is_none()) {
                const OptionalStates exclude_v(exclude);
                return no_gil([&] { return ham.project(kv, sv, eps, exclude_v.ptr()); });
            }

            if (!exclude.is_none()) throw std::invalid_argument("exclude is only valid when bras is None");

            const StateArray ba = nb::cast<StateArray>(bras);
            const auto bv = states(ba);
            return no_gil([&] { return ham.project(bv, kv, sv, eps); });
        }, "bras"_a.none(), "kets"_a.noconvert(), "scale"_a.noconvert(), "eps"_a = 0.0, "exclude"_a = nb::none())

        .def("conn", [](
            const libdet::rhf::Hamiltonian& ham,
            const StateArray& kets,
            double eps
        ) {
            const auto kv = states(kets);
            return no_gil([&] { return ham.conn(kv, eps); });
        }, "kets"_a.noconvert(), "eps"_a = 0.0)

        .def("sample_conn", [](
            const libdet::rhf::Hamiltonian& ham,
            const StateArray& kets,
            const I64Array& counts_arr,
            double eps1,
            double eps2,
            std::uint64_t seed
        ) {
            const auto kv = states(kets);
            const auto cv = counts(counts_arr, kv.n_dets);
            return no_gil([&] {
                return ham.sample_conn(
                    kv,
                    cv.data,
                    cv.n_stream,
                    eps1,
                    eps2,
                    seed
                );
            });
        }, "kets"_a.noconvert(), "counts"_a.noconvert(), "eps1"_a, "eps2"_a = 0.0, "seed"_a = std::uint64_t{0})


        .def("local_conn", [](
            const libdet::rhf::Hamiltonian& ham,
            const StateArray& kets,
            double eps1,
            double eps2,
            const I64Array& counts_arr,
            std::uint64_t seed
        ) {
            const auto kv = states(kets);
            const auto cv = counts(counts_arr, kv.n_dets);
            if (cv.n_stream != 1u) {
                throw std::invalid_argument("local_conn: counts must have shape (N,)");
            }
            return no_gil([&] {
                return ham.local_conn(kv, eps1, eps2, cv.data, seed);
            });
        }, "kets"_a.noconvert(), "eps1"_a, "eps2"_a, "counts"_a.noconvert(), "seed"_a = std::uint64_t{0})

        .def("sample_project", [](
            const libdet::rhf::Hamiltonian& ham,
            const StateArray& kets,
            const F64Vec& scale,
            const I64Array& counts_arr,
            double eps1,
            double eps2,
            nb::object exclude,
            std::uint64_t seed
        ) {
            const auto kv = states(kets);
            const auto sv = f64(scale);
            const auto cv = counts(counts_arr, kv.n_dets);
            const OptionalStates exclude_v(exclude);
            return no_gil([&] {
                return ham.sample_project(
                    kv,
                    sv,
                    cv.data,
                    cv.n_stream,
                    eps1,
                    eps2,
                    exclude_v.ptr(),
                    seed
                );
            });
        }, "kets"_a.noconvert(), "scale"_a.noconvert(), "counts"_a.noconvert(), "eps1"_a, "eps2"_a = 0.0, "exclude"_a = nb::none(), "seed"_a = std::uint64_t{0})

        .def("matrix", [](
            const libdet::rhf::Hamiltonian& ham,
            const StateArray& bras,
            const StateArray& kets
        ) {
            const auto bv = states(bras);
            const auto kv = states(kets);
            auto out = no_gil([&] { return ham.matrix(bv, kv); });
            return nb::make_tuple(
                own(std::move(out.indptr)),
                own(std::move(out.indices)),
                own(std::move(out.data)),
                nb::make_tuple(out.n_bra, out.n_ket)
            );
        }, "bras"_a.noconvert(), "kets"_a.noconvert())

        .def("matvec", [](
            const libdet::rhf::Hamiltonian& ham,
            const StateArray& bras,
            const StateArray& kets,
            const F64Vec& x
        ) {
            const auto bv = states(bras);
            const auto kv = states(kets);
            const auto xv = f64(x);
            return own(no_gil([&] { return ham.matvec(bv, kv, xv); }));
        }, "bras"_a.noconvert(), "kets"_a.noconvert(), "x"_a.noconvert())

        .def("matmat", [](
            const libdet::rhf::Hamiltonian& ham,
            const StateArray& bras,
            const StateArray& kets,
            const F64Mat& x
        ) {
            const auto bv = states(bras);
            const auto kv = states(kets);
            const auto xv = f64(x);
            const std::size_t nrhs = static_cast<std::size_t>(x.shape(1));
            return own(
                no_gil([&] { return ham.matmat(bv, kv, xv, nrhs); }),
                bv.n_dets,
                nrhs
            );
        }, "bras"_a.noconvert(), "kets"_a.noconvert(), "x"_a.noconvert());
}

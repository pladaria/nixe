/* This file is part of the dynarmic project.
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace Dynarmic::Backend::X64 {

// Deferred emission captures are bounded by the emitter's own call sites. Keep
// them in reusable stable-vector nodes instead of allocating each closure.
// The nodes need not move, and no callback outlives its block's EmitContext.
class DeferredEmit final {
public:
    template<typename F>
    explicit DeferredEmit(F&& function)
            : invoke([](void* data) { (*std::launder(static_cast<Function<F>*>(data)))(); })
            , destroy([](void* data) { std::destroy_at(std::launder(static_cast<Function<F>*>(data))); }) {
        static_assert(sizeof(Function<F>) <= sizeof(storage));
        static_assert(alignof(Function<F>) <= alignof(std::max_align_t));
        std::construct_at(reinterpret_cast<Function<F>*>(storage), std::forward<F>(function));
    }

    ~DeferredEmit() {
        destroy(storage);
    }

    DeferredEmit(const DeferredEmit&) = delete;
    DeferredEmit& operator=(const DeferredEmit&) = delete;
    DeferredEmit(DeferredEmit&&) = delete;
    DeferredEmit& operator=(DeferredEmit&&) = delete;

    void operator()() {
        invoke(storage);
    }

private:
    template<typename F>
    using Function = std::remove_cvref_t<F>;

    void (*invoke)(void*);
    void (*destroy)(void*);
    alignas(std::max_align_t) std::byte storage[128];
};

}  // namespace Dynarmic::Backend::X64

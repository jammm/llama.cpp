#include "loom-catalog/ggml-hrx-loom-catalog-runtime-internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

static ggml_backend_hrx_loom_catalog_entry make_entry(const unsigned char * source_data, size_t source_size) {
    return {
        /* .id                   = */ "add_f32",
        /* .op                   = */ "GGML_OP_ADD",
        /* .target               = */ "gfx1100",
        /* .source_name          = */ "sources/add_f32.loom",
        /* .source_data          = */ source_data,
        /* .source_size          = */ source_size,
        /* .source_format        = */ "loom-text",
        /* .symbol               = */ "hrx_add_f32",
        /* .workgroup_size       = */ {256, 1, 1},
        /* .binding_count        = */ 3,
        /* .parameter_count      = */ 4,
        /* .constant_byte_length = */ 8,
    };
}

static void set_binding(ggml_backend_hrx_loom_config_binding & binding, const char * name, const char * value) {
    std::memset(&binding, 0, sizeof(binding));
    std::strncpy(binding.name, name, sizeof(binding.name) - 1);
    std::strncpy(binding.value, value, sizeof(binding.value) - 1);
    binding.type = "i64";
}

static bool reference_is_metadata_op(const ggml_tensor * tensor) {
    if (!tensor) {
        return false;
    }
    switch (tensor->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

static bool reference_follows_metadata_path(
        const ggml_tensor * value,
        const ggml_tensor * source) {
    if (value == source) {
        return true;
    }
    while (value && reference_is_metadata_op(value)) {
        value = value->src[0] ? value->src[0] : value->view_src;
        if (value == source) {
            return true;
        }
    }
    return false;
}

static bool reference_is_graph_output(
        const ggml_backend_hrx_loom_op_request * request,
        const ggml_tensor * tensor) {
    if (!request || !request->cgraph || !tensor) {
        return true;
    }
    if ((tensor->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return true;
    }
    for (int i = 0; i < request->cgraph->n_nodes; ++i) {
        const ggml_tensor * node = request->cgraph->nodes[i];
        if (node && (node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 &&
            reference_follows_metadata_path(node, tensor)) {
            return true;
        }
    }
    return false;
}

static bool reference_node_is_listed(
        int node_index,
        const int * node_indices,
        int node_count) {
    for (int i = 0; i < node_count; ++i) {
        if (node_indices[i] == node_index) {
            return true;
        }
    }
    return false;
}

static bool reference_is_transient(
        const ggml_backend_hrx_loom_op_request * request,
        const ggml_tensor * tensor,
        const int * consumed_node_indices,
        int consumed_node_count) {
    if (!request || !request->cgraph || !tensor ||
        !consumed_node_indices || consumed_node_count <= 0 ||
        reference_is_graph_output(request, tensor)) {
        return false;
    }
    bool consumed_inside = false;
    for (int i = 0; i < request->cgraph->n_nodes; ++i) {
        const ggml_tensor * consumer = request->cgraph->nodes[i];
        if (!consumer || consumer == tensor ||
            reference_is_metadata_op(consumer) ||
            ggml_nelements(consumer) == 0) {
            continue;
        }
        bool consumes_tensor = false;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            consumes_tensor = consumes_tensor ||
                reference_follows_metadata_path(
                    consumer->src[j], tensor);
        }
        if (!consumes_tensor) {
            continue;
        }
        if (!reference_node_is_listed(
                i, consumed_node_indices, consumed_node_count)) {
            return false;
        }
        consumed_inside = true;
    }
    return consumed_inside;
}

static bool reference_consumers_through_view(
        const ggml_backend_hrx_loom_op_request * request,
        const ggml_tensor * owner,
        const ggml_tensor * view) {
    if (!request || !request->cgraph || !owner || !view ||
        owner == view ||
        !reference_follows_metadata_path(view, owner) ||
        (owner->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }
    for (int i = 0; i < request->cgraph->n_nodes; ++i) {
        const ggml_tensor * node = request->cgraph->nodes[i];
        if (!node) {
            continue;
        }
        if ((node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 &&
            reference_follows_metadata_path(node, owner) &&
            !reference_follows_metadata_path(node, view)) {
            return false;
        }
        if (reference_is_metadata_op(node) ||
            ggml_nelements(node) == 0) {
            continue;
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * input = node->src[j];
            if (reference_follows_metadata_path(input, owner) &&
                !reference_follows_metadata_path(input, view)) {
                return false;
            }
        }
    }
    return true;
}

struct graph_fact_fixture {
    std::vector<ggml_tensor>   tensors;
    std::vector<ggml_tensor *> nodes;
    std::vector<ggml_tensor *> visited_keys;
    std::vector<ggml_bitset_t> visited_used;
    ggml_cgraph                graph = {};

    explicit graph_fact_fixture(size_t count) :
        tensors(count),
        nodes(count),
        visited_keys(count * 4 + 1, nullptr),
        visited_used(
            ggml_bitset_size(visited_keys.size()),
            0) {
        for (size_t i = 0; i < count; ++i) {
            tensors[i]       = {};
            tensors[i].type  = GGML_TYPE_F32;
            tensors[i].op    = GGML_OP_ADD;
            tensors[i].ne[0] = 1;
            tensors[i].ne[1] = 1;
            tensors[i].ne[2] = 1;
            tensors[i].ne[3] = 1;
            nodes[i] = &tensors[i];
        }
        graph.n_nodes = static_cast<int>(nodes.size());
        graph.nodes   = nodes.data();
        graph.uid     = 41;
        graph.visited_hash_set = {
            visited_keys.size(),
            visited_used.data(),
            visited_keys.data(),
        };
        for (ggml_tensor * tensor : nodes) {
            ggml_hash_insert(&graph.visited_hash_set, tensor);
        }
    }

    ggml_backend_hrx_loom_op_request request(
            int node_index,
            uint64_t execution_epoch) {
        return {
            /* .op                    = */ nodes[node_index],
            /* .cgraph                = */ &graph,
            /* .node_index            = */ node_index,
            /* .stream                = */ nullptr,
            /* .bind_tensor           = */ nullptr,
            /* .bind_tensor_user_data = */ nullptr,
            /* .storage_layout        = */ nullptr,
            /* .storage_layout_user_data = */ nullptr,
            /* .invocation_context    = */ nullptr,
            /* .execution_epoch       = */ execution_epoch,
        };
    }
};

static ggml_backend_hrx_loom_kernel_plan make_plan(const ggml_backend_hrx_loom_catalog_entry * entry,
                                                   const char *                                 nelements,
                                                   const char *                                 workgroup_size_x) {
    ggml_backend_hrx_loom_kernel_plan plan = {};
    plan.entry                             = entry;
    set_binding(plan.config_bindings[0], "nelements", nelements);
    set_binding(plan.config_bindings[1], "workgroup_size_x", workgroup_size_x);
    plan.config_binding_count = 2;
    return plan;
}

static std::string cache_key(const ggml_backend_hrx_loom_kernel_plan & plan) {
    std::string key = ggml_backend_hrx_loom_cache_key("gfx1100", &plan);
    if (key.empty()) {
        std::fprintf(stderr, "failed to format cache key\n");
        std::abort();
    }
    return key;
}

static void expect_true(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::abort();
    }
}

struct fake_binding_context {
    hrx_buffer_t        buffer      = nullptr;
    const ggml_tensor * fail_tensor = nullptr;
};

static bool fake_bind_tensor(
        void * user_data,
        const ggml_tensor * tensor,
        hrx_buffer_ref_t * out_ref) {
    auto * context = static_cast<fake_binding_context *>(user_data);
    if (!context || !tensor || !out_ref ||
        tensor == context->fail_tensor) {
        return false;
    }
    *out_ref = {
        /* .buffer = */ context->buffer,
        /* .offset = */ reinterpret_cast<uintptr_t>(tensor->data),
        /* .length = */ 64,
    };
    return true;
}

static void test_graph_fact_cache() {
    graph_fact_fixture fixture(6);
    ggml_tensor * owner   = &fixture.tensors[0];
    ggml_tensor * reshape = &fixture.tensors[1];
    ggml_tensor * view    = &fixture.tensors[2];
    ggml_tensor * inside  = &fixture.tensors[3];
    ggml_tensor * outside = &fixture.tensors[4];
    ggml_tensor * unused  = &fixture.tensors[5];

    reshape->op     = GGML_OP_RESHAPE;
    reshape->src[0] = owner;
    view->op        = GGML_OP_VIEW;
    view->src[0]    = reshape;
    view->view_src  = reshape;
    inside->op      = GGML_OP_MUL;
    inside->src[0]  = view;
    inside->src[1]  = view;
    outside->op     = GGML_OP_ADD;
    outside->src[0] = owner;
    unused->op      = GGML_OP_ADD;

    auto request = fixture.request(0, 7);
    const int inside_only[] = {3};
    const int both_consumers[] = {3, 4};
    for (const auto & check : std::vector<std::pair<const ggml_tensor *, std::vector<int>>>{
             {owner, {3}},
             {owner, {3, 4}},
             {reshape, {3}},
             {view, {3}},
             {unused, {3}},
         }) {
        const bool expected = reference_is_transient(
            &request,
            check.first,
            check.second.data(),
            static_cast<int>(check.second.size()));
        const bool actual =
            ggml_backend_hrx_loom_tensor_is_transient_at_indices_cached(
                &request,
                check.first,
                check.second.data(),
                static_cast<int>(check.second.size()));
        expect_true(actual == expected,
                    "cached transient facts must match the graph-scan oracle");
    }
    expect_true(!ggml_backend_hrx_loom_tensor_is_transient_at_indices_cached(
                    &request, owner, inside_only, 1),
                "a consumer bypassing a metadata view must keep its owner live");
    expect_true(ggml_backend_hrx_loom_tensor_is_transient_at_indices_cached(
                    &request, owner, both_consumers, 2),
                "all unique consumers inside a fusion must make an owner transient");

    expect_true(
        ggml_backend_hrx_loom_tensor_consumers_through_view_cached(
            &request, owner, view) ==
        reference_consumers_through_view(&request, owner, view),
        "a direct owner consumer must fail cached through-view safety");
    expect_true(
        ggml_backend_hrx_loom_tensor_consumers_through_view_cached(
            &request, reshape, view) ==
        reference_consumers_through_view(&request, reshape, view),
        "duplicate source slots through one view must preserve through-view safety");
    expect_true(
        ggml_backend_hrx_loom_tensor_consumers_through_view_cached(
            &request, reshape, view),
        "all observations below one metadata view must be accepted");

    ggml_backend_hrx_loom_graph_fact_cache cache;
    const auto * first = cache.resolve(&request);
    expect_true(first && first->uid == fixture.graph.uid &&
                    first->execution_epoch == request.execution_epoch,
                "the first graph-fact lookup must build a valid cache entry");
    request.execution_epoch = 8;
    expect_true(cache.resolve(&request) == first,
                "nonzero graph UIDs must reuse facts across execution epochs");

    fixture.graph.uid = 42;
    const auto * changed_uid = cache.resolve(&request);
    expect_true(changed_uid && changed_uid->uid == 42 &&
                    cache.graphs.size() == 1,
                "a graph UID change must replace stale facts");

    std::vector<ggml_tensor *> replacement_nodes = fixture.nodes;
    fixture.graph.nodes = replacement_nodes.data();
    const auto * changed_storage = cache.resolve(&request);
    expect_true(changed_storage &&
                    changed_storage->nodes == replacement_nodes.data() &&
                    cache.graphs.size() == 1,
                "node-array identity changes must replace stale facts");
    fixture.graph.nodes = fixture.nodes.data();

    fixture.graph.uid = 0;
    request.execution_epoch = 9;
    const auto * uid_zero_epoch_9 = cache.resolve(&request);
    expect_true(uid_zero_epoch_9 &&
                    uid_zero_epoch_9->execution_epoch == 9,
                "UID-zero facts must build for the current execution epoch");
    expect_true(cache.resolve(&request) == uid_zero_epoch_9,
                "UID-zero facts must be reusable inside one execution epoch");
    request.execution_epoch = 10;
    const auto * uid_zero_epoch_10 = cache.resolve(&request);
    expect_true(uid_zero_epoch_10 &&
                    uid_zero_epoch_10->execution_epoch == 10 &&
                    cache.graphs.size() == 1,
                "UID-zero facts must rebuild at the next execution epoch");

    fixture.graph.uid = 43;
    request.execution_epoch = 11;
    view->flags |= GGML_TENSOR_FLAG_OUTPUT;
    expect_true(
        !ggml_backend_hrx_loom_tensor_is_transient_at_indices_cached(
            &request, owner, both_consumers, 2) &&
        !reference_is_transient(
            &request, owner, both_consumers, 2),
        "an output reached through metadata must keep every ancestor live");
    expect_true(
        ggml_backend_hrx_loom_tensor_consumers_through_view_cached(
            &request, reshape, view) ==
        reference_consumers_through_view(&request, reshape, view),
        "an output reached through the selected view must remain bounded");
    owner->flags |= GGML_TENSOR_FLAG_OUTPUT;
    expect_true(
        !ggml_backend_hrx_loom_tensor_consumers_through_view_cached(
            &request, owner, view),
        "a directly exported owner must never be hidden by a view");
    owner->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
    view->flags  &= ~GGML_TENSOR_FLAG_OUTPUT;

    fixture.graph.uid = 44;
    outside->ne[0] = 0;
    expect_true(
        ggml_backend_hrx_loom_tensor_is_transient_at_indices_cached(
            &request, owner, inside_only, 1) ==
        reference_is_transient(
            &request, owner, inside_only, 1),
        "zero-element consumers must be ignored exactly as before");
    expect_true(
        ggml_backend_hrx_loom_tensor_consumers_through_view_cached(
            &request, owner, view) ==
        reference_consumers_through_view(&request, owner, view),
        "zero-element consumers must not create an observation bypass");

    fixture.graph.uid = 45;
    unused->op        = GGML_OP_VIEW;
    unused->src[0]    = owner;
    unused->view_src  = owner;
    unused->flags    |= GGML_TENSOR_FLAG_OUTPUT;
    expect_true(
        !ggml_backend_hrx_loom_tensor_consumers_through_view_cached(
            &request, owner, view) &&
        !reference_consumers_through_view(&request, owner, view),
        "an output through a sibling metadata view must reject bounded ownership");

    graph_fact_fixture off_hash_fixture(3);
    ggml_tensor off_hash_owner = {};
    off_hash_owner.type  = GGML_TYPE_F32;
    off_hash_owner.op    = GGML_OP_NONE;
    off_hash_owner.ne[0] = 1;
    off_hash_owner.ne[1] = 1;
    off_hash_owner.ne[2] = 1;
    off_hash_owner.ne[3] = 1;
    ggml_tensor * off_hash_view     = &off_hash_fixture.tensors[0];
    ggml_tensor * through_consumer  = &off_hash_fixture.tensors[1];
    ggml_tensor * direct_consumer   = &off_hash_fixture.tensors[2];
    off_hash_view->op       = GGML_OP_VIEW;
    off_hash_view->src[0]   = &off_hash_owner;
    off_hash_view->view_src = &off_hash_owner;
    through_consumer->op     = GGML_OP_MUL;
    through_consumer->src[0] = off_hash_view;
    direct_consumer->op      = GGML_OP_ADD;
    direct_consumer->src[0]  = &off_hash_owner;

    auto off_hash_request = off_hash_fixture.request(0, 1);
    ggml_backend_hrx_loom_graph_fact_cache off_hash_cache;
    const auto * off_hash_facts = off_hash_cache.resolve(&off_hash_request);
    expect_true(
        off_hash_facts && !off_hash_facts->find(&off_hash_owner) &&
            off_hash_facts->find(off_hash_view),
        "an off-hash leaf must not invalidate facts for indexed tensors");
    const int off_hash_consumers[] = {1, 2};
    expect_true(
        ggml_backend_hrx_loom_tensor_is_transient_at_indices_cached(
            &off_hash_request,
            &off_hash_owner,
            off_hash_consumers,
            2) ==
        reference_is_transient(
            &off_hash_request,
            &off_hash_owner,
            off_hash_consumers,
            2),
        "an off-hash transient query must retain graph-scan semantics");
    const int through_consumer_only[] = {1};
    expect_true(
        ggml_backend_hrx_loom_tensor_is_transient_at_indices_cached(
            &off_hash_request,
            off_hash_view,
            through_consumer_only,
            1) ==
        reference_is_transient(
            &off_hash_request,
            off_hash_view,
            through_consumer_only,
            1),
        "indexed descendants of an off-hash leaf must use valid cached facts");
    expect_true(
        ggml_backend_hrx_loom_tensor_consumers_through_view_cached(
            &off_hash_request,
            &off_hash_owner,
            off_hash_view) ==
        reference_consumers_through_view(
            &off_hash_request,
            &off_hash_owner,
            off_hash_view),
        "an off-hash owner query must retain graph-scan bypass detection");

    direct_consumer->ne[0] = 0;
    ++off_hash_fixture.graph.uid;
    expect_true(
        off_hash_cache.resolve(&off_hash_request) &&
        ggml_backend_hrx_loom_tensor_consumers_through_view_cached(
            &off_hash_request,
            &off_hash_owner,
            off_hash_view) &&
        reference_consumers_through_view(
            &off_hash_request,
            &off_hash_owner,
            off_hash_view),
        "off-hash ownership through one indexed view must remain admissible");

    graph_fact_fixture overflow(10);
    ggml_tensor * overflow_owner = &overflow.tensors[0];
    for (int i = 1; i < overflow.graph.n_nodes; ++i) {
        overflow.tensors[i].op     = GGML_OP_MUL;
        overflow.tensors[i].src[0] = overflow_owner;
    }
    auto overflow_request = overflow.request(0, 1);
    const int first_eight[] = {1, 2, 3, 4, 5, 6, 7, 8};
    overflow.tensors[9].ne[0] = 0;
    expect_true(
        ggml_backend_hrx_loom_tensor_is_transient_at_indices_cached(
            &overflow_request, overflow_owner, first_eight, 8) &&
        reference_is_transient(
            &overflow_request, overflow_owner, first_eight, 8),
        "exactly eight consumers must fit the bounded fact representation");
    overflow.tensors[9].ne[0] = 1;
    ++overflow.graph.uid;
    expect_true(
        !ggml_backend_hrx_loom_tensor_is_transient_at_indices_cached(
            &overflow_request, overflow_owner, first_eight, 8) &&
        !reference_is_transient(
            &overflow_request, overflow_owner, first_eight, 8),
        "a ninth consumer must saturate facts and reject an eight-node fusion");
}

int main() {
    test_graph_fact_cache();

    static const unsigned char source_a[] = "kernel.def @a() { }";
    static const unsigned char source_b[] = "kernel.def @b() { }";

    const ggml_backend_hrx_loom_catalog_entry entry_a  = make_entry(source_a, sizeof(source_a) - 1);
    const ggml_backend_hrx_loom_catalog_entry entry_b  = make_entry(source_b, sizeof(source_b) - 1);
    const ggml_backend_hrx_loom_kernel_plan plan_a0 = make_plan(&entry_a, "257", "256");
    const ggml_backend_hrx_loom_kernel_plan plan_a1 = make_plan(&entry_a, "257", "256");
    const ggml_backend_hrx_loom_kernel_plan plan_b0 = make_plan(&entry_a, "2048", "256");
    const ggml_backend_hrx_loom_kernel_plan plan_c0 = make_plan(&entry_b, "257", "256");

    expect_true(cache_key(plan_a0) == cache_key(plan_a1), "identical plans must produce identical cache keys");
    expect_true(cache_key(plan_a0) != cache_key(plan_b0), "config changes must produce different cache keys");
    expect_true(cache_key(plan_a0) != cache_key(plan_c0), "source changes must produce different cache keys");
    expect_true(ggml_backend_hrx_loom_cache_key("gfx1100", nullptr).empty(), "null plan must produce empty cache key");

    ggml_tensor cached_op = {};
    ggml_tensor replacement_op = {};
    ggml_tensor * graph_nodes[] = {&cached_op};
    ggml_cgraph cached_graph = {};
    cached_graph.n_nodes = 1;
    cached_graph.nodes   = graph_nodes;
    cached_graph.uid     = 41;
    const auto * catalog_identity =
        reinterpret_cast<const ggml_backend_hrx_loom_catalog *>(
            0x1000);
    const auto * other_catalog_identity =
        reinterpret_cast<const ggml_backend_hrx_loom_catalog *>(
            0x2000);

    ggml_backend_hrx_loom_kernel_plan resolved_plan = plan_a0;
    resolved_plan.loaded_route_catalog = catalog_identity;
    resolved_plan.loaded_route =
        reinterpret_cast<ggml_backend_hrx_loaded_loom_route *>(0x3000);
    const ggml_backend_hrx_loom_kernel_plan queued_plan = resolved_plan;
    expect_true(
        queued_plan.loaded_route_catalog == catalog_identity &&
            queued_plan.loaded_route == resolved_plan.loaded_route,
        "a deferred plan copy must retain its resolved loaded route");
    expect_true(
        queued_plan.loaded_route_catalog != other_catalog_identity,
        "a loaded route must remain scoped to its owning catalog");

    ggml_backend_hrx_loom_op_request cached_request = {
        /* .op                    = */ &cached_op,
        /* .cgraph                = */ &cached_graph,
        /* .node_index            = */ 0,
        /* .stream                = */ nullptr,
        /* .bind_tensor           = */ nullptr,
        /* .bind_tensor_user_data = */ nullptr,
        /* .storage_layout        = */ nullptr,
        /* .storage_layout_user_data = */ nullptr,
        /* .invocation_context    = */ nullptr,
        /* .execution_epoch       = */ 7,
    };
    ggml_backend_hrx_loom_execution_plan cached_plan = {};
    cached_plan.main.entry                    = &entry_a;
    cached_plan.main.binding_count            = 1;
    cached_plan.main.bindings[0].buffer       =
        reinterpret_cast<hrx_buffer_t>(0x3000);
    cached_plan.main.bindings[0].offset       = 64;
    cached_plan.main.bindings[0].length       = 256;
    cached_plan.consumed_node_count           = 1;
    cached_plan.consumed_node_indices[0]      = 0;
    cached_plan.dispatch_owner_node_index     = 0;

    ggml_backend_hrx_loom_resolved_plan_cache resolved_plans;
    int prepare_count = 0;
    expect_true(
        resolved_plans.find(
            catalog_identity,
            &cached_request) == nullptr,
        "the first graph execution must miss the resolved-plan cache");
    ++prepare_count;
    expect_true(
        resolved_plans.publish(
            catalog_identity,
            &cached_request,
            "cached-route",
            cached_plan),
        "a fully prepared plan must publish");
    const auto * first_cached =
        resolved_plans.find(
            catalog_identity,
            &cached_request);
    expect_true(
        first_cached && first_cached->plan &&
            prepare_count == 1,
        "the repeated graph execution must bypass preparation");
    cached_plan.main.bindings[0].offset = 128;
    expect_true(
        first_cached->plan->main.bindings[0].offset == 64,
        "the resolved-plan cache must own an immutable plan copy");

    cached_request.execution_epoch = 8;
    expect_true(
        resolved_plans.find(
            catalog_identity,
            &cached_request) == first_cached,
        "execution epochs must not invalidate graph-identity plans");
    expect_true(
        resolved_plans.find(
            other_catalog_identity,
            &cached_request) == nullptr,
        "catalog identity changes must invalidate resolved plans");

    cached_graph.uid = 42;
    expect_true(
        resolved_plans.find(
            catalog_identity,
            &cached_request) == nullptr,
        "graph UID changes must invalidate resolved plans");
    cached_graph.uid = 41;
    cached_graph.n_nodes = 2;
    expect_true(
        resolved_plans.find(
            catalog_identity,
            &cached_request) == nullptr,
        "graph node-count changes must invalidate resolved plans");
    cached_graph.n_nodes = 1;

    graph_nodes[0] = &replacement_op;
    expect_true(
        resolved_plans.find(
            catalog_identity,
            &cached_request) == nullptr,
        "graph node replacement must invalidate resolved plans");
    cached_request.op = &replacement_op;
    expect_true(
        resolved_plans.find(
            catalog_identity,
            &cached_request) == nullptr,
        "a replacement operation must not inherit the old plan");
    cached_request.op = &cached_op;
    graph_nodes[0]    = &cached_op;

    ggml_cgraph other_graph = cached_graph;
    ggml_backend_hrx_loom_op_request other_graph_request =
        cached_request;
    other_graph_request.cgraph = &other_graph;
    expect_true(
        resolved_plans.find(
            catalog_identity,
            &other_graph_request) == nullptr,
        "graph pointer changes must invalidate resolved plans");

    ggml_backend_hrx_loom_execution_plan invalid_plan = cached_plan;
    invalid_plan.main.entry = nullptr;
    expect_true(
        !resolved_plans.publish(
            catalog_identity,
            &cached_request,
            "invalid-route",
            invalid_plan) &&
            resolved_plans.find(
                catalog_identity,
                &cached_request) == first_cached,
        "rejected preparation must preserve the prior valid plan");
    cached_graph.uid = 0;
    expect_true(
        resolved_plans.find(
            catalog_identity,
            &cached_request) == nullptr &&
            !resolved_plans.publish(
                catalog_identity,
                &cached_request,
                "uid-zero-route",
                cached_plan),
        "UID-zero graphs must remain uncached");
    cached_graph.uid = 41;

    const hrx_buffer_t scratch_buffer = reinterpret_cast<hrx_buffer_t>(0x1000);
    ggml_backend_hrx_loom_scratch_plan ordinary_scratch = {
        /* .name             = */ "mmid",
        /* .bytes            = */ 2097152,
        /* .minimum_capacity = */ 0,
    };
    ggml_backend_hrx_loom_scratch_plan graph_scratch = {
        /* .name             = */ "mmid",
        /* .bytes            = */ 11026432,
        /* .minimum_capacity = */ 11026432,
    };
    ggml_backend_hrx_loom_prepass_plan ordinary_prepass = {};
    ordinary_prepass.cache_scratch_index = 1;
    ordinary_prepass.cache_region_length = ordinary_scratch.bytes;
    ordinary_prepass.kernel.binding_scratch_index[1] = 1;
    ordinary_prepass.kernel.binding_scratch_length[1] = ordinary_scratch.bytes;
    ordinary_prepass.kernel.binding_count = 2;
    ggml_backend_hrx_loom_prepass_plan graph_prepass = ordinary_prepass;
    graph_prepass.kernel.binding_scratch_length[1] = graph_scratch.bytes;

    hrx_buffer_ref_t ordinary_region = {};
    hrx_buffer_ref_t graph_region = {};
    expect_true(ggml_backend_hrx_loom_resolve_cache_region(
                    ordinary_scratch, ordinary_prepass, scratch_buffer, &ordinary_region),
                "ordinary MMID cache region must resolve");
    expect_true(ggml_backend_hrx_loom_resolve_cache_region(
                    graph_scratch, graph_prepass, scratch_buffer, &graph_region),
                "graph MMID cache region must resolve");
    expect_true(ordinary_region.buffer == graph_region.buffer &&
                    ordinary_region.offset == graph_region.offset &&
                    ordinary_region.length == graph_region.length,
                "equal produced regions must not depend on scratch-plan capacity");
    expect_true(ggml_backend_hrx_loom_cache_region_is_bound(graph_prepass),
                "explicit cache region must fit its producer binding");

    ggml_backend_hrx_loom_scratch_cache_key ordinary_key = {
        /* .source_owner = */ reinterpret_cast<const ggml_tensor *>(0x2000),
        /* .source       = */ {reinterpret_cast<hrx_buffer_t>(0x3000), 64, 8192},
        /* .region       = */ ordinary_region,
        /* .epoch        = */ 7,
        /* .artifact     = */ "mmid_table",
    };
    ggml_backend_hrx_loom_scratch_cache_key graph_key = ordinary_key;
    graph_key.region = graph_region;
    expect_true(ggml_backend_hrx_loom_scratch_cache_key_matches(ordinary_key, graph_key),
                "equal source, artifact, and produced region must hit");
    graph_key.region.length += 256;
    expect_true(!ggml_backend_hrx_loom_scratch_cache_key_matches(ordinary_key, graph_key),
                "different produced-region lengths must miss");
    graph_key = ordinary_key;
    graph_key.source.offset += 4;
    expect_true(!ggml_backend_hrx_loom_scratch_cache_key_matches(ordinary_key, graph_key),
                "different source regions must miss");
    graph_key = ordinary_key;
    graph_key.artifact = "different_table";
    expect_true(!ggml_backend_hrx_loom_scratch_cache_key_matches(ordinary_key, graph_key),
                "different artifacts must miss");

    graph_prepass.cache_region_offset = graph_scratch.bytes;
    expect_true(!ggml_backend_hrx_loom_cache_region_is_bound(graph_prepass),
                "cache regions outside producer bindings must be rejected");

    const hrx_buffer_t shared_buffer =
        reinterpret_cast<hrx_buffer_t>(0x4000);
    const hrx_buffer_ref_t first_region = {
        /* .buffer = */ shared_buffer,
        /* .offset = */ 64,
        /* .length = */ 128,
    };
    const hrx_buffer_ref_t touching_region = {
        /* .buffer = */ shared_buffer,
        /* .offset = */ 192,
        /* .length = */ 64,
    };
    const hrx_buffer_ref_t overlapping_region = {
        /* .buffer = */ shared_buffer,
        /* .offset = */ 128,
        /* .length = */ 128,
    };
    expect_true(!ggml_backend_hrx_loom_deferred_refs_overlap(
                    first_region,
                    touching_region),
                "touching deferred buffer regions must remain disjoint");
    expect_true(ggml_backend_hrx_loom_deferred_refs_overlap(
                    first_region,
                    overlapping_region),
                "overlapping deferred buffer regions must be rejected");
    expect_true(!ggml_backend_hrx_loom_deferred_write_conflicts(
                    14, 14, overlapping_region, first_region),
                "an intended producer write at the read ready-index must be allowed");
    expect_true(ggml_backend_hrx_loom_deferred_write_conflicts(
                    15, 14, overlapping_region, first_region),
                "a later overlapping writer must invalidate a deferred read");

    ggml_tensor lifetime_nodes[6] = {};
    ggml_tensor intervening_input = {};
    for (auto & node : lifetime_nodes) {
        node.op    = GGML_OP_ADD;
        node.ne[0] = 16;
        node.ne[1] = 1;
        node.ne[2] = 1;
        node.ne[3] = 1;
    }
    intervening_input.op    = GGML_OP_NONE;
    intervening_input.ne[0] = 16;
    intervening_input.ne[1] = 1;
    intervening_input.ne[2] = 1;
    intervening_input.ne[3] = 1;
    lifetime_nodes[3].src[0] = &intervening_input;
    ggml_tensor * lifetime_graph_nodes[] = {
        &lifetime_nodes[0],
        &lifetime_nodes[1],
        &lifetime_nodes[2],
        &lifetime_nodes[3],
        &lifetime_nodes[4],
        &lifetime_nodes[5],
    };
    ggml_cgraph lifetime_graph = {};
    lifetime_graph.n_nodes = 6;
    lifetime_graph.nodes   = lifetime_graph_nodes;

    fake_binding_context binding_context = {
        /* .buffer      = */ reinterpret_cast<hrx_buffer_t>(0x6000),
        /* .fail_tensor = */ nullptr,
    };
    lifetime_nodes[0].data = reinterpret_cast<void *>(0);
    lifetime_nodes[1].data = reinterpret_cast<void *>(128);
    lifetime_nodes[2].data = reinterpret_cast<void *>(192);
    lifetime_nodes[3].data = reinterpret_cast<void *>(512);
    lifetime_nodes[4].data = reinterpret_cast<void *>(576);
    lifetime_nodes[5].data = reinterpret_cast<void *>(256);
    intervening_input.data = reinterpret_cast<void *>(640);

    ggml_backend_hrx_loom_op_request lifetime_request = {
        /* .op                    = */ &lifetime_nodes[0],
        /* .cgraph                = */ &lifetime_graph,
        /* .node_index            = */ 0,
        /* .stream                = */ nullptr,
        /* .bind_tensor           = */ fake_bind_tensor,
        /* .bind_tensor_user_data = */ &binding_context,
        /* .storage_layout        = */ nullptr,
        /* .storage_layout_user_data = */ nullptr,
        /* .invocation_context    = */ nullptr,
        /* .execution_epoch       = */ 1,
    };
    ggml_backend_hrx_loom_execution_plan lifetime_plan = {};
    lifetime_plan.dispatch_owner_node_index = 2;
    lifetime_plan.consumed_node_count = 4;
    lifetime_plan.consumed_node_indices[0] = 0;
    lifetime_plan.consumed_node_indices[1] = 2;
    lifetime_plan.consumed_node_indices[2] = 4;
    lifetime_plan.consumed_node_indices[3] = 5;
    lifetime_plan.main.binding_count = 2;
    lifetime_plan.main.bindings[0] = {
        /* .buffer = */ binding_context.buffer,
        /* .offset = */ 128,
        /* .length = */ 64,
    };
    lifetime_plan.main.deferred_read_tensors[0] =
        &lifetime_nodes[1];
    lifetime_plan.main.deferred_read_binding_mask = 1;
    lifetime_plan.main.bindings[1] = {
        /* .buffer = */ binding_context.buffer,
        /* .offset = */ 256,
        /* .length = */ 64,
    };
    lifetime_plan.main.deferred_write_tensors[1] =
        &lifetime_nodes[5];
    lifetime_plan.main.deferred_write_binding_mask = 2;

    expect_true(
        ggml_backend_hrx_loom_plan_can_be_selected(
            &lifetime_request,
            &lifetime_plan),
        "disjoint intervening accesses must preserve an early fused output");
    lifetime_nodes[3].data = reinterpret_cast<void *>(288);
    expect_true(
        !ggml_backend_hrx_loom_plan_can_be_selected(
            &lifetime_request,
            &lifetime_plan),
        "an intervening destination overlapping an early output must reject");
    lifetime_nodes[3].data = reinterpret_cast<void *>(512);
    intervening_input.data = reinterpret_cast<void *>(288);
    expect_true(
        !ggml_backend_hrx_loom_plan_can_be_selected(
            &lifetime_request,
            &lifetime_plan),
        "an intervening input overlapping an early output must reject");
    lifetime_plan.consumed_node_indices[
        lifetime_plan.consumed_node_count++] = 3;
    expect_true(
        ggml_backend_hrx_loom_plan_can_be_selected(
            &lifetime_request,
            &lifetime_plan),
        "a consumed intervening node must not create a false lifetime hazard");
    --lifetime_plan.consumed_node_count;
    intervening_input.data = reinterpret_cast<void *>(640);
    binding_context.fail_tensor = &intervening_input;
    expect_true(
        !ggml_backend_hrx_loom_plan_can_be_selected(
            &lifetime_request,
            &lifetime_plan),
        "failure to bind an intervening input must reject the early output");
    binding_context.fail_tensor = nullptr;

    ggml_backend_hrx_loom_execution_plan ssm_plan = {};
    ssm_plan.dispatch_owner_node_index = 6;
    ssm_plan.consumed_node_count = 5;
    ssm_plan.consumed_node_indices[0] = 0;
    ssm_plan.consumed_node_indices[1] = 2;
    ssm_plan.consumed_node_indices[2] = 3;
    ssm_plan.consumed_node_indices[3] = 6;
    ssm_plan.consumed_node_indices[4] = 7;

    ggml_backend_hrx_loom_execution_plan gdn_plan = {};
    gdn_plan.dispatch_owner_node_index = 16;
    gdn_plan.consumed_node_count = 3;
    gdn_plan.consumed_node_indices[0] = 5;
    gdn_plan.consumed_node_indices[1] = 16;
    gdn_plan.consumed_node_indices[2] = 17;

    ggml_backend_hrx_loom_execution_plan dual_projection_plan = {};
    dual_projection_plan.dispatch_owner_node_index = 14;
    dual_projection_plan.consumed_node_count = 6;
    for (int i = 0; i < dual_projection_plan.consumed_node_count; ++i) {
        dual_projection_plan.consumed_node_indices[i] = 10 + i;
    }

    expect_true(ggml_backend_hrx_loom_deferred_plans_can_coexist(
                    ssm_plan,
                    gdn_plan),
                "disjoint SSM and GDN deferred plans must coexist");
    expect_true(ggml_backend_hrx_loom_deferred_plans_can_coexist(
                    gdn_plan,
                    dual_projection_plan),
                "pending GDN and its dual-projection producer must coexist");

    ggml_backend_hrx_loom_execution_plan duplicate_owner = gdn_plan;
    expect_true(!ggml_backend_hrx_loom_deferred_plans_can_coexist(
                    gdn_plan,
                    duplicate_owner),
                "GDN state and qk-L2 alternatives must not share an owner");

    ggml_backend_hrx_loom_execution_plan overlapping_nodes =
        dual_projection_plan;
    overlapping_nodes.consumed_node_indices[0] =
        gdn_plan.consumed_node_indices[0];
    expect_true(!ggml_backend_hrx_loom_deferred_plans_can_coexist(
                    gdn_plan,
                    overlapping_nodes),
                "deferred plans must not consume the same graph node");

    const ggml_cgraph * graph =
        reinterpret_cast<const ggml_cgraph *>(0x5000);
    std::vector<ggml_backend_hrx_loom_deferred_plan> deferred;
    expect_true(ggml_backend_hrx_loom_enqueue_deferred_plan(
                    deferred,
                    {graph, 7, ssm_plan.dispatch_owner_node_index,
                     "ssm", &ssm_plan, nullptr}),
                "SSM state route must enqueue at cache-get");
    expect_true(ggml_backend_hrx_loom_enqueue_deferred_plan(
                    deferred,
                    {graph, 7, gdn_plan.dispatch_owner_node_index,
                     "gdn", &gdn_plan, nullptr}),
                "GDN state route must overlap the pending SSM route");
    expect_true(deferred.size() == 2 &&
                    deferred[0].owner_index ==
                        ssm_plan.dispatch_owner_node_index &&
                    deferred[1].owner_index ==
                        gdn_plan.dispatch_owner_node_index,
                "deferred plans must remain ordered by dispatch owner");
    ggml_backend_hrx_loom_deferred_plan ready = {};
    expect_true(ggml_backend_hrx_loom_take_deferred_plan(
                    deferred, graph, 7,
                    ssm_plan.dispatch_owner_node_index, &ready) ==
                    GGML_BACKEND_HRX_LOOM_DEFERRED_READY &&
                    std::strcmp(ready.route_id, "ssm") == 0,
                "SSM owner must take exactly the matching plan");
    expect_true(deferred.size() == 1 &&
                    deferred[0].owner_index ==
                        gdn_plan.dispatch_owner_node_index,
                "taking SSM must preserve the pending GDN plan");
    expect_true(ggml_backend_hrx_loom_enqueue_deferred_plan(
                    deferred,
                    {graph, 7,
                     dual_projection_plan.dispatch_owner_node_index,
                     "dual-projection", &dual_projection_plan, nullptr}),
                "dual projection must overlap the pending GDN route");
    expect_true(deferred.size() == 2 &&
                    deferred[0].owner_index ==
                        dual_projection_plan.dispatch_owner_node_index &&
                    deferred[1].owner_index ==
                        gdn_plan.dispatch_owner_node_index,
                "dual projection must dispatch before its GDN consumer");
    expect_true(!ggml_backend_hrx_loom_enqueue_deferred_plan(
                    deferred,
                    {graph, 7, duplicate_owner.dispatch_owner_node_index,
                     "qk-l2-alternative", &duplicate_owner, nullptr}),
                "qk-L2 alternative sharing the GDN owner must be rejected");
    expect_true(deferred.size() == 2,
                "rejected GDN alternative must not mutate the queue");
    expect_true(!ggml_backend_hrx_loom_deferred_queue_accepts_plan(
                    deferred,
                    duplicate_owner),
                "route selection must skip a conflicting deferred candidate");
    ggml_backend_hrx_loom_execution_plan ordinary_l2_plan = {};
    expect_true(ggml_backend_hrx_loom_deferred_queue_accepts_plan(
                    deferred,
                    ordinary_l2_plan),
                "route selection must allow the ordinary fallback");
    expect_true(deferred.size() == 2,
                "route selection checks must preserve pending plans");
    expect_true(ggml_backend_hrx_loom_take_deferred_plan(
                    deferred, graph, 7,
                    dual_projection_plan.dispatch_owner_node_index,
                    &ready) ==
                    GGML_BACKEND_HRX_LOOM_DEFERRED_READY &&
                    std::strcmp(ready.route_id, "dual-projection") == 0,
                "dual projection must materialize gate and beta before GDN");
    expect_true(deferred.size() == 1 &&
                    deferred[0].owner_index ==
                        gdn_plan.dispatch_owner_node_index,
                "taking dual projection must preserve pending GDN");
    expect_true(ggml_backend_hrx_loom_take_deferred_plan(
                    deferred, graph, 7,
                    gdn_plan.dispatch_owner_node_index, &ready) ==
                    GGML_BACKEND_HRX_LOOM_DEFERRED_READY &&
                    std::strcmp(ready.route_id, "gdn") == 0 &&
                    deferred.empty(),
                "GDN owner must take the remaining plan");

    expect_true(ggml_backend_hrx_loom_enqueue_deferred_plan(
                    deferred,
                    {graph, 7, ssm_plan.dispatch_owner_node_index,
                     "stale", &ssm_plan, nullptr}),
                "stale-plan setup must enqueue");
    expect_true(ggml_backend_hrx_loom_take_deferred_plan(
                    deferred, graph, 7,
                    ssm_plan.dispatch_owner_node_index + 1, &ready) ==
                    GGML_BACKEND_HRX_LOOM_DEFERRED_STALE &&
                    deferred.empty(),
                "a skipped owner must remove and report only its stale plan");

    expect_true(ggml_backend_hrx_loom_enqueue_deferred_plan(
                    deferred,
                    {graph, 7, ssm_plan.dispatch_owner_node_index,
                     "old-epoch", &ssm_plan, nullptr}),
                "epoch-reset setup must enqueue");
    expect_true(ggml_backend_hrx_loom_take_deferred_plan(
                    deferred, graph, 8, 0, &ready) ==
                    GGML_BACKEND_HRX_LOOM_DEFERRED_NONE &&
                    deferred.empty(),
                "new execution epoch must clear old deferred plans");

    ggml_backend_hrx_loom_consumed_nodes ready_nodes = {};
    expect_true(ggml_backend_hrx_loom_copy_consumed_nodes_from(
                    ssm_plan,
                    ssm_plan.dispatch_owner_node_index,
                    &ready_nodes),
                "dispatch-owner replay must retain ready consumed nodes");
    expect_true(ready_nodes.count == 2 &&
                    ready_nodes.indices[0] ==
                        ssm_plan.dispatch_owner_node_index &&
                    ready_nodes.indices[1] == 7 &&
                    ready_nodes.dispatch_owner_node_index == -1,
                "dispatch-owner replay must drop earlier claims and dispatch now");
    return 0;
}

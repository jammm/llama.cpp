#pragma once

#include "ggml-hrx-loom-catalog-runtime.h"
#include "ggml-hrx-loom-catalog.h"
#include "ggml-hrx-runtime-util.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

static constexpr size_t GGML_BACKEND_HRX_LOOM_MAX_BINDINGS        = 9;
static constexpr size_t GGML_BACKEND_HRX_LOOM_MAX_CONSTANTS_SIZE  = 256;
static constexpr size_t GGML_BACKEND_HRX_LOOM_MAX_CONFIG_BINDINGS = 64;
static constexpr size_t GGML_BACKEND_HRX_LOOM_MAX_SCRATCH         = 4;
static constexpr size_t GGML_BACKEND_HRX_LOOM_MAX_PREPASSES       = 4;
static constexpr size_t GGML_BACKEND_HRX_LOOM_CONFIG_NAME_BYTES   = 64;
static constexpr size_t GGML_BACKEND_HRX_LOOM_CONFIG_VALUE_BYTES  = 128;

struct ggml_backend_hrx_loom_config_binding {
    char         name[GGML_BACKEND_HRX_LOOM_CONFIG_NAME_BYTES];
    char         value[GGML_BACKEND_HRX_LOOM_CONFIG_VALUE_BYTES];
    const char * type;
};

struct ggml_backend_hrx_loaded_loom_route;

struct ggml_backend_hrx_loom_kernel_plan {
    const ggml_backend_hrx_loom_catalog_entry * entry                                                      = nullptr;
    mutable const ggml_backend_hrx_loom_catalog * loaded_route_catalog                                    = nullptr;
    mutable ggml_backend_hrx_loaded_loom_route *  loaded_route                                            = nullptr;
    hrx_dispatch_config_t                       dispatch                                                   = {};
    hrx_buffer_ref_t                            bindings[GGML_BACKEND_HRX_LOOM_MAX_BINDINGS]               = {};
    uint8_t                                     binding_scratch_index[GGML_BACKEND_HRX_LOOM_MAX_BINDINGS]  = {};
    size_t                                      binding_scratch_offset[GGML_BACKEND_HRX_LOOM_MAX_BINDINGS] = {};
    size_t                                      binding_scratch_length[GGML_BACKEND_HRX_LOOM_MAX_BINDINGS] = {};
    const ggml_tensor *                         deferred_read_tensors[GGML_BACKEND_HRX_LOOM_MAX_BINDINGS]  = {};
    uint16_t                                    deferred_read_binding_mask                                 = 0;
    const ggml_tensor *                         deferred_write_tensors[GGML_BACKEND_HRX_LOOM_MAX_BINDINGS] = {};
    uint16_t                                    deferred_write_binding_mask                                = 0;
    size_t                                      binding_count                                              = 0;
    uint8_t                                     constants[GGML_BACKEND_HRX_LOOM_MAX_CONSTANTS_SIZE]        = {};
    size_t                                      constants_size                                             = 0;
    ggml_backend_hrx_loom_config_binding        config_bindings[GGML_BACKEND_HRX_LOOM_MAX_CONFIG_BINDINGS] = {};
    size_t                                      config_binding_count                                       = 0;
};

struct ggml_backend_hrx_loom_scratch_plan {
    const char * name             = nullptr;
    size_t       bytes            = 0;
    size_t       minimum_capacity = 0;
};

struct ggml_backend_hrx_loom_prepass_plan {
    ggml_backend_hrx_loom_kernel_plan kernel;
    uint8_t                           cache_scratch_index = 0;
    size_t                            cache_region_offset  = 0;
    size_t                            cache_region_length  = 0;
    const ggml_tensor *               cache_source         = nullptr;
    hrx_buffer_ref_t                  cache_source_binding = {};
};

struct ggml_backend_hrx_loom_execution_plan {
    ggml_backend_hrx_loom_kernel_plan  main;
    ggml_backend_hrx_loom_scratch_plan scratch[GGML_BACKEND_HRX_LOOM_MAX_SCRATCH] = {};
    size_t                             scratch_count = 0;
    ggml_backend_hrx_loom_prepass_plan prepasses[GGML_BACKEND_HRX_LOOM_MAX_PREPASSES] = {};
    size_t                             prepass_count = 0;
    int                                consumed_node_indices[GGML_BACKEND_HRX_LOOM_MAX_CONSUMED_NODES] = {};
    int                                consumed_node_count = 0;
    int                                dispatch_owner_node_index = -1;
};

struct ggml_backend_hrx_loom_tensor_graph_facts {
    size_t  observation_count = 0;
    int     consumer_indices[
        GGML_BACKEND_HRX_LOOM_MAX_CONSUMED_NODES + 1] = {};
    uint8_t consumer_count = 0;
    bool    graph_output   = false;
};

struct ggml_backend_hrx_loom_graph_facts {
    const ggml_cgraph * cgraph             = nullptr;
    uint64_t            uid                = 0;
    uint64_t            execution_epoch    = 0;
    int                 n_nodes            = 0;
    ggml_tensor **      nodes              = nullptr;
    ggml_tensor **      visited_keys       = nullptr;
    ggml_bitset_t *     visited_used       = nullptr;
    size_t              visited_hash_size  = 0;
    std::vector<ggml_backend_hrx_loom_tensor_graph_facts> tensors;

    bool build(const ggml_backend_hrx_loom_op_request * request);

    bool matches(const ggml_backend_hrx_loom_op_request * request) const;

    const ggml_backend_hrx_loom_tensor_graph_facts * find(
        const ggml_tensor * tensor) const;
};

struct ggml_backend_hrx_loom_graph_fact_cache {
    std::vector<std::unique_ptr<ggml_backend_hrx_loom_graph_facts>> graphs;

    const ggml_backend_hrx_loom_graph_facts * resolve(
        const ggml_backend_hrx_loom_op_request * request);

    void clear() {
        graphs.clear();
    }
};

struct ggml_backend_hrx_loom_resolved_plan_cache_entry {
    const ggml_tensor *                              op       = nullptr;
    const char *                                     route_id = nullptr;
    std::unique_ptr<ggml_backend_hrx_loom_execution_plan> plan;
};

struct ggml_backend_hrx_loom_resolved_graph_cache {
    const ggml_backend_hrx_loom_catalog *                    catalog = nullptr;
    const ggml_cgraph *                                      cgraph  = nullptr;
    uint64_t                                                 uid     = 0;
    int                                                      n_nodes = 0;
    std::vector<ggml_backend_hrx_loom_resolved_plan_cache_entry> nodes;
};

struct ggml_backend_hrx_loom_resolved_plan_cache {
    std::vector<ggml_backend_hrx_loom_resolved_graph_cache> graphs;

    static bool request_is_cacheable(
        const ggml_backend_hrx_loom_catalog *          catalog,
        const ggml_backend_hrx_loom_op_request *       request) {
        return catalog && request && request->cgraph &&
               request->cgraph->uid != 0 &&
               request->cgraph->n_nodes > 0 &&
               request->node_index >= 0 &&
               request->node_index < request->cgraph->n_nodes &&
               request->cgraph->nodes &&
               request->cgraph->nodes[request->node_index] ==
                   request->op;
    }

    static bool plan_is_cacheable(
        const ggml_backend_hrx_loom_execution_plan & plan) {
        if (!plan.main.entry ||
            plan.main.binding_count >
                GGML_BACKEND_HRX_LOOM_MAX_BINDINGS ||
            plan.main.constants_size >
                GGML_BACKEND_HRX_LOOM_MAX_CONSTANTS_SIZE ||
            plan.main.config_binding_count >
                GGML_BACKEND_HRX_LOOM_MAX_CONFIG_BINDINGS ||
            plan.scratch_count >
                GGML_BACKEND_HRX_LOOM_MAX_SCRATCH ||
            plan.prepass_count >
                GGML_BACKEND_HRX_LOOM_MAX_PREPASSES ||
            plan.consumed_node_count < 1 ||
            plan.consumed_node_count >
                GGML_BACKEND_HRX_LOOM_MAX_CONSUMED_NODES) {
            return false;
        }
        for (size_t i = 0; i < plan.prepass_count; ++i) {
            const auto & kernel = plan.prepasses[i].kernel;
            if (!kernel.entry ||
                kernel.binding_count >
                    GGML_BACKEND_HRX_LOOM_MAX_BINDINGS ||
                kernel.constants_size >
                    GGML_BACKEND_HRX_LOOM_MAX_CONSTANTS_SIZE ||
                kernel.config_binding_count >
                    GGML_BACKEND_HRX_LOOM_MAX_CONFIG_BINDINGS) {
                return false;
            }
        }
        return true;
    }

    const ggml_backend_hrx_loom_resolved_plan_cache_entry * find(
        const ggml_backend_hrx_loom_catalog *    catalog,
        const ggml_backend_hrx_loom_op_request * request) const {
        if (!request_is_cacheable(catalog, request)) {
            return nullptr;
        }
        for (const auto & graph : graphs) {
            if (graph.catalog != catalog ||
                graph.cgraph != request->cgraph ||
                graph.uid != request->cgraph->uid ||
                graph.n_nodes != request->cgraph->n_nodes ||
                graph.nodes.size() !=
                    static_cast<size_t>(graph.n_nodes)) {
                continue;
            }
            const auto & entry = graph.nodes[request->node_index];
            return entry.op == request->op && entry.route_id &&
                   entry.plan ?
                &entry :
                nullptr;
        }
        return nullptr;
    }

    bool publish(
        const ggml_backend_hrx_loom_catalog *          catalog,
        const ggml_backend_hrx_loom_op_request *       request,
        const char *                                   route_id,
        const ggml_backend_hrx_loom_execution_plan &   plan) {
        if (!request_is_cacheable(catalog, request) || !route_id ||
            !plan_is_cacheable(plan)) {
            return false;
        }

        for (auto it = graphs.begin(); it != graphs.end();) {
            if (it->cgraph == request->cgraph &&
                (it->catalog != catalog ||
                 it->uid != request->cgraph->uid ||
                 it->n_nodes != request->cgraph->n_nodes)) {
                it = graphs.erase(it);
            } else {
                ++it;
            }
        }

        auto graph_it = std::find_if(
            graphs.begin(),
            graphs.end(),
            [catalog, request](
                const ggml_backend_hrx_loom_resolved_graph_cache & graph) {
                return graph.catalog == catalog &&
                       graph.cgraph == request->cgraph &&
                       graph.uid == request->cgraph->uid &&
                       graph.n_nodes == request->cgraph->n_nodes;
            });
        if (graph_it == graphs.end()) {
            ggml_backend_hrx_loom_resolved_graph_cache graph = {};
            graph.catalog = catalog;
            graph.cgraph  = request->cgraph;
            graph.uid     = request->cgraph->uid;
            graph.n_nodes = request->cgraph->n_nodes;
            graph.nodes.resize(
                static_cast<size_t>(graph.n_nodes));
            graphs.push_back(std::move(graph));
            graph_it = graphs.end();
            --graph_it;
        }

        auto & entry = graph_it->nodes[request->node_index];
        entry.op       = request->op;
        entry.route_id = route_id;
        entry.plan =
            std::make_unique<ggml_backend_hrx_loom_execution_plan>(
                plan);
        return true;
    }

    void clear() {
        graphs.clear();
    }
};

static inline bool ggml_backend_hrx_loom_deferred_refs_overlap(
    const hrx_buffer_ref_t & lhs,
    const hrx_buffer_ref_t & rhs) {
    if (!lhs.buffer || lhs.buffer != rhs.buffer ||
        lhs.length == 0 || rhs.length == 0) {
        return false;
    }
    return lhs.offset <= rhs.offset ?
        rhs.offset - lhs.offset < lhs.length :
        lhs.offset - rhs.offset < rhs.length;
}

static inline bool ggml_backend_hrx_loom_deferred_write_conflicts(
    int                      writer_node_index,
    int                      ready_node_index,
    const hrx_buffer_ref_t & writer,
    const hrx_buffer_ref_t & deferred_read) {
    return writer_node_index > ready_node_index &&
           ggml_backend_hrx_loom_deferred_refs_overlap(
               writer,
               deferred_read);
}

static inline bool ggml_backend_hrx_loom_copy_consumed_nodes_from(
    const ggml_backend_hrx_loom_execution_plan & plan,
    int                                          first_node_index,
    ggml_backend_hrx_loom_consumed_nodes *       out) {
    if (!out || plan.consumed_node_count < 1 ||
        plan.consumed_node_count > GGML_BACKEND_HRX_LOOM_MAX_CONSUMED_NODES) {
        return false;
    }
    *out = {};
    for (int i = 0; i < plan.consumed_node_count; ++i) {
        const int node_index = plan.consumed_node_indices[i];
        if (node_index < first_node_index) {
            continue;
        }
        if (out->count >= GGML_BACKEND_HRX_LOOM_MAX_CONSUMED_NODES) {
            return false;
        }
        out->indices[out->count++] = node_index;
    }
    return out->count > 0;
}

static inline bool ggml_backend_hrx_loom_deferred_plans_can_coexist(
    const ggml_backend_hrx_loom_execution_plan & lhs,
    const ggml_backend_hrx_loom_execution_plan & rhs) {
    if (lhs.dispatch_owner_node_index < 0 ||
        rhs.dispatch_owner_node_index < 0 ||
        lhs.dispatch_owner_node_index == rhs.dispatch_owner_node_index ||
        lhs.consumed_node_count < 1 ||
        lhs.consumed_node_count > GGML_BACKEND_HRX_LOOM_MAX_CONSUMED_NODES ||
        rhs.consumed_node_count < 1 ||
        rhs.consumed_node_count > GGML_BACKEND_HRX_LOOM_MAX_CONSUMED_NODES) {
        return false;
    }
    for (int i = 0; i < lhs.consumed_node_count; ++i) {
        for (int j = 0; j < rhs.consumed_node_count; ++j) {
            if (lhs.consumed_node_indices[i] ==
                rhs.consumed_node_indices[j]) {
                return false;
            }
        }
    }
    return true;
}

struct ggml_backend_hrx_loom_deferred_plan {
    const ggml_cgraph *                  cgraph          = nullptr;
    uint64_t                             execution_epoch = 0;
    int                                  owner_index     = -1;
    const char *                         route_id        = nullptr;
    const ggml_backend_hrx_loom_execution_plan * plan    = nullptr;
    std::unique_ptr<ggml_backend_hrx_loom_execution_plan> owned_plan;
};

static inline bool ggml_backend_hrx_loom_deferred_queue_accepts_plan(
    const std::vector<ggml_backend_hrx_loom_deferred_plan> & deferred,
    const ggml_backend_hrx_loom_execution_plan &              candidate) {
    if (candidate.dispatch_owner_node_index < 0) {
        return true;
    }
    for (const auto & pending : deferred) {
        if (!pending.plan ||
            !ggml_backend_hrx_loom_deferred_plans_can_coexist(
                *pending.plan,
                candidate)) {
            return false;
        }
    }
    return true;
}

enum ggml_backend_hrx_loom_deferred_take_result {
    GGML_BACKEND_HRX_LOOM_DEFERRED_NONE,
    GGML_BACKEND_HRX_LOOM_DEFERRED_READY,
    GGML_BACKEND_HRX_LOOM_DEFERRED_STALE,
};

static inline bool ggml_backend_hrx_loom_enqueue_deferred_plan(
    std::vector<ggml_backend_hrx_loom_deferred_plan> & deferred,
    ggml_backend_hrx_loom_deferred_plan                candidate) {
    if (!candidate.plan ||
        candidate.owner_index < 0 ||
        candidate.owner_index !=
            candidate.plan->dispatch_owner_node_index) {
        return false;
    }
    if (!ggml_backend_hrx_loom_deferred_queue_accepts_plan(
            deferred,
            *candidate.plan)) {
        return false;
    }
    const auto insertion = std::lower_bound(
        deferred.begin(),
        deferred.end(),
        candidate.owner_index,
        [](const ggml_backend_hrx_loom_deferred_plan & pending,
           int owner_index) {
            return pending.owner_index < owner_index;
        });
    deferred.insert(insertion, std::move(candidate));
    return true;
}

static inline ggml_backend_hrx_loom_deferred_take_result
ggml_backend_hrx_loom_take_deferred_plan(
    std::vector<ggml_backend_hrx_loom_deferred_plan> & deferred,
    const ggml_cgraph *                                cgraph,
    uint64_t                                           execution_epoch,
    int                                                node_index,
    ggml_backend_hrx_loom_deferred_plan *              out) {
    if (out) {
        *out = {};
    }
    const auto incompatible = std::find_if(
        deferred.begin(),
        deferred.end(),
        [cgraph, execution_epoch](
            const ggml_backend_hrx_loom_deferred_plan & pending) {
            return pending.cgraph != cgraph ||
                   pending.execution_epoch != execution_epoch;
        });
    if (incompatible != deferred.end()) {
        deferred.clear();
        return GGML_BACKEND_HRX_LOOM_DEFERRED_NONE;
    }
    if (deferred.empty() ||
        deferred.front().owner_index > node_index) {
        return GGML_BACKEND_HRX_LOOM_DEFERRED_NONE;
    }

    const bool stale = deferred.front().owner_index < node_index;
    if (out) {
        *out = std::move(deferred.front());
    }
    deferred.erase(deferred.begin());
    return stale ?
        GGML_BACKEND_HRX_LOOM_DEFERRED_STALE :
        GGML_BACKEND_HRX_LOOM_DEFERRED_READY;
}

bool ggml_backend_hrx_loom_plan_can_be_selected(
    const ggml_backend_hrx_loom_op_request * request,
    const ggml_backend_hrx_loom_execution_plan * plan);

bool ggml_backend_hrx_loom_tensor_is_transient_at_indices_cached(
    const ggml_backend_hrx_loom_op_request * request,
    const ggml_tensor *                      tensor,
    const int *                              consumed_node_indices,
    int                                      consumed_node_count);

bool ggml_backend_hrx_loom_tensor_consumers_through_view_cached(
    const ggml_backend_hrx_loom_op_request * request,
    const ggml_tensor *                      owner,
    const ggml_tensor *                      view);

struct ggml_backend_hrx_loom_scratch_cache_key {
    const ggml_tensor * source_owner = nullptr;
    hrx_buffer_ref_t    source       = {};
    hrx_buffer_ref_t    region       = {};
    uint64_t            epoch        = 0;
    const char *        artifact     = nullptr;
};

static inline bool ggml_backend_hrx_loom_scratch_cache_key_matches(
    const ggml_backend_hrx_loom_scratch_cache_key & lhs,
    const ggml_backend_hrx_loom_scratch_cache_key & rhs) {
    return lhs.source_owner == rhs.source_owner &&
           lhs.source.buffer == rhs.source.buffer &&
           lhs.source.offset == rhs.source.offset &&
           lhs.source.length == rhs.source.length &&
           lhs.region.buffer == rhs.region.buffer &&
           lhs.region.offset == rhs.region.offset &&
           lhs.region.length == rhs.region.length &&
           lhs.epoch == rhs.epoch &&
           lhs.artifact && rhs.artifact &&
           std::strcmp(lhs.artifact, rhs.artifact) == 0;
}

static inline bool ggml_backend_hrx_loom_resolve_cache_region(
    const ggml_backend_hrx_loom_scratch_plan & scratch,
    const ggml_backend_hrx_loom_prepass_plan & prepass,
    hrx_buffer_t                               buffer,
    hrx_buffer_ref_t *                         region) {
    if (!buffer || !region || scratch.bytes == 0 ||
        (prepass.cache_region_length == 0 && prepass.cache_region_offset != 0)) {
        return false;
    }
    const size_t offset = prepass.cache_region_length == 0 ? 0 : prepass.cache_region_offset;
    const size_t length = prepass.cache_region_length == 0 ? scratch.bytes : prepass.cache_region_length;
    if (offset > scratch.bytes || length == 0 || length > scratch.bytes - offset) {
        return false;
    }
    *region = {
        /* .buffer = */ buffer,
        /* .offset = */ offset,
        /* .length = */ length,
    };
    return true;
}

static inline bool ggml_backend_hrx_loom_cache_region_is_bound(
    const ggml_backend_hrx_loom_prepass_plan & prepass) {
    if (prepass.cache_scratch_index == 0 || prepass.cache_region_length == 0 ||
        prepass.kernel.binding_count > GGML_BACKEND_HRX_LOOM_MAX_BINDINGS) {
        return false;
    }
    for (size_t i = 0; i < prepass.kernel.binding_count; ++i) {
        if (prepass.kernel.binding_scratch_index[i] != prepass.cache_scratch_index) {
            continue;
        }
        const size_t binding_offset = prepass.kernel.binding_scratch_offset[i];
        const size_t binding_length = prepass.kernel.binding_scratch_length[i];
        if (binding_offset <= prepass.cache_region_offset &&
            prepass.cache_region_offset - binding_offset <= binding_length &&
            prepass.cache_region_length <=
                binding_length - (prepass.cache_region_offset - binding_offset)) {
            return true;
        }
    }
    return false;
}

static inline bool ggml_backend_hrx_loom_match_only(const ggml_backend_hrx_loom_execution_plan * plan) {
    return plan == nullptr;
}

struct ggml_backend_hrx_loom_compile_input {
    const void *                                 source_data;
    size_t                                       source_size;
    const char *                                 source_format;
    const char *                                 source_name;
    const char *                                 target;
    const char *                                 symbol;
    const ggml_backend_hrx_loom_config_binding * config_bindings;
    size_t                                       config_binding_count;
};

struct ggml_backend_hrx_loom_compile_output {
    void * executable_data;
    size_t executable_size;
    char * report_json;
    size_t report_json_size;
};

static inline uint64_t ggml_backend_hrx_loom_fnv1a64(const void * data, size_t size) {
    static constexpr uint64_t FNV_OFFSET_BASIS = UINT64_C(14695981039346656037);
    static constexpr uint64_t FNV_PRIME        = UINT64_C(1099511628211);

    uint64_t     result = FNV_OFFSET_BASIS;
    const auto * bytes  = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < size; ++i) {
        result ^= bytes[i];
        result *= FNV_PRIME;
    }
    return result;
}

static inline void ggml_backend_hrx_loom_append_key_field(std::string & key, const char * data, size_t size) {
    key += std::to_string(size);
    key += ':';
    if (data && size > 0) {
        key.append(data, size);
    }
}

static inline void ggml_backend_hrx_loom_append_key_field(std::string & key, const char * value) {
    if (!value) {
        ggml_backend_hrx_loom_append_key_field(key, "", 0);
        return;
    }
    ggml_backend_hrx_loom_append_key_field(key, value, std::strlen(value));
}

static inline void ggml_backend_hrx_loom_append_key_field(std::string & key, uint64_t value) {
    const std::string text = std::to_string(value);
    ggml_backend_hrx_loom_append_key_field(key, text.c_str(), text.size());
}

static inline std::string ggml_backend_hrx_loom_cache_key(const char *                              target,
                                                          const ggml_backend_hrx_loom_kernel_plan * plan) {
    std::string                                 key;
    const ggml_backend_hrx_loom_catalog_entry * entry = plan ? plan->entry : nullptr;
    if (!entry) {
        return key;
    }

    ggml_backend_hrx_loom_append_key_field(key, target);
    ggml_backend_hrx_loom_append_key_field(key, entry->id);
    ggml_backend_hrx_loom_append_key_field(key, entry->source_name);
    ggml_backend_hrx_loom_append_key_field(key, entry->source_format);
    ggml_backend_hrx_loom_append_key_field(key, static_cast<uint64_t>(entry->source_size));
    ggml_backend_hrx_loom_append_key_field(key, ggml_backend_hrx_loom_fnv1a64(entry->source_data, entry->source_size));
    ggml_backend_hrx_loom_append_key_field(key, entry->symbol);

    for (size_t i = 0; i < plan->config_binding_count; ++i) {
        const ggml_backend_hrx_loom_config_binding & binding = plan->config_bindings[i];
        ggml_backend_hrx_loom_append_key_field(key, binding.name);
        ggml_backend_hrx_loom_append_key_field(key, binding.type);
        ggml_backend_hrx_loom_append_key_field(key, binding.value);
    }
    return key;
}

bool ggml_backend_hrx_loom_compile(const ggml_backend_hrx_loom_compile_input * input,
                                   ggml_backend_hrx_loom_compile_output *      output);

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_unsupported(ggml_backend_hrx_loom_unsupported_reason reason);

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_supported(const char * route_id);

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_failed(const char * route_id);

const ggml_backend_hrx_loom_catalog_entry * ggml_backend_hrx_loom_find_entry(
    const ggml_backend_hrx_loom_catalog * catalog,
    const char *                          route_id);

int64_t ggml_backend_hrx_loom_next_power_of_2(int64_t value);

bool ggml_backend_hrx_loom_bind_tensor(const ggml_backend_hrx_loom_op_request * request,
                                       const ggml_tensor *                      tensor,
                                       hrx_buffer_ref_t *                       out_ref);

bool ggml_backend_hrx_loom_storage_layout_matches(
    const ggml_backend_hrx_loom_op_request * request,
    const ggml_tensor *                      tensor,
    const char *                             expected);

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_match_request(ggml_backend_hrx_loom_catalog *          catalog,
                                                                      const ggml_backend_hrx_loom_op_request * request);

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_prepare_plan(ggml_backend_hrx_loom_catalog *          catalog,
                                                                     const ggml_backend_hrx_loom_op_request * request,
                                                                     ggml_backend_hrx_loom_execution_plan *   plan);

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_match_or_prepare_request(
    ggml_backend_hrx_loom_catalog *          catalog,
    const ggml_backend_hrx_loom_op_request * request,
    ggml_backend_hrx_loom_execution_plan *   plan);

#include "ggml-hrx-loom-catalog-runtime-internal.h"
#include "ggml-hrx-runtime-util.h"

#include <loomc/loomc.h>
#include <loomc/target/amdgpu.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_backend_hrx_loaded_loom_route {
    const ggml_backend_hrx_loom_catalog_entry * entry = nullptr;
    std::vector<ggml_backend_hrx_loom_config_binding> config_bindings;
    std::string                  cache_key;
    hrx_executable_t             executable     = nullptr;
    uint32_t                     export_ordinal = 0;
    hrx_executable_export_info_t export_info    = {};
    std::string                  report_json;

    ~ggml_backend_hrx_loaded_loom_route() {
        if (executable) {
            hrx_executable_release(executable);
        }
    }
};

struct ggml_backend_hrx_loaded_loom_route_bucket {
    std::vector<ggml_backend_hrx_loaded_loom_route *> routes;
    ggml_backend_hrx_loaded_loom_route *              most_recent = nullptr;
};

namespace {

#define GGML_HRX_LOOM_CHECK(expr) ggml_backend_hrx_log_hrx_status((expr), #expr, __FILE__, __LINE__)

static void ggml_backend_hrx_loom_log_status(loomc_status_t status, const char * expr, const char * file, int line) {
    if (loomc_status_is_ok(status)) {
        return;
    }

    char               buffer[1024] = {};
    loomc_host_size_t  length       = 0;
    const bool         formatted    = loomc_status_format(status, sizeof(buffer), buffer, &length);
    const char * const message      = formatted ? buffer : loomc_status_code_string(loomc_status_code(status));
    GGML_LOG_ERROR("%s:%d: %s failed: %s\n", file, line, expr, message ? message : "unknown Loom error");
    loomc_status_free(status);
}

#define GGML_HRX_LOOMC_CHECK(expr) ggml_backend_hrx_loom_log_status((expr), #expr, __FILE__, __LINE__)

static ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_response(
    ggml_backend_hrx_loom_result             result,
    ggml_backend_hrx_loom_unsupported_reason unsupported_reason,
    const char *                             route_id) {
    return {
        /* .result             = */ result,
        /* .unsupported_reason = */ unsupported_reason,
        /* .route_id           = */ route_id,
    };
}

static void ggml_backend_hrx_loom_compile_output_free(ggml_backend_hrx_loom_compile_output * output) {
    if (!output) {
        return;
    }

    std::free(output->executable_data);
    std::free(output->report_json);
    *output = {};
}

static void ggml_backend_hrx_loom_log_result_diagnostics(const loomc_result_t * result) {
    if (!result) {
        return;
    }

    for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
        const loomc_diagnostic_t * diagnostic = loomc_result_diagnostic_at(result, i);
        if (!diagnostic) {
            continue;
        }
        GGML_LOG_ERROR("Loom diagnostic: %.*s: %.*s\n", static_cast<int>(diagnostic->code.size),
                       diagnostic->code.data, static_cast<int>(diagnostic->message.size), diagnostic->message.data);
    }
}

static const loomc_artifact_t * ggml_backend_hrx_loom_find_artifact(const loomc_result_t *     result,
                                                                    loomc_artifact_kind_t     kind,
                                                                    loomc_string_view_t       format) {
    if (!result) {
        return nullptr;
    }

    for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
        const loomc_artifact_t * artifact = loomc_result_artifact_at(result, i);
        if (!artifact) {
            continue;
        }
        if (artifact->kind == kind && loomc_string_view_equal(artifact->format, format)) {
            return artifact;
        }
    }
    return nullptr;
}

static bool ggml_backend_hrx_loom_require_success(loomc_result_t * result, const char * phase) {
    if (result && loomc_result_succeeded(result)) {
        return true;
    }

    GGML_LOG_ERROR("%s: Loom %s failed\n", __func__, phase);
    ggml_backend_hrx_loom_log_result_diagnostics(result);
    return false;
}

static bool ggml_backend_hrx_loom_copy_artifact_bytes(const loomc_artifact_t * artifact,
                                                      void **                  out_data,
                                                      size_t *                 out_size) {
    if (!artifact || !out_data || !out_size || !artifact->contents.data || artifact->contents.data_length == 0) {
        return false;
    }

    void * data = std::malloc(artifact->contents.data_length);
    if (!data) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes for Loom artifact\n", __func__,
                       static_cast<size_t>(artifact->contents.data_length));
        return false;
    }

    std::memcpy(data, artifact->contents.data, artifact->contents.data_length);
    *out_data = data;
    *out_size = artifact->contents.data_length;
    return true;
}

struct ggml_backend_hrx_loom_compile_state {
    loomc_target_environment_t * target_environment = nullptr;
    loomc_context_t *            context            = nullptr;
    loomc_workspace_t *          workspace          = nullptr;
    loomc_source_t *             source             = nullptr;
    loomc_module_t *             module             = nullptr;
    loomc_target_profile_t *     target_profile     = nullptr;
    loomc_compiler_t *           compiler           = nullptr;
    loomc_pass_program_t *       pass_program       = nullptr;
    loomc_result_t *             result             = nullptr;

    ~ggml_backend_hrx_loom_compile_state() {
        loomc_result_release(result);
        loomc_pass_program_release(pass_program);
        loomc_compiler_release(compiler);
        loomc_target_profile_release(target_profile);
        loomc_module_release(module);
        loomc_source_release(source);
        loomc_workspace_release(workspace);
        loomc_context_release(context);
        loomc_target_environment_release(target_environment);
    }

    void reset_result() {
        loomc_result_release(result);
        result = nullptr;
    }
};

}  // namespace

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_unsupported(ggml_backend_hrx_loom_unsupported_reason reason) {
    return ggml_backend_hrx_loom_response(GGML_BACKEND_HRX_LOOM_UNSUPPORTED, reason, nullptr);
}

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_supported(const char * route_id) {
    return ggml_backend_hrx_loom_response(GGML_BACKEND_HRX_LOOM_INVOKED, GGML_BACKEND_HRX_LOOM_UNSUPPORTED_NONE,
                                          route_id);
}

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_failed(const char * route_id) {
    return ggml_backend_hrx_loom_response(GGML_BACKEND_HRX_LOOM_FAILED, GGML_BACKEND_HRX_LOOM_UNSUPPORTED_NONE,
                                          route_id);
}

int64_t ggml_backend_hrx_loom_next_power_of_2(int64_t value) {
    int64_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

struct ggml_backend_hrx_loom_catalog {
    hrx_device_t                                                     device = nullptr;
    std::string                                                      architecture;
    std::string                                                      target;
    std::vector<std::unique_ptr<ggml_backend_hrx_loaded_loom_route>> routes;
    std::unordered_map<
        const ggml_backend_hrx_loom_catalog_entry *,
        ggml_backend_hrx_loaded_loom_route_bucket>                   route_buckets;

    ~ggml_backend_hrx_loom_catalog() {
        route_buckets.clear();
        routes.clear();
        if (device) {
            hrx_device_release(device);
        }
    }
};

namespace {

static void ggml_backend_hrx_loom_cache_loaded_route(
    const ggml_backend_hrx_loom_kernel_plan * plan,
    const ggml_backend_hrx_loom_catalog *     catalog,
    ggml_backend_hrx_loaded_loom_route *      route) {
    plan->loaded_route_catalog = catalog;
    plan->loaded_route         = route;
}

static bool ggml_backend_hrx_loom_loaded_route_matches(
    const ggml_backend_hrx_loaded_loom_route * route,
    const ggml_backend_hrx_loom_kernel_plan *  plan) {
    if (!route || !plan || route->entry != plan->entry ||
        route->config_bindings.size() != plan->config_binding_count) {
        return false;
    }
    for (size_t i = 0; i < plan->config_binding_count; ++i) {
        const auto & lhs = route->config_bindings[i];
        const auto & rhs = plan->config_bindings[i];
        if (!lhs.type || !rhs.type ||
            std::strcmp(lhs.name, rhs.name) != 0 ||
            std::strcmp(lhs.value, rhs.value) != 0 ||
            std::strcmp(lhs.type, rhs.type) != 0) {
            return false;
        }
    }
    return true;
}

static constexpr size_t GGML_BACKEND_HRX_LOOM_SCRATCH_ALIGNMENT = 256;

static size_t ggml_backend_hrx_loom_align_up(size_t value, size_t alignment) {
    if (alignment == 0) {
        return value;
    }
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

struct ggml_backend_hrx_loom_scratch_state {
    std::string               name;
    hrx_buffer_t              buffer          = nullptr;
    size_t                    capacity        = 0;
    std::vector<hrx_buffer_t> retired_buffers;
    const ggml_tensor *       cache_source    = nullptr;
    const ggml_tensor *       cache_owner     = nullptr;
    hrx_buffer_ref_t          cache_source_binding = {};
    hrx_buffer_ref_t          cache_region         = {};
    int                       cache_node_index = -1;
    uint64_t                  cache_epoch     = 0;
    std::string               cache_artifact;

    void clear_cache() {
        cache_source         = nullptr;
        cache_owner          = nullptr;
        cache_source_binding = {};
        cache_region         = {};
        cache_node_index     = -1;
        cache_epoch          = 0;
        cache_artifact.clear();
    }

    ~ggml_backend_hrx_loom_scratch_state() {
        if (buffer) {
            hrx_buffer_release(buffer);
        }
        for (hrx_buffer_t retired : retired_buffers) {
            if (retired) {
                hrx_buffer_release(retired);
            }
        }
    }
};

}  // namespace

struct ggml_backend_hrx_loom_invocation_context {
    hrx_device_t                                                   device = nullptr;
    std::vector<std::unique_ptr<ggml_backend_hrx_loom_scratch_state>> scratch;
    std::vector<ggml_backend_hrx_loom_deferred_plan>                deferred;
    ggml_backend_hrx_loom_graph_fact_cache                          graph_facts;
    ggml_backend_hrx_loom_resolved_plan_cache                       resolved_plans;

    ~ggml_backend_hrx_loom_invocation_context() {
        resolved_plans.clear();
        graph_facts.clear();
        deferred.clear();
        scratch.clear();
        if (device) {
            hrx_device_release(device);
        }
    }
};

static bool ggml_backend_hrx_loom_deferred_reads_survive(
    const ggml_backend_hrx_loom_op_request * request,
    const ggml_backend_hrx_loom_execution_plan & plan);

bool ggml_backend_hrx_loom_plan_can_be_selected(
    const ggml_backend_hrx_loom_op_request * request,
    const ggml_backend_hrx_loom_execution_plan * plan) {
    if (!request || !plan ||
        plan->dispatch_owner_node_index <= request->node_index) {
        return true;
    }
    if (!ggml_backend_hrx_loom_deferred_reads_survive(
            request,
            *plan)) {
        return false;
    }
    return !request->invocation_context ||
           ggml_backend_hrx_loom_deferred_queue_accepts_plan(
               request->invocation_context->deferred,
               *plan);
}

static bool ggml_backend_hrx_loom_cached_plan_can_be_selected(
    const ggml_backend_hrx_loom_op_request * request,
    const ggml_backend_hrx_loom_execution_plan * plan) {
    if (!request || !plan) {
        return false;
    }
    if (plan->dispatch_owner_node_index <= request->node_index) {
        return true;
    }
    return !request->invocation_context ||
           ggml_backend_hrx_loom_deferred_queue_accepts_plan(
               request->invocation_context->deferred,
               *plan);
}

const ggml_backend_hrx_loom_catalog_entry * ggml_backend_hrx_loom_find_entry(
    const ggml_backend_hrx_loom_catalog * catalog,
    const char *                          route_id) {
    if (!catalog || !route_id) {
        return nullptr;
    }

    size_t                                      count   = 0;
    const ggml_backend_hrx_loom_catalog_entry * entries = ggml_backend_hrx_loom_catalog_entries(&count);
    for (size_t i = 0; i < count; ++i) {
        if (std::strcmp(entries[i].id, route_id) == 0 && catalog->target == entries[i].target) {
            return &entries[i];
        }
    }
    return nullptr;
}

namespace {

static ggml_backend_hrx_loaded_loom_route * ggml_backend_hrx_loom_get_loaded_route(
    ggml_backend_hrx_loom_catalog *           catalog,
    const ggml_backend_hrx_loom_kernel_plan * plan) {
    if (!catalog || !plan || !plan->entry) {
        return nullptr;
    }
    if (plan->config_binding_count > GGML_BACKEND_HRX_LOOM_MAX_CONFIG_BINDINGS) {
        GGML_LOG_ERROR("%s: route %s has too many config bindings: %zu\n", __func__, plan->entry->id,
                       plan->config_binding_count);
        return nullptr;
    }
    if (plan->loaded_route_catalog == catalog && plan->loaded_route) {
        return plan->loaded_route;
    }
    auto bucket_it = catalog->route_buckets.find(plan->entry);
    if (bucket_it != catalog->route_buckets.end()) {
        auto & bucket = bucket_it->second;
        if (ggml_backend_hrx_loom_loaded_route_matches(
                bucket.most_recent,
                plan)) {
            ggml_backend_hrx_loom_cache_loaded_route(
                plan, catalog, bucket.most_recent);
            return bucket.most_recent;
        }
        for (auto * route : bucket.routes) {
            if (route == bucket.most_recent ||
                !ggml_backend_hrx_loom_loaded_route_matches(
                    route,
                    plan)) {
                continue;
            }
            bucket.most_recent = route;
            ggml_backend_hrx_loom_cache_loaded_route(
                plan, catalog, route);
            return route;
        }
    }

    const std::string cache_key = ggml_backend_hrx_loom_cache_key(catalog->target.c_str(), plan);
    ggml_backend_hrx_loom_compile_input compile_input = {
        /* .source_data          = */ plan->entry->source_data,
        /* .source_size          = */ plan->entry->source_size,
        /* .source_format        = */ plan->entry->source_format,
        /* .source_name          = */ plan->entry->source_name,
        /* .target               = */ catalog->target.c_str(),
        /* .symbol               = */ plan->entry->symbol,
        /* .config_bindings      = */ plan->config_bindings,
        /* .config_binding_count = */ plan->config_binding_count,
    };
    ggml_backend_hrx_loom_compile_output compile_output = {};
    if (!ggml_backend_hrx_loom_compile(&compile_input, &compile_output)) {
        ggml_backend_hrx_loom_compile_output_free(&compile_output);
        return nullptr;
    }

    hrx_executable_t executable = nullptr;
    if (!compile_output.executable_data || compile_output.executable_size == 0 ||
        !GGML_HRX_LOOM_CHECK(hrx_executable_load_data(catalog->device, compile_output.executable_data,
                                                      compile_output.executable_size, "amdgpu",
                                                      catalog->target.c_str(), &executable))) {
        ggml_backend_hrx_loom_compile_output_free(&compile_output);
        return nullptr;
    }

    uint32_t export_ordinal = 0;
    if (!GGML_HRX_LOOM_CHECK(hrx_executable_lookup_export_by_name(executable, plan->entry->symbol, &export_ordinal))) {
        hrx_executable_release(executable);
        ggml_backend_hrx_loom_compile_output_free(&compile_output);
        return nullptr;
    }

    hrx_executable_export_info_t export_info = {};
    if (!GGML_HRX_LOOM_CHECK(hrx_executable_export_info(executable, export_ordinal, &export_info))) {
        hrx_executable_release(executable);
        ggml_backend_hrx_loom_compile_output_free(&compile_output);
        return nullptr;
    }

    if (!ggml_backend_hrx_export_abi_matches(__func__, plan->entry->id, export_info, plan->entry->binding_count,
                                             plan->entry->parameter_count, plan->entry->constant_byte_length)) {
        hrx_executable_release(executable);
        ggml_backend_hrx_loom_compile_output_free(&compile_output);
        return nullptr;
    }

    auto route            = std::make_unique<ggml_backend_hrx_loaded_loom_route>();
    route->entry          = plan->entry;
    route->config_bindings.assign(
        plan->config_bindings,
        plan->config_bindings + plan->config_binding_count);
    route->cache_key      = cache_key;
    route->executable     = executable;
    route->export_ordinal = export_ordinal;
    route->export_info    = export_info;
    if (compile_output.report_json && compile_output.report_json_size > 0) {
        route->report_json.assign(compile_output.report_json, compile_output.report_json_size);
    }
    catalog->routes.push_back(std::move(route));
    ggml_backend_hrx_loom_compile_output_free(&compile_output);
    auto * loaded_route = catalog->routes.back().get();
    auto & bucket = catalog->route_buckets[plan->entry];
    bucket.routes.push_back(loaded_route);
    bucket.most_recent = loaded_route;
    ggml_backend_hrx_loom_cache_loaded_route(
        plan, catalog, loaded_route);
    return loaded_route;
}

static bool ggml_backend_hrx_loom_resolve_loaded_routes(
    ggml_backend_hrx_loom_catalog *              catalog,
    const ggml_backend_hrx_loom_execution_plan * plan) {
    for (size_t i = 0; i < plan->prepass_count; ++i) {
        if (!ggml_backend_hrx_loom_get_loaded_route(
                catalog, &plan->prepasses[i].kernel)) {
            return false;
        }
    }
    return ggml_backend_hrx_loom_get_loaded_route(
               catalog, &plan->main) != nullptr;
}

static bool ggml_backend_hrx_loom_dispatch_kernel_plan_impl(
    ggml_backend_hrx_loom_catalog *           catalog,
    hrx_stream_t                              stream,
    const ggml_backend_hrx_loom_kernel_plan * plan,
    const hrx_buffer_ref_t *                  bindings_override = nullptr) {
    if (!plan || !plan->entry) {
        return false;
    }

    auto * route = ggml_backend_hrx_loom_get_loaded_route(catalog, plan);
    if (!route || !stream) {
        return false;
    }

    if (!ggml_backend_hrx_dispatch_abi_matches(__func__, plan->entry->id, plan->binding_count,
                                               plan->entry->binding_count, plan->constants_size,
                                               plan->entry->constant_byte_length)) {
        return false;
    }

    return GGML_HRX_LOOM_CHECK(hrx_stream_dispatch(stream, route->executable, route->export_ordinal, &plan->dispatch,
                                                   plan->constants, plan->constants_size,
                                                   bindings_override ? bindings_override : plan->bindings,
                                                   plan->binding_count, HRX_DISPATCH_FLAG_NONE));
}

static ggml_backend_hrx_loom_scratch_state * ggml_backend_hrx_loom_find_scratch(
    ggml_backend_hrx_loom_invocation_context * context,
    const char *                               name) {
    if (!context || !name) {
        return nullptr;
    }
    for (const auto & state : context->scratch) {
        if (state->name == name) {
            return state.get();
        }
    }
    auto state  = std::make_unique<ggml_backend_hrx_loom_scratch_state>();
    state->name = name;
    context->scratch.push_back(std::move(state));
    return context->scratch.back().get();
}

static bool ggml_backend_hrx_loom_reserve_scratch(
    ggml_backend_hrx_loom_invocation_context * context,
    ggml_backend_hrx_loom_scratch_state *      state,
    const ggml_backend_hrx_loom_scratch_plan & plan) {
    if (!context || !context->device || !state || plan.bytes == 0) {
        return false;
    }
    if (state->buffer && state->capacity >= plan.bytes) {
        return true;
    }

    if (state->buffer) {
        state->retired_buffers.push_back(state->buffer);
        state->buffer   = nullptr;
        state->capacity = 0;
        state->clear_cache();
    }

    if (plan.bytes > SIZE_MAX / 2) {
        return false;
    }
    const size_t requested = std::max(plan.bytes * 2, plan.minimum_capacity);
    if (requested > SIZE_MAX - (GGML_BACKEND_HRX_LOOM_SCRATCH_ALIGNMENT - 1)) {
        return false;
    }
    const size_t capacity =
        ggml_backend_hrx_loom_align_up(requested, GGML_BACKEND_HRX_LOOM_SCRATCH_ALIGNMENT);
    hrx_buffer_params_t params = {
        /* .type           = */ HRX_MEMORY_TYPE_DEVICE_LOCAL,
        /* .access         = */ HRX_MEMORY_ACCESS_ALL,
        /* .usage          = */ HRX_BUFFER_USAGE_DEFAULT,
        /* .queue_affinity = */ 0,
    };
    if (!GGML_HRX_LOOM_CHECK(
            hrx_allocator_allocate_buffer(hrx_device_allocator(context->device), params, capacity, &state->buffer))) {
        state->buffer = nullptr;
        return false;
    }
    state->capacity = capacity;
    return true;
}

static const ggml_tensor * ggml_backend_hrx_loom_scratch_owner(const ggml_tensor * tensor) {
    const ggml_tensor * owner = tensor;
    while (owner && owner->view_src && owner->op == GGML_OP_RESHAPE && owner->view_offs == 0 &&
           owner->type == owner->view_src->type && ggml_is_contiguous(owner) && ggml_is_contiguous(owner->view_src) &&
           ggml_nbytes(owner) == ggml_nbytes(owner->view_src)) {
        owner = owner->view_src;
    }
    return owner;
}

static bool ggml_backend_hrx_loom_buffer_refs_overlap(const hrx_buffer_ref_t & lhs,
                                                      const hrx_buffer_ref_t & rhs) {
    return lhs.buffer == rhs.buffer && lhs.offset < rhs.offset + rhs.length && rhs.offset < lhs.offset + lhs.length;
}

static bool ggml_backend_hrx_loom_is_metadata_op(const ggml_tensor * node) {
    if (!node) {
        return true;
    }
    switch (node->op) {
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

static const ggml_tensor * ggml_backend_hrx_loom_metadata_parent(
    const ggml_tensor * tensor) {
    if (!tensor || !ggml_backend_hrx_loom_is_metadata_op(tensor)) {
        return nullptr;
    }
    return tensor->src[0] ? tensor->src[0] : tensor->view_src;
}

static bool ggml_backend_hrx_loom_tensor_follows_metadata_path_runtime(
    const ggml_tensor * value,
    const ggml_tensor * source) {
    if (value == source) {
        return true;
    }
    while (value && ggml_backend_hrx_loom_is_metadata_op(value)) {
        const ggml_tensor * next =
            ggml_backend_hrx_loom_metadata_parent(value);
        if (!next || next == value) {
            return false;
        }
        value = next;
        if (value == source) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx_loom_tensor_is_graph_output_runtime(
    const ggml_backend_hrx_loom_op_request * request,
    const ggml_tensor *                      tensor) {
    if (!request || !request->cgraph || !tensor) {
        return true;
    }
    if ((tensor->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return true;
    }
    for (int i = 0; i < request->cgraph->n_nodes; ++i) {
        const ggml_tensor * node = request->cgraph->nodes[i];
        if (node && (node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 &&
            ggml_backend_hrx_loom_tensor_follows_metadata_path_runtime(
                node, tensor)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx_loom_node_index_is_listed_runtime(
    int         node_index,
    const int * node_indices,
    int         node_count) {
    for (int i = 0; i < node_count; ++i) {
        if (node_indices[i] == node_index) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx_loom_tensor_is_transient_runtime(
    const ggml_backend_hrx_loom_op_request * request,
    const ggml_tensor *                      tensor,
    const int *                              consumed_node_indices,
    int                                      consumed_node_count) {
    if (!request || !request->cgraph || !tensor ||
        !consumed_node_indices || consumed_node_count <= 0 ||
        ggml_backend_hrx_loom_tensor_is_graph_output_runtime(
            request, tensor)) {
        return false;
    }
    bool consumed_inside = false;
    for (int i = 0; i < request->cgraph->n_nodes; ++i) {
        const ggml_tensor * consumer = request->cgraph->nodes[i];
        if (!consumer || consumer == tensor ||
            ggml_backend_hrx_loom_is_metadata_op(consumer) ||
            ggml_nelements(consumer) == 0) {
            continue;
        }
        bool consumes_tensor = false;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            consumes_tensor = consumes_tensor ||
                ggml_backend_hrx_loom_tensor_follows_metadata_path_runtime(
                    consumer->src[j], tensor);
        }
        if (!consumes_tensor) {
            continue;
        }
        if (!ggml_backend_hrx_loom_node_index_is_listed_runtime(
                i, consumed_node_indices, consumed_node_count)) {
            return false;
        }
        consumed_inside = true;
    }
    return consumed_inside;
}

static bool ggml_backend_hrx_loom_tensor_consumers_through_view_runtime(
    const ggml_backend_hrx_loom_op_request * request,
    const ggml_tensor *                      owner,
    const ggml_tensor *                      view) {
    if (!request || !request->cgraph || !owner || !view ||
        owner == view ||
        !ggml_backend_hrx_loom_tensor_follows_metadata_path_runtime(
            view, owner) ||
        (owner->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }
    for (int i = 0; i < request->cgraph->n_nodes; ++i) {
        const ggml_tensor * node = request->cgraph->nodes[i];
        if (!node) {
            continue;
        }
        if ((node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 &&
            ggml_backend_hrx_loom_tensor_follows_metadata_path_runtime(
                node, owner) &&
            !ggml_backend_hrx_loom_tensor_follows_metadata_path_runtime(
                node, view)) {
            return false;
        }
        if (ggml_backend_hrx_loom_is_metadata_op(node) ||
            ggml_nelements(node) == 0) {
            continue;
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * input = node->src[j];
            if (ggml_backend_hrx_loom_tensor_follows_metadata_path_runtime(
                    input, owner) &&
                !ggml_backend_hrx_loom_tensor_follows_metadata_path_runtime(
                    input, view)) {
                return false;
            }
        }
    }
    return true;
}

static size_t ggml_backend_hrx_loom_graph_fact_slot(
    const ggml_backend_hrx_loom_graph_facts & facts,
    const ggml_tensor *                       tensor) {
    if (!facts.cgraph || !tensor || facts.visited_hash_size == 0 ||
        !facts.visited_keys || !facts.visited_used ||
        facts.tensors.size() != facts.visited_hash_size) {
        return GGML_HASHSET_FULL;
    }
    const ggml_hash_set & visited = facts.cgraph->visited_hash_set;
    if (visited.size != facts.visited_hash_size ||
        visited.keys != facts.visited_keys ||
        visited.used != facts.visited_used) {
        return GGML_HASHSET_FULL;
    }
    const size_t slot = ggml_hash_find(&visited, tensor);
    if (slot == GGML_HASHSET_FULL || slot >= facts.tensors.size() ||
        !ggml_bitset_get(visited.used, slot) ||
        visited.keys[slot] != tensor) {
        return GGML_HASHSET_FULL;
    }
    return slot;
}

static bool ggml_backend_hrx_loom_record_graph_observation_path(
    ggml_backend_hrx_loom_graph_facts * facts,
    const ggml_tensor *                 endpoint,
    int                                 consumer_node_index,
    bool                                graph_output) {
    if (!facts || !endpoint) {
        return endpoint == nullptr;
    }
    const ggml_tensor * value = endpoint;
    size_t traversed = 0;
    while (value) {
        const size_t slot =
            ggml_backend_hrx_loom_graph_fact_slot(*facts, value);
        if (slot != GGML_HASHSET_FULL) {
            auto & tensor_facts = facts->tensors[slot];
            if (tensor_facts.observation_count == SIZE_MAX) {
                return false;
            }
            ++tensor_facts.observation_count;
            tensor_facts.graph_output =
                tensor_facts.graph_output || graph_output;

            if (consumer_node_index >= 0 &&
                consumer_node_index < facts->n_nodes &&
                facts->nodes[consumer_node_index] != value &&
                tensor_facts.consumer_count <=
                    GGML_BACKEND_HRX_LOOM_MAX_CONSUMED_NODES) {
                const uint8_t count = tensor_facts.consumer_count;
                if (count == 0 ||
                    tensor_facts.consumer_indices[count - 1] !=
                        consumer_node_index) {
                    tensor_facts.consumer_indices[count] =
                        consumer_node_index;
                    tensor_facts.consumer_count =
                        static_cast<uint8_t>(count + 1);
                }
            }
        }

        if (!ggml_backend_hrx_loom_is_metadata_op(value)) {
            break;
        }
        const ggml_tensor * next =
            ggml_backend_hrx_loom_metadata_parent(value);
        if (next == value || ++traversed > facts->visited_hash_size) {
            return false;
        }
        value = next;
    }
    return true;
}

static bool ggml_backend_hrx_loom_scratch_source_unchanged(
    const ggml_backend_hrx_loom_op_request * request,
    const ggml_backend_hrx_loom_scratch_state & state,
    const hrx_buffer_ref_t &                    source_binding) {
    if (!request || !request->cgraph || state.cache_node_index < 0 ||
        request->node_index <= state.cache_node_index || request->node_index > request->cgraph->n_nodes) {
        return false;
    }
    for (int i = state.cache_node_index + 1; i < request->node_index; ++i) {
        const ggml_tensor * between = request->cgraph->nodes[i];
        if (ggml_backend_hrx_loom_is_metadata_op(between)) {
            continue;
        }
        hrx_buffer_ref_t destination = {};
        if (!ggml_backend_hrx_loom_bind_tensor(request, between, &destination) ||
            ggml_backend_hrx_loom_buffer_refs_overlap(source_binding, destination)) {
            return false;
        }
    }
    return true;
}

static bool ggml_backend_hrx_loom_scratch_matches(
    const ggml_backend_hrx_loom_op_request * request,
    ggml_backend_hrx_loom_scratch_state *    state,
    const ggml_backend_hrx_loom_prepass_plan & prepass,
    const hrx_buffer_ref_t &                  region) {
    if (!request || !state || !prepass.kernel.entry || !prepass.cache_source) {
        return false;
    }
    const ggml_backend_hrx_loom_scratch_cache_key cached = {
        /* .source_owner = */ state->cache_owner,
        /* .source       = */ state->cache_source_binding,
        /* .region       = */ state->cache_region,
        /* .epoch        = */ state->cache_epoch,
        /* .artifact     = */ state->cache_artifact.c_str(),
    };
    const ggml_backend_hrx_loom_scratch_cache_key candidate = {
        /* .source_owner = */ ggml_backend_hrx_loom_scratch_owner(prepass.cache_source),
        /* .source       = */ prepass.cache_source_binding,
        /* .region       = */ region,
        /* .epoch        = */ request->execution_epoch,
        /* .artifact     = */ prepass.kernel.entry->id,
    };
    if (!ggml_backend_hrx_loom_scratch_cache_key_matches(cached, candidate)) {
        return false;
    }
    if (state->cache_source == prepass.cache_source) {
        return true;
    }
    if (!ggml_backend_hrx_loom_scratch_source_unchanged(request, *state, prepass.cache_source_binding)) {
        return false;
    }
    state->cache_source     = prepass.cache_source;
    state->cache_node_index = request->node_index;
    return true;
}

static void ggml_backend_hrx_loom_set_scratch_source(
    const ggml_backend_hrx_loom_op_request * request,
    ggml_backend_hrx_loom_scratch_state *    state,
    const ggml_backend_hrx_loom_prepass_plan & prepass,
    const hrx_buffer_ref_t &                  region) {
    state->cache_source         = prepass.cache_source;
    state->cache_owner          = ggml_backend_hrx_loom_scratch_owner(prepass.cache_source);
    state->cache_source_binding = prepass.cache_source_binding;
    state->cache_region         = region;
    state->cache_node_index     = request->node_index;
    state->cache_epoch          = request->execution_epoch;
    state->cache_artifact       = prepass.kernel.entry->id;
}

static bool ggml_backend_hrx_loom_patch_scratch_bindings(
    const ggml_backend_hrx_loom_kernel_plan *                                 kernel,
    const ggml_backend_hrx_loom_execution_plan *                              plan,
    const std::array<ggml_backend_hrx_loom_scratch_state *,
                     GGML_BACKEND_HRX_LOOM_MAX_SCRATCH> &                     states,
    std::array<hrx_buffer_ref_t, GGML_BACKEND_HRX_LOOM_MAX_BINDINGS> *         resolved,
    const hrx_buffer_ref_t **                                                  out_bindings) {
    if (!kernel || !plan || !resolved || !out_bindings ||
        kernel->binding_count > GGML_BACKEND_HRX_LOOM_MAX_BINDINGS) {
        return false;
    }
    *out_bindings = kernel->bindings;
    bool copied = false;
    for (size_t i = 0; i < kernel->binding_count; ++i) {
        const uint8_t encoded_index = kernel->binding_scratch_index[i];
        if (encoded_index == 0) {
            continue;
        }
        const size_t scratch_index = static_cast<size_t>(encoded_index - 1);
        if (scratch_index >= plan->scratch_count || scratch_index >= states.size() || !states[scratch_index]) {
            return false;
        }
        const size_t offset = kernel->binding_scratch_offset[i];
        const size_t length = kernel->binding_scratch_length[i];
        if (offset > plan->scratch[scratch_index].bytes || length > plan->scratch[scratch_index].bytes - offset) {
            return false;
        }
        if (!copied) {
            std::memcpy(
                resolved->data(),
                kernel->bindings,
                kernel->binding_count * sizeof(kernel->bindings[0]));
            *out_bindings = resolved->data();
            copied = true;
        }
        (*resolved)[i] = {
            /* .buffer = */ states[scratch_index]->buffer,
            /* .offset = */ offset,
            /* .length = */ length,
        };
    }
    return true;
}

}  // namespace

bool ggml_backend_hrx_loom_graph_facts::build(
    const ggml_backend_hrx_loom_op_request * request) {
    cgraph            = nullptr;
    uid               = 0;
    execution_epoch   = 0;
    n_nodes           = 0;
    nodes             = nullptr;
    visited_keys      = nullptr;
    visited_used      = nullptr;
    visited_hash_size = 0;
    tensors.clear();

    if (!request || !request->cgraph ||
        request->cgraph->n_nodes < 0 ||
        (request->cgraph->n_nodes > 0 && !request->cgraph->nodes) ||
        request->cgraph->visited_hash_set.size == 0 ||
        !request->cgraph->visited_hash_set.keys ||
        !request->cgraph->visited_hash_set.used) {
        return false;
    }

    cgraph            = request->cgraph;
    uid               = cgraph->uid;
    execution_epoch   = request->execution_epoch;
    n_nodes           = cgraph->n_nodes;
    nodes             = cgraph->nodes;
    visited_keys      = cgraph->visited_hash_set.keys;
    visited_used      = cgraph->visited_hash_set.used;
    visited_hash_size = cgraph->visited_hash_set.size;
    tensors.resize(visited_hash_size);

    for (int node_index = 0; node_index < n_nodes; ++node_index) {
        const ggml_tensor * node = nodes[node_index];
        if (!node) {
            continue;
        }
        if ((node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 &&
            !ggml_backend_hrx_loom_record_graph_observation_path(
                this,
                node,
                -1,
                true)) {
            tensors.clear();
            cgraph = nullptr;
            return false;
        }
        if (ggml_backend_hrx_loom_is_metadata_op(node) ||
            ggml_nelements(node) == 0) {
            continue;
        }
        for (int source_index = 0;
             source_index < GGML_MAX_SRC;
             ++source_index) {
            const ggml_tensor * source = node->src[source_index];
            if (source &&
                !ggml_backend_hrx_loom_record_graph_observation_path(
                    this,
                    source,
                    node_index,
                    false)) {
                tensors.clear();
                cgraph = nullptr;
                return false;
            }
        }
    }
    return true;
}

bool ggml_backend_hrx_loom_graph_facts::matches(
    const ggml_backend_hrx_loom_op_request * request) const {
    if (!request || !request->cgraph || !cgraph ||
        request->cgraph != cgraph ||
        request->cgraph->uid != uid ||
        request->cgraph->n_nodes != n_nodes ||
        request->cgraph->nodes != nodes ||
        request->cgraph->visited_hash_set.size != visited_hash_size ||
        request->cgraph->visited_hash_set.keys != visited_keys ||
        request->cgraph->visited_hash_set.used != visited_used ||
        tensors.size() != visited_hash_size) {
        return false;
    }
    return uid != 0 || request->execution_epoch == execution_epoch;
}

const ggml_backend_hrx_loom_tensor_graph_facts *
ggml_backend_hrx_loom_graph_facts::find(
    const ggml_tensor * tensor) const {
    const size_t slot =
        ggml_backend_hrx_loom_graph_fact_slot(*this, tensor);
    return slot == GGML_HASHSET_FULL ? nullptr : &tensors[slot];
}

const ggml_backend_hrx_loom_graph_facts *
ggml_backend_hrx_loom_graph_fact_cache::resolve(
    const ggml_backend_hrx_loom_op_request * request) {
    if (!request || !request->cgraph) {
        return nullptr;
    }
    for (auto it = graphs.begin(); it != graphs.end();) {
        if (!*it) {
            it = graphs.erase(it);
            continue;
        }
        if ((*it)->matches(request)) {
            return it->get();
        }
        if ((*it)->cgraph == request->cgraph) {
            it = graphs.erase(it);
            continue;
        }
        ++it;
    }

    auto facts =
        std::make_unique<ggml_backend_hrx_loom_graph_facts>();
    if (!facts->build(request)) {
        return nullptr;
    }
    graphs.push_back(std::move(facts));
    return graphs.back().get();
}

static const ggml_backend_hrx_loom_graph_facts *
ggml_backend_hrx_loom_resolve_graph_facts(
    const ggml_backend_hrx_loom_op_request * request,
    ggml_backend_hrx_loom_graph_facts *      uncached) {
    if (!request || !request->cgraph) {
        return nullptr;
    }
    if (request->invocation_context) {
        return request->invocation_context->graph_facts.resolve(request);
    }
    return uncached && uncached->build(request) ? uncached : nullptr;
}

bool ggml_backend_hrx_loom_tensor_is_transient_at_indices_cached(
    const ggml_backend_hrx_loom_op_request * request,
    const ggml_tensor *                      tensor,
    const int *                              consumed_node_indices,
    int                                      consumed_node_count) {
    if (!request || !request->cgraph || !tensor ||
        !consumed_node_indices || consumed_node_count <= 0 ||
        consumed_node_count >
            GGML_BACKEND_HRX_LOOM_MAX_CONSUMED_NODES ||
        (tensor->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }
    ggml_backend_hrx_loom_graph_facts uncached;
    const auto * graph_facts =
        ggml_backend_hrx_loom_resolve_graph_facts(
            request,
            &uncached);
    const auto * tensor_facts =
        graph_facts ? graph_facts->find(tensor) : nullptr;
    if (!tensor_facts) {
        return ggml_backend_hrx_loom_tensor_is_transient_runtime(
            request,
            tensor,
            consumed_node_indices,
            consumed_node_count);
    }
    if (tensor_facts->graph_output ||
        tensor_facts->consumer_count == 0 ||
        tensor_facts->consumer_count >
            GGML_BACKEND_HRX_LOOM_MAX_CONSUMED_NODES) {
        return false;
    }
    for (uint8_t i = 0; i < tensor_facts->consumer_count; ++i) {
        bool consumed = false;
        for (int j = 0; j < consumed_node_count; ++j) {
            consumed = consumed ||
                consumed_node_indices[j] ==
                    tensor_facts->consumer_indices[i];
        }
        if (!consumed) {
            return false;
        }
    }
    return true;
}

bool ggml_backend_hrx_loom_tensor_consumers_through_view_cached(
    const ggml_backend_hrx_loom_op_request * request,
    const ggml_tensor *                      owner,
    const ggml_tensor *                      view) {
    if (!request || !request->cgraph || !owner || !view ||
        owner == view ||
        (owner->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
        !ggml_backend_hrx_loom_tensor_follows_metadata_path_runtime(
            view,
            owner)) {
        return false;
    }
    ggml_backend_hrx_loom_graph_facts uncached;
    const auto * graph_facts =
        ggml_backend_hrx_loom_resolve_graph_facts(
            request,
            &uncached);
    const auto * owner_facts =
        graph_facts ? graph_facts->find(owner) : nullptr;
    const auto * view_facts =
        graph_facts ? graph_facts->find(view) : nullptr;
    if (!owner_facts || !view_facts) {
        return ggml_backend_hrx_loom_tensor_consumers_through_view_runtime(
            request, owner, view);
    }
    return owner_facts->observation_count ==
           view_facts->observation_count;
}

bool ggml_backend_hrx_loom_bind_tensor(const ggml_backend_hrx_loom_op_request * request,
                                       const ggml_tensor *                      tensor,
                                       hrx_buffer_ref_t *                       out_ref) {
    return request && request->bind_tensor && request->bind_tensor(request->bind_tensor_user_data, tensor, out_ref);
}

bool ggml_backend_hrx_loom_storage_layout_matches(
    const ggml_backend_hrx_loom_op_request * request,
    const ggml_tensor *                      tensor,
    const char *                             expected) {
    if (!request || !tensor || !expected || !request->storage_layout) {
        return false;
    }
    const char * actual =
        request->storage_layout(request->storage_layout_user_data, tensor);
    return actual && std::strcmp(actual, expected) == 0;
}

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_match_request(
    ggml_backend_hrx_loom_catalog * catalog, const ggml_backend_hrx_loom_op_request * request) {
    return ggml_backend_hrx_loom_match_or_prepare_request(catalog, request, nullptr);
}

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_prepare_plan(
    ggml_backend_hrx_loom_catalog *          catalog,
    const ggml_backend_hrx_loom_op_request * request,
    ggml_backend_hrx_loom_execution_plan *   plan) {
    return ggml_backend_hrx_loom_match_or_prepare_request(catalog, request, plan);
}

bool ggml_backend_hrx_loom_compile(const ggml_backend_hrx_loom_compile_input * input,
                                   ggml_backend_hrx_loom_compile_output *      output) {
    if (output) {
        *output = {};
    }
    if (!input || !output || !input->source_data || input->source_size == 0 || !input->source_format ||
        !input->source_name || !input->target || !input->symbol) {
        GGML_LOG_ERROR("%s: invalid Loom compile input\n", __func__);
        return false;
    }
    if (input->config_binding_count > 0 && !input->config_bindings) {
        GGML_LOG_ERROR("%s: invalid Loom compile config bindings\n", __func__);
        return false;
    }
    if (std::strcmp(input->source_format, "loom-text") != 0) {
        GGML_LOG_ERROR("%s: unsupported Loom source format %s\n", __func__, input->source_format);
        return false;
    }

    std::vector<loomc_config_binding_t> config_bindings;
    config_bindings.reserve(input->config_binding_count);
    for (size_t i = 0; i < input->config_binding_count; ++i) {
        const ggml_backend_hrx_loom_config_binding & binding = input->config_bindings[i];
        config_bindings.push_back({
            /* .key   = */ loomc_make_cstring_view(binding.name),
            /* .value = */ loomc_make_cstring_view(binding.value),
        });
    }

    ggml_backend_hrx_loom_compile_state state;
    loomc_status_t status =
        loomc_target_environment_create_amdgpu(loomc_allocator_system(), &state.target_environment);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }

    loomc_context_target_options_t context_target_options = {
        /* .type               = */ LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        /* .structure_size     = */ sizeof(loomc_context_target_options_t),
        /* .next               = */ nullptr,
        /* .target_environment = */ state.target_environment,
    };
    loomc_context_options_t context_options = {
        /* .type           = */ LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        /* .structure_size = */ sizeof(loomc_context_options_t),
        /* .next           = */ &context_target_options,
    };
    status = loomc_context_create(&context_options, loomc_allocator_system(), &state.context);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }

    status = loomc_workspace_create(nullptr, loomc_allocator_system(), &state.workspace);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }

    loomc_source_options_t source_options = {
        /* .type              = */ LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        /* .structure_size    = */ sizeof(loomc_source_options_t),
        /* .next              = */ nullptr,
        /* .format            = */ LOOMC_SOURCE_FORMAT_TEXT,
        /* .identifier        = */ loomc_make_cstring_view(input->source_name),
        /* .contents          = */ loomc_make_byte_span(input->source_data, input->source_size),
        /* .storage           = */ LOOMC_SOURCE_STORAGE_BORROWED,
        /* .release           = */ nullptr,
        /* .release_user_data = */ nullptr,
    };
    status = loomc_source_create(&source_options, loomc_allocator_system(), &state.source);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }

    loomc_amdgpu_profile_options_t profile_options = {
        /* .type           = */ LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
        /* .structure_size = */ sizeof(loomc_amdgpu_profile_options_t),
        /* .next           = */ nullptr,
        /* .identifier     = */ loomc_make_cstring_view(input->target),
        /* .processor      = */ loomc_make_cstring_view(input->target),
    };
    status = loomc_target_profile_create_amdgpu(state.target_environment, &profile_options, loomc_allocator_system(),
                                                &state.target_profile);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }
    status = loomc_compiler_create(state.context, nullptr, loomc_allocator_system(), &state.compiler);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }

    loomc_target_pipeline_options_t pipeline_options = {
        /* .type                     = */ LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
        /* .structure_size           = */ sizeof(loomc_target_pipeline_options_t),
        /* .next                     = */ nullptr,
        /* .identifier               = */ loomc_make_cstring_view("hrx-loom-prepared-low"),
        /* .kind                     = */ LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
        /* .control_flow_lowering    = */ LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
        /* .source_to_low_max_errors = */ 20,
    };
    status = loomc_pass_program_create_from_target_pipeline(state.context, &pipeline_options, loomc_allocator_system(),
                                                            &state.pass_program, &state.result);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }
    if (!ggml_backend_hrx_loom_require_success(state.result, "target pipeline preparation")) {
        return false;
    }
    state.reset_result();

    status = loomc_module_deserialize_from_source(state.context, state.workspace, state.source, nullptr,
                                                  loomc_allocator_system(), &state.module, &state.result);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }
    if (!ggml_backend_hrx_loom_require_success(state.result, "source deserialization")) {
        return false;
    }
    state.reset_result();

    const loomc_target_specialization_t target_specialization = {
        /* .function_symbol = */ loomc_make_cstring_view(input->symbol),
        /* .target_profile  = */ state.target_profile,
    };
    loomc_target_specialization_options_t target_options = {
        /* .type                 = */ LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
        /* .structure_size       = */ sizeof(loomc_target_specialization_options_t),
        /* .next                 = */ nullptr,
        /* .specializations      = */ &target_specialization,
        /* .specialization_count = */ 1,
    };
    loomc_compile_options_t compile_options = {
        /* .type           = */ LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        /* .structure_size = */ sizeof(loomc_compile_options_t),
        /* .next           = */ &target_options,
        /* .module_name    = */ loomc_make_cstring_view(input->symbol),
        /* .artifact_flags = */ 0,
        /* .config         = */
            {
                /* .bindings      = */ config_bindings.empty() ? nullptr : config_bindings.data(),
                /* .binding_count = */ config_bindings.size(),
                /* .json_object   = */ loomc_string_view_empty(),
                /* .flags         = */ LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
                                      LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
            },
    };
    status = loomc_compile_module(state.compiler, state.workspace, state.pass_program, state.module, &compile_options,
                                  loomc_allocator_system(), &state.result);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }
    if (!ggml_backend_hrx_loom_require_success(state.result, "compilation")) {
        return false;
    }
    state.reset_result();

    const loomc_option_entry_t emit_entries[] = {
        {
            /* .key   = */ loomc_make_cstring_view(LOOMC_EMIT_OPTION_KEY_IDENTIFIER),
            /* .value = */ loomc_make_cstring_view(input->symbol),
        },
    };
    loomc_amdgpu_emit_options_t amdgpu_options = {
        /* .type            = */ LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
        /* .structure_size  = */ sizeof(loomc_amdgpu_emit_options_t),
        /* .next            = */ nullptr,
        /* .runtime_globals = */ LOOMC_AMDGPU_RUNTIME_GLOBAL_NONE,
    };
    loomc_compile_report_options_t report_options = {
        /* .type           = */ LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
        /* .structure_size = */ sizeof(loomc_compile_report_options_t),
        /* .next           = */ &amdgpu_options,
        /* .mode           = */ LOOMC_COMPILE_REPORT_MODE_SUMMARY,
        /* .identifier     = */ loomc_string_view_empty(),
    };
    loomc_option_dict_t option_dict = {
        /* .type           = */ LOOMC_STRUCTURE_TYPE_OPTION_DICT,
        /* .structure_size = */ sizeof(loomc_option_dict_t),
        /* .next           = */ &report_options,
        /* .entries        = */ emit_entries,
        /* .entry_count    = */ 1,
    };
    loomc_emit_options_t emit_options = {
        /* .type            = */ LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
        /* .structure_size  = */ sizeof(loomc_emit_options_t),
        /* .next            = */ &option_dict,
        /* .artifact_format = */ loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
        /* .identifier      = */ loomc_make_cstring_view(input->symbol),
        /* .artifact_flags  = */ LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
    };
    status = loomc_emit_module(state.target_environment, state.workspace, state.module, &emit_options,
                               loomc_allocator_system(), &state.result);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }
    if (!ggml_backend_hrx_loom_require_success(state.result, "AMDGPU emission")) {
        return false;
    }

    const loomc_artifact_t * executable =
        ggml_backend_hrx_loom_find_artifact(state.result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
                                            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO));
    if (!ggml_backend_hrx_loom_copy_artifact_bytes(executable, &output->executable_data, &output->executable_size)) {
        ggml_backend_hrx_loom_compile_output_free(output);
        return false;
    }

    const loomc_artifact_t * report =
        ggml_backend_hrx_loom_find_artifact(state.result, LOOMC_ARTIFACT_KIND_REPORT,
                                            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON));
    if (report && !ggml_backend_hrx_loom_copy_artifact_bytes(report, reinterpret_cast<void **>(&output->report_json),
                                                            &output->report_json_size)) {
        ggml_backend_hrx_loom_compile_output_free(output);
        return false;
    }

    return true;
}

ggml_backend_hrx_loom_catalog * ggml_backend_hrx_loom_catalog_new(hrx_device_t device, const char * architecture) {
    if (!device || !architecture || architecture[0] == '\0') {
        return nullptr;
    }

    auto * catalog = new (std::nothrow) ggml_backend_hrx_loom_catalog();
    if (!catalog) {
        return nullptr;
    }

    hrx_device_retain(device);
    catalog->device       = device;
    catalog->architecture = architecture;
    catalog->target       = ggml_backend_hrx_architecture_base(architecture);
    return catalog;
}

void ggml_backend_hrx_loom_catalog_free(ggml_backend_hrx_loom_catalog * catalog) {
    delete catalog;
}

ggml_backend_hrx_loom_invocation_context * ggml_backend_hrx_loom_invocation_context_new(hrx_device_t device) {
    if (!device) {
        return nullptr;
    }
    auto * context = new (std::nothrow) ggml_backend_hrx_loom_invocation_context();
    if (!context) {
        return nullptr;
    }
    hrx_device_retain(device);
    context->device = device;
    return context;
}

void ggml_backend_hrx_loom_invocation_context_free(ggml_backend_hrx_loom_invocation_context * context) {
    delete context;
}

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_supports_op(
    ggml_backend_hrx_loom_catalog *         catalog,
    const ggml_tensor *                     op,
    ggml_backend_hrx_loom_storage_layout_fn storage_layout,
    void *                                  storage_layout_user_data) {
    if (!catalog || !op) {
        return ggml_backend_hrx_loom_unsupported(GGML_BACKEND_HRX_LOOM_UNSUPPORTED_NO_ROUTE);
    }

    const ggml_backend_hrx_loom_op_request request = {
        /* .op                    = */ op,
        /* .cgraph                = */ nullptr,
        /* .node_index            = */ -1,
        /* .stream                = */ nullptr,
        /* .bind_tensor           = */ nullptr,
        /* .bind_tensor_user_data = */ nullptr,
        /* .storage_layout        = */ storage_layout,
        /* .storage_layout_user_data = */ storage_layout_user_data,
        /* .invocation_context    = */ nullptr,
        /* .execution_epoch       = */ 0,
    };
    return ggml_backend_hrx_loom_match_request(catalog, &request);
}

static bool ggml_backend_hrx_loom_dispatch_execution_plan(
    ggml_backend_hrx_loom_catalog *          catalog,
    const ggml_backend_hrx_loom_op_request * request,
    const ggml_backend_hrx_loom_execution_plan * plan) {
    if (!catalog || !request || !request->op || !request->stream || !plan ||
        plan->scratch_count > GGML_BACKEND_HRX_LOOM_MAX_SCRATCH ||
        plan->prepass_count > GGML_BACKEND_HRX_LOOM_MAX_PREPASSES ||
        ((plan->scratch_count > 0 || plan->prepass_count > 0) &&
         !request->invocation_context)) {
        return false;
    }

    std::array<ggml_backend_hrx_loom_scratch_state *, GGML_BACKEND_HRX_LOOM_MAX_SCRATCH> scratch_states = {};
    for (size_t i = 0; i < plan->scratch_count; ++i) {
        auto * state = ggml_backend_hrx_loom_find_scratch(
            request->invocation_context,
            plan->scratch[i].name);
        if (!state ||
            !ggml_backend_hrx_loom_reserve_scratch(
                request->invocation_context,
                state,
                plan->scratch[i])) {
            return false;
        }
        scratch_states[i] = state;
    }

    for (size_t i = 0; i < plan->prepass_count; ++i) {
        const auto & prepass = plan->prepasses[i];
        std::array<hrx_buffer_ref_t, GGML_BACKEND_HRX_LOOM_MAX_BINDINGS> resolved;
        const hrx_buffer_ref_t * bindings = nullptr;
        if (!ggml_backend_hrx_loom_patch_scratch_bindings(
                &prepass.kernel,
                plan,
                scratch_states,
                &resolved,
                &bindings)) {
            return false;
        }

        bool already_done = false;
        ggml_backend_hrx_loom_scratch_state * cache_state = nullptr;
        hrx_buffer_ref_t cache_region = {};
        if (prepass.cache_scratch_index != 0) {
            const size_t cache_index = static_cast<size_t>(prepass.cache_scratch_index - 1);
            if (cache_index >= plan->scratch_count || !scratch_states[cache_index] ||
                !ggml_backend_hrx_loom_resolve_cache_region(
                    plan->scratch[cache_index],
                    prepass,
                    scratch_states[cache_index]->buffer,
                    &cache_region) ||
                (prepass.cache_region_length != 0 &&
                 !ggml_backend_hrx_loom_cache_region_is_bound(prepass))) {
                return false;
            }
            cache_state = scratch_states[cache_index];
            already_done =
                ggml_backend_hrx_loom_scratch_matches(request, cache_state, prepass, cache_region);
        }
        if (already_done) {
            continue;
        }
        if (!ggml_backend_hrx_loom_dispatch_kernel_plan_impl(
                catalog,
                request->stream,
                &prepass.kernel,
                bindings) ||
            !GGML_HRX_LOOM_CHECK(hrx_stream_execution_barrier(request->stream))) {
            return false;
        }
        if (cache_state) {
            ggml_backend_hrx_loom_set_scratch_source(request, cache_state, prepass, cache_region);
        }
    }

    std::array<hrx_buffer_ref_t, GGML_BACKEND_HRX_LOOM_MAX_BINDINGS> resolved;
    const hrx_buffer_ref_t * bindings = nullptr;
    return ggml_backend_hrx_loom_patch_scratch_bindings(
               &plan->main,
               plan,
               scratch_states,
               &resolved,
               &bindings) &&
           ggml_backend_hrx_loom_dispatch_kernel_plan_impl(
               catalog,
               request->stream,
               &plan->main,
               bindings);
}

static const ggml_tensor * ggml_backend_hrx_loom_data_producer(
    const ggml_tensor * tensor) {
    const ggml_tensor * value = tensor;
    while (value &&
           ggml_backend_hrx_loom_is_metadata_op(value)) {
        const ggml_tensor * next =
            value->src[0] ? value->src[0] : value->view_src;
        if (!next || next == value) {
            return nullptr;
        }
        value = next;
    }
    return value;
}

static int ggml_backend_hrx_loom_graph_node_index(
    const ggml_cgraph * graph,
    const ggml_tensor * node) {
    if (!graph || !node) {
        return -1;
    }
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i] == node) {
            return i;
        }
    }
    return -1;
}

static bool ggml_backend_hrx_loom_plan_consumes_node(
    const ggml_backend_hrx_loom_execution_plan & plan,
    int                                           node_index) {
    for (int i = 0; i < plan.consumed_node_count; ++i) {
        if (plan.consumed_node_indices[i] == node_index) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx_loom_kernel_access_masks_are_valid(
    const ggml_backend_hrx_loom_kernel_plan & kernel) {
    if (kernel.binding_count > GGML_BACKEND_HRX_LOOM_MAX_BINDINGS) {
        return false;
    }
    const uint16_t valid_binding_mask =
        static_cast<uint16_t>(
            (UINT16_C(1) << kernel.binding_count) - 1);
    return ((kernel.deferred_read_binding_mask |
             kernel.deferred_write_binding_mask) &
            static_cast<uint16_t>(~valid_binding_mask)) == 0;
}

static bool ggml_backend_hrx_loom_deferred_reads_survive(
    const ggml_backend_hrx_loom_op_request * request,
    const ggml_backend_hrx_loom_execution_plan & plan) {
    const int owner_index = plan.dispatch_owner_node_index;
    if (!request || !request->cgraph || !request->bind_tensor ||
        owner_index <= request->node_index ||
        owner_index >= request->cgraph->n_nodes ||
        plan.prepass_count > GGML_BACKEND_HRX_LOOM_MAX_PREPASSES ||
        !ggml_backend_hrx_loom_kernel_access_masks_are_valid(
            plan.main)) {
        return false;
    }
    for (size_t i = 0; i < plan.prepass_count; ++i) {
        if (!ggml_backend_hrx_loom_kernel_access_masks_are_valid(
                plan.prepasses[i].kernel)) {
            return false;
        }
    }

    struct deferred_read {
        hrx_buffer_ref_t binding    = {};
        int              ready_index = -1;
    };
    static constexpr size_t max_deferred_reads =
        (GGML_BACKEND_HRX_LOOM_MAX_PREPASSES + 1) *
        GGML_BACKEND_HRX_LOOM_MAX_BINDINGS;
    std::array<deferred_read, max_deferred_reads> deferred_reads = {};
    size_t deferred_read_count = 0;
    struct deferred_write {
        hrx_buffer_ref_t binding      = {};
        int              logical_index = -1;
    };
    std::array<deferred_write, max_deferred_reads> deferred_writes = {};
    size_t deferred_write_count = 0;

    const auto collect_deferred_accesses =
        [&](const ggml_backend_hrx_loom_kernel_plan & kernel) {
        for (size_t i = 0; i < kernel.binding_count; ++i) {
            const uint16_t binding_bit = UINT16_C(1) << i;
            const hrx_buffer_ref_t & binding = kernel.bindings[i];
            if ((kernel.deferred_read_binding_mask & binding_bit) != 0) {
                if (!kernel.deferred_read_tensors[i] ||
                    !binding.buffer || binding.length == 0) {
                    return false;
                }
                const ggml_tensor * producer =
                    ggml_backend_hrx_loom_data_producer(
                        kernel.deferred_read_tensors[i]);
                const int producer_index =
                    ggml_backend_hrx_loom_graph_node_index(
                        request->cgraph,
                        producer);
                if (producer_index >= owner_index ||
                    (producer_index >= 0 &&
                     ggml_backend_hrx_loom_plan_consumes_node(
                         plan,
                         producer_index)) ||
                    deferred_read_count >= deferred_reads.size()) {
                    return false;
                }
                deferred_reads[deferred_read_count++] = {
                    /* .binding     = */ binding,
                    /* .ready_index = */ producer_index > request->node_index ?
                        producer_index : request->node_index,
                };
            }
            if ((kernel.deferred_write_binding_mask & binding_bit) != 0) {
                if (!kernel.deferred_write_tensors[i] ||
                    !binding.buffer || binding.length == 0) {
                    return false;
                }
                const ggml_tensor * logical_producer =
                    ggml_backend_hrx_loom_data_producer(
                        kernel.deferred_write_tensors[i]);
                const int logical_index =
                    ggml_backend_hrx_loom_graph_node_index(
                        request->cgraph,
                        logical_producer);
                if (logical_index > owner_index) {
                    if (!ggml_backend_hrx_loom_plan_consumes_node(
                            plan,
                            logical_index) ||
                        deferred_write_count >= deferred_writes.size()) {
                        return false;
                    }
                    deferred_writes[deferred_write_count++] = {
                        /* .binding       = */ binding,
                        /* .logical_index = */ logical_index,
                    };
                }
            }
        }
        return true;
    };

    for (size_t i = 0; i < plan.prepass_count; ++i) {
        if (!collect_deferred_accesses(plan.prepasses[i].kernel)) {
            return false;
        }
    }
    if (!collect_deferred_accesses(plan.main)) {
        return false;
    }

    for (int node_index = request->node_index + 1;
         node_index < owner_index;
         ++node_index) {
        const ggml_tensor * writer =
            request->cgraph->nodes[node_index];
        if (!writer ||
            ggml_backend_hrx_loom_is_metadata_op(writer) ||
            ggml_nelements(writer) == 0 ||
            ggml_backend_hrx_loom_plan_consumes_node(
                plan,
                node_index)) {
            continue;
        }
        hrx_buffer_ref_t writer_ref = {};
        if (!request->bind_tensor(
                request->bind_tensor_user_data,
                writer,
                &writer_ref)) {
            return false;
        }
        for (size_t i = 0; i < deferred_read_count; ++i) {
            if (ggml_backend_hrx_loom_deferred_write_conflicts(
                    node_index,
                    deferred_reads[i].ready_index,
                    writer_ref,
                    deferred_reads[i].binding)) {
                return false;
            }
        }
    }
    for (int node_index = owner_index + 1;
         node_index < request->cgraph->n_nodes;
         ++node_index) {
        bool needs_audit = false;
        for (size_t i = 0; i < deferred_write_count; ++i) {
            needs_audit =
                needs_audit ||
                node_index < deferred_writes[i].logical_index;
        }
        if (!needs_audit) {
            break;
        }
        const ggml_tensor * accessor =
            request->cgraph->nodes[node_index];
        if (!accessor ||
            ggml_backend_hrx_loom_is_metadata_op(accessor) ||
            ggml_nelements(accessor) == 0 ||
            ggml_backend_hrx_loom_plan_consumes_node(
                plan,
                node_index)) {
            continue;
        }
        const auto access_conflicts_with_early_output =
            [&](const ggml_tensor * tensor) {
                if (!tensor || ggml_nelements(tensor) == 0) {
                    return false;
                }
                hrx_buffer_ref_t access_ref = {};
                if (!request->bind_tensor(
                        request->bind_tensor_user_data,
                        tensor,
                        &access_ref)) {
                    return true;
                }
                for (size_t i = 0; i < deferred_write_count; ++i) {
                    if (node_index < deferred_writes[i].logical_index &&
                        ggml_backend_hrx_loom_deferred_refs_overlap(
                            access_ref,
                            deferred_writes[i].binding)) {
                        return true;
                    }
                }
                return false;
            };
        if (access_conflicts_with_early_output(accessor)) {
            return false;
        }
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            if (access_conflicts_with_early_output(accessor->src[i])) {
                return false;
            }
        }
    }
    return true;
}

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_invoke(
    ggml_backend_hrx_loom_catalog *          catalog,
    const ggml_backend_hrx_loom_op_request * request,
    ggml_backend_hrx_loom_consumed_nodes *   consumed_nodes) {
    if (consumed_nodes) {
        *consumed_nodes = {};
    }
    if (!catalog || !request || !request->op) {
        return ggml_backend_hrx_loom_failed(nullptr);
    }

    auto * invocation_context = request->invocation_context;
    if (invocation_context && !invocation_context->deferred.empty()) {
        ggml_backend_hrx_loom_deferred_plan ready = {};
        const auto take_result =
            ggml_backend_hrx_loom_take_deferred_plan(
                invocation_context->deferred,
                request->cgraph,
                request->execution_epoch,
                request->node_index,
                &ready);
        if (take_result ==
            GGML_BACKEND_HRX_LOOM_DEFERRED_STALE) {
            return ggml_backend_hrx_loom_failed(ready.route_id);
        }
        if (take_result ==
            GGML_BACKEND_HRX_LOOM_DEFERRED_READY) {
            if (!ready.plan ||
                !consumed_nodes ||
                !ggml_backend_hrx_loom_dispatch_execution_plan(
                    catalog,
                    request,
                    ready.plan) ||
                !ggml_backend_hrx_loom_copy_consumed_nodes_from(
                    *ready.plan,
                    request->node_index,
                    consumed_nodes)) {
                return ggml_backend_hrx_loom_failed(ready.route_id);
            }
            return ggml_backend_hrx_loom_supported(ready.route_id);
        }
    }

    std::unique_ptr<ggml_backend_hrx_loom_execution_plan> prepared_plan;
    const ggml_backend_hrx_loom_execution_plan * plan         = nullptr;
    ggml_backend_hrx_loom_op_response response =
        ggml_backend_hrx_loom_unsupported(
            GGML_BACKEND_HRX_LOOM_UNSUPPORTED_NO_ROUTE);
    bool prepared = false;
    const auto * resolved =
        invocation_context ?
        invocation_context->resolved_plans.find(catalog, request) :
        nullptr;
    if (resolved &&
        ggml_backend_hrx_loom_cached_plan_can_be_selected(
            request,
            resolved->plan.get())) {
        plan = resolved->plan.get();
        response = ggml_backend_hrx_loom_supported(
            resolved->route_id);
    } else {
        prepared_plan.reset(
            new (std::nothrow)
                ggml_backend_hrx_loom_execution_plan());
        if (!prepared_plan) {
            return ggml_backend_hrx_loom_failed(nullptr);
        }
        response = ggml_backend_hrx_loom_prepare_plan(
            catalog,
            request,
            prepared_plan.get());
        plan     = prepared_plan.get();
        prepared = true;
    }
    if (response.result != GGML_BACKEND_HRX_LOOM_INVOKED) {
        return response;
    }
    if (plan->dispatch_owner_node_index > request->node_index) {
        if (!invocation_context || !request->cgraph || !consumed_nodes ||
            plan->dispatch_owner_node_index >= request->cgraph->n_nodes ||
            plan->consumed_node_count < 1 ||
            plan->consumed_node_count >
                GGML_BACKEND_HRX_LOOM_MAX_CONSUMED_NODES) {
            return ggml_backend_hrx_loom_failed(response.route_id);
        }
        bool owner_is_consumed = false;
        for (int i = 0; i < plan->consumed_node_count; ++i) {
            owner_is_consumed =
                owner_is_consumed ||
                plan->consumed_node_indices[i] ==
                    plan->dispatch_owner_node_index;
        }
        if (!owner_is_consumed) {
            return ggml_backend_hrx_loom_failed(response.route_id);
        }
        if (!ggml_backend_hrx_loom_resolve_loaded_routes(
                catalog, plan)) {
            return ggml_backend_hrx_loom_failed(response.route_id);
        }
        if (!ggml_backend_hrx_loom_copy_consumed_nodes_from(
                *plan,
                request->node_index,
                consumed_nodes)) {
            return ggml_backend_hrx_loom_failed(response.route_id);
        }
        ggml_backend_hrx_loom_deferred_plan deferred_plan = {
            /* .cgraph          = */ request->cgraph,
            /* .execution_epoch = */ request->execution_epoch,
            /* .owner_index     = */ plan->dispatch_owner_node_index,
            /* .route_id        = */ response.route_id,
            /* .plan            = */ plan,
            /* .owned_plan      = */ nullptr,
        };
        if (prepared) {
            deferred_plan.owned_plan = std::move(prepared_plan);
            deferred_plan.plan = deferred_plan.owned_plan.get();
        }
        if (!ggml_backend_hrx_loom_enqueue_deferred_plan(
                invocation_context->deferred,
                std::move(deferred_plan))) {
            return ggml_backend_hrx_loom_unsupported(
                GGML_BACKEND_HRX_LOOM_UNSUPPORTED_NO_ROUTE);
        }
        consumed_nodes->dispatch_owner_node_index =
            plan->dispatch_owner_node_index;
        if (prepared) {
            invocation_context->resolved_plans.publish(
                catalog,
                request,
                response.route_id,
                *plan);
        }
        return response;
    }
    if (plan->dispatch_owner_node_index >= 0 &&
        plan->dispatch_owner_node_index != request->node_index) {
        return ggml_backend_hrx_loom_failed(response.route_id);
    }
    if (!ggml_backend_hrx_loom_dispatch_execution_plan(
            catalog,
            request,
            plan)) {
        return ggml_backend_hrx_loom_failed(response.route_id);
    }
    if (consumed_nodes &&
        !ggml_backend_hrx_loom_copy_consumed_nodes_from(
            *plan,
            request->node_index,
            consumed_nodes)) {
        return ggml_backend_hrx_loom_failed(response.route_id);
    }
    if (prepared && invocation_context) {
        invocation_context->resolved_plans.publish(
            catalog,
            request,
            response.route_id,
            *plan);
    }
    return response;
}

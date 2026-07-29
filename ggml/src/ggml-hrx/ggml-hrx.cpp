#include "ggml-hrx.h"

#include "ggml-hrx-catalog.h"
#include "ggml-hrx-q5-down-group4.h"
#include "ggml-hrx-test.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "loom-jit/ggml-hrx-loom-jit.h"

#include "hrx_runtime.h"

#include <cerrno>
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

static constexpr size_t GGML_HRX_ALIGNMENT = 256;
static constexpr uintptr_t GGML_HRX_FAKE_PTR_BASE = 0x1000;
static constexpr size_t GGML_HRX_STAGING_ARENA_DEFAULT_SIZE = 8 * 1024 * 1024;
static constexpr size_t GGML_HRX_SSM_X_SNAPSHOT_ROWS = 61;

struct ggml_backend_hrx_reg_context;
struct ggml_backend_hrx_context;

struct ggml_backend_hrx_test_dispatch_recorder {
    std::mutex mutex;
    bool enabled = false;
    std::map<std::string, ggml_backend_hrx_test_route_record> routes;
};

static ggml_backend_hrx_test_dispatch_recorder g_ggml_backend_hrx_test_dispatch_recorder;

struct ggml_backend_hrx_compiled_route {
    const ggml_backend_hrx_catalog_route * route = nullptr;
    hrx_executable_t executable = nullptr;
    uint32_t export_ordinal = 0;
    hrx_executable_export_info_t export_info = {};
    ggml_hrx_loom_jit_launch_config_t launch_config = {};
};

struct ggml_backend_hrx_compiled_route_deleter {
    void operator()(ggml_backend_hrx_compiled_route * route) const;
};

using ggml_backend_hrx_compiled_route_ptr =
    std::unique_ptr<ggml_backend_hrx_compiled_route, ggml_backend_hrx_compiled_route_deleter>;

struct ggml_backend_hrx_options {
    std::string catalog_dir;
    std::string evidence_dir;
    std::string trace_jsonl_path;
    std::string loom_sanitizer;
    std::string loom_sanitizer_reporting;
    bool trace_routes = false;
    bool trace_graph = false;
    size_t staging_arena_size = GGML_HRX_STAGING_ARENA_DEFAULT_SIZE;
};

struct ggml_backend_hrx_recurrent_cache_layer {
    int layer = -1;
    int cache_r_get_index = -1;
    int concat_index = -1;
    int cache_r_write_index = -1;
    int cache_s_get_index = -1;
    int ssm_conv_index = -1;
    int silu_index = -1;
    int gated_delta_net_index = -1;
    int cache_s_write_index = -1;
    const ggml_tensor * cache_r_get = nullptr;
    const ggml_tensor * concat = nullptr;
    const ggml_tensor * cache_r_write = nullptr;
    const ggml_tensor * cache_s_get = nullptr;
    const ggml_tensor * ssm_conv = nullptr;
    const ggml_tensor * silu = nullptr;
    const ggml_tensor * gated_delta_net = nullptr;
    const ggml_tensor * cache_s_write = nullptr;
    const ggml_tensor * cache_r_owner = nullptr;
    const ggml_tensor * cache_s_owner = nullptr;
    const ggml_tensor * conv_x = nullptr;
    const ggml_tensor * conv_filter = nullptr;
    const ggml_tensor * conv_dst = nullptr;
    const ggml_tensor * attention_dst = nullptr;
};

struct ggml_backend_hrx_recurrent_cache_plan {
    uint64_t graph_uid = 0;
    int node_count = 0;
    const ggml_backend_hrx_catalog * catalog = nullptr;
    bool valid = false;
    bool ready = false;
    std::vector<ggml_backend_hrx_recurrent_cache_layer> layers;
    std::vector<uint8_t> skip_mask;
};

struct ggml_backend_hrx_gdn_qk_scale_layer {
    int layer = -1;
    int q_norm_index = -1;
    int k_norm_index = -1;
    int gated_delta_net_index = -1;
    const ggml_tensor * q_norm = nullptr;
    const ggml_tensor * k_norm = nullptr;
    const ggml_tensor * gated_delta_net = nullptr;
    const ggml_tensor * raw_q = nullptr;
    const ggml_tensor * raw_k = nullptr;
};

struct ggml_backend_hrx_gdn_qk_scale_plan {
    uint64_t graph_uid = 0;
    int node_count = 0;
    const ggml_backend_hrx_catalog * catalog = nullptr;
    bool valid = false;
    bool ready = false;
    std::vector<ggml_backend_hrx_gdn_qk_scale_layer> layers;
    std::vector<uint8_t> skip_mask;
};

struct ggml_backend_hrx_ssm_conv_silu_layer {
    int layer = -1;
    int ssm_conv_index = -1;
    int silu_index = -1;
    bool x_dst_alias = false;
    uint64_t node_signature = 0;
    uint64_t request_fingerprint = 0;
    const ggml_tensor * ssm_conv = nullptr;
    const ggml_tensor * silu = nullptr;
    const ggml_tensor * window = nullptr;
    const ggml_tensor * state = nullptr;
    const ggml_tensor * x = nullptr;
    const ggml_tensor * filter = nullptr;
    const ggml_tensor * dst = nullptr;
};

struct ggml_backend_hrx_ssm_conv_silu_plan {
    uint64_t graph_uid = 0;
    int node_count = 0;
    const ggml_backend_hrx_catalog * catalog = nullptr;
    const ggml_backend_hrx_context * owner = nullptr;
    bool valid = false;
    bool ready = false;
    size_t alias_layers = 0;
    size_t disjoint_layers = 0;
    std::vector<ggml_backend_hrx_ssm_conv_silu_layer> layers;
    std::vector<uint8_t> skip_mask;
};

struct ggml_backend_hrx_ssm_conv_silu_dispatch_seed {
    const ggml_tensor * node = nullptr;
    uint64_t signature = 0;
    uint64_t graph_uid = 0;
    const ggml_backend_hrx_catalog_route * route = nullptr;
    std::vector<ggml_backend_hrx_catalog_binding> bindings;
    std::vector<int64_t> workload;
    std::vector<const ggml_tensor *> tensors;
    std::vector<uint8_t> constants;
    ggml_backend_hrx_compiled_route * compiled = nullptr;
};

static void ggml_backend_hrx_test_record_jit_compile(const std::string & route_id) {
    std::lock_guard<std::mutex> lock(g_ggml_backend_hrx_test_dispatch_recorder.mutex);
    if (!g_ggml_backend_hrx_test_dispatch_recorder.enabled) {
        return;
    }
    g_ggml_backend_hrx_test_dispatch_recorder.routes[route_id].jit_compile_count++;
}

static void ggml_backend_hrx_test_record_jit_cache_hit(const std::string & route_id) {
    std::lock_guard<std::mutex> lock(g_ggml_backend_hrx_test_dispatch_recorder.mutex);
    if (!g_ggml_backend_hrx_test_dispatch_recorder.enabled) {
        return;
    }
    g_ggml_backend_hrx_test_dispatch_recorder.routes[route_id].jit_cache_hit_count++;
}

static void ggml_backend_hrx_test_record_dispatch(const std::string & route_id) {
    std::lock_guard<std::mutex> lock(g_ggml_backend_hrx_test_dispatch_recorder.mutex);
    if (!g_ggml_backend_hrx_test_dispatch_recorder.enabled) {
        return;
    }
    g_ggml_backend_hrx_test_dispatch_recorder.routes[route_id].dispatch_count++;
}

struct ggml_backend_hrx_staging_arena {
    hrx_stream_t stream = nullptr;
    hrx_buffer_t buffer = nullptr;
    uint8_t * mapped = nullptr;
    size_t capacity = 0;
    size_t offset = 0;
    std::vector<hrx_buffer_t> retired_buffers;
};

struct ggml_backend_hrx_moe_router_tail_layer_plan {
    int layer = -1;
    int down_index = -1;
    int mul_index = -1;
    std::array<int, 7> add_indices = {};
    const ggml_tensor * down = nullptr;
    const ggml_tensor * mul = nullptr;
    std::array<const ggml_tensor *, 7> adds = {};
    const ggml_tensor * selected = nullptr;
    const ggml_tensor * dst = nullptr;
    const ggml_tensor * router_source = nullptr;
    int64_t ntokens = 0;
    int64_t router_source_stride = 0;
    int64_t router_source_offset = 0;
};

struct ggml_backend_hrx_moe_router_tail_graph_plan {
    uint64_t graph_uid = 0;
    int node_count = 0;
    const ggml_backend_hrx_catalog * catalog = nullptr;
    bool examined = false;
    bool ready = false;
    int64_t ntokens = 0;
    std::array<ggml_backend_hrx_moe_router_tail_layer_plan, 40> layers = {};
    // Exactly ADD0..ADD5 from each layer. The 40 router MULs are suppressed
    // separately as fused producers; the terminal ADD6 remains the routed node.
    std::vector<uint8_t> add_skip_mask;
};

struct ggml_backend_hrx_terminal_qact_layer_plan {
    int layer = -1;
    int glu_index = -1;
    int down_index = -1;
    const ggml_tensor * glu = nullptr;
    const ggml_tensor * down = nullptr;
    const ggml_tensor * ids = nullptr;
};

struct ggml_backend_hrx_terminal_qact_graph_plan {
    uint64_t graph_uid = 0;
    int node_count = 0;
    const ggml_backend_hrx_catalog * catalog = nullptr;
    bool examined = false;
    bool ready = false;
    size_t layer_count = 0;
    std::array<
        ggml_backend_hrx_terminal_qact_layer_plan, 40> layers = {};
};

struct ggml_backend_hrx_shared_expert_terminal_layer_plan {
    int layer = -1;
    int down_index = -1;
    int raw_gate_index = -1;
    int sigmoid_index = -1;
    int mul_index = -1;
    int terminal_index = -1;
    const ggml_tensor * down = nullptr;
    const ggml_tensor * raw_gate = nullptr;
    const ggml_tensor * sigmoid = nullptr;
    const ggml_tensor * mul = nullptr;
    const ggml_tensor * addend = nullptr;
    const ggml_tensor * terminal = nullptr;
};

struct ggml_backend_hrx_shared_expert_terminal_graph_plan {
    uint64_t graph_uid = 0;
    int node_count = 0;
    const ggml_backend_hrx_catalog * catalog = nullptr;
    bool examined = false;
    bool ready = false;
    std::array<
        ggml_backend_hrx_shared_expert_terminal_layer_plan, 40> layers = {};
    // Exactly the shared-down GEMM, scalar sigmoid, and broadcast MUL from
    // each layer. The raw scalar projection and terminal ADD remain scheduled.
    std::vector<uint8_t> skip_mask;
};

struct ggml_backend_hrx_gdn_rms_side_layer_plan {
    int layer = -1;
    int gated_delta_net_index = -1;
    int rms_norm_index = -1;
    int side_index = -1;
    int q8_index = -1;
    int silu_index = -1;
    int terminal_index = -1;
    const ggml_tensor * raw = nullptr;
    const ggml_tensor * rms_norm = nullptr;
    const ggml_tensor * norm_weight = nullptr;
    const ggml_tensor * side = nullptr;
    const ggml_tensor * q8_gemm = nullptr;
    const ggml_tensor * silu = nullptr;
    const ggml_tensor * terminal = nullptr;
};

struct ggml_backend_hrx_gdn_rms_side_graph_plan {
    uint64_t graph_uid = 0;
    int node_count = 0;
    const ggml_backend_hrx_catalog * catalog = nullptr;
    bool examined = false;
    bool ready = false;
    std::array<
        ggml_backend_hrx_gdn_rms_side_layer_plan, 30> layers = {};
    // Exactly RMS_NORM, its weighted side materialization, Q8 GEMM, and SiLU
    // for each of the thirty recurrent layers. The terminal MUL remains.
    std::vector<uint8_t> skip_mask;
};

struct ggml_backend_hrx_device_context {
    struct resolved_prepass {
        size_t descriptor_index = 0;
        const ggml_backend_hrx_catalog_prepass * descriptor_identity = nullptr;
        ggml_backend_hrx_catalog_prepass descriptor;
        int64_t elements = 0;
        size_t output_count = 0;
        std::array<size_t, 3> output_offsets = {};
        std::array<size_t, 3> output_lengths = {};
        size_t scratch_bytes = 0;
        std::vector<ggml_backend_hrx_catalog_binding> bindings;
        ggml_backend_hrx_compiled_route * compiled = nullptr;
        hrx_executable_t executable = nullptr;
        uint32_t export_ordinal = 0;
        hrx_executable_export_info_t export_info = {};
        std::string export_name;
        ggml_hrx_loom_jit_launch_config_t launch_config = {};
        hrx_dispatch_config_t dispatch_config = {};
    };
    struct resolved_prepass_plan {
        uint64_t graph_uid = 0;
        int node_index = -1;
        int node_count = 0;
        const ggml_backend_hrx_catalog * catalog = nullptr;
        const ggml_backend_hrx_catalog_route * route = nullptr;
        std::string route_id;
        ggml_backend_hrx_compiled_route * compiled = nullptr;
        hrx_executable_t executable = nullptr;
        uint32_t export_ordinal = 0;
        hrx_executable_export_info_t export_info = {};
        std::string export_name;
        ggml_hrx_loom_jit_launch_config_t launch_config = {};
        std::vector<uint8_t> constants;
        std::vector<int64_t> workload;
        std::vector<hrx_buffer_ref_t> direct_bindings;
        std::unordered_map<std::string, int64_t> shape;
        std::vector<resolved_prepass> prepasses;
    };
    // supports_op probes each route/shape once; see the compile probe there.
    struct resolved_dispatch {
        uint64_t signature = 0;
        uint64_t graph_uid = 0;
        const ggml_backend_hrx_catalog_route * route = nullptr;
        std::vector<ggml_backend_hrx_catalog_binding> bindings;
        std::vector<int64_t> workload;
        std::vector<const ggml_tensor *> tensors;
        std::vector<uint8_t> constants;
        bool prepass_free = false;
        ggml_backend_hrx_compiled_route * compiled = nullptr;
        std::optional<resolved_prepass_plan> prepass_plan;
    };
    std::unordered_map<const ggml_tensor *, resolved_dispatch> resolved_dispatches;
    // supports_op runs for every node on every graph build and otherwise repeats
    // the whole request build and route scan each time.
    std::unordered_map<uint64_t, bool> supports_answers;
    std::unordered_set<uint64_t> supports_probe_ok;
    std::unordered_set<uint64_t> supports_probe_bad;
    ggml_backend_hrx_reg_context * reg_context = nullptr;
    const ggml_backend_hrx_options * options = nullptr;
    hrx_device_t device = nullptr;
    hrx_stream_t transfer_stream = nullptr;
    ggml_hrx_loom_jit_amdgpu_t jit = nullptr;
    std::string name;
    std::string description;
    std::string architecture;
    size_t memory_total = 0;
    std::mutex streams_mutex;
    std::vector<hrx_stream_t> live_streams;
    std::vector<ggml_backend_hrx_staging_arena> staging_arenas;
    hrx_stream_t active_stream = nullptr;
    std::mutex compiled_routes_mutex;
    // Keyed by a 64-bit hash of the route and its resolved bindings; building a
    // string key per dispatch showed up in prompt throughput.
    std::unordered_map<uint64_t, ggml_backend_hrx_compiled_route_ptr> compiled_routes;
    // Destination of the activation-quantization prepass. One region is enough
    // because only the activations of the node being dispatched are live, and
    // consecutive nodes that share a source (expert gate/up, attention qkv/gate)
    // reuse the previous result instead of re-running the prepass.
    hrx_buffer_t quant_scratch = nullptr;
    size_t quant_scratch_capacity = 0;
    // The cache key is graph-local and storage-based. Qwen's expert gate sees a
    // full-span RESHAPE of attn_post_norm while the shared-expert gate later
    // sees the owner tensor itself; pointer identity alone needlessly
    // re-quantizes the same bytes.
    const ggml_tensor * quant_scratch_source = nullptr;
    const ggml_tensor * quant_scratch_source_owner = nullptr;
    hrx_buffer_t quant_scratch_source_buffer = nullptr;
    size_t quant_scratch_source_offset = 0;
    size_t quant_scratch_source_length = 0;
    int quant_scratch_source_node = -1;
    uint64_t quant_scratch_epoch = 0;
    size_t quant_scratch_bytes = 0;
    std::string quant_scratch_artifact;
    // Grown buffers are retired rather than released: dispatches already queued
    // on the stream still reference the old one.
    std::vector<hrx_buffer_t> retired_quant_scratch;
    // Weights never change, so a persistent prepass (q8_0 -> f16 expansion for
    // the matrix path) is bump-allocated here once and kept for the whole run.
    hrx_buffer_t weight_arena = nullptr;
    size_t weight_arena_capacity = 0;
    size_t weight_arena_used = 0;
    std::map<std::pair<hrx_buffer_t, size_t>, size_t> weight_arena_offsets;
    std::map<std::string, hrx_buffer_t> class_scratch;
    std::map<std::string, size_t> class_scratch_bytes;
    std::map<std::string, const ggml_tensor *> class_scratch_source;
    std::map<std::string, uint64_t> class_scratch_epoch;
    // Bumped per graph, so a cached prepass result can never be reused
    // across graphs where the same tensor address holds new contents.
    uint64_t graph_epoch = 0;
    // Set for the duration of a graph walk. Only consulted by fusions that
    // have to know a value's other consumers; null during supports_op.
    const ggml_cgraph * current_graph = nullptr;
    const ggml_backend_hrx_moe_router_tail_graph_plan * current_moe_router_tail_plan = nullptr;
    const ggml_backend_hrx_terminal_qact_graph_plan *
        current_terminal_qact_plan = nullptr;
    const ggml_backend_hrx_shared_expert_terminal_graph_plan *
        current_shared_expert_terminal_plan = nullptr;
    const ggml_backend_hrx_gdn_rms_side_graph_plan *
        current_gdn_rms_side_plan = nullptr;
    int current_node_index = -1;
    const ggml_backend_hrx_recurrent_cache_plan * current_recurrent_cache_plan = nullptr;
    const ggml_backend_hrx_gdn_qk_scale_plan * current_gdn_qk_scale_plan = nullptr;
    // SSM_CONV/SiLU memo entries live in the device-global resolved_dispatches
    // table, so their validated graph plan and owner must have the same scope.
    // Keep these fields last so the established hot-field offsets do not move.
    std::unique_ptr<ggml_backend_hrx_ssm_conv_silu_plan> ssm_conv_silu_plan;
};

struct ggml_backend_hrx_reg_context {
    ggml_backend_hrx_options options;
    ggml_backend_hrx_catalog_ptr catalog;
    std::ofstream trace_jsonl;
    std::mutex trace_mutex;
    bool gpu_initialized = false;
    std::vector<std::unique_ptr<ggml_backend_hrx_device_context>> device_contexts;
    std::vector<ggml_backend_device> devices;

    ~ggml_backend_hrx_reg_context();
};

struct ggml_backend_hrx_buffer_type_context {
    ggml_backend_hrx_device_context * device_context = nullptr;
    std::string name;
    hrx_buffer_params_t params = {};
};

struct ggml_backend_hrx_buffer_context {
    ggml_backend_hrx_device_context * device_context = nullptr;
    hrx_buffer_t buffer = nullptr;
    uint8_t * base = nullptr;
    // Small metadata registry for the capacity-neutral packed weights. It
    // prevents a renamed/full-span view from being mistaken for canonical
    // bytes. Device storage remains one allocation with no duplicate payload.
    std::mutex q5_down_group4_mutex;
    std::vector<std::pair<size_t, size_t>> q5_down_group4_ranges;
};

struct ggml_backend_hrx_context {
    ggml_backend_hrx_device_context * device_context = nullptr;
    hrx_stream_t stream = nullptr;
    std::string name;
    // A nonzero graph UID identifies one scheduler graph generation. Keep the
    // cache backend-local so simultaneous streams cannot race on its contents.
    // Direct backend graphs have UID 0 and intentionally recompute every time.
    uint64_t fusion_graph_uid = 0;
    std::vector<uint8_t> fusion_producer_mask;
    ggml_backend_hrx_moe_router_tail_graph_plan moe_router_tail_plan;
    ggml_backend_hrx_terminal_qact_graph_plan terminal_qact_plan;
    ggml_backend_hrx_shared_expert_terminal_graph_plan
        shared_expert_terminal_plan;
    ggml_backend_hrx_gdn_rms_side_graph_plan gdn_rms_side_plan;
    ggml_backend_hrx_recurrent_cache_plan recurrent_cache_plan;
    ggml_backend_hrx_gdn_qk_scale_plan gdn_qk_scale_plan;
};

static void
ggml_backend_hrx_clear_ssm_conv_silu_plan(
        ggml_backend_hrx_device_context * device_context);

struct ggml_backend_hrx_dispatch_request {
    ggml_backend_hrx_catalog_problem problem;
    std::vector<const ggml_tensor *> tensors;
    std::vector<uint8_t> constants;
};

static bool ggml_backend_hrx_log_status(hrx_status_t status, const char * expr, const char * file, int line) {
    if (hrx_status_is_ok(status)) {
        return true;
    }

    char * message = nullptr;
    size_t length = 0;
    hrx_status_to_string(status, &message, &length);
    GGML_LOG_ERROR("%s:%d: %s failed: %s\n", file, line, expr, message ? message : "unknown HRX error");
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return false;
}

#define GGML_HRX_CHECK(expr) ggml_backend_hrx_log_status((expr), #expr, __FILE__, __LINE__)

void ggml_backend_hrx_compiled_route_deleter::operator()(ggml_backend_hrx_compiled_route * route) const {
    if (route && route->executable) {
        hrx_executable_release(route->executable);
        route->executable = nullptr;
    }
    delete route;
}

static size_t ggml_backend_hrx_align_up(size_t value, size_t alignment) {
    GGML_ASSERT(alignment > 0);
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

static size_t ggml_backend_hrx_staging_arena_capacity(const ggml_backend_hrx_device_context * device_context) {
    const size_t requested =
        device_context && device_context->options ?
        device_context->options->staging_arena_size :
        GGML_HRX_STAGING_ARENA_DEFAULT_SIZE;
    return ggml_backend_hrx_align_up(std::max(requested, GGML_HRX_ALIGNMENT), GGML_HRX_ALIGNMENT);
}

static std::string ggml_backend_hrx_test_case_index_key(const std::string & target_key, const std::string & family) {
    return target_key + "\n" + family;
}

static ggml_guid_t ggml_backend_hrx_guid(void) {
    static ggml_guid guid = {
        0x1c, 0x65, 0x79, 0x0a, 0x31, 0x8b, 0x4d, 0xa6,
        0x9e, 0x16, 0x6f, 0x13, 0x39, 0xb2, 0xe7, 0x5c,
    };
    return &guid;
}

static ggml_backend_hrx_device_context * ggml_backend_hrx_get_device_context(ggml_backend_dev_t dev) {
    return static_cast<ggml_backend_hrx_device_context *>(dev->context);
}

static ggml_backend_hrx_buffer_type_context * ggml_backend_hrx_get_buft_context(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context);
}

static ggml_backend_hrx_buffer_context * ggml_backend_hrx_get_buffer_context(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_backend_hrx_buffer_context *>(buffer->context);
}

static void * ggml_backend_hrx_buffer_get_base(ggml_backend_buffer_t buffer);
static const char * ggml_backend_hrx_buffer_type_get_name(
        ggml_backend_buffer_type_t buft);

static const char * ggml_backend_hrx_getenv_once(const char * name) {
    return std::getenv(name);
}

static std::string ggml_backend_hrx_env_string(const char * name) {
    const char * value = ggml_backend_hrx_getenv_once(name);
    return value ? std::string(value) : std::string();
}

static bool ggml_backend_hrx_parse_bool_value(const std::string & value) {
    return !value.empty() && value != "0" && value != "false" && value != "FALSE" && value != "off" && value != "OFF";
}

static bool ggml_backend_hrx_env_bool(const char * name) {
    return ggml_backend_hrx_parse_bool_value(ggml_backend_hrx_env_string(name));
}

static std::optional<size_t> ggml_backend_hrx_parse_size_value(const std::string & value) {
    if (value.empty()) {
        return std::nullopt;
    }

    errno = 0;
    char * end = nullptr;
    unsigned long long parsed = std::strtoull(value.c_str(), &end, 0);
    if (errno != 0 || end == value.c_str()) {
        return std::nullopt;
    }

    size_t multiplier = 1;
    if (*end != '\0') {
        if (end[1] != '\0') {
            if ((end[1] != 'b' && end[1] != 'B') || end[2] != '\0') {
                return std::nullopt;
            }
        }
        switch (*end) {
            case 'k':
            case 'K':
                multiplier = 1024;
                break;
            case 'm':
            case 'M':
                multiplier = 1024 * 1024;
                break;
            case 'g':
            case 'G':
                multiplier = 1024 * 1024 * 1024;
                break;
            default:
                return std::nullopt;
        }
    }

    if (parsed > std::numeric_limits<size_t>::max() / multiplier) {
        return std::nullopt;
    }
    return static_cast<size_t>(parsed) * multiplier;
}

static ggml_backend_hrx_options ggml_backend_hrx_parse_options() {
    ggml_backend_hrx_options options;
    options.catalog_dir = ggml_backend_hrx_env_string("GGML_HRX_CATALOG_DIR");
    options.evidence_dir = ggml_backend_hrx_env_string("GGML_HRX_EVIDENCE_DIR");
    options.trace_jsonl_path = ggml_backend_hrx_env_string("GGML_HRX_TRACE_JSONL");
    options.loom_sanitizer = ggml_backend_hrx_env_string("GGML_HRX_LOOM_SANITIZER");
    options.loom_sanitizer_reporting = ggml_backend_hrx_env_string("GGML_HRX_LOOM_SANITIZER_REPORTING");
    options.trace_routes = ggml_backend_hrx_env_bool("GGML_HRX_TRACE_ROUTES");
    options.trace_graph = ggml_backend_hrx_env_bool("GGML_HRX_TRACE_GRAPH");

    const std::string staging_size = ggml_backend_hrx_env_string("GGML_HRX_STAGING_ARENA_SIZE");
    if (!staging_size.empty()) {
        if (auto parsed = ggml_backend_hrx_parse_size_value(staging_size)) {
            options.staging_arena_size = *parsed;
        } else {
            GGML_LOG_ERROR(
                "%s: ignoring invalid GGML_HRX_STAGING_ARENA_SIZE=%s\n",
                __func__, staging_size.c_str());
        }
    }
    return options;
}

static const char * ggml_backend_hrx_optional_c_str(const std::string & value) {
    return value.empty() ? nullptr : value.c_str();
}

static bool ggml_backend_hrx_trace_enabled(const ggml_backend_hrx_reg_context * reg_context) {
    return reg_context && reg_context->trace_jsonl.is_open();
}

static void ggml_backend_hrx_trace_event(
        ggml_backend_hrx_reg_context * reg_context,
        nlohmann::json event) {
    if (!reg_context || !reg_context->trace_jsonl.is_open()) {
        return;
    }
    event["backend"] = GGML_HRX_NAME;
    std::lock_guard<std::mutex> lock(reg_context->trace_mutex);
    reg_context->trace_jsonl << event.dump() << '\n';
    reg_context->trace_jsonl.flush();
}

static void ggml_backend_hrx_write_evidence_file(
        ggml_backend_hrx_device_context * device_context,
        const std::string & name,
        const void * data,
        size_t size) {
    if (!device_context || !device_context->options || device_context->options->evidence_dir.empty() ||
        !data || size == 0) {
        return;
    }
    std::string path = device_context->options->evidence_dir;
    if (!path.empty() && path.back() != '/') {
        path += '/';
    }
    path += name;
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        GGML_LOG_WARN("%s: failed to open evidence file %s\n", __func__, path.c_str());
        return;
    }
    file.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
}

static size_t ggml_backend_hrx_tensor_offset(const ggml_backend_hrx_buffer_context * context, const ggml_tensor * tensor) {
    return static_cast<size_t>(static_cast<const uint8_t *>(tensor->data) - context->base);
}

static bool ggml_backend_hrx_is_q5_down_group4_tensor(
        const ggml_tensor * tensor) {
    if (!tensor) {
        return false;
    }
    const int64_t ne[4] = {
        tensor->ne[0],
        tensor->ne[1],
        tensor->ne[2],
        tensor->ne[3],
    };
    return ggml_hrx_q5_down_group4::is_exact_tensor(
        tensor->type == GGML_TYPE_Q5_K,
        ggml_is_contiguous(tensor),
        ne,
        ggml_nbytes(tensor),
        ggml_get_name(tensor));
}

using ggml_backend_hrx_q5_down_group4_range_state =
    ggml_hrx_q5_down_group4::range_state;

static bool ggml_backend_hrx_ranges_overlap(
        size_t left_offset,
        size_t left_length,
        size_t right_offset,
        size_t right_length) {
    return left_length != 0 &&
           right_length != 0 &&
           left_offset < right_offset + right_length &&
           right_offset < left_offset + left_length;
}

static ggml_backend_hrx_q5_down_group4_range_state
ggml_backend_hrx_get_q5_down_group4_range_state(
        ggml_backend_hrx_buffer_context * context,
        size_t offset,
        size_t length) {
    if (!context || length == 0 ||
        offset > std::numeric_limits<size_t>::max() - length) {
        return ggml_backend_hrx_q5_down_group4_range_state::none;
    }
    std::lock_guard<std::mutex> lock(context->q5_down_group4_mutex);
    for (const auto & range : context->q5_down_group4_ranges) {
        if (range.first == offset && range.second == length) {
            return ggml_backend_hrx_q5_down_group4_range_state::exact;
        }
        if (ggml_backend_hrx_ranges_overlap(
                offset, length, range.first, range.second)) {
            return ggml_backend_hrx_q5_down_group4_range_state::overlap;
        }
    }
    return ggml_backend_hrx_q5_down_group4_range_state::none;
}

static bool ggml_backend_hrx_register_q5_down_group4_range(
        ggml_backend_hrx_buffer_context * context,
        size_t offset,
        size_t length) {
    if (!context || length != ggml_hrx_q5_down_group4::kTensorBytes ||
        offset > std::numeric_limits<size_t>::max() - length) {
        return false;
    }
    std::lock_guard<std::mutex> lock(context->q5_down_group4_mutex);
    for (const auto & range : context->q5_down_group4_ranges) {
        if (range.first == offset && range.second == length) {
            return true;
        }
        if (ggml_backend_hrx_ranges_overlap(
                offset, length, range.first, range.second)) {
            return false;
        }
    }
    context->q5_down_group4_ranges.emplace_back(offset, length);
    return true;
}

static ggml_backend_hrx_q5_down_group4_range_state
ggml_backend_hrx_tensor_q5_down_group4_range_state(
        const ggml_tensor * tensor) {
    if (!tensor) {
        return ggml_backend_hrx_q5_down_group4_range_state::none;
    }
    ggml_backend_buffer_t buffer =
        tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buffer ||
        buffer->iface.get_base != ggml_backend_hrx_buffer_get_base) {
        return ggml_backend_hrx_q5_down_group4_range_state::none;
    }
    // select_weight_buft() probes supports_op with a zero-byte HRX buffer
    // before tensor allocation. Do not subtract null/fake pointers or mistake
    // that capability query for live registered storage.
    if (buffer->size == 0 || !tensor->data) {
        return ggml_backend_hrx_q5_down_group4_range_state::none;
    }
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    const uintptr_t base =
        reinterpret_cast<uintptr_t>(context->base);
    const uintptr_t data =
        reinterpret_cast<uintptr_t>(tensor->data);
    const size_t length = ggml_nbytes(tensor);
    if (data < base) {
        return ggml_backend_hrx_q5_down_group4_range_state::none;
    }
    const size_t offset = static_cast<size_t>(data - base);
    if (offset > buffer->size || length > buffer->size - offset) {
        return ggml_backend_hrx_q5_down_group4_range_state::none;
    }
    return ggml_backend_hrx_get_q5_down_group4_range_state(
        context,
        offset,
        length);
}

static bool
ggml_backend_hrx_is_q5_down_group4_zero_size_capability_probe(
        const ggml_tensor * tensor) {
    if (!tensor || tensor->view_src || tensor->data) {
        return false;
    }
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (!buffer || buffer->size != 0 || !buffer->buft ||
        buffer->buft->iface.get_name !=
            ggml_backend_hrx_buffer_type_get_name) {
        return false;
    }
    // ggml_backend_buft_alloc_buffer(buft, 0) returns this exact generic
    // sentinel without invoking the backend allocator: no interface callbacks
    // and no context. A live HRX allocation never has this representation.
    return buffer->context == nullptr &&
           buffer->iface.free_buffer == nullptr &&
           buffer->iface.get_base == nullptr &&
           buffer->iface.set_tensor == nullptr;
}

static void ggml_backend_hrx_register_stream(ggml_backend_hrx_device_context * device_context, hrx_stream_t stream) {
    if (!device_context || !stream) {
        return;
    }
    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    if (std::find(device_context->live_streams.begin(), device_context->live_streams.end(), stream) ==
            device_context->live_streams.end()) {
        device_context->live_streams.push_back(stream);
    }
}

static void ggml_backend_hrx_reset_staging_arena_locked(ggml_backend_hrx_staging_arena & arena) {
    for (hrx_buffer_t buffer : arena.retired_buffers) {
        hrx_buffer_release(buffer);
    }
    arena.retired_buffers.clear();
    arena.offset = 0;
}

static void ggml_backend_hrx_release_staging_arena_locked(ggml_backend_hrx_staging_arena & arena) {
    if (arena.buffer) {
        hrx_buffer_release(arena.buffer);
    }
    for (hrx_buffer_t buffer : arena.retired_buffers) {
        hrx_buffer_release(buffer);
    }
    arena = {};
}

static ggml_backend_hrx_staging_arena * ggml_backend_hrx_find_staging_arena_locked(
        ggml_backend_hrx_device_context * device_context,
        hrx_stream_t stream) {
    for (auto & arena : device_context->staging_arenas) {
        if (arena.stream == stream) {
            return &arena;
        }
    }
    return nullptr;
}

static ggml_backend_hrx_staging_arena * ggml_backend_hrx_get_staging_arena_locked(
        ggml_backend_hrx_device_context * device_context,
        hrx_stream_t stream) {
    if (auto * arena = ggml_backend_hrx_find_staging_arena_locked(device_context, stream)) {
        return arena;
    }
    device_context->staging_arenas.push_back({});
    auto & arena = device_context->staging_arenas.back();
    arena.stream = stream;
    return &arena;
}

static void ggml_backend_hrx_unregister_stream(ggml_backend_hrx_device_context * device_context, hrx_stream_t stream) {
    if (!device_context || !stream) {
        return;
    }

    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    auto & streams = device_context->live_streams;
    streams.erase(std::remove(streams.begin(), streams.end(), stream), streams.end());
    auto & arenas = device_context->staging_arenas;
    auto arena_it = std::find_if(
        arenas.begin(), arenas.end(),
        [stream](const ggml_backend_hrx_staging_arena & arena) { return arena.stream == stream; });
    if (arena_it != arenas.end()) {
        ggml_backend_hrx_release_staging_arena_locked(*arena_it);
        arenas.erase(arena_it);
    }
    if (device_context->active_stream == stream) {
        device_context->active_stream = nullptr;
    }
}

static hrx_stream_t ggml_backend_hrx_retain_timeline_stream(ggml_backend_hrx_device_context * device_context) {
    if (!device_context) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    hrx_stream_t stream = device_context->active_stream;
    if (!stream) {
        stream = device_context->transfer_stream;
    }
    if (stream) {
        hrx_stream_retain(stream);
    }
    return stream;
}

static bool ggml_backend_hrx_sync_streams(ggml_backend_hrx_device_context * device_context) {
    if (!device_context) {
        return true;
    }

    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    bool ok = true;
    for (hrx_stream_t stream : device_context->live_streams) {
        ok = GGML_HRX_CHECK(hrx_stream_synchronize(stream)) && ok;
        if (auto * arena = ggml_backend_hrx_find_staging_arena_locked(device_context, stream)) {
            ggml_backend_hrx_reset_staging_arena_locked(*arena);
        }
    }
    return ok;
}

static bool ggml_backend_hrx_sync_graph_entry_streams(
        ggml_backend_hrx_device_context * device_context,
        hrx_stream_t graph_stream) {
    if (!device_context) {
        return true;
    }

    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    hrx_stream_t streams[] = {
        device_context->active_stream,
        device_context->transfer_stream,
    };

    bool ok = true;
    for (hrx_stream_t stream : streams) {
        if (!stream || stream == graph_stream) {
            continue;
        }
        ok = GGML_HRX_CHECK(hrx_stream_synchronize(stream)) && ok;
        if (auto * arena = ggml_backend_hrx_find_staging_arena_locked(device_context, stream)) {
            ggml_backend_hrx_reset_staging_arena_locked(*arena);
        }
    }
    return ok;
}

static bool ggml_backend_hrx_prepare_stream_signal(
        hrx_stream_t stream,
        hrx_semaphore_t * semaphore,
        uint64_t * signal_value,
        hrx_semaphore_list_t * wait_list,
        hrx_semaphore_list_t * signal_list,
        hrx_semaphore_t * wait_semaphores,
        uint64_t * wait_values,
        hrx_semaphore_t * signal_semaphores,
        uint64_t * signal_values) {
    hrx_timeline_point_t position = {};
    if (!GGML_HRX_CHECK(hrx_stream_flush(stream)) ||
        !GGML_HRX_CHECK(hrx_stream_get_timeline_position(stream, &position)) ||
        !GGML_HRX_CHECK(hrx_stream_get_semaphore(stream, semaphore))) {
        return false;
    }

    *signal_value = position.value + 1;
    if (position.value > 0) {
        wait_semaphores[0] = *semaphore;
        wait_values[0] = position.value;
        *wait_list = {
            /* .semaphores = */ wait_semaphores,
            /* .values     = */ wait_values,
            /* .count      = */ 1,
        };
    } else {
        *wait_list = {};
    }

    signal_semaphores[0] = *semaphore;
    signal_values[0] = *signal_value;
    *signal_list = {
        /* .semaphores = */ signal_semaphores,
        /* .values     = */ signal_values,
        /* .count      = */ 1,
    };
    return true;
}

static bool ggml_backend_hrx_finish_stream_signal(hrx_stream_t stream, uint64_t signal_value) {
    uint64_t advanced_value = 0;
    if (!GGML_HRX_CHECK(hrx_stream_advance_timeline(stream, &advanced_value))) {
        return false;
    }
    if (advanced_value != signal_value) {
        GGML_LOG_ERROR(
            "%s: stream timeline advanced to %" PRIu64 ", expected %" PRIu64 "\n",
            __func__, advanced_value, signal_value);
        return false;
    }
    return GGML_HRX_CHECK(hrx_stream_wait(stream));
}

static bool ggml_backend_hrx_queue_fill_stream_sync(
        ggml_backend_hrx_device_context * device_context,
        hrx_buffer_t buffer,
        size_t offset,
        size_t size,
        const void * pattern,
        size_t pattern_size) {
    hrx_stream_t stream = ggml_backend_hrx_retain_timeline_stream(device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX stream registered for synchronous fill\n", __func__);
        return false;
    }

    hrx_semaphore_t semaphore = nullptr;
    uint64_t signal_value = 0;
    hrx_semaphore_t wait_semaphores[1] = {};
    uint64_t wait_values[1] = {};
    hrx_semaphore_t signal_semaphores[1] = {};
    uint64_t signal_values[1] = {};
    hrx_semaphore_list_t wait_list = {};
    hrx_semaphore_list_t signal_list = {};
    bool ok = ggml_backend_hrx_prepare_stream_signal(
        stream, &semaphore, &signal_value, &wait_list, &signal_list,
        wait_semaphores, wait_values, signal_semaphores, signal_values);
    ok = ok && GGML_HRX_CHECK(hrx_queue_fill(
        device_context->device, 0,
        wait_list.count ? &wait_list : nullptr,
        &signal_list, buffer, offset, size, pattern, pattern_size));
    ok = ok && ggml_backend_hrx_finish_stream_signal(stream, signal_value);
    hrx_stream_release(stream);
    return ok;
}

static bool ggml_backend_hrx_queue_copy_stream_sync(
        ggml_backend_hrx_device_context * device_context,
        hrx_buffer_t src,
        size_t src_offset,
        hrx_buffer_t dst,
        size_t dst_offset,
        size_t size) {
    hrx_stream_t stream = ggml_backend_hrx_retain_timeline_stream(device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX stream registered for synchronous copy\n", __func__);
        return false;
    }

    hrx_semaphore_t semaphore = nullptr;
    uint64_t signal_value = 0;
    hrx_semaphore_t wait_semaphores[1] = {};
    uint64_t wait_values[1] = {};
    hrx_semaphore_t signal_semaphores[1] = {};
    uint64_t signal_values[1] = {};
    hrx_semaphore_list_t wait_list = {};
    hrx_semaphore_list_t signal_list = {};
    bool ok = ggml_backend_hrx_prepare_stream_signal(
        stream, &semaphore, &signal_value, &wait_list, &signal_list,
        wait_semaphores, wait_values, signal_semaphores, signal_values);
    ok = ok && GGML_HRX_CHECK(hrx_queue_copy(
        device_context->device, 0,
        wait_list.count ? &wait_list : nullptr,
        &signal_list, src, src_offset, dst, dst_offset, size));
    ok = ok && ggml_backend_hrx_finish_stream_signal(stream, signal_value);
    hrx_stream_release(stream);
    return ok;
}

static bool ggml_backend_hrx_ensure_staging_buffer_locked(
        ggml_backend_hrx_device_context * device_context,
        ggml_backend_hrx_staging_arena * arena,
        size_t required_capacity) {
    if (arena->buffer && arena->capacity >= required_capacity && arena->mapped) {
        return true;
    }

    if (arena->buffer) {
        arena->retired_buffers.push_back(arena->buffer);
        arena->buffer = nullptr;
        arena->mapped = nullptr;
        arena->capacity = 0;
        arena->offset = 0;
    }

    const size_t capacity = ggml_backend_hrx_align_up(
        std::max(required_capacity, ggml_backend_hrx_staging_arena_capacity(device_context)),
        GGML_HRX_ALIGNMENT);
    hrx_buffer_params_t params = {
        /* .type = */ HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
        /* .access = */ HRX_MEMORY_ACCESS_ALL,
        /* .usage = */ HRX_BUFFER_USAGE_DEFAULT |
                       HRX_BUFFER_USAGE_MAPPING_SCOPED |
                       HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
        /* .queue_affinity = */ 0,
    };
    if (!GGML_HRX_CHECK(hrx_allocator_allocate_buffer(
            hrx_device_allocator(device_context->device), params, capacity, &arena->buffer))) {
        return false;
    }

    void * mapped = nullptr;
    if (!GGML_HRX_CHECK(hrx_buffer_map(arena->buffer, HRX_MAP_READ | HRX_MAP_WRITE, 0, capacity, &mapped))) {
        hrx_buffer_release(arena->buffer);
        arena->buffer = nullptr;
        return false;
    }
    arena->mapped = static_cast<uint8_t *>(mapped);
    arena->capacity = capacity;
    arena->offset = 0;
    return true;
}

static bool ggml_backend_hrx_stage_and_copy_tensor(
        ggml_backend_hrx_buffer_context * context,
        const ggml_tensor * tensor,
        const void * data,
        size_t buffer_offset,
        size_t buffer_size,
        size_t size) {
    if (!context || !context->buffer || !data) {
        return false;
    }
    if (buffer_offset > buffer_size || size > buffer_size - buffer_offset) {
        GGML_LOG_ERROR(
            "%s: upload for tensor %s exceeds HRX buffer bounds: offset=%zu size=%zu buffer_size=%zu\n",
            __func__, tensor ? tensor->name : "<unknown>", buffer_offset, size, buffer_size);
        return false;
    }

    hrx_stream_t stream = ggml_backend_hrx_retain_timeline_stream(context->device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX stream available for tensor upload\n", __func__);
        return false;
    }

    std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
    auto * arena = ggml_backend_hrx_get_staging_arena_locked(context->device_context, stream);
    if (!arena ||
        !ggml_backend_hrx_ensure_staging_buffer_locked(
            context->device_context, arena, ggml_backend_hrx_staging_arena_capacity(context->device_context))) {
        hrx_stream_release(stream);
        return false;
    }

    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    size_t uploaded = 0;
    bool ok = true;
    while (uploaded < size) {
        size_t staging_offset = ggml_backend_hrx_align_up(arena->offset, GGML_HRX_ALIGNMENT);
        if (staging_offset >= arena->capacity) {
            ok = GGML_HRX_CHECK(hrx_stream_flush(stream)) && GGML_HRX_CHECK(hrx_stream_wait(stream));
            if (!ok) {
                break;
            }
            ggml_backend_hrx_reset_staging_arena_locked(*arena);
            staging_offset = 0;
        }

        const size_t available = arena->capacity - staging_offset;
        const size_t chunk_size = std::min(size - uploaded, available);
        if (chunk_size == 0) {
            GGML_LOG_ERROR("%s: HRX staging arena has no available space\n", __func__);
            ok = false;
            break;
        }

        std::memcpy(arena->mapped + staging_offset, bytes + uploaded, chunk_size);
        ok = GGML_HRX_CHECK(hrx_stream_copy_buffer(
            stream,
            arena->buffer,
            staging_offset,
            context->buffer,
            buffer_offset + uploaded,
            chunk_size));
        if (!ok) {
            break;
        }

        arena->offset = ggml_backend_hrx_align_up(staging_offset + chunk_size, GGML_HRX_ALIGNMENT);
        uploaded += chunk_size;
    }

    hrx_stream_release(stream);
    return ok;
}

static bool ggml_backend_hrx_copy_tensor_to_staging(
        ggml_backend_hrx_buffer_context * context,
        const ggml_tensor * tensor,
        size_t buffer_offset,
        size_t buffer_size,
        void * data,
        size_t size) {
    if (!context || !context->buffer || !data) {
        return false;
    }
    if (buffer_offset > buffer_size || size > buffer_size - buffer_offset) {
        GGML_LOG_ERROR(
            "%s: readback for tensor %s exceeds HRX buffer bounds: offset=%zu size=%zu buffer_size=%zu\n",
            __func__, tensor ? tensor->name : "<unknown>", buffer_offset, size, buffer_size);
        return false;
    }

    hrx_stream_t stream = ggml_backend_hrx_retain_timeline_stream(context->device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX stream available for tensor readback\n", __func__);
        return false;
    }

    auto * out_bytes = static_cast<uint8_t *>(data);
    size_t copied = 0;
    bool ok = true;
    {
        std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
        auto * arena = ggml_backend_hrx_get_staging_arena_locked(context->device_context, stream);
        if (!arena ||
            !ggml_backend_hrx_ensure_staging_buffer_locked(
                context->device_context, arena, ggml_backend_hrx_staging_arena_capacity(context->device_context))) {
            hrx_stream_release(stream);
            return false;
        }

        while (copied < size) {
            size_t staging_offset = ggml_backend_hrx_align_up(arena->offset, GGML_HRX_ALIGNMENT);
            if (staging_offset >= arena->capacity) {
                ok = GGML_HRX_CHECK(hrx_stream_synchronize(stream));
                if (!ok) {
                    break;
                }
                ggml_backend_hrx_reset_staging_arena_locked(*arena);
                staging_offset = 0;
            }

            const size_t chunk_size = std::min(size - copied, arena->capacity - staging_offset);
            if (chunk_size == 0) {
                GGML_LOG_ERROR("%s: HRX staging arena has no available space\n", __func__);
                ok = false;
                break;
            }

            ok = GGML_HRX_CHECK(hrx_stream_copy_buffer(
                stream,
                context->buffer,
                buffer_offset + copied,
                arena->buffer,
                staging_offset,
                chunk_size));
            if (!ok) {
                break;
            }

            ok = GGML_HRX_CHECK(hrx_stream_synchronize(stream));
            if (!ok) {
                break;
            }
            std::memcpy(out_bytes + copied, arena->mapped + staging_offset, chunk_size);
            copied += chunk_size;
            ggml_backend_hrx_reset_staging_arena_locked(*arena);
        }
    }

    hrx_stream_release(stream);
    return ok;
}

static bool ggml_backend_hrx_write_q5_down_group4_canonical(
        ggml_backend_hrx_buffer_context * context,
        const ggml_tensor * tensor,
        const void * canonical,
        size_t canonical_size,
        size_t buffer_offset,
        size_t buffer_size) {
    try {
        std::vector<uint8_t> packed(
            ggml_hrx_q5_down_group4::kTensorBytes);
        if (!ggml_hrx_q5_down_group4::pack(
                canonical,
                canonical_size,
                packed.data(),
                packed.size())) {
            return false;
        }
        return ggml_backend_hrx_stage_and_copy_tensor(
            context,
            tensor,
            packed.data(),
            buffer_offset,
            buffer_size,
            packed.size());
    } catch (const std::bad_alloc &) {
        GGML_LOG_ERROR(
            "%s: transient Q5-down group4 pack allocation failed for %s\n",
            __func__, tensor ? tensor->name : "<unknown>");
        return false;
    }
}

static bool ggml_backend_hrx_read_q5_down_group4_canonical(
        ggml_backend_hrx_buffer_context * context,
        const ggml_tensor * tensor,
        size_t buffer_offset,
        size_t buffer_size,
        void * canonical,
        size_t canonical_size) {
    try {
        std::vector<uint8_t> packed(
            ggml_hrx_q5_down_group4::kTensorBytes);
        if (!ggml_backend_hrx_copy_tensor_to_staging(
                context,
                tensor,
                buffer_offset,
                buffer_size,
                packed.data(),
                packed.size())) {
            return false;
        }
        return ggml_hrx_q5_down_group4::unpack(
            packed.data(),
            packed.size(),
            canonical,
            canonical_size);
    } catch (const std::bad_alloc &) {
        GGML_LOG_ERROR(
            "%s: transient Q5-down group4 unpack allocation failed for %s\n",
            __func__, tensor ? tensor->name : "<unknown>");
        return false;
    }
}

static bool ggml_backend_hrx_set_q5_down_group4_semantic(
        ggml_backend_hrx_buffer_context * context,
        const ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size,
        size_t buffer_offset,
        size_t buffer_size) {
    const size_t tensor_size =
        ggml_hrx_q5_down_group4::kTensorBytes;
    if (offset > tensor_size || size > tensor_size - offset) {
        return false;
    }
    if (offset == 0 && size == tensor_size) {
        return ggml_backend_hrx_write_q5_down_group4_canonical(
            context,
            tensor,
            data,
            size,
            buffer_offset,
            buffer_size);
    }

    try {
        std::vector<uint8_t> canonical(tensor_size);
        if (!ggml_backend_hrx_read_q5_down_group4_canonical(
                context,
                tensor,
                buffer_offset,
                buffer_size,
                canonical.data(),
                canonical.size())) {
            return false;
        }
        std::memcpy(canonical.data() + offset, data, size);
        return ggml_backend_hrx_write_q5_down_group4_canonical(
            context,
            tensor,
            canonical.data(),
            canonical.size(),
            buffer_offset,
            buffer_size);
    } catch (const std::bad_alloc &) {
        GGML_LOG_ERROR(
            "%s: transient Q5-down group4 RMW allocation failed for %s\n",
            __func__, tensor ? tensor->name : "<unknown>");
        return false;
    }
}

static bool ggml_backend_hrx_get_q5_down_group4_semantic(
        ggml_backend_hrx_buffer_context * context,
        const ggml_tensor * tensor,
        void * data,
        size_t offset,
        size_t size,
        size_t buffer_offset,
        size_t buffer_size) {
    const size_t tensor_size =
        ggml_hrx_q5_down_group4::kTensorBytes;
    if (offset > tensor_size || size > tensor_size - offset) {
        return false;
    }
    if (offset == 0 && size == tensor_size) {
        return ggml_backend_hrx_read_q5_down_group4_canonical(
            context,
            tensor,
            buffer_offset,
            buffer_size,
            data,
            size);
    }

    try {
        std::vector<uint8_t> canonical(tensor_size);
        if (!ggml_backend_hrx_read_q5_down_group4_canonical(
                context,
                tensor,
                buffer_offset,
                buffer_size,
                canonical.data(),
                canonical.size())) {
            return false;
        }
        std::memcpy(data, canonical.data() + offset, size);
        return true;
    } catch (const std::bad_alloc &) {
        GGML_LOG_ERROR(
            "%s: transient Q5-down group4 partial-read allocation failed for %s\n",
            __func__, tensor ? tensor->name : "<unknown>");
        return false;
    }
}

static size_t ggml_backend_hrx_total_memory(hrx_device_t device) {
    uint64_t memory_total = 0;
    if (!GGML_HRX_CHECK(hrx_device_get_property(
            device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY,
            &memory_total, sizeof(memory_total)))) {
        return 0;
    }
    return static_cast<size_t>(memory_total);
}

static std::string ggml_backend_hrx_device_architecture(hrx_device_t device) {
    std::array<char, 128> architecture = {};
    if (!GGML_HRX_CHECK(hrx_device_get_property(
            device, HRX_DEVICE_PROPERTY_ARCHITECTURE,
            architecture.data(), architecture.size()))) {
        return std::string();
    }
    return std::string(architecture.data());
}

static std::string ggml_backend_hrx_device_description(hrx_device_t device) {
    std::array<char, 128> name = {};
    std::array<char, 128> architecture = {};

    if (!GGML_HRX_CHECK(hrx_device_get_property(
            device, HRX_DEVICE_PROPERTY_NAME, name.data(), name.size()))) {
        std::snprintf(name.data(), name.size(), "unknown");
    }

    if (!GGML_HRX_CHECK(hrx_device_get_property(
            device, HRX_DEVICE_PROPERTY_ARCHITECTURE,
            architecture.data(), architecture.size()))) {
        std::snprintf(architecture.data(), architecture.size(), "unknown");
    }

    std::string description(name.data());
    if (!description.empty() && architecture[0] != '\0') {
        description += " (";
        description += architecture.data();
        description += ")";
    }
    return description.empty() ? std::string("HRX GPU") : description;
}

static const char * ggml_backend_hrx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return ggml_backend_hrx_get_buft_context(buft)->name.c_str();
}

static void ggml_backend_hrx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (context->buffer) {
        hrx_buffer_release(context->buffer);
    }
    delete context;
}

static void * ggml_backend_hrx_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ggml_backend_hrx_get_buffer_context(buffer)->base;
}

static void ggml_backend_hrx_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (size == 0 || !context->buffer) {
        return;
    }

    const size_t tensor_size = ggml_nbytes(tensor);
    if (offset > tensor_size || size > tensor_size - offset) {
        GGML_LOG_ERROR(
            "%s: tensor memset range is out of bounds for %s\n",
            __func__, tensor->name);
        return;
    }
    const size_t tensor_buffer_offset =
        ggml_backend_hrx_tensor_offset(context, tensor);
    const bool packed_tensor =
        ggml_backend_hrx_is_q5_down_group4_tensor(tensor);
    const auto packed_range =
        ggml_backend_hrx_get_q5_down_group4_range_state(
            context, tensor_buffer_offset, tensor_size);
    if ((!packed_tensor &&
         packed_range !=
             ggml_backend_hrx_q5_down_group4_range_state::none) ||
        (packed_tensor &&
         packed_range ==
             ggml_backend_hrx_q5_down_group4_range_state::overlap)) {
        GGML_LOG_ERROR(
            "%s: refusing non-semantic memset through an alias of packed "
            "Q5-down storage (%s)\n",
            __func__, tensor->name);
        return;
    }
    if (packed_tensor &&
        !(offset == 0 &&
          size == ggml_hrx_q5_down_group4::kTensorBytes)) {
        bool ok = false;
        try {
            std::vector<uint8_t> canonical(
                ggml_hrx_q5_down_group4::kTensorBytes);
            if (!ggml_backend_hrx_read_q5_down_group4_canonical(
                    context,
                    tensor,
                    tensor_buffer_offset,
                    buffer->size,
                    canonical.data(),
                    canonical.size())) {
                GGML_LOG_ERROR(
                    "%s: failed to read packed tensor %s for semantic memset\n",
                    __func__, tensor->name);
                return;
            }
            std::memset(canonical.data() + offset, value, size);
            ok = ggml_backend_hrx_write_q5_down_group4_canonical(
                    context,
                    tensor,
                    canonical.data(),
                    canonical.size(),
                    tensor_buffer_offset,
                    buffer->size);
            if (!ok) {
                GGML_LOG_ERROR(
                    "%s: failed to write packed tensor %s after semantic memset\n",
                    __func__, tensor->name);
            }
        } catch (const std::bad_alloc &) {
            GGML_LOG_ERROR(
                "%s: transient Q5-down group4 memset allocation failed for %s\n",
                __func__, tensor->name);
        }
        if (ok && !ggml_backend_hrx_register_q5_down_group4_range(
                context, tensor_buffer_offset, tensor_size)) {
            GGML_LOG_ERROR(
                "%s: packed Q5-down range registration conflict for %s\n",
                __func__, tensor->name);
        }
        return;
    }

    if (!ggml_backend_hrx_sync_streams(context->device_context)) {
        return;
    }

    const size_t buffer_offset = tensor_buffer_offset + offset;
    const bool ok = ggml_backend_hrx_queue_fill_stream_sync(
        context->device_context, context->buffer, buffer_offset, size,
        &value, sizeof(value));
    if (ok && packed_tensor &&
        !ggml_backend_hrx_register_q5_down_group4_range(
            context, tensor_buffer_offset, tensor_size)) {
        GGML_LOG_ERROR(
            "%s: packed Q5-down range registration conflict for %s\n",
            __func__, tensor->name);
    }
}

static void ggml_backend_hrx_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (size == 0 || !context->buffer) {
        return;
    }

    const size_t tensor_buffer_offset =
        ggml_backend_hrx_tensor_offset(context, tensor);
    const size_t tensor_size = ggml_nbytes(tensor);
    const bool packed_tensor =
        ggml_backend_hrx_is_q5_down_group4_tensor(tensor);
    const auto packed_range =
        ggml_backend_hrx_get_q5_down_group4_range_state(
            context, tensor_buffer_offset, tensor_size);
    if ((!packed_tensor &&
         packed_range !=
             ggml_backend_hrx_q5_down_group4_range_state::none) ||
        (packed_tensor &&
         packed_range ==
             ggml_backend_hrx_q5_down_group4_range_state::overlap)) {
        GGML_LOG_ERROR(
            "%s: refusing non-semantic upload through an alias of packed "
            "Q5-down storage (%s)\n",
            __func__, tensor->name);
        return;
    }
    const bool ok = packed_tensor
        ? ggml_backend_hrx_set_q5_down_group4_semantic(
              context,
              tensor,
              data,
              offset,
              size,
              tensor_buffer_offset,
              buffer->size)
        : ggml_backend_hrx_stage_and_copy_tensor(
              context,
              tensor,
              data,
              tensor_buffer_offset + offset,
              buffer->size,
              size);
    if (!ok) {
        GGML_LOG_ERROR("%s: failed to upload tensor %s through HRX staging\n", __func__, tensor->name);
    } else if (packed_tensor &&
               !ggml_backend_hrx_register_q5_down_group4_range(
                   context, tensor_buffer_offset, tensor_size)) {
        GGML_LOG_ERROR(
            "%s: packed Q5-down range registration conflict for %s\n",
            __func__, tensor->name);
    }
}

static void ggml_backend_hrx_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (size == 0 || !context->buffer) {
        return;
    }

    const size_t tensor_buffer_offset =
        ggml_backend_hrx_tensor_offset(context, tensor);
    const size_t tensor_size = ggml_nbytes(tensor);
    const bool packed_tensor =
        ggml_backend_hrx_is_q5_down_group4_tensor(tensor);
    const auto packed_range =
        ggml_backend_hrx_get_q5_down_group4_range_state(
            context, tensor_buffer_offset, tensor_size);
    if ((!packed_tensor &&
         packed_range !=
             ggml_backend_hrx_q5_down_group4_range_state::none) ||
        (packed_tensor &&
         packed_range !=
             ggml_backend_hrx_q5_down_group4_range_state::exact)) {
        GGML_LOG_ERROR(
            "%s: refusing read through an unregistered/aliased packed "
            "Q5-down range (%s)\n",
            __func__, tensor->name);
        return;
    }
    const bool ok = packed_tensor
        ? ggml_backend_hrx_get_q5_down_group4_semantic(
              context,
              tensor,
              data,
              offset,
              size,
              tensor_buffer_offset,
              buffer->size)
        : ggml_backend_hrx_copy_tensor_to_staging(
              context,
              tensor,
              tensor_buffer_offset + offset,
              buffer->size,
              data,
              size);
    if (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor %s through HRX staging\n", __func__, tensor->name);
    }
}

static bool ggml_backend_hrx_buffer_cpy_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t src_buffer = src->view_src ? src->view_src->buffer : src->buffer;
    if (!src_buffer || src_buffer->iface.get_base != ggml_backend_hrx_buffer_get_base) {
        return false;
    }

    auto * dst_context = ggml_backend_hrx_get_buffer_context(buffer);
    auto * src_context = ggml_backend_hrx_get_buffer_context(src_buffer);
    if (dst_context->device_context != src_context->device_context ||
        !dst_context->buffer || !src_context->buffer) {
        return false;
    }

    if (!ggml_backend_hrx_sync_streams(dst_context->device_context)) {
        return false;
    }

    const size_t src_offset = ggml_backend_hrx_tensor_offset(src_context, src);
    const size_t dst_offset = ggml_backend_hrx_tensor_offset(dst_context, dst);
    const size_t size = ggml_nbytes(src);
    const bool src_packed =
        ggml_backend_hrx_is_q5_down_group4_tensor(src);
    const bool dst_packed =
        ggml_backend_hrx_is_q5_down_group4_tensor(dst);
    const auto src_range =
        ggml_backend_hrx_get_q5_down_group4_range_state(
            src_context, src_offset, size);
    const size_t dst_size = ggml_nbytes(dst);
    const auto dst_range =
        ggml_backend_hrx_get_q5_down_group4_range_state(
            dst_context, dst_offset, dst_size);
    if ((!src_packed &&
         src_range !=
             ggml_backend_hrx_q5_down_group4_range_state::none) ||
        (src_packed &&
         src_range !=
             ggml_backend_hrx_q5_down_group4_range_state::exact) ||
        (!dst_packed &&
         dst_range !=
             ggml_backend_hrx_q5_down_group4_range_state::none) ||
        (dst_packed &&
         dst_range ==
             ggml_backend_hrx_q5_down_group4_range_state::overlap)) {
        GGML_LOG_ERROR(
            "%s: refusing device copy through an unregistered/aliased "
            "Q5-down packed range (%s -> %s)\n",
            __func__, src->name, dst->name);
        return false;
    }
    if (src_packed != dst_packed) {
        if (size != dst_size ||
            size != ggml_hrx_q5_down_group4::kTensorBytes) {
            return false;
        }
        try {
            std::vector<uint8_t> canonical(size);
            const bool read_ok = src_packed
                ? ggml_backend_hrx_read_q5_down_group4_canonical(
                      src_context,
                      src,
                      src_offset,
                      src_buffer->size,
                      canonical.data(),
                      canonical.size())
                : ggml_backend_hrx_copy_tensor_to_staging(
                      src_context,
                      src,
                      src_offset,
                      src_buffer->size,
                      canonical.data(),
                      canonical.size());
            if (!read_ok) {
                return false;
            }
            const bool write_ok = dst_packed
                ? ggml_backend_hrx_write_q5_down_group4_canonical(
                      dst_context,
                      dst,
                      canonical.data(),
                      canonical.size(),
                      dst_offset,
                      buffer->size)
                : ggml_backend_hrx_stage_and_copy_tensor(
                      dst_context,
                      dst,
                      canonical.data(),
                      dst_offset,
                      buffer->size,
                      canonical.size());
            if (write_ok && dst_packed &&
                !ggml_backend_hrx_register_q5_down_group4_range(
                    dst_context, dst_offset, dst_size)) {
                GGML_LOG_ERROR(
                    "%s: packed Q5-down destination range conflict for %s\n",
                    __func__, dst->name);
                return false;
            }
            return write_ok;
        } catch (const std::bad_alloc &) {
            GGML_LOG_ERROR(
                "%s: transient Q5-down group4 copy allocation failed (%s -> %s)\n",
                __func__, src->name, dst->name);
            return false;
        }
    }
    const bool copy_ok = ggml_backend_hrx_queue_copy_stream_sync(
        dst_context->device_context,
        src_context->buffer, src_offset,
        dst_context->buffer, dst_offset,
        size);
    if (copy_ok && dst_packed &&
        !ggml_backend_hrx_register_q5_down_group4_range(
            dst_context, dst_offset, dst_size)) {
        GGML_LOG_ERROR(
            "%s: packed Q5-down destination range conflict for %s\n",
            __func__, dst->name);
        return false;
    }
    return copy_ok;
}

static void ggml_backend_hrx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (buffer->size == 0 || !context->buffer) {
        return;
    }

    if (!ggml_backend_hrx_sync_streams(context->device_context)) {
        return;
    }

    (void) ggml_backend_hrx_queue_fill_stream_sync(
        context->device_context, context->buffer, 0, buffer->size, &value, sizeof(value));
}

static const ggml_backend_buffer_i ggml_backend_hrx_buffer_i = {
    /* .free_buffer   = */ ggml_backend_hrx_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_hrx_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_hrx_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_hrx_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_hrx_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ ggml_backend_hrx_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_hrx_buffer_clear,
    /* .reset         = */ nullptr,
};

static ggml_backend_buffer_t ggml_backend_hrx_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    auto * buft_context = ggml_backend_hrx_get_buft_context(buft);

    hrx_buffer_t hrx_buffer = nullptr;
    if (size > 0 &&
        !GGML_HRX_CHECK(hrx_allocator_allocate_buffer(
            hrx_device_allocator(buft_context->device_context->device),
            buft_context->params, size, &hrx_buffer))) {
        return nullptr;
    }

    auto * context =
        new (std::nothrow) ggml_backend_hrx_buffer_context;
    if (!context) {
        if (hrx_buffer) {
            hrx_buffer_release(hrx_buffer);
        }
        return nullptr;
    }
    context->device_context = buft_context->device_context;
    context->buffer = hrx_buffer;
    context->base =
        reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE);

    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(
        buft, ggml_backend_hrx_buffer_i, context, size);
    if (!buffer) {
        if (context->buffer) {
            hrx_buffer_release(context->buffer);
        }
        delete context;
    }
    return buffer;
}

static size_t ggml_backend_hrx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_HRX_ALIGNMENT;
}

static size_t ggml_backend_hrx_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * buft_context = ggml_backend_hrx_get_buft_context(buft);
    return buft_context->device_context->memory_total > 0 ?
        buft_context->device_context->memory_total :
        std::numeric_limits<size_t>::max();
}

static const ggml_backend_buffer_type_i ggml_backend_hrx_buffer_type_i = {
    /* .get_name       = */ ggml_backend_hrx_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_hrx_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_hrx_buffer_type_get_alignment,
    /* .get_max_size   = */ ggml_backend_hrx_buffer_type_get_max_size,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ nullptr,
};

static ggml_backend_buffer_type_t ggml_backend_hrx_device_buffer_type(ggml_backend_dev_t dev) {
    auto * device_context = ggml_backend_hrx_get_device_context(dev);
    static std::vector<std::unique_ptr<ggml_backend_buffer_type>> buffer_types;
    static std::vector<std::unique_ptr<ggml_backend_hrx_buffer_type_context>> contexts;

    for (const auto & buft : buffer_types) {
        auto * context = ggml_backend_hrx_get_buft_context(buft.get());
        if (context->device_context == device_context) {
            return buft.get();
        }
    }

    auto * context = new ggml_backend_hrx_buffer_type_context {
        /* .device_context = */ device_context,
        /* .name           = */ device_context->name,
        /* .params         = */ {
            /* .type = */ HRX_MEMORY_TYPE_DEVICE_LOCAL,
            /* .access = */ HRX_MEMORY_ACCESS_ALL,
            /* .usage = */ HRX_BUFFER_USAGE_DEFAULT,
            /* .queue_affinity = */ 0,
        },
    };

    auto * buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_hrx_buffer_type_i,
        /* .device  = */ dev,
        /* .context = */ context,
    };

    contexts.emplace_back(context);
    buffer_types.emplace_back(buft);
    return buft;
}

static const char * ggml_backend_hrx_get_name(ggml_backend_t backend) {
    return static_cast<ggml_backend_hrx_context *>(backend->context)->name.c_str();
}

static void ggml_backend_hrx_free(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    if (context->stream) {
        GGML_HRX_CHECK(hrx_stream_synchronize(context->stream));
        ggml_backend_hrx_unregister_stream(context->device_context, context->stream);
        hrx_stream_release(context->stream);
    }
    if (context->device_context->ssm_conv_silu_plan &&
        context->device_context->ssm_conv_silu_plan->owner == context) {
        ggml_backend_hrx_clear_ssm_conv_silu_plan(
            context->device_context);
    }
    delete context;
    delete backend;
}

static void ggml_backend_hrx_synchronize(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    if (context->stream) {
        GGML_HRX_CHECK(hrx_stream_synchronize(context->stream));
        std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
        if (auto * arena = ggml_backend_hrx_find_staging_arena_locked(context->device_context, context->stream)) {
            ggml_backend_hrx_reset_staging_arena_locked(*arena);
        }
    }
}

// A node with no output elements has nothing to compute. ggml emits these --
// the recurrent-state cache copy is empty whenever no sequence needs its state
// rolled forward -- and no catalog route matches a zero-element shape, so they
// have to be recognised as no-ops rather than left to route lookup, which would
// reject them and abort the scheduler on a pre-allocated tensor whose backend
// "cannot run" a copy of nothing.
static bool ggml_backend_hrx_is_empty_op(const ggml_tensor * op) {
    return ggml_nelements(op) == 0;
}

static bool ggml_backend_hrx_is_metadata_op(const ggml_tensor * op) {
    switch (op->op) {
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

static const char * ggml_backend_hrx_catalog_type_name(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return "F32";
        case GGML_TYPE_F16:
            return "F16";
        case GGML_TYPE_BF16:
            return "BF16";
        case GGML_TYPE_I32:
            return "I32";
        case GGML_TYPE_Q4_K:
            return "Q4_K";
        case GGML_TYPE_Q5_K:
            return "Q5_K";
        case GGML_TYPE_Q6_K:
            return "Q6_K";
        case GGML_TYPE_Q8_0:
            return "Q8_0";
        case GGML_TYPE_Q8_1:
            return "Q8_1";
        case GGML_TYPE_I64:
            return "I64";
        default:
            return ggml_type_name(type);
    }
}

static void ggml_backend_hrx_set_shape_alias(
        ggml_backend_hrx_catalog_problem * problem,
        const char * prefix,
        const char * key,
        int64_t value) {
    problem->shape[key] = value;
    std::string namespaced = prefix;
    namespaced += ".";
    namespaced += key;
    problem->shape[std::move(namespaced)] = value;
}

template <typename T>
static void ggml_backend_hrx_append_constant(std::vector<uint8_t> * constants, const T & value) {
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(&value);
    constants->insert(constants->end(), bytes, bytes + sizeof(T));
}

static int64_t ggml_backend_hrx_tensor_row_count(const ggml_tensor * tensor) {
    return tensor ? tensor->ne[1] * tensor->ne[2] * tensor->ne[3] : 0;
}

static int64_t ggml_backend_hrx_tensor_row_stride_elements(const ggml_tensor * tensor) {
    return tensor && tensor->type == GGML_TYPE_F32 ? static_cast<int64_t>(tensor->nb[1] / sizeof(float)) : 0;
}

// Fully dense f32: no padding anywhere, so element i of the tensor lives at
// data[i]. This is what the pointwise, norm, softmax and GLU kernels actually
// require. They are handed ncols, a row count that flattens ne1/ne2/ne3, and at
// most a *source* row stride -- never a destination row stride and never an
// ne2/ne3 stride -- so a tensor that is merely row-contiguous (nb[0] == 4) but
// padded between rows or higher dimensions is addressed incorrectly and yields
// silently wrong results rather than a rejection. Requiring full density sends
// those to the CPU instead.
static bool ggml_backend_hrx_is_f32_dense(const ggml_tensor * tensor) {
    return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor);
}

// Row contiguity alone. Only safe for kernels that are given every stride
// explicitly, which today means the strided-flatten CONT.
static bool ggml_backend_hrx_is_f32_row_contiguous(const ggml_tensor * tensor) {
    return tensor && tensor->type == GGML_TYPE_F32 && tensor->nb[0] == sizeof(float);
}

// Dense rows at a uniform stride: each row's ne0 elements are adjacent, and
// every row across ne1/ne2/ne3 sits at an exact multiple of nb[1]. That is the
// precondition for flattening a tensor to (row_count, row_stride), which is how
// the pointwise kernels address their sources -- `row = linear / ncols` then
// `row * srcN_row_stride`. A column slice of a wider tensor qualifies (nb[1] is
// simply larger than ne0), but padding between ne2/ne3 planes does not, because
// the single row stride cannot express it.
static bool ggml_backend_hrx_is_f32_row_strided(const ggml_tensor * tensor) {
    if (!ggml_backend_hrx_is_f32_row_contiguous(tensor)) {
        return false;
    }
    return tensor->nb[2] == tensor->ne[1] * tensor->nb[1] &&
           tensor->nb[3] == tensor->ne[2] * tensor->nb[2];
}

static bool ggml_backend_hrx_is_row_contiguous(const ggml_tensor * tensor) {
    return tensor && tensor->nb[0] == ggml_type_size(tensor->type);
}

static void ggml_backend_hrx_add_tensor_facts(
        ggml_backend_hrx_catalog_problem * problem,
        const char * prefix,
        const ggml_tensor * tensor) {
    if (!tensor) {
        return;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        problem->facts[std::string(prefix) + ".ne" + std::to_string(i)] = tensor->ne[i];
        problem->facts[std::string(prefix) + ".nb" + std::to_string(i)] = tensor->nb[i];
    }
}

static ggml_backend_buffer_t ggml_backend_hrx_tensor_storage_buffer(const ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buffer || buffer->iface.get_base != ggml_backend_hrx_buffer_get_base) {
        return nullptr;
    }
    return buffer;
}

static bool ggml_backend_hrx_tensor_storage_range(
        const ggml_tensor * tensor,
        ggml_backend_buffer_t * out_buffer,
        size_t * out_offset,
        size_t * out_length) {
    ggml_backend_buffer_t buffer = ggml_backend_hrx_tensor_storage_buffer(tensor);
    if (!buffer || !tensor || !tensor->data) {
        return false;
    }
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    *out_buffer = buffer;
    *out_offset = ggml_backend_hrx_tensor_offset(context, tensor);
    *out_length = ggml_nbytes(tensor);
    return true;
}

static void ggml_backend_hrx_add_tensor_overlap_facts(
        ggml_backend_hrx_catalog_problem * problem,
        const std::vector<const ggml_tensor *> & tensors) {
    for (size_t i = 0; i < tensors.size(); ++i) {
        for (size_t j = i + 1; j < tensors.size(); ++j) {
            int64_t overlaps = 0;
            ggml_backend_buffer_t buffer_i = nullptr;
            ggml_backend_buffer_t buffer_j = nullptr;
            size_t offset_i = 0;
            size_t offset_j = 0;
            size_t length_i = 0;
            size_t length_j = 0;
            if (ggml_backend_hrx_tensor_storage_range(tensors[i], &buffer_i, &offset_i, &length_i) &&
                ggml_backend_hrx_tensor_storage_range(tensors[j], &buffer_j, &offset_j, &length_j) &&
                buffer_i == buffer_j) {
                const size_t end_i = offset_i + length_i;
                const size_t end_j = offset_j + length_j;
                overlaps = std::max(offset_i, offset_j) < std::min(end_i, end_j) ? 1 : 0;
            }
            problem->facts["tensor_overlap." + std::to_string(i) + "_" + std::to_string(j)] = overlaps;
        }
    }
}

static bool ggml_backend_hrx_request_matches_loaded_route(
        ggml_backend_hrx_device_context * device_context,
        const ggml_backend_hrx_dispatch_request & request,
        const char * family) {
    if (!device_context || !device_context->reg_context || !device_context->reg_context->catalog) {
        return false;
    }
    ggml_backend_hrx_catalog_problem problem = request.problem;
    ggml_backend_hrx_add_tensor_overlap_facts(&problem, request.tensors);
    const auto * route = ggml_backend_hrx_catalog_find_route(*device_context->reg_context->catalog, problem);
    return route && (!family || route->family == family);
}

static const ggml_tensor * ggml_backend_hrx_zero_offset_source_chain_target(
        const ggml_tensor * tensor,
        enum ggml_op op) {
    for (const ggml_tensor * cur = tensor; cur != nullptr;) {
        if (cur->op == op) {
            return cur;
        }
        if (cur->view_offs != 0) {
            return nullptr;
        }
        if (cur->view_src) {
            cur = cur->view_src;
            continue;
        }
        if ((cur->op == GGML_OP_VIEW ||
             cur->op == GGML_OP_RESHAPE ||
             cur->op == GGML_OP_PERMUTE ||
             cur->op == GGML_OP_TRANSPOSE) &&
            cur->src[0]) {
            cur = cur->src[0];
            continue;
        }
        return nullptr;
    }
    return nullptr;
}

static bool ggml_backend_hrx_make_mul_mat_problem(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_catalog_problem * out_problem) {
    if (!device_context || !node || node->op != GGML_OP_MUL_MAT || !node->src[0] || !node->src[1] || !out_problem) {
        return false;
    }
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(node)) {
        return false;
    }
    out_problem->op = "MUL_MAT";
    out_problem->target_key = device_context->architecture;
    out_problem->supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(src0->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(src1->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32) ?
            "batched_attention_broadcast" : "contiguous"},
    };
    out_problem->shape = {
        {"k", src0->ne[0]},
        {"rows", src0->ne[1]},
        {"cols", src1->ne[1]},
    };
    out_problem->shape["mul_mat_f16.k"] = src0->ne[0];
    out_problem->shape["mul_mat_f16.rows"] = src0->ne[1];
    out_problem->shape["mul_mat_f16.cols"] = src1->ne[1];
    out_problem->shape["mul_mat_f16.dst_ne2"] = node->ne[2];
    out_problem->shape["mul_mat_f16.dst_ne3"] = node->ne[3];
    out_problem->shape["mul_mat_f16.src0_ne2"] = src0->ne[2];
    out_problem->shape["mul_mat_f16.src0_ne3"] = src0->ne[3];
    out_problem->shape["mul_mat_f16.src0_stride_row"] = src0->nb[1];
    out_problem->shape["mul_mat_f16.src0_stride_ne2"] = src0->nb[2];
    out_problem->shape["mul_mat_f16.src0_stride_ne3"] = src0->nb[3];
    out_problem->shape["mul_mat_f16.src1_stride_col"] = src1->nb[1];
    out_problem->shape["mul_mat_f16.src1_stride_ne2"] = src1->nb[2];
    out_problem->shape["mul_mat_f16.src1_stride_ne3"] = src1->nb[3];
    out_problem->shape["mul_mat_f16.dst_stride_col"] = node->nb[1];
    out_problem->shape["mul_mat_f16.dst_stride_ne2"] = node->nb[2];
    out_problem->shape["mul_mat_f16.dst_stride_ne3"] = node->nb[3];
    out_problem->facts = {
        {"src0.ne0", src0->ne[0]},
        {"src0.ne1", src0->ne[1]},
        {"src0.ne2", src0->ne[2]},
        {"src0.ne3", src0->ne[3]},
        {"src1.ne0", src1->ne[0]},
        {"src1.ne1", src1->ne[1]},
        {"src1.ne2", src1->ne[2]},
        {"src1.ne3", src1->ne[3]},
        {"dst.ne0", node->ne[0]},
        {"dst.ne1", node->ne[1]},
        {"dst.ne2", node->ne[2]},
        {"dst.ne3", node->ne[3]},
    };
    return true;
}

struct ggml_backend_hrx_gdn_q8_silu_mul_match {
    const ggml_tensor * q8_gemm = nullptr;
    const ggml_tensor * silu = nullptr;
    const ggml_tensor * side = nullptr;
};

static bool ggml_backend_hrx_same_storage_span(
        const ggml_tensor * lhs,
        const ggml_tensor * rhs) {
    ggml_backend_buffer_t lhs_buffer = nullptr;
    ggml_backend_buffer_t rhs_buffer = nullptr;
    size_t lhs_offset = 0;
    size_t rhs_offset = 0;
    size_t lhs_length = 0;
    size_t rhs_length = 0;
    return ggml_backend_hrx_tensor_storage_range(
               lhs, &lhs_buffer, &lhs_offset, &lhs_length) &&
           ggml_backend_hrx_tensor_storage_range(
               rhs, &rhs_buffer, &rhs_offset, &rhs_length) &&
           lhs_buffer == rhs_buffer &&
           lhs_offset == rhs_offset &&
           lhs_length == rhs_length;
}

static bool ggml_backend_hrx_disjoint_storage_spans(
        const ggml_tensor * lhs,
        const ggml_tensor * rhs) {
    ggml_backend_buffer_t lhs_buffer = nullptr;
    ggml_backend_buffer_t rhs_buffer = nullptr;
    size_t lhs_offset = 0;
    size_t rhs_offset = 0;
    size_t lhs_length = 0;
    size_t rhs_length = 0;
    if (!ggml_backend_hrx_tensor_storage_range(
            lhs, &lhs_buffer, &lhs_offset, &lhs_length) ||
        !ggml_backend_hrx_tensor_storage_range(
            rhs, &rhs_buffer, &rhs_offset, &rhs_length)) {
        return false;
    }
    return lhs_buffer != rhs_buffer ||
           lhs_offset + lhs_length <= rhs_offset ||
           rhs_offset + rhs_length <= lhs_offset;
}

// True only for a zero-offset metadata/view chain in which every view covers
// the target's complete physical span. This is deliberately narrower than the
// general source-chain helper: the fused epilogue relies on flat-index identity
// between [4096,512] GEMM output and its [128,16384] reshape.
static bool ggml_backend_hrx_zero_offset_full_span_chain_reaches(
        const ggml_tensor * tensor,
        const ggml_tensor * target) {
    if (!tensor || !target) {
        return false;
    }
    const size_t target_bytes = ggml_nbytes(target);
    const ggml_tensor * cur = tensor;
    for (int depth = 0; cur && depth < 16; ++depth) {
        if (cur == target) {
            return true;
        }
        if (cur->view_offs != 0 || ggml_nbytes(cur) != target_bytes) {
            return false;
        }
        if (!ggml_backend_hrx_is_metadata_op(cur)) {
            return false;
        }
        if (cur->src[0]) {
            cur = cur->src[0];
            continue;
        }
        if (cur->view_src) {
            cur = cur->view_src;
            continue;
        }
        return false;
    }
    return false;
}

// Detect metadata descendants independently of offset and span. Fusion
// consumer proofs use this stricter ancestry walk first, then separately admit
// only the exact zero-offset/full-span chain they can safely absorb.
static bool ggml_backend_hrx_metadata_chain_reaches(
        const ggml_tensor * tensor,
        const ggml_tensor * target) {
    if (!tensor || !target) {
        return false;
    }
    const ggml_tensor * cur = tensor;
    for (int depth = 0; cur && depth < 16; ++depth) {
        if (cur == target) {
            return true;
        }
        if (!ggml_backend_hrx_is_metadata_op(cur)) {
            return false;
        }
        if (cur->src[0]) {
            cur = cur->src[0];
            continue;
        }
        if (cur->view_src) {
            cur = cur->view_src;
            continue;
        }
        return false;
    }
    return false;
}

static bool ggml_backend_hrx_match_gdn_q8_silu_mul(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_gdn_q8_silu_mul_match * out_match) {
    if (!device_context || !device_context->current_graph || !node || !out_match ||
        node->op != GGML_OP_MUL || !node->src[0] || !node->src[1] ||
        node->type != GGML_TYPE_F32 ||
        node->src[0]->type != GGML_TYPE_F32 ||
        node->src[1]->type != GGML_TYPE_F32) {
        return false;
    }

    // The measured graph is MUL(side, SILU(reshape(q8 GEMM))) and the MUL
    // writes in place over side. Keep the operand orientation exact instead of
    // admitting a commuted shape whose alias/lifetime contract was not traced.
    const ggml_tensor * side = node->src[0];
    const ggml_tensor * silu = node->src[1];
    if (silu->op != GGML_OP_UNARY ||
        ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU ||
        !silu->src[0]) {
        return false;
    }
    const ggml_tensor * q8_gemm =
        ggml_backend_hrx_zero_offset_source_chain_target(
            silu->src[0], GGML_OP_MUL_MAT);
    if (!q8_gemm || !q8_gemm->src[0] || !q8_gemm->src[1] ||
        q8_gemm->src[0]->type != GGML_TYPE_Q8_0 ||
        q8_gemm->src[1]->type != GGML_TYPE_F32 ||
        q8_gemm->type != GGML_TYPE_F32) {
        return false;
    }

    constexpr int64_t expected_bytes = 4096ll * 512ll * sizeof(float);
    if (q8_gemm->src[0]->ne[0] != 2048 ||
        q8_gemm->src[0]->ne[1] != 4096 ||
        q8_gemm->src[0]->ne[2] != 1 ||
        q8_gemm->src[0]->ne[3] != 1 ||
        q8_gemm->src[1]->ne[0] != 2048 ||
        q8_gemm->src[1]->ne[1] != 512 ||
        q8_gemm->src[1]->ne[2] != 1 ||
        q8_gemm->src[1]->ne[3] != 1 ||
        q8_gemm->ne[0] != 4096 ||
        q8_gemm->ne[1] != 512 ||
        q8_gemm->ne[2] != 1 ||
        q8_gemm->ne[3] != 1 ||
        silu->src[0]->ne[0] != 128 ||
        ggml_backend_hrx_tensor_row_count(silu->src[0]) != 16384 ||
        silu->ne[0] != 128 ||
        ggml_backend_hrx_tensor_row_count(silu) != 16384 ||
        side->ne[0] != 128 ||
        ggml_backend_hrx_tensor_row_count(side) != 16384 ||
        node->ne[0] != 128 ||
        ggml_backend_hrx_tensor_row_count(node) != 16384 ||
        !ggml_are_same_shape(side, node) ||
        !ggml_are_same_shape(silu, node) ||
        q8_gemm->src[0]->nb[1] != 2176 ||
        q8_gemm->src[1]->nb[1] != 2048 * sizeof(float) ||
        q8_gemm->nb[1] != 4096 * sizeof(float) ||
        ggml_nbytes(q8_gemm) != expected_bytes ||
        ggml_nbytes(silu->src[0]) != expected_bytes ||
        ggml_nbytes(silu) != expected_bytes ||
        ggml_nbytes(side) != expected_bytes ||
        ggml_nbytes(node) != expected_bytes ||
        !ggml_is_contiguous(q8_gemm->src[0]) ||
        !ggml_is_contiguous(q8_gemm->src[1]) ||
        !ggml_backend_hrx_is_f32_dense(q8_gemm) ||
        !ggml_backend_hrx_is_f32_dense(silu->src[0]) ||
        !ggml_backend_hrx_is_f32_dense(silu) ||
        !ggml_backend_hrx_is_f32_dense(side) ||
        !ggml_backend_hrx_is_f32_dense(node) ||
        !ggml_backend_hrx_zero_offset_full_span_chain_reaches(
            silu->src[0], q8_gemm) ||
        !ggml_backend_hrx_same_storage_span(q8_gemm, silu->src[0]) ||
        !ggml_backend_hrx_same_storage_span(side, node) ||
        !ggml_backend_hrx_disjoint_storage_spans(q8_gemm, silu) ||
        !ggml_backend_hrx_disjoint_storage_spans(q8_gemm, side) ||
        !ggml_backend_hrx_disjoint_storage_spans(silu, side) ||
        !ggml_backend_hrx_disjoint_storage_spans(q8_gemm->src[0], side) ||
        !ggml_backend_hrx_disjoint_storage_spans(q8_gemm->src[1], side) ||
        (q8_gemm->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        (silu->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const ggml_tensor * q8_chain = silu->src[0];
    for (int depth = 0;
         q8_chain && q8_chain != q8_gemm && depth < 16;
         ++depth) {
        if ((q8_chain->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            !ggml_backend_hrx_is_metadata_op(q8_chain)) {
            return false;
        }
        q8_chain = q8_chain->src[0] ?
            q8_chain->src[0] : q8_chain->view_src;
    }
    if (q8_chain != q8_gemm) {
        return false;
    }

    const ggml_cgraph * cgraph = device_context->current_graph;
    int q8_index = -1;
    int silu_index = -1;
    int terminal_index = -1;
    int silu_uses = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * candidate = cgraph->nodes[i];
        if (!candidate) {
            continue;
        }
        if (candidate == q8_gemm) {
            q8_index = i;
        } else if (candidate == silu) {
            silu_index = i;
        } else if (candidate == node) {
            terminal_index = i;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * source = candidate->src[s];
            if (!source) {
                continue;
            }
            if (ggml_backend_hrx_metadata_chain_reaches(source, silu)) {
                if (candidate != node || source != silu) {
                    return false;
                }
                ++silu_uses;
            }
            if (!ggml_backend_hrx_metadata_chain_reaches(source, q8_gemm)) {
                continue;
            }
            const bool exact_silu_input =
                candidate == silu && source == silu->src[0] &&
                ggml_backend_hrx_zero_offset_full_span_chain_reaches(
                    source, q8_gemm);
            const bool exact_metadata_link =
                ggml_backend_hrx_is_metadata_op(candidate) &&
                ggml_backend_hrx_zero_offset_full_span_chain_reaches(
                    candidate, q8_gemm) &&
                ggml_backend_hrx_zero_offset_full_span_chain_reaches(
                    silu->src[0], candidate);
            if ((!exact_silu_input && !exact_metadata_link) ||
                (exact_metadata_link &&
                 (candidate->flags & GGML_TENSOR_FLAG_OUTPUT))) {
                return false;
            }
        }
    }
    if (silu_uses != 1 ||
        q8_index < 0 || silu_index < 0 || terminal_index < 0 ||
        !(q8_index < silu_index && silu_index < terminal_index)) {
        return false;
    }
    // Delaying the q8 work to the terminal MUL is safe only across the exact
    // absorbed chain; another compute node could reuse/overwrite its activation
    // allocation before the fused dispatch reads it.
    for (int i = q8_index + 1; i < terminal_index; ++i) {
        const ggml_tensor * between = cgraph->nodes[i];
        if (between && between != silu &&
            !ggml_backend_hrx_is_metadata_op(between) &&
            !ggml_backend_hrx_is_empty_op(between)) {
            return false;
        }
    }

    out_match->q8_gemm = q8_gemm;
    out_match->silu = silu;
    out_match->side = side;
    return true;
}

static bool ggml_backend_hrx_make_gdn_q8_silu_mul_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request,
        ggml_backend_hrx_gdn_q8_silu_mul_match * out_match = nullptr) {
    if (!device_context || !out_request) {
        return false;
    }
    ggml_backend_hrx_gdn_q8_silu_mul_match match = {};
    if (!ggml_backend_hrx_match_gdn_q8_silu_mul(
            device_context, node, &match)) {
        return false;
    }

    ggml_backend_hrx_dispatch_request request = {};
    if (!ggml_backend_hrx_make_mul_mat_problem(
            device_context, match.q8_gemm, &request.problem)) {
        return false;
    }
    request.problem.supports["fusion"] =
        "MUL_MAT_GDN_SILU_MUL_EPILOGUE";
    // Keep the matrix-output facts from make_mul_mat_problem: the fused
    // destination is a byte-identical reshape, but the base route's constraints
    // intentionally describe its logical [4096,512] matrix result.
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "side", match.side);
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "terminal_dst", node);
    request.tensors = {
        match.q8_gemm->src[0],
        match.q8_gemm->src[1],
        match.side,
        node,
    };
    request.constants.clear();

    ggml_backend_hrx_catalog_problem problem = request.problem;
    ggml_backend_hrx_add_tensor_overlap_facts(&problem, request.tensors);
    const auto * route =
        device_context->reg_context && device_context->reg_context->catalog
            ? ggml_backend_hrx_catalog_find_route(
                  *device_context->reg_context->catalog, problem)
            : nullptr;
    const auto fusion =
        route ? route->supports.find("fusion")
              : decltype(route->supports.find("fusion")){};
    if (!route ||
        route->id !=
            "mul_mat_q8_0_f32_wmmai8_gdn_silu_mul_epilogue" ||
        route->family != "mul_mat_q8_0_f32_tiled" ||
        route->op != "MUL_MAT" ||
        route->source_id !=
            "mul_mat_q8_0_wmmai8_gdn_silu_mul_epilogue" ||
        route->artifact_id !=
            "mul_mat_q8_0_wmmai8_gdn_silu_mul_epilogue_loombc" ||
        route->root_symbol !=
            "@hrx2_mul_mat_q8_0_f32_wmmai8_gdn_silu_mul_epilogue" ||
        route->export_name !=
            "hrx2_mul_mat_q8_0_f32_wmmai8_gdn_silu_mul_epilogue" ||
        route->binding_count != 7 ||
        route->parameter_count != 7 ||
        route->constant_byte_length != 0 ||
        fusion == route->supports.end() ||
        fusion->second != "MUL_MAT_GDN_SILU_MUL_EPILOGUE") {
        return false;
    }

    *out_request = std::move(request);
    if (out_match) {
        *out_match = match;
    }
    return true;
}

static bool ggml_backend_hrx_gdn_rms_side_route_is_exact(
        const ggml_backend_hrx_catalog_route * route) {
    if (!route ||
        route->id !=
            "mul_mat_q8_0_f32_wmmai8_gdn_rms_side_epilogue" ||
        route->family != "mul_mat_q8_0_f32_tiled" ||
        route->op != "MUL_MAT" ||
        route->source_id !=
            "mul_mat_q8_0_wmmai8_gdn_rms_side_epilogue" ||
        route->artifact_id !=
            "mul_mat_q8_0_wmmai8_gdn_rms_side_epilogue_loombc" ||
        route->root_symbol !=
            "@hrx2_mul_mat_q8_0_f32_wmmai8_gdn_rms_side_epilogue" ||
        route->export_name !=
            "hrx2_mul_mat_q8_0_f32_wmmai8_gdn_rms_side_epilogue" ||
        route->binding_count != 9 ||
        route->parameter_count != 9 ||
        route->constant_byte_length != 0 ||
        route->prepasses.size() != 2) {
        return false;
    }
    const auto fusion = route->supports.find("fusion");
    const auto & scale = route->prepasses[0];
    const auto & qact = route->prepasses[1];
    return fusion != route->supports.end() &&
           fusion->second ==
               "MUL_MAT_GDN_RMS_SIDE_SILU_MUL_EPILOGUE" &&
           scale.enabled &&
           scale.artifact_id ==
               "gdn_rms_side_scale_prepass_loombc" &&
           scale.root_symbol ==
               "@hrx2_gdn_rms_side_scales_f32" &&
           scale.export_name ==
               "hrx2_gdn_rms_side_scales_f32" &&
           scale.src_index == 2 &&
           scale.src_indices.size() == 1 &&
           scale.src_indices[0] == 2 &&
           scale.dst_index == -1 &&
           scale.bytes_per_element == sizeof(float) &&
           scale.scratch_element_sources ==
               std::vector<std::string>({
                   "shape.gdn_rms_side.nrows",
               }) &&
           !scale.persistent &&
           scale.scratch_class == "gdn_rms_side_scale" &&
           qact.enabled &&
           qact.artifact_id == "quant_act_q8_loombc" &&
           qact.root_symbol == "@hrx2_quant_act_q8" &&
           qact.export_name == "hrx2_quant_act_q8" &&
           qact.src_index == 1 &&
           qact.src_indices.size() == 1 &&
           qact.src_indices[0] == 1 &&
           qact.dst_index == -1 &&
           qact.bytes_per_element == 0 &&
           qact.scratch_class.empty();
}

static const ggml_backend_hrx_gdn_rms_side_layer_plan *
ggml_backend_hrx_find_gdn_rms_side_layer(
        const ggml_backend_hrx_gdn_rms_side_graph_plan * plan,
        const ggml_tensor * terminal) {
    if (!plan || !plan->ready || !terminal) {
        return nullptr;
    }
    for (const auto & layer : plan->layers) {
        if (layer.terminal == terminal) {
            return &layer;
        }
    }
    return nullptr;
}

static bool ggml_backend_hrx_make_gdn_rms_side_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_backend_hrx_gdn_rms_side_layer_plan & layer,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !layer.q8_gemm ||
        !layer.q8_gemm->src[0] || !layer.q8_gemm->src[1] ||
        !layer.raw || !layer.norm_weight || !layer.terminal) {
        return false;
    }
    ggml_backend_hrx_dispatch_request request = {};
    if (!ggml_backend_hrx_make_mul_mat_problem(
            device_context, layer.q8_gemm, &request.problem)) {
        return false;
    }
    request.problem.supports["fusion"] =
        "MUL_MAT_GDN_RMS_SIDE_SILU_MUL_EPILOGUE";
    ggml_backend_hrx_set_shape_alias(
        &request.problem, "gdn_rms_side", "ncols", 128);
    ggml_backend_hrx_set_shape_alias(
        &request.problem, "gdn_rms_side", "nrows", 16384);
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "raw_side", layer.raw);
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "norm_weight", layer.norm_weight);
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "terminal_dst", layer.terminal);
    request.tensors = {
        layer.q8_gemm->src[0],
        layer.q8_gemm->src[1],
        layer.raw,
        layer.norm_weight,
        layer.terminal,
    };
    request.constants.clear();

    ggml_backend_hrx_catalog_problem problem = request.problem;
    ggml_backend_hrx_add_tensor_overlap_facts(
        &problem, request.tensors);
    const auto * route =
        device_context->reg_context &&
                device_context->reg_context->catalog
            ? ggml_backend_hrx_catalog_find_route(
                  *device_context->reg_context->catalog, problem)
            : nullptr;
    if (!ggml_backend_hrx_gdn_rms_side_route_is_exact(route)) {
        return false;
    }
    *out_request = std::move(request);
    return true;
}

static bool ggml_backend_hrx_make_current_gdn_rms_side_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context) {
        return false;
    }
    const auto * layer =
        ggml_backend_hrx_find_gdn_rms_side_layer(
            device_context->current_gdn_rms_side_plan, node);
    return layer &&
           ggml_backend_hrx_make_gdn_rms_side_request(
               device_context, *layer, out_request);
}

struct ggml_backend_hrx_decode_gdn_q8_silu_mul_match {
    const ggml_tensor * q8_gemm = nullptr;
    const ggml_tensor * silu = nullptr;
    const ggml_tensor * side = nullptr;
};

static bool ggml_backend_hrx_match_decode_gdn_q8_silu_mul(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_decode_gdn_q8_silu_mul_match * out_match) {
    if (!device_context || !device_context->current_graph || !node ||
        !out_match || node->op != GGML_OP_MUL ||
        !node->src[0] || !node->src[1] ||
        node->type != GGML_TYPE_F32 ||
        node->src[0]->type != GGML_TYPE_F32 ||
        node->src[1]->type != GGML_TYPE_F32) {
        return false;
    }

    // The admitted one-token graph is exactly
    // MUL(side, SILU(reshape(Q8_GEMM))) with dst == side. Do not admit the
    // commuted MUL orientation: only binding2/binding3 alias was observed and
    // proved safe for the four-binding kernel.
    const ggml_tensor * side = node->src[0];
    const ggml_tensor * silu = node->src[1];
    if (silu->op != GGML_OP_UNARY ||
        ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU ||
        !silu->src[0]) {
        return false;
    }
    const ggml_tensor * q8_gemm =
        ggml_backend_hrx_zero_offset_source_chain_target(
            silu->src[0], GGML_OP_MUL_MAT);
    if (!q8_gemm || !q8_gemm->src[0] || !q8_gemm->src[1] ||
        q8_gemm->src[0]->type != GGML_TYPE_Q8_0 ||
        q8_gemm->src[1]->type != GGML_TYPE_F32 ||
        q8_gemm->type != GGML_TYPE_F32) {
        return false;
    }

    constexpr int64_t expected_bytes = 4096ll * sizeof(float);
    if (q8_gemm->src[0]->ne[0] != 2048 ||
        q8_gemm->src[0]->ne[1] != 4096 ||
        q8_gemm->src[0]->ne[2] != 1 ||
        q8_gemm->src[0]->ne[3] != 1 ||
        q8_gemm->src[1]->ne[0] != 2048 ||
        q8_gemm->src[1]->ne[1] != 1 ||
        q8_gemm->src[1]->ne[2] != 1 ||
        q8_gemm->src[1]->ne[3] != 1 ||
        q8_gemm->ne[0] != 4096 ||
        q8_gemm->ne[1] != 1 ||
        q8_gemm->ne[2] != 1 ||
        q8_gemm->ne[3] != 1 ||
        silu->src[0]->ne[0] != 128 ||
        silu->src[0]->ne[1] != 32 ||
        silu->src[0]->ne[2] != 1 ||
        silu->src[0]->ne[3] != 1 ||
        silu->ne[0] != 128 ||
        silu->ne[1] != 32 ||
        silu->ne[2] != 1 ||
        silu->ne[3] != 1 ||
        side->ne[0] != 128 ||
        side->ne[1] != 32 ||
        side->ne[2] != 1 ||
        side->ne[3] != 1 ||
        node->ne[0] != 128 ||
        node->ne[1] != 32 ||
        node->ne[2] != 1 ||
        node->ne[3] != 1 ||
        !ggml_are_same_shape(side, node) ||
        !ggml_are_same_shape(silu, node) ||
        q8_gemm->src[0]->nb[1] != 2176 ||
        q8_gemm->src[1]->nb[1] != 2048 * sizeof(float) ||
        q8_gemm->nb[1] != 4096 * sizeof(float) ||
        ggml_nbytes(q8_gemm->src[0]) != 8912896 ||
        ggml_nbytes(q8_gemm->src[1]) != 8192 ||
        ggml_nbytes(q8_gemm) != expected_bytes ||
        ggml_nbytes(silu->src[0]) != expected_bytes ||
        ggml_nbytes(silu) != expected_bytes ||
        ggml_nbytes(side) != expected_bytes ||
        ggml_nbytes(node) != expected_bytes ||
        !ggml_is_contiguous(q8_gemm->src[0]) ||
        !ggml_is_contiguous(q8_gemm->src[1]) ||
        !ggml_backend_hrx_is_f32_dense(q8_gemm) ||
        !ggml_backend_hrx_is_f32_dense(silu->src[0]) ||
        !ggml_backend_hrx_is_f32_dense(silu) ||
        !ggml_backend_hrx_is_f32_dense(side) ||
        !ggml_backend_hrx_is_f32_dense(node) ||
        !ggml_backend_hrx_zero_offset_full_span_chain_reaches(
            silu->src[0], q8_gemm) ||
        !ggml_backend_hrx_same_storage_span(q8_gemm, silu->src[0]) ||
        !ggml_backend_hrx_same_storage_span(side, node) ||
        !ggml_backend_hrx_disjoint_storage_spans(
            q8_gemm->src[0], q8_gemm->src[1]) ||
        !ggml_backend_hrx_disjoint_storage_spans(
            q8_gemm->src[0], q8_gemm) ||
        !ggml_backend_hrx_disjoint_storage_spans(
            q8_gemm->src[1], q8_gemm) ||
        !ggml_backend_hrx_disjoint_storage_spans(q8_gemm, silu) ||
        !ggml_backend_hrx_disjoint_storage_spans(q8_gemm, side) ||
        !ggml_backend_hrx_disjoint_storage_spans(silu, side) ||
        !ggml_backend_hrx_disjoint_storage_spans(
            q8_gemm->src[0], side) ||
        !ggml_backend_hrx_disjoint_storage_spans(
            q8_gemm->src[1], side) ||
        (q8_gemm->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        (silu->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    // Every metadata node between the Q8 output and SiLU input must preserve
    // the complete zero-offset span and remain private.
    const ggml_tensor * q8_chain = silu->src[0];
    for (int depth = 0;
         q8_chain && q8_chain != q8_gemm && depth < 16;
         ++depth) {
        if ((q8_chain->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            !ggml_backend_hrx_is_metadata_op(q8_chain)) {
            return false;
        }
        q8_chain = q8_chain->src[0] ?
            q8_chain->src[0] : q8_chain->view_src;
    }
    if (q8_chain != q8_gemm) {
        return false;
    }

    const ggml_cgraph * cgraph = device_context->current_graph;
    int q8_index = -1;
    int silu_index = -1;
    int terminal_index = -1;
    int silu_uses = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * candidate = cgraph->nodes[i];
        if (!candidate) {
            continue;
        }
        if (candidate == q8_gemm) {
            q8_index = i;
        } else if (candidate == silu) {
            silu_index = i;
        } else if (candidate == node) {
            terminal_index = i;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * source = candidate->src[s];
            if (!source) {
                continue;
            }
            if (ggml_backend_hrx_metadata_chain_reaches(source, silu)) {
                if (candidate != node || source != silu) {
                    return false;
                }
                ++silu_uses;
            }
            if (!ggml_backend_hrx_metadata_chain_reaches(
                    source, q8_gemm)) {
                continue;
            }
            const bool exact_silu_input =
                candidate == silu && source == silu->src[0] &&
                ggml_backend_hrx_zero_offset_full_span_chain_reaches(
                    source, q8_gemm);
            const bool exact_metadata_link =
                ggml_backend_hrx_is_metadata_op(candidate) &&
                ggml_backend_hrx_zero_offset_full_span_chain_reaches(
                    candidate, q8_gemm) &&
                ggml_backend_hrx_zero_offset_full_span_chain_reaches(
                    silu->src[0], candidate);
            if ((!exact_silu_input && !exact_metadata_link) ||
                (exact_metadata_link &&
                 (candidate->flags & GGML_TENSOR_FLAG_OUTPUT))) {
                return false;
            }
        }
    }
    if (silu_uses != 1 ||
        q8_index < 0 || silu_index < 0 || terminal_index < 0 ||
        !(q8_index < silu_index && silu_index < terminal_index)) {
        return false;
    }
    // The delayed terminal launch is safe only when no unrelated compute can
    // alter an input or reuse the in-place destination between the producer
    // and terminal MUL.
    for (int i = q8_index + 1; i < terminal_index; ++i) {
        const ggml_tensor * between = cgraph->nodes[i];
        if (between && between != silu &&
            !ggml_backend_hrx_is_metadata_op(between) &&
            !ggml_backend_hrx_is_empty_op(between)) {
            return false;
        }
    }

    out_match->q8_gemm = q8_gemm;
    out_match->silu = silu;
    out_match->side = side;
    return true;
}

static bool ggml_backend_hrx_decode_gdn_q8_silu_mul_route_is_exact(
        const ggml_backend_hrx_catalog_route * route) {
    static const char * route_id =
        "mul_mat_q8_0_f32_packed_decode_k2048_r4096_c1_wg256_"
        "scfunroll2_gdn_silu_mul_epilogue";
    static const char * source_id =
        "mul_mat_q8_0_f32_packed_decode_gdn_silu_mul_epilogue";
    static const char * symbol =
        "hrx2_mul_mat_q8_0_f32_static_packed_scf_unroll_"
        "decode_gdn_silu_mul_epilogue";
    if (!route ||
        route->id != route_id ||
        route->family != "mul_mat_q8_0_f32" ||
        route->op != "MUL_MAT" ||
        route->source_id != source_id ||
        route->artifact_id != std::string(source_id) + "_loombc" ||
        route->root_symbol != std::string("@") + symbol ||
        route->export_name != symbol ||
        route->binding_count != 4 ||
        route->parameter_count != 4 ||
        route->constant_byte_length != 0 ||
        !route->prepasses.empty() ||
        route->constraints.size() != 15 ||
        route->bindings.size() != 5) {
        return false;
    }
    const auto fusion = route->supports.find("fusion");
    if (fusion == route->supports.end() ||
        fusion->second !=
            "MUL_MAT_GDN_SILU_MUL_EPILOGUE_DECODE") {
        return false;
    }
    const std::array<std::pair<const char *, int64_t>, 3>
        exact_shapes = {{
            {"k", 2048},
            {"rows", 4096},
            {"cols", 1},
        }};
    for (const auto & [name, value] : exact_shapes) {
        const auto minimum = route->shape_min.find(name);
        const auto maximum = route->shape_max.find(name);
        if (minimum == route->shape_min.end() ||
            maximum == route->shape_max.end() ||
            minimum->second != value ||
            maximum->second != value) {
            return false;
        }
    }
    const std::array<std::pair<const char *, const char *>, 3>
        shape_bindings = {{
            {"@hrx2.shape.k", "k"},
            {"@hrx2.shape.rows", "rows"},
            {"@hrx2.shape.cols", "cols"},
        }};
    for (size_t i = 0; i < shape_bindings.size(); ++i) {
        if (route->bindings[i].key != shape_bindings[i].first ||
            route->bindings[i].shape_source !=
                shape_bindings[i].second ||
            !route->bindings[i].value.empty()) {
            return false;
        }
    }
    if (route->bindings[3].key !=
            "@hrx2.tuning.workgroup_size" ||
        route->bindings[3].value != "256" ||
        !route->bindings[3].shape_source.empty() ||
        route->bindings[4].key !=
            "@hrx2.tuning.q8_0_f32.unroll_factor" ||
        route->bindings[4].value != "2" ||
        !route->bindings[4].shape_source.empty()) {
        return false;
    }
    const auto has_overlap = [route](
            const char * source, int64_t value) {
        size_t matches = 0;
        for (const auto & constraint : route->constraints) {
            if (constraint.source == source &&
                constraint.has_eq_value &&
                constraint.eq_value == value) {
                ++matches;
            }
        }
        return matches == 1;
    };
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            const std::string source =
                "tensor_overlap." + std::to_string(i) + "_" +
                std::to_string(j);
            if (!has_overlap(
                    source.c_str(), (i == 2 && j == 3) ? 1 : 0)) {
                return false;
            }
        }
    }
    return true;
}

static bool ggml_backend_hrx_make_decode_gdn_q8_silu_mul_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request,
        ggml_backend_hrx_decode_gdn_q8_silu_mul_match *
            out_match = nullptr) {
    if (!device_context || !out_request) {
        return false;
    }
    ggml_backend_hrx_decode_gdn_q8_silu_mul_match match = {};
    if (!ggml_backend_hrx_match_decode_gdn_q8_silu_mul(
            device_context, node, &match)) {
        return false;
    }

    ggml_backend_hrx_dispatch_request request = {};
    if (!ggml_backend_hrx_make_mul_mat_problem(
            device_context, match.q8_gemm, &request.problem)) {
        return false;
    }
    request.problem.supports["fusion"] =
        "MUL_MAT_GDN_SILU_MUL_EPILOGUE_DECODE";
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "side", match.side);
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "terminal_dst", node);
    request.tensors = {
        match.q8_gemm->src[0],
        match.q8_gemm->src[1],
        match.side,
        node,
    };
    request.constants.clear();

    ggml_backend_hrx_catalog_problem problem = request.problem;
    ggml_backend_hrx_add_tensor_overlap_facts(
        &problem, request.tensors);
    const auto * route =
        device_context->reg_context &&
                device_context->reg_context->catalog
            ? ggml_backend_hrx_catalog_find_route(
                  *device_context->reg_context->catalog, problem)
            : nullptr;
    if (!ggml_backend_hrx_decode_gdn_q8_silu_mul_route_is_exact(
            route)) {
        return false;
    }

    *out_request = std::move(request);
    if (out_match) {
        *out_match = match;
    }
    return true;
}

struct ggml_backend_hrx_fa_gate_epilogue_match;

static bool ggml_backend_hrx_make_fa_gate_epilogue_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request,
        ggml_backend_hrx_fa_gate_epilogue_match * out_match = nullptr);

struct ggml_backend_hrx_binary_f32_layout {
    std::string key;
    bool rhs_row_broadcast = false;
};

static bool ggml_backend_hrx_describe_binary_f32_layout(
        const ggml_tensor * node,
        ggml_backend_hrx_binary_f32_layout * out_layout) {
    if (!node || !node->src[0] || !node->src[1] || !out_layout) {
        return false;
    }
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    // The sources are addressed through explicit row strides, so a column slice
    // of a wider tensor is fine; the destination is written by flat linear index
    // and has to be dense.
    if (!ggml_backend_hrx_is_f32_row_strided(src0) ||
        !ggml_backend_hrx_is_f32_row_strided(src1) ||
        !ggml_backend_hrx_is_f32_dense(node)) {
        return false;
    }

    const int64_t ncols = node->ne[0];
    const int64_t nrows = ggml_backend_hrx_tensor_row_count(node);
    if (src0->ne[0] != ncols || ggml_backend_hrx_tensor_row_count(src0) != nrows) {
        return false;
    }
    if (src1->ne[0] == ncols && ggml_backend_hrx_tensor_row_count(src1) == nrows) {
        out_layout->key = "contiguous";
        out_layout->rhs_row_broadcast = false;
        return true;
    }
    if (src1->ne[0] == 1 && ggml_backend_hrx_tensor_row_count(src1) == nrows) {
        out_layout->key = "contiguous_src0_rhs_column_broadcast";
        out_layout->rhs_row_broadcast = false;
        return true;
    }
    if (src1->ne[0] == ncols && ggml_backend_hrx_tensor_row_count(src1) == 1) {
        out_layout->key = "contiguous_src0_rhs_row_broadcast";
        out_layout->rhs_row_broadcast = true;
        return true;
    }
    return false;
}

static void ggml_backend_hrx_set_pointwise_shape(
        ggml_backend_hrx_catalog_problem * problem,
        const ggml_tensor * node,
        bool rhs_row_broadcast) {
    ggml_backend_hrx_set_shape_alias(problem, "pointwise", "ncols", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(problem, "pointwise", "nrows", ggml_backend_hrx_tensor_row_count(node));
    ggml_backend_hrx_set_shape_alias(
        problem, "pointwise", "src0_row_stride",
        ggml_backend_hrx_tensor_row_stride_elements(node->src[0]));
    ggml_backend_hrx_set_shape_alias(
        problem, "pointwise", "src1_row_stride",
        rhs_row_broadcast ? 0 : ggml_backend_hrx_tensor_row_stride_elements(node->src[1]));
    ggml_backend_hrx_set_shape_alias(problem, "pointwise", "src1_ncols", node->src[1]->ne[0]);
}

static const ggml_backend_hrx_moe_router_tail_layer_plan *
ggml_backend_hrx_find_moe_router_tail_layer(
        const ggml_backend_hrx_moe_router_tail_graph_plan * plan,
        const ggml_tensor * terminal) {
    if (!plan || !plan->ready || !terminal) {
        return nullptr;
    }
    for (const auto & layer : plan->layers) {
        if (layer.adds[6] == terminal) {
            return &layer;
        }
    }
    return nullptr;
}

static bool ggml_backend_hrx_make_moe_router_tail_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_backend_hrx_moe_router_tail_layer_plan & layer,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !layer.selected || !layer.dst ||
        !layer.router_source ||
        (layer.ntokens != 1 && layer.ntokens != 512) ||
        (layer.ntokens == 1 &&
         (layer.router_source_stride != 8 ||
          layer.router_source_offset != 0)) ||
        (layer.ntokens == 512 &&
         (layer.router_source_stride != 256 ||
          layer.router_source_offset != 8))) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "ADD";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", "F32"},
        {"dst_type", "F32"},
        {"layout", "moe_slice_sum"},
        {"fusion", "ADD_CHAIN_SLICE_SUM"},
    };
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "sumslices", "n", 2048);
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "sumslices", "ntokens", layer.ntokens);
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "sumslices", "nslices", 8);
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "sumslices", "src_slice_stride", 2048);
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "sumslices", "src_token_stride", 16384);
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "sumslices", "dst_token_stride", 2048);
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "sumslices", "router_source_stride",
        layer.router_source_stride);
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "sumslices", "router_source_offset",
        layer.router_source_offset);
    ggml_backend_hrx_add_tensor_facts(
        &out_request->problem, "src0", layer.selected);
    ggml_backend_hrx_add_tensor_facts(
        &out_request->problem, "dst", layer.dst);
    ggml_backend_hrx_add_tensor_facts(
        &out_request->problem, "router_source", layer.router_source);
    // The mandatory prepass reads binding 2 and appends the stable snapshot as
    // binding 3. Binding 2 remains present but is intentionally unused by the
    // reducer because it may share its owner with binding 1.
    out_request->tensors = {
        layer.selected,
        layer.dst,
        layer.router_source,
    };
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_current_moe_router_tail_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context) {
        return false;
    }
    const auto * layer = ggml_backend_hrx_find_moe_router_tail_layer(
        device_context->current_moe_router_tail_plan, node);
    return layer &&
           ggml_backend_hrx_make_moe_router_tail_request(
               device_context, *layer, out_request);
}

static bool ggml_backend_hrx_shared_expert_terminal_route_is_exact(
        const ggml_backend_hrx_catalog_route * route) {
    if (!route ||
        route->id !=
            "mul_mat_q8_0_f32_wmmai8_shared_expert_terminal_epilogue" ||
        route->family != "mul_mat_q8_0_f32_tiled" ||
        route->op != "MUL_MAT" ||
        route->source_id !=
            "mul_mat_q8_0_wmmai8_shared_expert_terminal_epilogue" ||
        route->artifact_id !=
            "mul_mat_q8_0_wmmai8_shared_expert_terminal_epilogue_loombc" ||
        route->root_symbol !=
            "@hrx2_mul_mat_q8_0_f32_wmmai8_"
            "shared_expert_terminal_epilogue" ||
        route->export_name !=
            "hrx2_mul_mat_q8_0_f32_wmmai8_"
            "shared_expert_terminal_epilogue" ||
        route->binding_count != 8 ||
        route->parameter_count != 8 ||
        route->constant_byte_length != 0 ||
        route->prepasses.size() != 1) {
        return false;
    }
    const auto fusion = route->supports.find("fusion");
    const auto & prepass = route->prepasses[0];
    return fusion != route->supports.end() &&
           fusion->second ==
               "MUL_MAT_SHARED_EXPERT_SIGMOID_MUL_ADD_EPILOGUE" &&
           prepass.enabled &&
           prepass.artifact_id == "quant_act_q8_loombc" &&
           prepass.root_symbol == "@hrx2_quant_act_q8" &&
           prepass.export_name == "hrx2_quant_act_q8" &&
           prepass.src_index == 1 &&
           prepass.dst_index == -1;
}

static const ggml_backend_hrx_shared_expert_terminal_layer_plan *
ggml_backend_hrx_find_shared_expert_terminal_layer(
        const ggml_backend_hrx_shared_expert_terminal_graph_plan * plan,
        const ggml_tensor * terminal) {
    if (!plan || !plan->ready || !terminal) {
        return nullptr;
    }
    for (const auto & layer : plan->layers) {
        if (layer.terminal == terminal) {
            return &layer;
        }
    }
    return nullptr;
}

static bool ggml_backend_hrx_make_shared_expert_terminal_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_backend_hrx_shared_expert_terminal_layer_plan & layer,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !layer.down ||
        !layer.raw_gate || !layer.addend || !layer.terminal ||
        !layer.down->src[0] || !layer.down->src[1]) {
        return false;
    }
    ggml_backend_hrx_dispatch_request request = {};
    if (!ggml_backend_hrx_make_mul_mat_problem(
            device_context, layer.down, &request.problem)) {
        return false;
    }
    request.problem.supports["fusion"] =
        "MUL_MAT_SHARED_EXPERT_SIGMOID_MUL_ADD_EPILOGUE";
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "raw_gate", layer.raw_gate);
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "addend", layer.addend);
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "terminal_dst", layer.terminal);
    request.tensors = {
        layer.down->src[0],
        layer.down->src[1],
        layer.raw_gate,
        layer.addend,
        layer.terminal,
    };
    request.constants.clear();

    ggml_backend_hrx_catalog_problem problem = request.problem;
    ggml_backend_hrx_add_tensor_overlap_facts(
        &problem, request.tensors);
    const auto * route =
        device_context->reg_context &&
                device_context->reg_context->catalog
            ? ggml_backend_hrx_catalog_find_route(
                  *device_context->reg_context->catalog, problem)
            : nullptr;
    if (!ggml_backend_hrx_shared_expert_terminal_route_is_exact(route)) {
        return false;
    }
    *out_request = std::move(request);
    return true;
}

static bool ggml_backend_hrx_make_current_shared_expert_terminal_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context) {
        return false;
    }
    const auto * layer =
        ggml_backend_hrx_find_shared_expert_terminal_layer(
            device_context->current_shared_expert_terminal_plan, node);
    return layer &&
           ggml_backend_hrx_make_shared_expert_terminal_request(
               device_context, *layer, out_request);
}

static bool ggml_backend_hrx_make_add_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_ADD) {
        return false;
    }
    if (ggml_backend_hrx_make_current_shared_expert_terminal_request(
            device_context, node, out_request)) {
        return true;
    }
    if (ggml_backend_hrx_make_current_moe_router_tail_request(
            device_context, node, out_request)) {
        return true;
    }
    ggml_backend_hrx_binary_f32_layout layout;
    if (!ggml_backend_hrx_describe_binary_f32_layout(node, &layout)) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "ADD";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(node->src[1]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", layout.key},
    };
    ggml_backend_hrx_set_pointwise_shape(&out_request->problem, node, layout.rhs_row_broadcast);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", node->src[1]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node->src[1], node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_mul_mat_id_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request);

static constexpr size_t GGML_HRX_MMID_TABLE_BYTES = 8 * 1024 * 1024;
static constexpr size_t GGML_HRX_MMID_QACT_OFFSET =
    GGML_HRX_MMID_TABLE_BYTES + 16 * 1024;
static constexpr size_t GGML_HRX_MMID_QACT_END =
    GGML_HRX_MMID_QACT_OFFSET + 512 * 8 * 640;

static const ggml_backend_hrx_terminal_qact_layer_plan *
ggml_backend_hrx_find_terminal_qact_layer(
        const ggml_backend_hrx_terminal_qact_graph_plan * plan,
        const ggml_tensor * node) {
    if (!plan || !plan->ready || !node) {
        return nullptr;
    }
    for (const auto & layer : plan->layers) {
        if (layer.glu == node || layer.down == node) {
            return &layer;
        }
    }
    return nullptr;
}

static bool ggml_backend_hrx_make_rms_norm_mul_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_MUL ||
        !node->src[0] || !node->src[1] || node->src[0]->op != GGML_OP_RMS_NORM ||
        !node->src[0]->src[0] || node->src[0]->type != GGML_TYPE_F32 ||
        node->src[0]->src[0]->type != GGML_TYPE_F32 || node->src[1]->type != GGML_TYPE_F32 ||
        node->type != GGML_TYPE_F32 ||
        !ggml_backend_hrx_is_f32_dense(node->src[0]->src[0]) ||
        !ggml_backend_hrx_is_f32_dense(node->src[1]) ||
        !ggml_backend_hrx_is_f32_dense(node)) {
        return false;
    }
    const ggml_tensor * src = node->src[0]->src[0];
    const ggml_tensor * weight = node->src[1];
    const int64_t ncols = node->ne[0];
    const int64_t nrows = ggml_backend_hrx_tensor_row_count(node);
    if (src->ne[0] != ncols || ggml_backend_hrx_tensor_row_count(src) != nrows ||
        weight->ne[0] != ncols || ggml_backend_hrx_tensor_row_count(weight) != 1) {
        return false;
    }

    out_request->problem = {};
    out_request->problem.op = "MUL";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(src->type)},
        {"weight_type", ggml_backend_hrx_catalog_type_name(weight->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous_src_row_broadcast_weight"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rms_norm_mul", "ncols", ncols);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rms_norm_mul", "nrows", nrows);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", src);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", weight);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {src, weight, node};
    out_request->constants.clear();
    float eps = 0.0f;
    std::memcpy(&eps, node->src[0]->op_params, sizeof(float));
    ggml_backend_hrx_append_constant(&out_request->constants, eps);
    return true;
}

static bool ggml_backend_hrx_make_add_rms_norm_mul_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_MUL ||
        !node->src[0] || !node->src[1] || node->src[0]->op != GGML_OP_RMS_NORM ||
        !node->src[0]->src[0] || node->src[0]->src[0]->op != GGML_OP_ADD ||
        !node->src[0]->src[0]->src[0] || !node->src[0]->src[0]->src[1] ||
        node->src[0]->type != GGML_TYPE_F32 ||
        node->src[0]->src[0]->type != GGML_TYPE_F32 ||
        node->src[0]->src[0]->src[0]->type != GGML_TYPE_F32 ||
        node->src[0]->src[0]->src[1]->type != GGML_TYPE_F32 ||
        node->src[1]->type != GGML_TYPE_F32 ||
        node->type != GGML_TYPE_F32 ||
        !ggml_backend_hrx_is_f32_dense(node->src[0]->src[0]->src[0]) ||
        !ggml_backend_hrx_is_f32_dense(node->src[0]->src[0]->src[1]) ||
        !ggml_backend_hrx_is_f32_dense(node->src[0]->src[0]) ||
        !ggml_backend_hrx_is_f32_dense(node->src[1]) ||
        !ggml_backend_hrx_is_f32_dense(node)) {
        return false;
    }
    const ggml_tensor * add = node->src[0]->src[0];
    const ggml_tensor * src0 = add->src[0];
    const ggml_tensor * src1 = add->src[1];
    const ggml_tensor * weight = node->src[1];
    const int64_t ncols = node->ne[0];
    const int64_t nrows = ggml_backend_hrx_tensor_row_count(node);
    if (src0->ne[0] != ncols || ggml_backend_hrx_tensor_row_count(src0) != nrows ||
        src1->ne[0] != ncols || ggml_backend_hrx_tensor_row_count(src1) != nrows ||
        weight->ne[0] != ncols || ggml_backend_hrx_tensor_row_count(weight) != 1 ||
        add->ne[0] != ncols || ggml_backend_hrx_tensor_row_count(add) != nrows) {
        return false;
    }

    out_request->problem = {};
    out_request->problem.op = "MUL";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src_type", ggml_backend_hrx_catalog_type_name(src0->type)},
        {"weight_type", ggml_backend_hrx_catalog_type_name(weight->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous_same_shape_add"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "add_rms_norm_mul", "ncols", ncols);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "add_rms_norm_mul", "nrows", nrows);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", src0);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", src1);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "add_dst", add);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "weight", weight);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {src0, src1, add, weight, node};
    out_request->constants.clear();
    float eps = 0.0f;
    std::memcpy(&eps, node->src[0]->op_params, sizeof(float));
    ggml_backend_hrx_append_constant(&out_request->constants, eps);
    return true;
}

static bool ggml_backend_hrx_make_mul_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_MUL) {
        return false;
    }
    ggml_backend_hrx_dispatch_request decode_gdn_q8_fused;
    if (ggml_backend_hrx_make_decode_gdn_q8_silu_mul_request(
            device_context, node, &decode_gdn_q8_fused)) {
        *out_request = std::move(decode_gdn_q8_fused);
        return true;
    }
    if (ggml_backend_hrx_make_current_gdn_rms_side_request(
            device_context, node, out_request)) {
        return true;
    }
    ggml_backend_hrx_dispatch_request gdn_q8_fused;
    if (ggml_backend_hrx_make_gdn_q8_silu_mul_request(
            device_context, node, &gdn_q8_fused)) {
        *out_request = std::move(gdn_q8_fused);
        return true;
    }
    ggml_backend_hrx_dispatch_request fa_gate_fused;
    if (ggml_backend_hrx_make_fa_gate_epilogue_request(
            device_context, node, &fa_gate_fused)) {
        *out_request = std::move(fa_gate_fused);
        return true;
    }
    ggml_backend_hrx_dispatch_request fused_request;
    const bool have_catalog = device_context->reg_context && device_context->reg_context->catalog;
    if (ggml_backend_hrx_make_add_rms_norm_mul_request(device_context, node, &fused_request) &&
        have_catalog &&
        ggml_backend_hrx_catalog_find_route(*device_context->reg_context->catalog, fused_request.problem)) {
        *out_request = std::move(fused_request);
        return true;
    }
    if (ggml_backend_hrx_make_rms_norm_mul_request(device_context, node, &fused_request) &&
        have_catalog &&
        ggml_backend_hrx_catalog_find_route(*device_context->reg_context->catalog, fused_request.problem)) {
        *out_request = std::move(fused_request);
        return true;
    }
    ggml_backend_hrx_binary_f32_layout layout;
    if (!ggml_backend_hrx_describe_binary_f32_layout(node, &layout)) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "MUL";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(node->src[1]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", layout.key},
    };
    ggml_backend_hrx_set_pointwise_shape(&out_request->problem, node, layout.rhs_row_broadcast);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", node->src[1]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node->src[1], node};
    out_request->constants.clear();
    return true;
}

struct ggml_backend_hrx_topk_moe_softmax_norm_match {
    const ggml_tensor * logits          = nullptr;
    const ggml_tensor * softmax         = nullptr;
    const ggml_tensor * probs_reshape   = nullptr;
    const ggml_tensor * argsort         = nullptr;
    const ggml_tensor * ids_view        = nullptr;
    const ggml_tensor * get_rows        = nullptr;
    const ggml_tensor * weights_reshape = nullptr;
    const ggml_tensor * sum_rows        = nullptr;
    const ggml_tensor * clamp           = nullptr;
    const ggml_tensor * div             = nullptr;
};

static bool ggml_backend_hrx_has_exact_shape(
        const ggml_tensor * tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2 = 1,
        int64_t ne3 = 1) {
    return tensor &&
           tensor->ne[0] == ne0 &&
           tensor->ne[1] == ne1 &&
           tensor->ne[2] == ne2 &&
           tensor->ne[3] == ne3;
}

// Qwen3.6's PP512 router is:
//
//   SOFT_MAX(logits) -----------------------> ARGSORT -> VIEW(top 8 ids)
//        |                                       |              |
//        +-> RESHAPE -> GET_ROWS <---------------+              +-> expert GEMMs
//                           |
//                       RESHAPE ----+-> SUM_ROWS -> CLAMP --+
//                                  +-----------------------> DIV
//
// The fused route is deliberately exact. In particular, the ids VIEW is not
// dense: each token starts 256 i32s after the previous one because it views the
// full ARGSORT destination. Bind the full destination so the kernel can retain
// that row stride and the later expert GEMMs can keep consuming the view.
static bool ggml_backend_hrx_match_topk_moe_early_softmax_norm(
        const ggml_tensor * node,
        ggml_backend_hrx_topk_moe_softmax_norm_match * out_match) {
    constexpr int64_t nexperts = 256;
    constexpr int64_t nselected = 8;
    constexpr float clamp_min = 6.103515625e-5f;

    if (!node || !out_match || node->op != GGML_OP_DIV ||
        !node->src[0] || !node->src[1]) {
        return false;
    }

    ggml_backend_hrx_topk_moe_softmax_norm_match match = {};
    match.div = node;
    match.weights_reshape = node->src[0];
    match.clamp = node->src[1];
    if (match.weights_reshape->op != GGML_OP_RESHAPE ||
        !match.weights_reshape->src[0] ||
        match.clamp->op != GGML_OP_CLAMP ||
        !match.clamp->src[0]) {
        return false;
    }

    match.get_rows = match.weights_reshape->src[0];
    match.sum_rows = match.clamp->src[0];
    if (match.get_rows->op != GGML_OP_GET_ROWS ||
        !match.get_rows->src[0] || !match.get_rows->src[1] ||
        match.sum_rows->op != GGML_OP_SUM_ROWS ||
        match.sum_rows->src[0] != match.weights_reshape) {
        return false;
    }

    match.probs_reshape = match.get_rows->src[0];
    match.ids_view = match.get_rows->src[1];
    if (match.probs_reshape->op != GGML_OP_RESHAPE ||
        !match.probs_reshape->src[0] ||
        match.ids_view->op != GGML_OP_VIEW ||
        !match.ids_view->src[0] ||
        match.ids_view->view_offs != 0) {
        return false;
    }

    match.softmax = match.probs_reshape->src[0];
    match.argsort = match.ids_view->src[0];
    if (match.softmax->op != GGML_OP_SOFT_MAX ||
        !match.softmax->src[0] ||
        match.softmax->src[1] != nullptr ||
        match.softmax->src[2] != nullptr ||
        match.argsort->op != GGML_OP_ARGSORT ||
        match.argsort->src[0] != match.softmax ||
        match.ids_view->view_src != match.argsort ||
        ggml_get_op_params_i32(match.argsort, 0) !=
            static_cast<int32_t>(GGML_SORT_ORDER_DESC)) {
        return false;
    }
    match.logits = match.softmax->src[0];
    const int64_t ntokens = match.logits->ne[1];
    if (ntokens != 1 && ntokens != 512) {
        return false;
    }

    float softmax_scale = 0.0f;
    float softmax_max_bias = 0.0f;
    float actual_clamp_min = 0.0f;
    float actual_clamp_max = 0.0f;
    std::memcpy(&softmax_scale, match.softmax->op_params, sizeof(float));
    std::memcpy(
        &softmax_max_bias,
        reinterpret_cast<const uint8_t *>(match.softmax->op_params) + sizeof(float),
        sizeof(float));
    std::memcpy(&actual_clamp_min, match.clamp->op_params, sizeof(float));
    std::memcpy(
        &actual_clamp_max,
        reinterpret_cast<const uint8_t *>(match.clamp->op_params) + sizeof(float),
        sizeof(float));
    if (softmax_scale != 1.0f || softmax_max_bias != 0.0f ||
        actual_clamp_min != clamp_min ||
        !std::isinf(actual_clamp_max) || actual_clamp_max < 0.0f) {
        return false;
    }

    // Every materialized f32 value is dense. The ids view is intentionally
    // strided, but both it and its full backing have the exact row strides used
    // by ggml_argsort_top_k.
    if (!ggml_backend_hrx_has_exact_shape(match.logits, nexperts, ntokens) ||
        !ggml_backend_hrx_has_exact_shape(match.softmax, nexperts, ntokens) ||
        !ggml_backend_hrx_has_exact_shape(match.probs_reshape, 1, nexperts, ntokens) ||
        !ggml_backend_hrx_has_exact_shape(match.argsort, nexperts, ntokens) ||
        !ggml_backend_hrx_has_exact_shape(match.ids_view, nselected, ntokens) ||
        !ggml_backend_hrx_has_exact_shape(match.get_rows, 1, nselected, ntokens) ||
        !ggml_backend_hrx_has_exact_shape(match.weights_reshape, nselected, ntokens) ||
        !ggml_backend_hrx_has_exact_shape(match.sum_rows, 1, ntokens) ||
        !ggml_backend_hrx_has_exact_shape(match.clamp, 1, ntokens) ||
        !ggml_backend_hrx_has_exact_shape(match.div, nselected, ntokens) ||
        !ggml_backend_hrx_is_f32_dense(match.logits) ||
        !ggml_backend_hrx_is_f32_dense(match.softmax) ||
        !ggml_backend_hrx_is_f32_dense(match.probs_reshape) ||
        match.argsort->type != GGML_TYPE_I32 ||
        !ggml_is_contiguous(match.argsort) ||
        match.ids_view->type != GGML_TYPE_I32 ||
        !ggml_are_same_stride(match.ids_view, match.argsort) ||
        !ggml_backend_hrx_is_f32_dense(match.get_rows) ||
        !ggml_backend_hrx_is_f32_dense(match.weights_reshape) ||
        !ggml_backend_hrx_is_f32_dense(match.sum_rows) ||
        !ggml_backend_hrx_is_f32_dense(match.clamp) ||
        !ggml_backend_hrx_is_f32_dense(match.div)) {
        return false;
    }

    *out_match = match;
    return true;
}

static bool ggml_backend_hrx_topk_moe_graph_match_is_safe(
        const ggml_cgraph * cgraph,
        const ggml_backend_hrx_topk_moe_softmax_norm_match & match);

static void ggml_backend_hrx_set_topk_moe_shape(
        ggml_backend_hrx_catalog_problem * problem,
        const ggml_backend_hrx_topk_moe_softmax_norm_match & match) {
    ggml_backend_hrx_set_shape_alias(
        problem, "topk_moe", "nexperts", match.logits->ne[0]);
    ggml_backend_hrx_set_shape_alias(
        problem, "topk_moe", "ntokens", match.logits->ne[1]);
    ggml_backend_hrx_set_shape_alias(
        problem, "topk_moe", "nselected", match.ids_view->ne[0]);
    ggml_backend_hrx_set_shape_alias(
        problem, "topk_moe", "logits_token_stride",
        static_cast<int64_t>(match.logits->nb[1] / sizeof(float)));
    ggml_backend_hrx_set_shape_alias(
        problem, "topk_moe", "ids_token_stride",
        static_cast<int64_t>(match.ids_view->nb[1] / sizeof(int32_t)));
    ggml_backend_hrx_set_shape_alias(
        problem, "topk_moe", "weights_token_stride",
        static_cast<int64_t>(match.div->nb[1] / sizeof(float)));
}

static bool ggml_backend_hrx_make_topk_moe_stage_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_backend_hrx_topk_moe_softmax_norm_match & match,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request) {
        return false;
    }
    ggml_backend_hrx_dispatch_request request = {};
    request.problem.op = "ARGSORT";
    request.problem.target_key = device_context->architecture;
    request.problem.supports = {
        {"src0_type", "F32"},
        {"dst_type", "I32"},
        {"layout", "topk_moe_softmax_norm_stage"},
        {"fusion", "TOPK_MOE_EARLY_SOFTMAX_NORM_STAGE"},
    };
    ggml_backend_hrx_set_topk_moe_shape(&request.problem, match);
    ggml_backend_hrx_add_tensor_facts(&request.problem, "src0", match.logits);
    ggml_backend_hrx_add_tensor_facts(&request.problem, "dst", match.argsort);
    request.tensors = {match.logits, match.argsort};
    request.constants.clear();
    *out_request = std::move(request);
    return true;
}

static bool ggml_backend_hrx_make_topk_moe_copy_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_backend_hrx_topk_moe_softmax_norm_match & match,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request) {
        return false;
    }
    ggml_backend_hrx_dispatch_request request = {};
    request.problem.op = "DIV";
    request.problem.target_key = device_context->architecture;
    request.problem.supports = {
        {"src0_type", "I32"},
        {"dst_type", "F32"},
        {"layout", "topk_moe_weights_copy"},
        {"fusion", "TOPK_MOE_WEIGHTS_COPY"},
    };
    ggml_backend_hrx_set_topk_moe_shape(&request.problem, match);
    ggml_backend_hrx_add_tensor_facts(&request.problem, "src0", match.argsort);
    ggml_backend_hrx_add_tensor_facts(&request.problem, "dst", match.div);
    request.tensors = {match.argsort, match.div};
    request.constants.clear();
    *out_request = std::move(request);
    return true;
}

static bool ggml_backend_hrx_make_topk_moe_decode_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_backend_hrx_topk_moe_softmax_norm_match & match,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || match.logits->ne[1] != 1) {
        return false;
    }
    ggml_backend_hrx_dispatch_request request = {};
    request.problem.op = "DIV";
    request.problem.target_key = device_context->architecture;
    request.problem.supports = {
        {"src0_type", "F32"},
        {"ids_type", "I32"},
        {"dst_type", "F32"},
        {"layout", "topk_moe_softmax_norm_decode"},
        {"fusion", "TOPK_MOE_SOFTMAX_NORM_DECODE"},
    };
    ggml_backend_hrx_set_topk_moe_shape(&request.problem, match);
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "src0", match.logits);
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "ids", match.argsort);
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "dst", match.div);
    request.tensors = {match.logits, match.argsort, match.div};
    request.constants.clear();
    *out_request = std::move(request);
    return true;
}

static bool ggml_backend_hrx_make_available_topk_moe_requests(
        ggml_backend_hrx_device_context * device_context,
        const ggml_backend_hrx_topk_moe_softmax_norm_match & match,
        ggml_backend_hrx_dispatch_request * out_stage,
        ggml_backend_hrx_dispatch_request * out_copy) {
    if (match.logits->ne[1] != 512) {
        return false;
    }
    ggml_backend_hrx_dispatch_request stage = {};
    ggml_backend_hrx_dispatch_request copy = {};
    if (!ggml_backend_hrx_make_topk_moe_stage_request(
            device_context, match, &stage) ||
        !ggml_backend_hrx_make_topk_moe_copy_request(
            device_context, match, &copy) ||
        !ggml_backend_hrx_request_matches_loaded_route(
            device_context, stage, "topk_moe_softmax_norm_stage_f32") ||
        !ggml_backend_hrx_request_matches_loaded_route(
            device_context, copy, "topk_moe_weights_copy_f32")) {
        return false;
    }
    if (out_stage) {
        *out_stage = std::move(stage);
    }
    if (out_copy) {
        *out_copy = std::move(copy);
    }
    return true;
}

static bool ggml_backend_hrx_make_available_topk_moe_decode_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_backend_hrx_topk_moe_softmax_norm_match & match,
        ggml_backend_hrx_dispatch_request * out_decode) {
    ggml_backend_hrx_dispatch_request decode = {};
    if (!ggml_backend_hrx_make_topk_moe_decode_request(
            device_context, match, &decode) ||
        !ggml_backend_hrx_request_matches_loaded_route(
            device_context, decode,
            "topk_moe_softmax_norm_decode_f32")) {
        return false;
    }
    if (out_decode) {
        *out_decode = std::move(decode);
    }
    return true;
}

// The stage dispatch sits at ARGSORT, but the exact fusion can only be proven
// from its downstream terminal DIV. During supports_op there is no active graph,
// so leave ARGSORT and DIV on their ordinary request paths. During graph compute,
// find the complete private chain and require both routes before selecting either
// half of the two-dispatch implementation.
static bool ggml_backend_hrx_find_current_topk_moe_match(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_topk_moe_softmax_norm_match * out_match) {
    if (!device_context || !node || !out_match || !device_context->current_graph) {
        return false;
    }
    const ggml_cgraph * cgraph = device_context->current_graph;
    if (node->op == GGML_OP_DIV) {
        ggml_backend_hrx_topk_moe_softmax_norm_match match = {};
        if (!ggml_backend_hrx_match_topk_moe_early_softmax_norm(node, &match) ||
            !ggml_backend_hrx_topk_moe_graph_match_is_safe(cgraph, match)) {
            return false;
        }
        *out_match = match;
        return true;
    }
    if (node->op != GGML_OP_ARGSORT) {
        return false;
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * terminal = cgraph->nodes[i];
        if (!terminal || terminal->op != GGML_OP_DIV) {
            continue;
        }
        ggml_backend_hrx_topk_moe_softmax_norm_match match = {};
        if (ggml_backend_hrx_match_topk_moe_early_softmax_norm(terminal, &match) &&
            match.argsort == node &&
            ggml_backend_hrx_topk_moe_graph_match_is_safe(cgraph, match)) {
            *out_match = match;
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx_make_div_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_DIV) {
        return false;
    }
    ggml_backend_hrx_topk_moe_softmax_norm_match match = {};
    if (ggml_backend_hrx_find_current_topk_moe_match(
            device_context, node, &match)) {
        ggml_backend_hrx_dispatch_request fused = {};
        const bool available = match.logits->ne[1] == 1 ?
            ggml_backend_hrx_make_available_topk_moe_decode_request(
                device_context, match, &fused) :
            ggml_backend_hrx_make_available_topk_moe_requests(
                device_context, match, nullptr, &fused);
        if (available) {
            *out_request = std::move(fused);
            return true;
        }
    }
    ggml_backend_hrx_binary_f32_layout layout;
    if (!ggml_backend_hrx_describe_binary_f32_layout(node, &layout)) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "DIV";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(node->src[1]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", layout.key},
    };
    ggml_backend_hrx_set_pointwise_shape(&out_request->problem, node, layout.rhs_row_broadcast);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", node->src[1]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node->src[1], node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_mul_mat_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!out_request || !ggml_backend_hrx_make_mul_mat_problem(device_context, node, &out_request->problem)) {
        return false;
    }
    out_request->tensors = {node->src[0], node->src[1], node};
    out_request->constants.clear();
    return true;
}


static bool ggml_backend_hrx_make_mul_mat_id_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_MUL_MAT_ID ||
        !node->src[0] || !node->src[1] || !node->src[2]) { return false; }
    if (node->src[1]->type != GGML_TYPE_F32 || node->src[2]->type != GGML_TYPE_I32 ||
        node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node->src[0]) ||
        !ggml_is_contiguous(node->src[1]) ||
        node->src[2]->nb[0] != ggml_type_size(node->src[2]->type) ||
        !ggml_is_contiguous(node)) {
        return false;
    }
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const ggml_tensor * ids = node->src[2];
    const char * layout = nullptr;
    const bool q5_down_group4 =
        ggml_backend_hrx_is_q5_down_group4_tensor(src0);
    const auto q5_down_group4_range =
        ggml_backend_hrx_tensor_q5_down_group4_range_state(src0);
    const bool q5_down_group4_capability_probe =
        q5_down_group4 &&
        ggml_backend_hrx_is_q5_down_group4_zero_size_capability_probe(
            src0);
    if (!ggml_hrx_q5_down_group4::request_storage_allowed(
            q5_down_group4,
            q5_down_group4_range,
            q5_down_group4_capability_probe)) {
        // A renamed/view alias must never route a packed range through a
        // canonical consumer. Only the loader's exact direct zero-byte
        // capability probe is exempt from the completed-upload requirement.
        return false;
    }
    if (src0->type == GGML_TYPE_Q4_K) {
        layout = "q4_k_expert_planes_rhs_token_or_selected_rows";
    } else if (src0->type == GGML_TYPE_Q5_K) {
        layout = q5_down_group4
            ? "q5_k_expert_group4_down_rhs_token_or_selected_rows"
            : "q5_k_expert_planes_rhs_token_or_selected_rows";
    } else if (src0->type == GGML_TYPE_Q6_K) {
        layout = "q6_k_expert_planes_rhs_token_or_selected_rows";
    } else {
        return false;
    }

    out_request->problem = {};
    out_request->problem.op = "MUL_MAT_ID";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(src0->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(src1->type)},
        {"src2_type", ggml_backend_hrx_catalog_type_name(ids->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", layout},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "mul_mat_id", "k", src0->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "mul_mat_id", "rows", src0->ne[1]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "mul_mat_id", "nexperts", src0->ne[2]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "mul_mat_id", "nselected", ids->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "mul_mat_id", "ntokens", ids->ne[1]);
    out_request->problem.shape["k"] = src0->ne[0];
    out_request->problem.shape["rows"] = src0->ne[1];
    out_request->problem.shape["cols"] = ids->ne[0];
    out_request->problem.shape["nrows"] = ids->ne[1];
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "mul_mat_id", "src1_selected_stride",
        src1->ne[1] == 1 ? 0 : static_cast<int64_t>(src1->nb[1] / sizeof(float)));
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "mul_mat_id", "src1_token_stride",
        static_cast<int64_t>(src1->nb[2] / sizeof(float)));
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "mul_mat_id", "idx_token_stride",
        static_cast<int64_t>(ids->nb[1] / sizeof(int32_t)));
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "mul_mat_id", "dst_token_stride",
        static_cast<int64_t>(node->nb[2] / sizeof(float)));
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", src0);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", src1);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src2", ids);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {src0, src1, ids, node};
    out_request->constants.clear();
    const auto * terminal_qact_layer =
        ggml_backend_hrx_find_terminal_qact_layer(
            device_context->current_terminal_qact_plan, node);
    if (terminal_qact_layer &&
        terminal_qact_layer->down == node) {
        out_request->problem.supports["fusion"] =
            "MUL_MAT_ID_Q5_DOWN_TERMINAL_QACT";
    }
    return true;
}

// hrx2_scale_f32 is flat and grid-strided: it derives one linear index from
// ncols*nrows and never uses the two separately, so how a contiguous tensor is
// split between them is free. Routes still gate on ncols, and the declared
// range of @hrx2.shape.pointwise.ncols stops at 65536, so a flat tensor longer
// than that -- the SSM state cache slice reshaped to 524288 elements -- would
// find no route despite every kernel being able to run it. Move the excess into
// rows. Only factors of two are taken, which is bounded work and covers the
// shapes that occur; anything left over is passed through unchanged and simply
// finds no route, as before.
static void ggml_backend_hrx_balance_flat_pointwise_shape(int64_t * ncols, int64_t * nrows) {
    const int64_t ncols_max = 65536;
    while (*ncols > ncols_max && (*ncols % 2) == 0) {
        *ncols /= 2;
        *nrows *= 2;
    }
}

static bool ggml_backend_hrx_make_scale_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_SCALE || !node->src[0] ||
        !ggml_backend_hrx_is_f32_dense(node->src[0]) ||
        !ggml_backend_hrx_is_f32_dense(node)) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "SCALE";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous"},
    };
    int64_t ncols = node->ne[0];
    int64_t nrows = ggml_backend_hrx_tensor_row_count(node);
    if (ggml_is_contiguous(node) && ggml_is_contiguous(node->src[0])) {
        ggml_backend_hrx_balance_flat_pointwise_shape(&ncols, &nrows);
    }
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "pointwise", "ncols", ncols);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "pointwise", "nrows", nrows);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node};
    out_request->constants.clear();
    float scale = 0.0f;
    float bias = 0.0f;
    std::memcpy(&scale, node->op_params, sizeof(float));
    std::memcpy(&bias, reinterpret_cast<const uint8_t *>(node->op_params) + sizeof(float), sizeof(float));
    ggml_backend_hrx_append_constant(&out_request->constants, scale);
    ggml_backend_hrx_append_constant(&out_request->constants, bias);
    return true;
}

static bool ggml_backend_hrx_make_clamp_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_CLAMP || !node->src[0] ||
        !ggml_backend_hrx_is_f32_dense(node->src[0]) ||
        !ggml_backend_hrx_is_f32_dense(node)) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "CLAMP";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "pointwise", "ncols", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "pointwise", "nrows", ggml_backend_hrx_tensor_row_count(node));
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node};
    out_request->constants.clear();
    float min_value = 0.0f;
    float max_value = 0.0f;
    std::memcpy(&min_value, node->op_params, sizeof(float));
    std::memcpy(&max_value, reinterpret_cast<const uint8_t *>(node->op_params) + sizeof(float), sizeof(float));
    ggml_backend_hrx_append_constant(&out_request->constants, min_value);
    ggml_backend_hrx_append_constant(&out_request->constants, max_value);
    return true;
}

// SILU / SIGMOID / SOFTPLUS. The Qwen3.6 Gated-DeltaNet layers use all three
// (conv-output gating, beta and alpha gating, shared-expert gate) and none had
// a request builder, so every occurrence fell back off the device. Each maps to
// a single Loom scalar primitive, so one flat elementwise request covers them.
static bool ggml_backend_hrx_make_unary_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_UNARY || !node->src[0] ||
        !ggml_backend_hrx_is_f32_dense(node->src[0]) ||
        !ggml_backend_hrx_is_f32_dense(node)) {
        return false;
    }
    const char * unary_name = nullptr;
    switch (ggml_get_unary_op(node)) {
        case GGML_UNARY_OP_SILU:     unary_name = "SILU";     break;
        case GGML_UNARY_OP_SIGMOID:  unary_name = "SIGMOID";  break;
        case GGML_UNARY_OP_SOFTPLUS: unary_name = "SOFTPLUS"; break;
        case GGML_UNARY_OP_EXP:      unary_name = "EXP";      break;
        case GGML_UNARY_OP_NEG:      unary_name = "NEG";      break;
        default: return false;
    }
    out_request->problem = {};
    out_request->problem.op = ggml_op_desc(node);
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"unary_op", unary_name},
        {"layout", "contiguous"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "pointwise", "ncols", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "pointwise", "nrows", ggml_backend_hrx_tensor_row_count(node));
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node};
    out_request->constants.clear();
    return true;
}

// Fused Gated DeltaNet. Claiming this is what stops llama.cpp decomposing the
// 30 SSM layers into a chunked delta rule (which would need SOLVE_TRI, CUMSUM,
// DIAG, TRI and batched 4D MUL_MAT, and tens of thousands of dispatches).
// Scalar-gate form only; the kernel is specialised to S_v == 128 with wave32.
static bool ggml_backend_hrx_make_gated_delta_net_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_GATED_DELTA_NET) {
        return false;
    }
    for (int i = 0; i < 6; ++i) {
        if (!node->src[i] || node->src[i]->type != GGML_TYPE_F32) {
            return false;
        }
    }
    const ggml_tensor * q     = node->src[0];
    const ggml_tensor * k     = node->src[1];
    const ggml_tensor * v     = node->src[2];
    const ggml_tensor * g     = node->src[3];
    const ggml_tensor * beta  = node->src[4];
    const ggml_tensor * state = node->src[5];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    // Only the scalar-gate (non-KDA) form and the S_v=128 specialisation.
    if (g->ne[0] != 1 || S_v != 128) {
        return false;
    }
    // The kernel appends one state snapshot, the final one. K > 1 asks for the
    // last K states at token boundaries, which it does not write.
    if (ggml_get_op_params_i32(node, 0) != 1) {
        return false;
    }
    if (node->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_are_same_stride(q, k) || q->ne[1] != k->ne[1]) {
        return false;
    }
    for (const ggml_tensor * t : {q, v, beta}) {
        for (int i = 1; i < 4; ++i) {
            if (t->nb[i] % sizeof(float) != 0) {
                return false;
            }
        }
    }

    out_request->problem = {};
    out_request->problem.op = "GATED_DELTA_NET";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(q->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"gate", "scalar"},
        {"layout", "state_transposed_column_per_wave"},
    };
    auto put = [&](const char * key, int64_t value) {
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "gdn", key, value);
    };
    put("s_v", S_v);
    put("n_heads", H);
    put("n_tokens", v->ne[2]);
    put("n_seqs", v->ne[3]);
    put("sq1", static_cast<int64_t>(q->nb[1] / sizeof(float)));
    put("sq2", static_cast<int64_t>(q->nb[2] / sizeof(float)));
    put("sq3", static_cast<int64_t>(q->nb[3] / sizeof(float)));
    put("sv1", static_cast<int64_t>(v->nb[1] / sizeof(float)));
    put("sv2", static_cast<int64_t>(v->nb[2] / sizeof(float)));
    put("sv3", static_cast<int64_t>(v->nb[3] / sizeof(float)));
    put("sb1", static_cast<int64_t>(beta->nb[1] / sizeof(float)));
    put("sb2", static_cast<int64_t>(beta->nb[2] / sizeof(float)));
    put("sb3", static_cast<int64_t>(beta->nb[3] / sizeof(float)));
    put("neqk1", q->ne[1]);
    put("rq3", v->ne[3] / q->ne[3]);
    out_request->problem.shape["ncols"] = S_v;
    out_request->problem.shape["nrows"] = H * v->ne[2] * v->ne[3];

    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", q);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {q, k, v, g, beta, state, node};
    out_request->constants.clear();
    const float scale = 1.0f / sqrtf(static_cast<float>(S_v));
    ggml_backend_hrx_append_constant(&out_request->constants, scale);
    return true;
}

static const ggml_backend_hrx_gdn_qk_scale_layer *
ggml_backend_hrx_find_gdn_qk_scale_layer(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node) {
    if (!device_context || !node ||
        !device_context->current_gdn_qk_scale_plan ||
        !device_context->current_gdn_qk_scale_plan->ready) {
        return nullptr;
    }
    for (const auto & layer :
         device_context->current_gdn_qk_scale_plan->layers) {
        if (layer.gated_delta_net == node) {
            return &layer;
        }
    }
    return nullptr;
}

static const ggml_backend_hrx_gdn_qk_scale_layer *
ggml_backend_hrx_find_gdn_qk_scale_layer(
        const ggml_backend_hrx_device_context * device_context,
        int layer_number) {
    if (!device_context ||
        !device_context->current_gdn_qk_scale_plan ||
        !device_context->current_gdn_qk_scale_plan->ready) {
        return nullptr;
    }
    for (const auto & layer :
         device_context->current_gdn_qk_scale_plan->layers) {
        if (layer.layer == layer_number) {
            return &layer;
        }
    }
    return nullptr;
}

static bool ggml_backend_hrx_make_gdn_qk_scale_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        const ggml_backend_hrx_gdn_qk_scale_layer & layer,
        ggml_backend_hrx_dispatch_request * out_request) {
    ggml_backend_hrx_dispatch_request request = {};
    if (layer.gated_delta_net != node ||
        !layer.q_norm || !layer.k_norm ||
        !layer.raw_q || !layer.raw_k ||
        !ggml_backend_hrx_make_gated_delta_net_request(
            device_context, node, &request) ||
        request.tensors.size() != 7 ||
        request.tensors[0] != layer.q_norm ||
        request.tensors[1] != layer.k_norm) {
        return false;
    }

    request.problem.supports["layout"] =
        "state_transposed_column_per_wave_qk_l2_full_head";
    request.problem.supports["fusion"] =
        "GATED_DELTA_NET_QK_L2_FULL_HEAD";
    auto put = [&](const char * key, int64_t value) {
        ggml_backend_hrx_set_shape_alias(
            &request.problem, "gdn_qk_scale", key, value);
    };
    put("ncols", layer.raw_q->ne[0]);
    put("ne1", layer.raw_q->ne[1]);
    put("ne2", layer.raw_q->ne[2]);
    put("ne3", layer.raw_q->ne[3]);
    put("s1", static_cast<int64_t>(
        layer.raw_q->nb[1] / sizeof(float)));
    put("s2", static_cast<int64_t>(
        layer.raw_q->nb[2] / sizeof(float)));
    put("s3", static_cast<int64_t>(
        layer.raw_q->nb[3] / sizeof(float)));

    request.tensors = {
        layer.raw_q,
        layer.raw_k,
        node->src[2],
        node->src[3],
        node->src[4],
        node->src[5],
        node,
    };
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "src0", layer.raw_q);
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "src1", layer.raw_k);
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "dst", node);
    *out_request = std::move(request);
    return true;
}

// CONCAT along dim 0, once per Gated-DeltaNet layer to prepend the cached conv
// state to the qkv projection. The second operand is a transpose view there, so
// the kernel takes explicit element strides for both sources.
static bool ggml_backend_hrx_make_concat_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_CONCAT ||
        !node->src[0] || !node->src[1]) {
        return false;
    }
    if (node->type != GGML_TYPE_F32 || node->src[0]->type != GGML_TYPE_F32 ||
        node->src[1]->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(node) || node->ne[3] != 1) {
        return false;
    }
    const int32_t dim = ggml_get_op_params_i32(node, 0);
    // A dim-1 concatenation of fully contiguous operands is a linear
    // concatenation: dst holds all of src0's elements followed by all of src1's.
    // Presenting it to the dim-0 kernel as a single row of ne0*ne1*ne2 elements
    // with unit strides is exact, and it is how the Gated-DeltaNet conv window
    // is built on newer llama.cpp bases -- which emit dim=1 here where older
    // ones emitted dim=0, so without this every conv_input falls to the CPU and
    // splits the graph 61 ways.
    const bool linear_dim1 = dim == 1 &&
        ggml_is_contiguous(node->src[0]) && ggml_is_contiguous(node->src[1]);
    if (dim != 0 && !linear_dim1) {
        return false;
    }
    // Strides are converted to elements; a source whose byte strides are not
    // float-aligned cannot be indexed this way.
    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];
    for (const ggml_tensor * t : {a, b}) {
        for (int i = 0; i < 3; ++i) {
            if (t->nb[i] % sizeof(float) != 0) {
                return false;
            }
        }
    }
    out_request->problem = {};
    out_request->problem.op = "CONCAT";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(a->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(b->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "strided_sources_contiguous_dst_dim0"},
    };
    auto put = [&](const char * key, int64_t value) {
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "concat", key, value);
    };
    if (linear_dim1) {
        const int64_t total = node->ne[0] * node->ne[1] * node->ne[2];
        put("ne0", total);
        put("ne1", 1);
        put("ne2", 1);
        put("src0_ne0", ggml_nelements(a));
        put("src0_nb0", 1);
        put("src0_nb1", ggml_nelements(a));
        put("src0_nb2", ggml_nelements(a));
        put("src1_nb0", 1);
        put("src1_nb1", ggml_nelements(b));
        put("src1_nb2", ggml_nelements(b));
        out_request->problem.shape["ncols"] = total;
        out_request->problem.shape["nrows"] = 1;
    } else {
        put("ne0", node->ne[0]);
        put("ne1", node->ne[1]);
        put("ne2", node->ne[2]);
        put("src0_ne0", a->ne[0]);
        put("src0_nb0", static_cast<int64_t>(a->nb[0] / sizeof(float)));
        put("src0_nb1", static_cast<int64_t>(a->nb[1] / sizeof(float)));
        put("src0_nb2", static_cast<int64_t>(a->nb[2] / sizeof(float)));
        put("src1_nb0", static_cast<int64_t>(b->nb[0] / sizeof(float)));
        put("src1_nb1", static_cast<int64_t>(b->nb[1] / sizeof(float)));
        put("src1_nb2", static_cast<int64_t>(b->nb[2] / sizeof(float)));
        out_request->problem.shape["ncols"] = node->ne[0];
        out_request->problem.shape["nrows"] = node->ne[1] * node->ne[2];
    }
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", a);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", b);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {a, b, node};
    out_request->constants.clear();
    return true;
}

// The conv window that SSM_CONV reads is built by a CONCAT that prepends the
// d_conv-1 carried state rows to this layer's x. That concat writes the entire
// 16.9 MB production window even when the planned convolution needs only the
// carried state and the x prefix whose arena storage is overwritten before use.
// Returns the two pieces when the window is a concat the fused kernel can read
// directly: channels-first, dim-1 concatenation, state first, both sides
// row-contiguous with the same channel count.
static bool ggml_backend_hrx_ssm_conv_window_pieces(
        const ggml_tensor * node,
        const ggml_tensor ** out_state,
        const ggml_tensor ** out_x) {
    if (!node) {
        return false;
    }
    const ggml_tensor * cc = node->src[0];
    if (!cc || cc->op != GGML_OP_CONCAT || cc->type != GGML_TYPE_F32 ||
        !cc->src[0] || !cc->src[1]) {
        return false;
    }
    if (ggml_get_op_params_i32(cc, 0) != 1) {
        return false;
    }
    const ggml_tensor * st = cc->src[0];
    const ggml_tensor * x  = cc->src[1];
    if (st->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 ||
        st->nb[0] != sizeof(float) || x->nb[0] != sizeof(float)) {
        return false;
    }
    const int64_t d_conv  = node->src[1] ? node->src[1]->ne[0] : 0;
    const int64_t d_inner = node->src[1] ? node->src[1]->ne[1] : 0;
    const int64_t n_t     = node->ne[1];
    // Channels-first only, and the split has to fall exactly on the carried rows.
    if (d_conv != 4 || st->ne[0] != d_inner || x->ne[0] != d_inner ||
        st->ne[1] != d_conv - 1 || x->ne[1] != n_t) {
        return false;
    }
    // Higher dimensions must be plain sequence axes with matching extents.
    if (st->ne[2] != x->ne[2] || st->ne[3] != 1 || x->ne[3] != 1) {
        return false;
    }
    *out_state = st;
    *out_x = x;
    return true;
}

// True when every consumer of this conv-window CONCAT other than SSM_CONV reads
// only the carried tail -- window rows [n_t, n_t+d_conv-1). The graph-scoped
// sparse route snapshots the carried state and overwritten x prefix while also
// performing that sole cache update.
static bool ggml_backend_hrx_concat_window_tail_only(
        const ggml_cgraph * cgraph,
        const ggml_tensor * concat,
        const ggml_tensor ** out_x,
        int64_t * out_d_conv,
        int64_t * out_n_t,
        const ggml_tensor ** out_copy) {
    if (!cgraph || !concat || concat->op != GGML_OP_CONCAT) {
        return false;
    }
    const ggml_tensor * conv = nullptr;
    for (int j = 0; j < cgraph->n_nodes && !conv; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n && n->op == GGML_OP_SSM_CONV && n->src[0] == concat) {
            conv = n;
        }
    }
    if (!conv) {
        return false;
    }
    const ggml_tensor * st = nullptr;
    const ggml_tensor * x  = nullptr;
    if (!ggml_backend_hrx_ssm_conv_window_pieces(conv, &st, &x)) {
        return false;
    }
    const int64_t d_conv = conv->src[1] ? conv->src[1]->ne[0] : 0;
    const int64_t n_t    = conv->ne[1];
    // The tail is sourced entirely from x only when x has at least d_conv-1
    // rows. Decode has n_t=1, so its live tail also contains carried-state
    // rows and must stay on the ordinary CONCAT path.
    if (d_conv < 2 || n_t < d_conv - 1 || concat->nb[1] == 0) {
        return false;
    }
    // Byte range of the live tail inside the window.
    const size_t tail_begin = (size_t) n_t * concat->nb[1];
    const char * const base = (const char *) concat->data;
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * u = cgraph->nodes[j];
        if (!u || u == concat || u == conv) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * t = u->src[s];
            if (!t) {
                continue;
            }
            const ggml_tensor * root = t->view_src ? t->view_src : t;
            if (root != concat) {
                continue;
            }
            // A view-producing node does not read anything itself; what matters is
            // where its window starts, since everything downstream sits inside it.
            const char * const p =
                (const char *) (ggml_backend_hrx_is_metadata_op(u) ? u->data : t->data);
            // Any read that starts before the tail keeps the whole window live.
            if (!p || p < base || (size_t) (p - base) < tail_begin) {
                return false;
            }
        }
    }
    // The tail's reader is a copy into the conv state cache. If it is exactly
    // one contiguous copy of the whole tail, this kernel can write the cache
    // itself and the copy becomes dead.
    const ggml_tensor * copy = nullptr;
    int copies = 0;
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * u = cgraph->nodes[j];
        if (!u || u == concat || u == conv || ggml_backend_hrx_is_metadata_op(u)) {
            continue;
        }
        for (int sl = 0; sl < GGML_MAX_SRC; ++sl) {
            const ggml_tensor * t = u->src[sl];
            const ggml_tensor * root = t && t->view_src ? t->view_src : t;
            if (root == concat) {
                ++copies;
                copy = u;
            }
        }
    }
    if (out_copy) {
        *out_copy = nullptr;
        if (copies == 1 && copy &&
            (copy->op == GGML_OP_CPY || copy->op == GGML_OP_CONT) &&
            copy->type == GGML_TYPE_F32 && ggml_is_contiguous(copy) &&
            ggml_nelements(copy) == (d_conv - 1) * x->ne[0] * x->ne[2]) {
            *out_copy = copy;
        }
    }
    *out_x = x;
    *out_d_conv = d_conv;
    *out_n_t = n_t;
    return true;
}

static const ggml_backend_hrx_ssm_conv_silu_layer *
ggml_backend_hrx_find_active_ssm_conv_silu_layer(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node) {
    if (!device_context || !node || !device_context->current_graph ||
        device_context->current_graph->uid == 0 ||
        !device_context->ssm_conv_silu_plan ||
        !device_context->reg_context ||
        !device_context->reg_context->catalog) {
        return nullptr;
    }
    const ggml_cgraph * cgraph = device_context->current_graph;
    const auto & plan = *device_context->ssm_conv_silu_plan;
    const auto * catalog = device_context->reg_context->catalog.get();
    if (!plan.owner || !plan.valid || !plan.ready ||
        plan.graph_uid != cgraph->uid ||
        plan.node_count != cgraph->n_nodes ||
        plan.catalog != catalog ||
        plan.layers.size() != 30) {
        return nullptr;
    }
    for (const auto & layer : plan.layers) {
        if ((layer.ssm_conv != node && layer.window != node) ||
            !layer.ssm_conv || !layer.window ||
            layer.ssm_conv->src[0] != layer.window ||
            layer.ssm_conv_index < 0 ||
            layer.ssm_conv_index >= cgraph->n_nodes ||
            cgraph->nodes[layer.ssm_conv_index] != layer.ssm_conv) {
            continue;
        }
        return &layer;
    }
    return nullptr;
}

static bool ggml_backend_hrx_make_concat_window_tail_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_cgraph * cgraph,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->type != GGML_TYPE_F32) {
        return false;
    }
    const auto * layer =
        ggml_backend_hrx_find_active_ssm_conv_silu_layer(
            device_context, node);
    if (!layer || layer->window != node ||
        device_context->current_graph != cgraph) {
        return false;
    }
    const ggml_tensor * state = node->src[0];
    const ggml_tensor * x = nullptr;
    int64_t d_conv = 0;
    int64_t n_t = 0;
    const ggml_tensor * copy = nullptr;
    if (!ggml_backend_hrx_concat_window_tail_only(cgraph, node, &x, &d_conv, &n_t,
                                                  &copy) ||
        !state || !copy ||
        state != layer->state || x != layer->x ||
        layer->ssm_conv->src[0] != node ||
        state->type != GGML_TYPE_F32 ||
        x->type != GGML_TYPE_F32 ||
        copy->type != GGML_TYPE_F32 ||
        state->nb[0] != sizeof(float) ||
        state->nb[1] % sizeof(float) != 0 ||
        x->nb[1] % sizeof(float) != 0 ||
        node->nb[1] % sizeof(float) != 0 ||
        !ggml_backend_hrx_disjoint_storage_spans(state, x) ||
        !ggml_backend_hrx_disjoint_storage_spans(state, node) ||
        !ggml_backend_hrx_disjoint_storage_spans(state, copy) ||
        !ggml_backend_hrx_disjoint_storage_spans(x, node) ||
        !ggml_backend_hrx_disjoint_storage_spans(x, copy) ||
        !ggml_backend_hrx_disjoint_storage_spans(node, copy)) {
        return false;
    }
    const int64_t d_inner = node->ne[0];
    const int64_t n_s = node->ne[2];
    out_request->problem = {};
    out_request->problem.op = "CONCAT";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(state->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(x->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "conv_window_state_x61_snapshot"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "concat_tail", "d_conv", d_conv);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "concat_tail", "d_inner", d_inner);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "concat_tail", "n_t", n_t);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "concat_tail", "n_s", n_s);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "concat_tail", "state_row_stride",
                                     static_cast<int64_t>(state->nb[1] / sizeof(float)));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "concat_tail", "x_row_stride",
                                     static_cast<int64_t>(x->nb[1] / sizeof(float)));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "concat_tail", "dst_row_stride",
                                     static_cast<int64_t>(node->nb[1] / sizeof(float)));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "concat_tail", "cache_row_stride",
                                     d_inner);
    out_request->problem.shape["ncols"] = d_inner;
    out_request->problem.shape["nrows"] =
        (d_conv - 1 +
         static_cast<int64_t>(GGML_HRX_SSM_X_SNAPSHOT_ROWS)) * n_s;
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", state);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", x);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {state, x, node, copy};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_ssm_conv_concat_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_SSM_CONV ||
        !node->src[1] || node->src[1]->type != GGML_TYPE_F32 ||
        node->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(node->src[1]) || !ggml_is_contiguous(node)) {
        return false;
    }
    const ggml_tensor * st = nullptr;
    const ggml_tensor * x  = nullptr;
    if (!ggml_backend_hrx_ssm_conv_window_pieces(node, &st, &x)) {
        return false;
    }
    // llama.cpp schedules the recurrent-cache update before SSM_CONV and the
    // arena may reuse st immediately after CONCAT. The reduced CONCAT route
    // preserves those old rows in its own output, whose lifetime extends
    // through this node; always read the state prefix from that snapshot.
    const ggml_tensor * state_snapshot = node->src[0];
    if (!state_snapshot || state_snapshot->type != GGML_TYPE_F32 ||
        state_snapshot->nb[0] != sizeof(float) ||
        state_snapshot->nb[1] % sizeof(float) != 0) {
        return false;
    }
    const int64_t d_conv  = node->src[1]->ne[0];
    const int64_t d_inner = node->src[1]->ne[1];
    const int64_t n_t     = node->ne[1];
    const int64_t n_s     = node->ne[2];
    if (node->ne[0] != d_inner) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "SSM_CONV";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(state_snapshot->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(node->src[1]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "conv_window_channels_first_concat"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "ssm_conv", "d_conv", d_conv);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "ssm_conv", "d_inner", d_inner);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "ssm_conv", "n_t", n_t);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "ssm_conv", "n_s", n_s);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "ssm_conv", "state_row_stride",
                                     static_cast<int64_t>(state_snapshot->nb[1] / sizeof(float)));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "ssm_conv", "x_row_stride",
                                     static_cast<int64_t>(x->nb[1] / sizeof(float)));
    out_request->problem.shape["ncols"] = d_inner;
    out_request->problem.shape["nrows"] = n_t * n_s;
    out_request->problem.shape["d_conv"] = d_conv;
    out_request->problem.shape["n_t"]   = n_t;
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", state_snapshot);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", node->src[1]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {state_snapshot, x, node->src[1], node};
    out_request->constants.clear();
    return true;
}

static bool
ggml_backend_hrx_make_ssm_conv_silu_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        const ggml_backend_hrx_ssm_conv_silu_layer & layer,
        ggml_backend_hrx_dispatch_request * out_request) {
    ggml_backend_hrx_dispatch_request request = {};
    if (layer.ssm_conv != node ||
        !layer.silu || !layer.window || !layer.state || !layer.x ||
        !layer.filter || !layer.dst ||
        node->src[0] != layer.window ||
        !ggml_backend_hrx_make_ssm_conv_concat_request(
            device_context, node, &request) ||
        request.tensors.size() != 4 ||
        request.tensors[0] != layer.window ||
        request.tensors[1] != layer.x ||
        request.tensors[2] != layer.filter) {
        return false;
    }
    request.problem.supports["layout"] =
        "conv_window_channels_first_concat_silu_regblock_wg1024";
    request.problem.supports["fusion"] =
        "SSM_CONV_SILU_REGBLOCK_WG1024";
    request.tensors[3] = layer.dst;
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "dst", layer.dst);
    *out_request = std::move(request);
    return true;
}

// Depthwise causal conv over the Gated-DeltaNet conv window, once per SSM
// layer. src0 is {d_conv-1+n_t, d_inner, n_s}, src1 is {d_conv, d_inner},
// dst is {d_inner, n_t, n_s}.
static bool ggml_backend_hrx_make_ssm_conv_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_SSM_CONV ||
        !node->src[0] || !node->src[1] ||
        node->src[0]->type != GGML_TYPE_F32 || node->src[1]->type != GGML_TYPE_F32 ||
        node->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(node->src[1]) || !ggml_is_contiguous(node)) {
        return false;
    }
    // The kernel indexes src0 with a single row stride, so the window must be
    // row-contiguous with a uniform stride between channels.
    if (node->src[0]->nb[0] != sizeof(float)) {
        return false;
    }
    const int64_t d_conv  = node->src[1]->ne[0];
    const int64_t d_inner = node->src[1]->ne[1];
    const int64_t n_t     = node->ne[1];
    const int64_t n_s     = node->ne[2];
    if (node->ne[0] != d_inner) {
        return false;
    }
    // The window's orientation decides the kernel. Older bases hand over
    // {d_conv-1+n_t, d_inner, n_s} with tokens contiguous; newer ones transpose
    // it to {d_inner, d_conv-1+n_t, n_s} with channels contiguous. Both give a
    // uniform nb[1] stride, so the same shape aliases describe either, but the
    // kernels are different: the tokens-contiguous one must transpose through
    // LDS, while the channels-contiguous one is coalesced with one output per
    // lane. Running either on the wrong layout is a 7x loss.
    // Identify the orientation from both dimensions, not just ne[0]: a test shape
    // where d_inner happens to equal the window length would otherwise be read as
    // channels-first and computed with the wrong strides. When both readings fit,
    // prefer the tokens-contiguous one, which is what the tiled kernel expects.
    const int64_t window = d_conv - 1 + n_t;
    const bool last_fits  = node->src[0]->ne[0] == window  && node->src[0]->ne[1] == d_inner;
    const bool first_fits = node->src[0]->ne[0] == d_inner && node->src[0]->ne[1] == window;
    if (!last_fits && !first_fits) {
        return false;
    }
    const bool channels_first = first_fits && !last_fits;
    out_request->problem = {};
    out_request->problem.op = "SSM_CONV";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(node->src[1]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", channels_first ? "conv_window_channels_first"
                                  : "conv_window_channels_last"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "ssm_conv", "d_conv", d_conv);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "ssm_conv", "d_inner", d_inner);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "ssm_conv", "n_t", n_t);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "ssm_conv", "n_s", n_s);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "ssm_conv", "src0_row_stride",
                                     static_cast<int64_t>(node->src[0]->nb[1] / sizeof(float)));
    out_request->problem.shape["ncols"] = d_inner;
    out_request->problem.shape["nrows"] = n_t * n_s;
    // Exposed so a route can restrict itself to a tap count and token count it
    // has been specialised for.
    out_request->problem.shape["d_conv"] = d_conv;
    out_request->problem.shape["n_t"]   = n_t;
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", node->src[1]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node->src[1], node};
    out_request->constants.clear();
    return true;
}

static const ggml_backend_hrx_recurrent_cache_layer *
ggml_backend_hrx_find_recurrent_cache_layer(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node) {
    if (!device_context || !node ||
        !device_context->current_recurrent_cache_plan ||
        !device_context->current_recurrent_cache_plan->ready) {
        return nullptr;
    }
    for (const auto & layer :
         device_context->current_recurrent_cache_plan->layers) {
        if (layer.ssm_conv == node ||
            layer.gated_delta_net == node) {
            return &layer;
        }
    }
    return nullptr;
}

static bool ggml_backend_hrx_make_ssm_conv_state_cache_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        const ggml_backend_hrx_recurrent_cache_layer & layer,
        ggml_backend_hrx_dispatch_request * out_request) {
    ggml_backend_hrx_dispatch_request request = {};
    if (layer.ssm_conv != node ||
        !layer.cache_r_owner || !layer.conv_x ||
        !layer.conv_filter || !layer.conv_dst ||
        !ggml_backend_hrx_make_ssm_conv_concat_request(
            device_context, node, &request) ||
        request.tensors.size() != 4 ||
        request.tensors[1] != layer.conv_x ||
        request.tensors[2] != layer.conv_filter ||
        request.tensors[3] != layer.conv_dst) {
        return false;
    }
    request.problem.supports["layout"] =
        "conv_state_cache_decode_inplace";
    request.problem.supports["fusion"] =
        "SSM_CONV_STATE_CACHE_DECODE";
    request.tensors[0] = layer.cache_r_owner;
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "src0", layer.cache_r_owner);
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "state_io", layer.cache_r_owner);
    *out_request = std::move(request);
    return true;
}

static bool ggml_backend_hrx_make_gdn_state_cache_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        const ggml_backend_hrx_recurrent_cache_layer & layer,
        ggml_backend_hrx_dispatch_request * out_request) {
    ggml_backend_hrx_dispatch_request request = {};
    if (layer.gated_delta_net != node ||
        !layer.cache_s_owner || !layer.attention_dst ||
        !ggml_backend_hrx_make_gated_delta_net_request(
            device_context, node, &request) ||
        request.tensors.size() != 7) {
        return false;
    }
    request.problem.supports["layout"] =
        "state_cache_decode_inplace";
    request.problem.supports["fusion"] =
        "GATED_DELTA_NET_STATE_CACHE_DECODE";
    request.tensors[5] = layer.cache_s_owner;
    request.tensors[6] = layer.attention_dst;
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "state_io", layer.cache_s_owner);
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "dst", layer.attention_dst);
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "attn_dst", layer.attention_dst);
    *out_request = std::move(request);
    return true;
}

// Fused attention. One workgroup per (query token, query head) and one lane per
// head-dimension element, so the workgroup size is the head dimension and every
// K/V load is unit-stride in d. Everything the kernel needs is a stride, so the
// only real constraints are the ones it cannot express: ALiBi bias, logit
// softcap, a missing mask, and a head dimension that is not a multiple of 32.
static bool ggml_backend_hrx_make_flash_attn_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    const ggml_tensor * m = node->src[3];
    if (!q || !k || !v || !m) {
        return false;
    }
    // A fifth source (attention sinks) would change the softmax normalisation.
    if (GGML_MAX_SRC > 4 && node->src[4]) {
        return false;
    }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 ||
        v->type != GGML_TYPE_F16 || m->type != GGML_TYPE_F16 ||
        node->type != GGML_TYPE_F32) {
        return false;
    }
    float scale = 0.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    std::memcpy(&scale, node->op_params, sizeof(float));
    std::memcpy(&max_bias, reinterpret_cast<const uint8_t *>(node->op_params) + sizeof(float),
                sizeof(float));
    std::memcpy(&logit_softcap,
                reinterpret_cast<const uint8_t *>(node->op_params) + 2 * sizeof(float),
                sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    const int64_t dhead   = q->ne[0];
    const int64_t ntokens = q->ne[1];
    const int64_t nheads  = q->ne[2];
    const int64_t nkv     = k->ne[1];
    const int64_t nkvhead = k->ne[2];
    if (dhead % 32 != 0 || dhead > 1024 || nkvhead == 0 || nheads % nkvhead != 0 ||
        k->ne[0] != dhead || v->ne[0] != dhead || node->ne[0] != dhead ||
        node->ne[1] != nheads || node->ne[2] != ntokens ||
        q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || node->ne[3] != 1 ||
        v->ne[1] != nkv || v->ne[2] != nkvhead || m->ne[0] < nkv) {
        return false;
    }
    // Unit stride along the reduced dimension is what makes the loads coalesced;
    // everything else is passed as an explicit stride.
    if (q->nb[0] != sizeof(float) || node->nb[0] != sizeof(float) ||
        k->nb[0] != sizeof(ggml_fp16_t) || v->nb[0] != sizeof(ggml_fp16_t) ||
        m->nb[0] != sizeof(ggml_fp16_t)) {
        return false;
    }
    const size_t f32 = sizeof(float);
    const size_t f16 = sizeof(ggml_fp16_t);
    if (q->nb[1] % f32 || q->nb[2] % f32 || node->nb[1] % f32 || node->nb[2] % f32 ||
        k->nb[1] % f16 || k->nb[2] % f16 || v->nb[1] % f16 || v->nb[2] % f16 ||
        m->nb[1] % f16) {
        return false;
    }

    out_request->problem = {};
    out_request->problem.op = "FLASH_ATTN_EXT";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(q->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(k->type)},
        {"src2_type", ggml_backend_hrx_catalog_type_name(v->type)},
        {"src3_type", ggml_backend_hrx_catalog_type_name(m->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"mask", "required"},
        {"max_bias", "0"},
        {"logit_softcap", "0"},
        {"layout", "head_major_qkv_online_softmax"},
    };
    auto putf = [&](const char * key, int64_t value) {
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "fa", key, value);
    };
    putf("d", dhead);
    putf("ntokens", ntokens);
    putf("nheads", nheads);
    putf("nkv", nkv);
    putf("gqa", nheads / nkvhead);
    putf("q_stride_token", static_cast<int64_t>(q->nb[1] / f32));
    putf("q_stride_head", static_cast<int64_t>(q->nb[2] / f32));
    putf("k_stride_pos", static_cast<int64_t>(k->nb[1] / f16));
    putf("k_stride_head", static_cast<int64_t>(k->nb[2] / f16));
    putf("v_stride_pos", static_cast<int64_t>(v->nb[1] / f16));
    putf("v_stride_head", static_cast<int64_t>(v->nb[2] / f16));
    putf("mask_stride_token", static_cast<int64_t>(m->nb[1] / f16));
    putf("dst_stride_head", static_cast<int64_t>(node->nb[1] / f32));
    putf("dst_stride_token", static_cast<int64_t>(node->nb[2] / f32));
    out_request->problem.shape["ncols"] = dhead;
    out_request->problem.shape["nrows"] = ntokens * nheads;
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", q);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {q, k, v, m, node};
    out_request->constants.clear();
    ggml_backend_hrx_append_constant(&out_request->constants, scale);
    return true;
}

struct ggml_backend_hrx_fa_gate_epilogue_match {
    const ggml_tensor * flash_attn = nullptr;
    const ggml_tensor * pregate = nullptr;
    const ggml_tensor * cont = nullptr;
    const ggml_tensor * sigmoid = nullptr;
    const ggml_tensor * raw_gate = nullptr;
};

static bool ggml_backend_hrx_match_fa_gate_epilogue(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_fa_gate_epilogue_match * out_match) {
    if (!device_context || !device_context->current_graph || !node ||
        !out_match || node->op != GGML_OP_MUL ||
        !node->src[0] || !node->src[1] ||
        node->type != GGML_TYPE_F32) {
        return false;
    }

    const ggml_tensor * pregate = node->src[0];
    const ggml_tensor * sigmoid = node->src[1];
    if (pregate->op != GGML_OP_RESHAPE || !pregate->src[0] ||
        pregate->view_src != pregate->src[0] ||
        sigmoid->op != GGML_OP_UNARY ||
        ggml_get_unary_op(sigmoid) != GGML_UNARY_OP_SIGMOID ||
        !sigmoid->src[0]) {
        return false;
    }
    const ggml_tensor * flash_attn = pregate->src[0];
    const ggml_tensor * cont = sigmoid->src[0];
    if (flash_attn->op != GGML_OP_FLASH_ATTN_EXT ||
        cont->op != GGML_OP_CONT || !cont->src[0]) {
        return false;
    }
    const ggml_tensor * raw_gate = cont->src[0];
    const ggml_tensor * q = flash_attn->src[0];
    const ggml_tensor * k = flash_attn->src[1];
    const ggml_tensor * v = flash_attn->src[2];
    const ggml_tensor * mask = flash_attn->src[3];
    if (!q || !k || !v || !mask ||
        (GGML_MAX_SRC > 4 && flash_attn->src[4]) ||
        raw_gate->op != GGML_OP_VIEW ||
        !raw_gate->src[0] ||
        raw_gate->view_src != raw_gate->src[0]) {
        return false;
    }

    auto exact_shape = [](const ggml_tensor * tensor,
                          int64_t ne0, int64_t ne1,
                          int64_t ne2, int64_t ne3) {
        return tensor &&
               tensor->ne[0] == ne0 && tensor->ne[1] == ne1 &&
               tensor->ne[2] == ne2 && tensor->ne[3] == ne3;
    };
    constexpr size_t output_bytes =
        256ull * 16ull * 512ull * sizeof(float);
    constexpr size_t raw_gate_bytes = 16776192;
    constexpr size_t joint_q_gate_bytes =
        512ull * 8192ull * sizeof(float);
    if (!exact_shape(q, 256, 512, 16, 1) ||
        !exact_shape(k, 256, 512, 2, 1) ||
        !exact_shape(v, 256, 512, 2, 1) ||
        !exact_shape(mask, 512, 512, 1, 1) ||
        !exact_shape(flash_attn, 256, 16, 512, 1) ||
        !exact_shape(pregate, 4096, 512, 1, 1) ||
        !exact_shape(raw_gate, 256, 16, 512, 1) ||
        !exact_shape(cont, 4096, 512, 1, 1) ||
        !exact_shape(sigmoid, 4096, 512, 1, 1) ||
        !exact_shape(node, 4096, 512, 1, 1) ||
        q->type != GGML_TYPE_F32 ||
        k->type != GGML_TYPE_F16 ||
        v->type != GGML_TYPE_F16 ||
        mask->type != GGML_TYPE_F16 ||
        flash_attn->type != GGML_TYPE_F32 ||
        pregate->type != GGML_TYPE_F32 ||
        raw_gate->type != GGML_TYPE_F32 ||
        cont->type != GGML_TYPE_F32 ||
        sigmoid->type != GGML_TYPE_F32 ||
        q->op != GGML_OP_PERMUTE ||
        k->op != GGML_OP_PERMUTE ||
        v->op != GGML_OP_PERMUTE ||
        q->nb[0] != sizeof(float) ||
        q->nb[1] != 4096 * sizeof(float) ||
        q->nb[2] != 256 * sizeof(float) ||
        k->nb[0] != sizeof(ggml_fp16_t) ||
        k->nb[1] != 512 * sizeof(ggml_fp16_t) ||
        k->nb[2] != 256 * sizeof(ggml_fp16_t) ||
        v->nb[0] != sizeof(ggml_fp16_t) ||
        v->nb[1] != 512 * sizeof(ggml_fp16_t) ||
        v->nb[2] != 256 * sizeof(ggml_fp16_t) ||
        mask->nb[0] != sizeof(ggml_fp16_t) ||
        mask->nb[1] != 512 * sizeof(ggml_fp16_t) ||
        flash_attn->nb[0] != sizeof(float) ||
        flash_attn->nb[1] != 256 * sizeof(float) ||
        flash_attn->nb[2] != 4096 * sizeof(float) ||
        raw_gate->nb[0] != sizeof(float) ||
        raw_gate->nb[1] != 512 * sizeof(float) ||
        raw_gate->nb[2] != 8192 * sizeof(float) ||
        raw_gate->nb[3] != 4194304 * sizeof(float) ||
        raw_gate->view_offs != 256 * sizeof(float) ||
        ggml_nbytes(raw_gate) != raw_gate_bytes ||
        ggml_nbytes(raw_gate->view_src) != joint_q_gate_bytes ||
        ggml_nbytes(q) != output_bytes ||
        ggml_nbytes(flash_attn) != output_bytes ||
        ggml_nbytes(pregate) != output_bytes ||
        ggml_nbytes(cont) != output_bytes ||
        ggml_nbytes(sigmoid) != output_bytes ||
        ggml_nbytes(node) != output_bytes ||
        !ggml_backend_hrx_is_f32_dense(flash_attn) ||
        !ggml_backend_hrx_is_f32_dense(pregate) ||
        !ggml_backend_hrx_is_f32_dense(cont) ||
        !ggml_backend_hrx_is_f32_dense(sigmoid) ||
        !ggml_backend_hrx_is_f32_dense(node) ||
        !ggml_backend_hrx_zero_offset_full_span_chain_reaches(
            pregate, flash_attn) ||
        (flash_attn->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        (pregate->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        (cont->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        (sigmoid->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        (q->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    ggml_backend_buffer_t raw_buffer = nullptr;
    ggml_backend_buffer_t raw_parent_buffer = nullptr;
    ggml_backend_buffer_t q_buffer = nullptr;
    size_t raw_offset = 0;
    size_t raw_parent_offset = 0;
    size_t q_offset = 0;
    size_t raw_length = 0;
    size_t raw_parent_length = 0;
    size_t q_length = 0;
    if (!ggml_backend_hrx_tensor_storage_range(
            raw_gate, &raw_buffer, &raw_offset, &raw_length) ||
        !ggml_backend_hrx_tensor_storage_range(
            raw_gate->view_src, &raw_parent_buffer,
            &raw_parent_offset, &raw_parent_length) ||
        !ggml_backend_hrx_tensor_storage_range(
            q, &q_buffer, &q_offset, &q_length) ||
        raw_buffer != raw_parent_buffer ||
        raw_offset != raw_parent_offset + 256 * sizeof(float) ||
        raw_length != raw_gate_bytes ||
        raw_parent_length != joint_q_gate_bytes ||
        raw_buffer != q_buffer ||
        raw_offset + raw_length != q_offset ||
        q_length != output_bytes ||
        !ggml_backend_hrx_same_storage_span(q, node) ||
        !ggml_backend_hrx_same_storage_span(flash_attn, pregate) ||
        !ggml_backend_hrx_same_storage_span(cont, sigmoid) ||
        !ggml_backend_hrx_same_storage_span(sigmoid, node) ||
        !ggml_backend_hrx_disjoint_storage_spans(
            flash_attn, node) ||
        !ggml_backend_hrx_disjoint_storage_spans(
            flash_attn, raw_gate)) {
        return false;
    }

    const std::array<const ggml_tensor *, 6> tensors = {
        q, k, v, mask, raw_gate, node,
    };
    for (size_t i = 0; i < tensors.size(); ++i) {
        for (size_t j = i + 1; j < tensors.size(); ++j) {
            const bool q_dst = i == 0 && j == 5;
            if ((q_dst &&
                 !ggml_backend_hrx_same_storage_span(
                     tensors[i], tensors[j])) ||
                (!q_dst &&
                 !ggml_backend_hrx_disjoint_storage_spans(
                     tensors[i], tensors[j]))) {
                return false;
            }
        }
    }
    for (const ggml_tensor * tensor : tensors) {
        if (tensor != node &&
            !ggml_backend_hrx_disjoint_storage_spans(
                flash_attn, tensor)) {
            return false;
        }
    }

    const ggml_cgraph * cgraph = device_context->current_graph;
    int flash_index = -1;
    int cont_index = -1;
    int sigmoid_index = -1;
    int terminal_index = -1;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * candidate = cgraph->nodes[i];
        if (!candidate) {
            continue;
        }
        if (candidate == flash_attn) {
            flash_index = i;
        } else if (candidate == cont) {
            cont_index = i;
        } else if (candidate == sigmoid) {
            sigmoid_index = i;
        } else if (candidate == node) {
            terminal_index = i;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * source = candidate->src[s];
            if (!source) {
                continue;
            }
            if (ggml_backend_hrx_metadata_chain_reaches(source, q) &&
                (candidate != flash_attn || source != q)) {
                return false;
            }
            if (ggml_backend_hrx_metadata_chain_reaches(
                    source, flash_attn)) {
                const bool exact_pregate =
                    candidate == pregate && source == flash_attn;
                const bool exact_terminal =
                    candidate == node && source == pregate;
                if (!exact_pregate && !exact_terminal) {
                    return false;
                }
            }
            if (ggml_backend_hrx_metadata_chain_reaches(
                    source, raw_gate) &&
                (candidate != cont || source != raw_gate)) {
                return false;
            }
            if (ggml_backend_hrx_metadata_chain_reaches(source, cont) &&
                (candidate != sigmoid || source != cont)) {
                return false;
            }
            if (ggml_backend_hrx_metadata_chain_reaches(
                    source, sigmoid) &&
                (candidate != node || source != sigmoid)) {
                return false;
            }
        }
    }
    if (flash_index < 0 || cont_index < 0 ||
        sigmoid_index < 0 || terminal_index < 0 ||
        !(flash_index < cont_index &&
          cont_index < sigmoid_index &&
          sigmoid_index < terminal_index)) {
        return false;
    }
    for (int i = flash_index + 1; i < terminal_index; ++i) {
        const ggml_tensor * between = cgraph->nodes[i];
        if (between && between != cont && between != sigmoid &&
            !ggml_backend_hrx_is_metadata_op(between) &&
            !ggml_backend_hrx_is_empty_op(between)) {
            return false;
        }
    }

    out_match->flash_attn = flash_attn;
    out_match->pregate = pregate;
    out_match->cont = cont;
    out_match->sigmoid = sigmoid;
    out_match->raw_gate = raw_gate;
    return true;
}

static bool ggml_backend_hrx_make_fa_gate_epilogue_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request,
        ggml_backend_hrx_fa_gate_epilogue_match * out_match) {
    if (!device_context || !out_request) {
        return false;
    }
    ggml_backend_hrx_fa_gate_epilogue_match match = {};
    if (!ggml_backend_hrx_match_fa_gate_epilogue(
            device_context, node, &match)) {
        return false;
    }

    ggml_backend_hrx_dispatch_request request = {};
    if (!ggml_backend_hrx_make_flash_attn_request(
            device_context, match.flash_attn, &request) ||
        request.constants.size() != sizeof(float)) {
        return false;
    }
    request.problem.supports["fusion"] =
        "FLASH_ATTN_CONT_SIGMOID_MUL_EPILOGUE";
    ggml_backend_hrx_set_shape_alias(
        &request.problem, "fa", "gate_stride_head",
        static_cast<int64_t>(
            match.raw_gate->nb[1] / sizeof(float)));
    ggml_backend_hrx_set_shape_alias(
        &request.problem, "fa", "gate_stride_token",
        static_cast<int64_t>(
            match.raw_gate->nb[2] / sizeof(float)));
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "fa_dst", match.flash_attn);
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "raw_gate", match.raw_gate);
    ggml_backend_hrx_add_tensor_facts(
        &request.problem, "terminal_dst", node);
    request.tensors = {
        match.flash_attn->src[0],
        match.flash_attn->src[1],
        match.flash_attn->src[2],
        match.flash_attn->src[3],
        match.raw_gate,
        node,
    };
    ggml_backend_hrx_add_tensor_overlap_facts(
        &request.problem, request.tensors);

    const auto * route =
        device_context->reg_context &&
        device_context->reg_context->catalog
            ? ggml_backend_hrx_catalog_find_route(
                  *device_context->reg_context->catalog,
                  request.problem)
            : nullptr;
    const auto fusion =
        route ? route->supports.find("fusion")
              : decltype(route->supports.find("fusion")){};
    if (!route ||
        route->id !=
            "flash_attn_ext_f32_f16_wmma_gate_epilogue" ||
        route->family != "flash_attn_ext_f32" ||
        route->op != "FLASH_ATTN_EXT" ||
        route->source_id != "flash_attn_wmma_gate_epilogue" ||
        route->artifact_id !=
            "flash_attn_wmma_gate_epilogue_loombc" ||
        route->root_symbol !=
            "@hrx2_flash_attn_ext_f32_f16_wmma_gate_epilogue" ||
        route->export_name !=
            "hrx2_flash_attn_ext_f32_f16_wmma_gate_epilogue" ||
        route->binding_count != 6 ||
        route->parameter_count != 7 ||
        route->constant_byte_length != sizeof(float) ||
        fusion == route->supports.end() ||
        fusion->second !=
            "FLASH_ATTN_CONT_SIGMOID_MUL_EPILOGUE") {
        return false;
    }

    *out_request = std::move(request);
    if (out_match) {
        *out_match = match;
    }
    return true;
}

// Row-wise norm whose source is a strided view and whose destination is dense.
// The row index is decomposed into (i1,i2,i3) inside the kernel and s1/s2/s3 are
// applied independently, so this covers padding between rows *and* between
// planes -- a per-head slice of a fused projection, which is what every norm in
// the Qwen3.6 graph reads.
static bool ggml_backend_hrx_make_strided_norm_request(
        ggml_backend_hrx_device_context * device_context,
        const char * op_name,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    const ggml_tensor * src = node->src[0];
    if (!ggml_backend_hrx_is_f32_row_contiguous(src) ||
        !ggml_backend_hrx_is_f32_dense(node) ||
        src->ne[0] != node->ne[0] || src->ne[1] != node->ne[1] ||
        src->ne[2] != node->ne[2] || src->ne[3] != node->ne[3]) {
        return false;
    }
    for (int i = 1; i < 4; ++i) {
        if (src->nb[i] % sizeof(float) != 0) {
            return false;
        }
    }
    out_request->problem = {};
    out_request->problem.op = op_name;
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(src->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "strided_src_to_dense_dst"},
    };
    auto putn = [&](const char * key, int64_t value) {
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "normx", key, value);
    };
    putn("ncols", src->ne[0]);
    putn("ne1", src->ne[1]);
    putn("ne2", src->ne[2]);
    putn("ne3", src->ne[3]);
    putn("s1", static_cast<int64_t>(src->nb[1] / sizeof(float)));
    putn("s2", static_cast<int64_t>(src->nb[2] / sizeof(float)));
    putn("s3", static_cast<int64_t>(src->nb[3] / sizeof(float)));
    out_request->problem.shape["ncols"] = src->ne[0];
    out_request->problem.shape["nrows"] = src->ne[1] * src->ne[2] * src->ne[3];
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", src);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {src, node};
    out_request->constants.clear();
    float eps = 0.0f;
    std::memcpy(&eps, node->op_params, sizeof(float));
    ggml_backend_hrx_append_constant(&out_request->constants, eps);
    return true;
}

// L2_NORM normalizes the convolved q and k projections in every Gated-DeltaNet
// layer, so it appears twice per SSM layer and had no builder.
static bool ggml_backend_hrx_make_l2_norm_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_L2_NORM || !node->src[0] ||
        !ggml_backend_hrx_is_f32_dense(node)) {
        return false;
    }
    if (!ggml_backend_hrx_is_f32_dense(node->src[0])) {
        return ggml_backend_hrx_make_strided_norm_request(
            device_context, "L2_NORM", node, out_request);
    }
    out_request->problem = {};
    out_request->problem.op = "L2_NORM";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "l2_norm", "ncols", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "l2_norm", "nrows", ggml_backend_hrx_tensor_row_count(node));
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node};
    out_request->constants.clear();
    float eps = 0.0f;
    std::memcpy(&eps, node->op_params, sizeof(float));
    ggml_backend_hrx_append_constant(&out_request->constants, eps);
    return true;
}

static bool ggml_backend_hrx_make_rms_norm_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_RMS_NORM || !node->src[0] ||
        !ggml_backend_hrx_is_f32_dense(node)) {
        return false;
    }
    if (!ggml_backend_hrx_is_f32_dense(node->src[0])) {
        return ggml_backend_hrx_make_strided_norm_request(
            device_context, "RMS_NORM", node, out_request);
    }
    out_request->problem = {};
    out_request->problem.op = "RMS_NORM";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rms_norm", "ncols", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rms_norm", "nrows", ggml_backend_hrx_tensor_row_count(node));
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node};
    out_request->constants.clear();
    float eps = 0.0f;
    std::memcpy(&eps, node->op_params, sizeof(float));
    ggml_backend_hrx_append_constant(&out_request->constants, eps);
    return true;
}

static bool ggml_backend_hrx_make_sum_rows_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_SUM_ROWS || !node->src[0] ||
        !ggml_backend_hrx_is_f32_dense(node->src[0]) ||
        !ggml_backend_hrx_is_f32_dense(node)) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "SUM_ROWS";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous_row_reduction"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "sum_rows", "ncols", node->src[0]->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "sum_rows", "nrows", ggml_backend_hrx_tensor_row_count(node->src[0]));
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "sum_rows", "src0_row_stride",
        ggml_backend_hrx_tensor_row_stride_elements(node->src[0]));
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_soft_max_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_SOFT_MAX || !node->src[0] ||
        !ggml_backend_hrx_is_f32_dense(node->src[0]) ||
        !ggml_backend_hrx_is_f32_dense(node)) {
        return false;
    }
    const ggml_tensor * mask = node->src[1];
    if (mask && !ggml_backend_hrx_is_f32_dense(mask)) {
        return false;
    }
    // src[2] carries attention sinks and op_params[1] carries the ALiBi
    // max_bias. Every catalog route advertises sinks=none / max_bias=0 and no
    // Loom softmax kernel implements either, but both facts used to be emitted
    // as unconditional literals below. That made the route match anyway and
    // the sink / slope contribution was silently dropped from the result.
    if (node->src[2] != nullptr) {
        return false;
    }
    float soft_max_bias = 0.0f;
    std::memcpy(&soft_max_bias, reinterpret_cast<const char *>(node->op_params) + sizeof(float),
                sizeof(float));
    if (soft_max_bias != 0.0f) {
        return false;
    }

    out_request->problem = {};
    out_request->problem.op = "SOFT_MAX";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"src1_type", mask ? ggml_backend_hrx_catalog_type_name(mask->type) : "none"},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"mask", mask ? "required" : "none"},
        {"max_bias", "0"},
        {"sinks", "none"},
        {"layout", "contiguous"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "ncols", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "nrows", ggml_backend_hrx_tensor_row_count(node));
    out_request->problem.shape["rows"] = 0;
    out_request->problem.shape["cols"] = 0;
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "ne01", node->src[0]->ne[1]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "ne02", node->src[0]->ne[2]);
    // The Loom softmax kernels view the mask as f32 elements, so the mask
    // strides must be expressed in elements. Passing ggml byte strides here
    // makes live attention read the wrong causal-mask rows.
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "mask_nb1", mask ? mask->nb[1] / sizeof(float) : 0);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "mask_nb2", mask ? mask->nb[2] / sizeof(float) : 0);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "mask_nb3", mask ? mask->nb[3] / sizeof(float) : 0);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "mask_ne1", mask ? mask->ne[1] : 0);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "mask_ne2", mask ? mask->ne[2] : 0);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "mask_ne3", mask ? mask->ne[3] : 0);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    if (mask) {
        ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", mask);
    }
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = mask ? std::vector<const ggml_tensor *>{node->src[0], mask, node} :
        std::vector<const ggml_tensor *>{node->src[0], node};
    out_request->constants.clear();
    float scale = 0.0f;
    std::memcpy(&scale, node->op_params, sizeof(float));
    ggml_backend_hrx_append_constant(&out_request->constants, scale);
    return true;
}

static bool ggml_backend_hrx_make_argsort_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_ARGSORT || !node->src[0] ||
        node->type != GGML_TYPE_I32 ||
        node->op_params[0] != static_cast<int32_t>(GGML_SORT_ORDER_DESC) ||
        !ggml_backend_hrx_is_f32_dense(node->src[0]) ||
        !ggml_is_contiguous(node)) {
        return false;
    }
    ggml_backend_hrx_topk_moe_softmax_norm_match match = {};
    ggml_backend_hrx_dispatch_request stage = {};
    if (ggml_backend_hrx_find_current_topk_moe_match(
            device_context, node, &match) &&
        ggml_backend_hrx_make_available_topk_moe_requests(
            device_context, match, &stage, nullptr)) {
        *out_request = std::move(stage);
        return true;
    }
    out_request->problem = {};
    out_request->problem.op = "ARGSORT";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"order", "DESC"},
        {"layout", "contiguous_rows"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "argsort", "ncols", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "argsort", "nrows", ggml_backend_hrx_tensor_row_count(node));
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_get_rows_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_GET_ROWS ||
        !node->src[0] || !node->src[1] || node->src[1]->type != GGML_TYPE_I32 ||
        !ggml_backend_hrx_is_row_contiguous(node->src[0]) ||
        !ggml_backend_hrx_is_f32_dense(node)) {
        return false;
    }
    const bool quantized_src = ggml_is_quantized(node->src[0]->type);
    if (node->src[0]->type != GGML_TYPE_F32 && !quantized_src) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "GET_ROWS";
    out_request->problem.target_key = device_context->architecture;
    const bool moe_weights_view =
        node->src[0]->type == GGML_TYPE_F32 &&
        node->src[0]->ne[0] == 1 &&
        node->src[0]->ne[2] == node->ne[2] &&
        node->src[0]->ne[3] == node->ne[3] &&
        node->src[1]->ne[0] == node->ne[1] &&
        node->src[1]->ne[1] == node->ne[2] &&
        node->src[1]->ne[2] == node->ne[3] &&
        node->ne[0] == 1 &&
        node->ne[1] > 0 &&
        node->ne[2] > 0;
    // ggml gathers per (i11, i12) slice of src0, so dims 2 and 3 are real work,
    // not padding. Kernels that flatten to ncols x nrows only fill the first
    // slice; they are gated off batched shapes by this key rather than silently
    // computing them, and the batch-aware gather is the one left matching.
    // The MoE gather is excluded: its dst is [1, topk, ntokens], so ne[2] is
    // always > 1, and that kernel already walks tokens with explicit strides.
    const bool batched = !moe_weights_view && node->ne[2] * node->ne[3] > 1;
    // The batched gather folds (i11, i12) into a row offset, which only holds
    // when a slice is a packed block of whole rows.
    if (batched && !ggml_is_contiguous(node->src[0])) {
        return false;
    }
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(node->src[1]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", moe_weights_view ? "moe_weights_topk_view" :
            (quantized_src ? "quantized_embedding_rows_1d" : "embedding_rows_1d")},
        {"batched", batched ? "true" : "false"},
    };
    if (moe_weights_view) {
        // Route domains for the MoE gather are keyed by the logical output
        // matrix [topk, tokens], while the underlying ggml_get_rows tensor is
        // shaped [1, topk, tokens]. Keep both fact families available without
        // letting the standard get_rows aliases overwrite the route keys.
        out_request->problem.shape["ncols"] = node->ne[1];
        out_request->problem.shape["nrows"] = node->ne[2];
        out_request->problem.shape["get_rows.ncols"] = node->ne[0];
        out_request->problem.shape["get_rows.nrows"] = node->ne[1];
    } else {
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "get_rows", "ncols", node->ne[0]);
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "get_rows", "nrows", node->ne[1]);
    }
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "get_rows", "src0_nrows", node->src[0]->ne[1]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "get_rows_moe", "nexperts", node->src[0]->ne[1]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "get_rows_moe", "nselected", node->src[1]->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "get_rows_moe", "ntokens", node->src[1]->ne[1]);
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "get_rows_moe", "src0_token_stride",
        static_cast<int64_t>(node->src[0]->nb[2] / sizeof(float)));
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "get_rows_moe", "idx_token_stride",
        static_cast<int64_t>(node->src[1]->nb[1] / sizeof(int32_t)));
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "get_rows_moe", "dst_token_stride",
        static_cast<int64_t>(node->nb[2] / sizeof(float)));
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "get_rows", "idx_row_stride",
        node->src[1]->ne[1] == 1 ? 1 : node->src[1]->nb[1] / sizeof(int32_t));
    if (!quantized_src) {
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "get_rows", "nbatch2", node->ne[2]);
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "get_rows", "nbatch3", node->ne[3]);
        ggml_backend_hrx_set_shape_alias(
            &out_request->problem, "get_rows", "idx_s0",
            static_cast<int64_t>(node->src[1]->nb[0] / sizeof(int32_t)));
        ggml_backend_hrx_set_shape_alias(
            &out_request->problem, "get_rows", "idx_s1",
            static_cast<int64_t>(node->src[1]->nb[1] / sizeof(int32_t)));
        ggml_backend_hrx_set_shape_alias(
            &out_request->problem, "get_rows", "idx_s2",
            static_cast<int64_t>(node->src[1]->nb[2] / sizeof(int32_t)));
    }
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", node->src[1]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node->src[1], node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_set_rows_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_SET_ROWS ||
        !node->src[0] || !node->src[1] || !node->src[2] ||
        node->src[0]->type != GGML_TYPE_F32 ||
        (node->src[1]->type != GGML_TYPE_I64 && node->src[1]->type != GGML_TYPE_I32) ||
        !ggml_backend_hrx_is_f32_dense(node->src[0]) ||
        !ggml_is_contiguous(node->src[1]) ||
        !ggml_backend_hrx_is_row_contiguous(node)) {
        return false;
    }

    auto make_cont_set_rows = [&]() -> bool {
        const ggml_tensor * cont = ggml_backend_hrx_zero_offset_source_chain_target(node->src[0], GGML_OP_CONT);
        if (!cont || !cont->src[0] || cont->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F16 ||
            node->src[1]->type != GGML_TYPE_I64 ||
            !ggml_backend_hrx_is_f32_dense(cont->src[0]) ||
            !ggml_backend_hrx_is_f32_dense(cont) ||
            cont->view_src != nullptr ||
            ggml_nelements(node->src[0]) != ggml_nelements(cont)) {
            return false;
        }

        ggml_backend_hrx_dispatch_request request = {};
        request.problem.op = "SET_ROWS";
        request.problem.target_key = device_context->architecture;
        request.problem.supports = {
            {"src0_type", ggml_backend_hrx_catalog_type_name(cont->type)},
            {"src1_type", ggml_backend_hrx_catalog_type_name(node->src[1]->type)},
            {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
            {"layout", "cont_src0_to_reshape_set_rows"},
            {"cont_ncols", std::to_string(cont->ne[0])},
        };
        ggml_backend_hrx_set_shape_alias(&request.problem, "cont", "ncols", cont->ne[0]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "cont", "ne1", cont->ne[1]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "cont", "ne2", cont->ne[2]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "cont", "src_nb1", cont->src[0]->nb[1] / sizeof(float));
        ggml_backend_hrx_set_shape_alias(&request.problem, "cont", "src_nb2", cont->src[0]->nb[2] / sizeof(float));
        ggml_backend_hrx_set_shape_alias(&request.problem, "cont", "src_nb3", cont->src[0]->nb[3] / sizeof(float));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "nc", node->ne[0]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "nr", node->src[0]->ne[1]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "ne02", node->src[0]->ne[2]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "ne03", node->src[0]->ne[3]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "ne1", node->ne[1]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "ne11", node->src[1]->ne[1]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "ne12", node->src[1]->ne[2]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "src0_nb1", node->src[0]->nb[1] / sizeof(float));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "src0_nb2", node->src[0]->nb[2] / sizeof(float));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "src0_nb3", node->src[0]->nb[3] / sizeof(float));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "idx_nb0", node->src[1]->nb[0] / sizeof(int64_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "idx_nb1", node->src[1]->nb[1] / sizeof(int64_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "idx_nb2", node->src[1]->nb[2] / sizeof(int64_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "dst_nb1", node->nb[1] / sizeof(ggml_fp16_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "dst_nb2", node->nb[2] / sizeof(ggml_fp16_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "dst_nb3", node->nb[3] / sizeof(ggml_fp16_t));
        request.problem.shape["ncols"] = node->ne[0];
        request.problem.shape["nrows"] = node->src[0]->ne[1];
        ggml_backend_hrx_add_tensor_facts(&request.problem, "src0", cont->src[0]);
        ggml_backend_hrx_add_tensor_facts(&request.problem, "src1", node->src[1]);
        ggml_backend_hrx_add_tensor_facts(&request.problem, "dst", node);
        request.tensors = {cont->src[0], node->src[1], node};
        request.constants.clear();
        if (!ggml_backend_hrx_request_matches_loaded_route(device_context, request, "cont_set_rows_f32")) {
            return false;
        }
        *out_request = std::move(request);
        return true;
    };

    auto make_rope_set_rows = [&]() -> bool {
        const ggml_tensor * rope = ggml_backend_hrx_zero_offset_source_chain_target(node->src[0], GGML_OP_ROPE);
        if (!rope || !rope->src[0] || !rope->src[1] || !rope->src[2] ||
            rope->src[0]->type != GGML_TYPE_F32 || rope->src[1]->type != GGML_TYPE_I32 ||
            rope->src[2]->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32 ||
            node->src[1]->type != GGML_TYPE_I64 || node->type != GGML_TYPE_F16 ||
            !ggml_backend_hrx_is_f32_dense(rope->src[0]) ||
            !ggml_is_contiguous(rope->src[1]) ||
            !ggml_is_contiguous(rope->src[2]) ||
            !ggml_backend_hrx_is_f32_dense(rope) ||
            rope->view_src != nullptr ||
            node->src[0]->ne[0] != rope->ne[0] * rope->ne[1] ||
            node->src[0]->ne[1] != rope->ne[2] ||
            node->src[0]->ne[2] != 1 ||
            node->src[0]->ne[3] != 1 ||
            node->src[0]->nb[0] != sizeof(float) ||
            node->src[0]->nb[1] != rope->nb[2] ||
            node->ne[0] != rope->ne[0] * rope->ne[1]) {
            return false;
        }

        int32_t n_dims = 0;
        int32_t mode = 0;
        float freq_base = 10000.0f;
        float freq_scale = 1.0f;
        float ext_factor = 0.0f;
        float attn_factor = 1.0f;
        std::memcpy(&n_dims, reinterpret_cast<const uint8_t *>(rope->op_params) + sizeof(int32_t), sizeof(int32_t));
        std::memcpy(&mode, reinterpret_cast<const uint8_t *>(rope->op_params) + 2 * sizeof(int32_t), sizeof(int32_t));
        std::memcpy(&freq_base, reinterpret_cast<const uint8_t *>(rope->op_params) + 5 * sizeof(float), sizeof(float));
        std::memcpy(&freq_scale, reinterpret_cast<const uint8_t *>(rope->op_params) + 6 * sizeof(float), sizeof(float));
        std::memcpy(&ext_factor, reinterpret_cast<const uint8_t *>(rope->op_params) + 7 * sizeof(float), sizeof(float));
        std::memcpy(&attn_factor, reinterpret_cast<const uint8_t *>(rope->op_params) + 8 * sizeof(float), sizeof(float));
        if (ext_factor != 0.0f || (mode & ~(GGML_ROPE_TYPE_NEOX)) != 0) {
            return false;
        }

        ggml_backend_hrx_dispatch_request request = {};
        request.problem.op = "SET_ROWS";
        request.problem.target_key = device_context->architecture;
        request.problem.supports = {
            {"src0_type", ggml_backend_hrx_catalog_type_name(rope->src[0]->type)},
            {"src1_type", ggml_backend_hrx_catalog_type_name(rope->src[1]->type)},
            {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
            {"mode", (mode & GGML_ROPE_TYPE_NEOX) ? "NEOX" : "NORMAL"},
            {"freq_factor", "present"},
            {"ext_factor", "0"},
            {"layout", "rope-view-set-rows-f16"},
        };
        const int64_t nheads = rope->src[0]->ne[1];
        const int64_t ntokens = rope->src[0]->ne[2];
        ggml_backend_hrx_set_shape_alias(&request.problem, "rope", "ncols", rope->src[0]->ne[0]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "rope", "n_dims", n_dims);
        ggml_backend_hrx_set_shape_alias(&request.problem, "rope", "nheads", nheads);
        ggml_backend_hrx_set_shape_alias(&request.problem, "rope", "ntokens", ntokens);
        ggml_backend_hrx_set_shape_alias(&request.problem, "rope", "src0_head_stride", rope->src[0]->nb[1] / sizeof(float));
        ggml_backend_hrx_set_shape_alias(&request.problem, "rope", "src0_token_stride", rope->src[0]->nb[2] / sizeof(float));
        ggml_backend_hrx_set_shape_alias(&request.problem, "rope", "pos_token_stride", rope->src[1]->nb[0] / sizeof(int32_t));
        request.problem.shape["ncols"] = rope->src[0]->ne[0];
        request.problem.shape["nrows"] = nheads * ntokens;
        request.problem.shape["n_dims"] = n_dims;
        request.problem.shape["rows"] = nheads;
        request.problem.shape["cols"] = ntokens;
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "ne1", node->ne[1]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "ne11", node->src[1]->ne[1]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "ne12", node->src[1]->ne[2]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "idx_nb0", node->src[1]->nb[0] / sizeof(int64_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "idx_nb1", node->src[1]->nb[1] / sizeof(int64_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "idx_nb2", node->src[1]->nb[2] / sizeof(int64_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "dst_nb1", node->nb[1] / sizeof(ggml_fp16_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "dst_nb2", node->nb[2] / sizeof(ggml_fp16_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "dst_nb3", node->nb[3] / sizeof(ggml_fp16_t));
        ggml_backend_hrx_add_tensor_facts(&request.problem, "src0", rope->src[0]);
        ggml_backend_hrx_add_tensor_facts(&request.problem, "src1", rope->src[1]);
        ggml_backend_hrx_add_tensor_facts(&request.problem, "src2", rope->src[2]);
        ggml_backend_hrx_add_tensor_facts(&request.problem, "idx", node->src[1]);
        ggml_backend_hrx_add_tensor_facts(&request.problem, "dst", node);
        request.tensors = {rope->src[0], rope->src[1], rope->src[2], node->src[1], node};
        request.constants.clear();
        ggml_backend_hrx_append_constant<float>(&request.constants, std::pow(freq_base, -2.0f / static_cast<float>(n_dims)));
        ggml_backend_hrx_append_constant<float>(&request.constants, freq_scale);
        ggml_backend_hrx_append_constant<float>(&request.constants, attn_factor);
        if (!ggml_backend_hrx_request_matches_loaded_route(device_context, request, "rope_set_rows_f32")) {
            return false;
        }
        *out_request = std::move(request);
        return true;
    };

    if (make_rope_set_rows() || make_cont_set_rows()) {
        return true;
    }

    const ggml_tensor * src = node->src[0];
    const ggml_tensor * idx = node->src[1];
    out_request->problem = {};
    out_request->problem.op = "SET_ROWS";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(src->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(idx->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous_rows"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "nc", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "nr", src->ne[1]);
    out_request->problem.shape["ncols"] = node->ne[0];
    out_request->problem.shape["nrows"] = src->ne[1];
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "ne02", src->ne[2]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "ne03", src->ne[3]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "ne1", node->ne[1]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "ne11", idx->ne[1]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "ne12", idx->ne[2]);
    const size_t dst_element_size = node->type == GGML_TYPE_F16 ? sizeof(ggml_fp16_t) : sizeof(float);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "src0_nb1", src->nb[1] / sizeof(float));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "src0_nb2", src->nb[2] / sizeof(float));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "src0_nb3", src->nb[3] / sizeof(float));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "idx_nb0", idx->nb[0] / sizeof(int64_t));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "idx_nb1", idx->nb[1] / sizeof(int64_t));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "idx_nb2", idx->nb[2] / sizeof(int64_t));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "dst_nb1", node->nb[1] / dst_element_size);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "dst_nb2", node->nb[2] / dst_element_size);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "dst_nb3", node->nb[3] / dst_element_size);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", src);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", idx);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src2", node->src[2]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {src, idx, node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_mul_mat_f16_cont_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_CONT ||
        !node->src[0] || node->src[0]->op != GGML_OP_PERMUTE ||
        !node->src[0]->src[0] || node->src[0]->src[0]->op != GGML_OP_MUL_MAT ||
        !node->src[0]->src[0]->src[0] || !node->src[0]->src[0]->src[1] ||
        node->type != GGML_TYPE_F32 || node->view_src != nullptr ||
        !ggml_is_contiguous(node)) {
        return false;
    }
    const ggml_tensor * permute = node->src[0];
    const ggml_tensor * mul_mat = permute->src[0];
    if (ggml_get_op_params_i32(permute, 0) != 0 ||
        ggml_get_op_params_i32(permute, 1) != 2 ||
        ggml_get_op_params_i32(permute, 2) != 1 ||
        ggml_get_op_params_i32(permute, 3) != 3 ||
        !ggml_backend_hrx_make_mul_mat_problem(device_context, mul_mat, &out_request->problem)) {
        return false;
    }
    const int64_t rows = mul_mat->src[0]->ne[1];
    const int64_t cols = mul_mat->src[1]->ne[1];
    if (mul_mat->src[0]->type != GGML_TYPE_F16 ||
        mul_mat->src[1]->type != GGML_TYPE_F32 ||
        mul_mat->type != GGML_TYPE_F32 ||
        permute->ne[0] != mul_mat->ne[0] ||
        permute->ne[1] != mul_mat->ne[2] ||
        permute->ne[2] != mul_mat->ne[1] ||
        permute->ne[3] != mul_mat->ne[3] ||
        node->ne[0] != rows * mul_mat->ne[2] ||
        node->ne[1] != cols ||
        node->ne[2] != mul_mat->ne[3] ||
        node->ne[3] != 1) {
        return false;
    }

    out_request->problem.op = "CONT";
    out_request->problem.supports["layout"] = "decode_kqv_permute_contiguous_noalias";
    out_request->tensors = {mul_mat->src[0], mul_mat->src[1], node};
    out_request->constants.clear();
    if (!ggml_backend_hrx_request_matches_loaded_route(device_context, *out_request, "mul_mat_f16_f32_batched_cont")) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx_make_softmax_kqv_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_CONT ||
        !node->src[0] || node->src[0]->op != GGML_OP_PERMUTE ||
        !node->src[0]->src[0] || node->src[0]->src[0]->op != GGML_OP_MUL_MAT ||
        !node->src[0]->src[0]->src[0] || !node->src[0]->src[0]->src[1] ||
        node->src[0]->src[0]->src[1]->op != GGML_OP_SOFT_MAX ||
        !node->src[0]->src[0]->src[1]->src[0] ||
        !node->src[0]->src[0]->src[1]->src[1] ||
        node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node)) {
        return false;
    }
    const ggml_tensor * permute = node->src[0];
    const ggml_tensor * kqv = permute->src[0];
    const ggml_tensor * softmax = kqv->src[1];
    const ggml_tensor * kq = softmax->src[0];
    const ggml_tensor * mask = softmax->src[1];
    const ggml_tensor * v = kqv->src[0];
    if (ggml_get_op_params_i32(permute, 0) != 0 ||
        ggml_get_op_params_i32(permute, 1) != 2 ||
        ggml_get_op_params_i32(permute, 2) != 1 ||
        ggml_get_op_params_i32(permute, 3) != 3 ||
        kq->type != GGML_TYPE_F32 ||
        mask->type != GGML_TYPE_F32 ||
        softmax->type != GGML_TYPE_F32 ||
        v->type != GGML_TYPE_F16 ||
        kqv->type != GGML_TYPE_F32 ||
        kqv->src[1] != softmax ||
        !ggml_is_contiguous(kq) ||
        !ggml_is_contiguous(mask) ||
        !ggml_is_contiguous(v)) {
        return false;
    }
    const int64_t kv = kq->ne[0];
    const int64_t n = kq->ne[1];
    const int64_t nheads = kq->ne[2];
    const int64_t d = v->ne[1];
    if (kq->ne[3] != 1 ||
        mask->ne[0] != kv || mask->ne[1] != n || mask->ne[2] != nheads || mask->ne[3] != 1 ||
        v->ne[0] != kv ||
        kqv->ne[0] != d || kqv->ne[1] != n || kqv->ne[2] != nheads || kqv->ne[3] != 1 ||
        node->ne[0] != d * nheads || node->ne[1] != n || node->ne[2] != 1 || node->ne[3] != 1) {
        return false;
    }

    ggml_backend_hrx_dispatch_request request = {};
    request.problem.op = "CONT";
    request.problem.target_key = device_context->architecture;
    request.problem.supports = {
        {"kq_type", ggml_backend_hrx_catalog_type_name(kq->type)},
        {"mask_type", ggml_backend_hrx_catalog_type_name(mask->type)},
        {"v_type", ggml_backend_hrx_catalog_type_name(v->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "phi4_decode_kq_softmax_kqv_permute_cont"},
        {"fusion", "SOFT_MAX + MUL_MAT(F16xF32) + PERMUTE + CONT"},
    };
    request.problem.shape = {
        {"k", kv},
        {"rows", n},
        {"cols", nheads},
    };
    ggml_backend_hrx_add_tensor_facts(&request.problem, "src0", kq);
    ggml_backend_hrx_add_tensor_facts(&request.problem, "src1", mask);
    ggml_backend_hrx_add_tensor_facts(&request.problem, "src2", v);
    ggml_backend_hrx_add_tensor_facts(&request.problem, "dst", node);
    request.tensors = {kq, mask, v, node};
    request.constants.clear();
    float scale = 1.0f;
    std::memcpy(&scale, softmax->op_params, sizeof(scale));
    ggml_backend_hrx_append_constant<float>(&request.constants, scale);
    if (!ggml_backend_hrx_request_matches_loaded_route(device_context, request, "softmax_kqv_f32_f16")) {
        return false;
    }
    *out_request = std::move(request);
    return true;
}

// Gather a strided f32 source into a flat contiguous destination of the same
// element count. Shared by CONT and CPY: ggml labels the operation differently
// depending on how the graph was built, but the kernel is the same. llama.cpp
// expresses the GDN conv-state window shift as a CPY from a strided view
// (ne=[3,8192], nb1=16 -- three floats out of every four) into a flat buffer,
// which is exactly the flattening ggml_cont_2d case.
static bool ggml_backend_hrx_make_strided_flatten_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * src,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    // src may be strided: this kernel receives s0..s3 and applies them. dst must
    // be dense, since it is written by flat linear index.
    if (!device_context || !out_request || !src || !node ||
        !ggml_backend_hrx_is_f32_row_contiguous(src) ||
        !ggml_backend_hrx_is_f32_dense(node) ||
        ggml_nelements(src) != ggml_nelements(node)) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (src->nb[i] % sizeof(float) != 0) {
            return false;
        }
    }
    out_request->problem = {};
    out_request->problem.op = "CONT";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(src->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "strided_src_to_flat_contiguous_dst"},
    };
    auto putc = [&](const char * key, int64_t value) {
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "contx", key, value);
    };
    putc("ne0", src->ne[0]);
    putc("ne1", src->ne[1]);
    putc("ne2", src->ne[2]);
    putc("ne3", src->ne[3]);
    putc("s0", static_cast<int64_t>(src->nb[0] / sizeof(float)));
    putc("s1", static_cast<int64_t>(src->nb[1] / sizeof(float)));
    putc("s2", static_cast<int64_t>(src->nb[2] / sizeof(float)));
    putc("s3", static_cast<int64_t>(src->nb[3] / sizeof(float)));
    out_request->problem.shape["ncols"] = src->ne[0];
    out_request->problem.shape["nrows"] = ggml_nelements(src) / src->ne[0];
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", src);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {src, node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_cont_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (ggml_backend_hrx_make_softmax_kqv_request(device_context, node, out_request)) {
        return true;
    }
    if (ggml_backend_hrx_make_mul_mat_f16_cont_request(device_context, node, out_request)) {
        return true;
    }
    if (!device_context || !out_request || !node || node->op != GGML_OP_CONT || !node->src[0] ||
        !ggml_backend_hrx_is_f32_dense(node)) {
        return false;
    }
    // The generic Loom CONT kernel indexes i0 directly across node->ne[0] and
    // reads its source densely, so hand anything else to the strided variant,
    // which decomposes the linear index and applies s0..s3 independently. That
    // covers both a flattened leading dimension (ggml_cont_2d over the gate half
    // of the fused QG projection) and a source view with padding between rows or
    // planes, which is what a per-head slice of a fused QKV projection is.
    if (node->src[0]->ne[0] != node->ne[0] ||
        !ggml_backend_hrx_is_f32_dense(node->src[0])) {
        return ggml_backend_hrx_make_strided_flatten_request(
            device_context, node->src[0], node, out_request);
    }
    out_request->problem = {};
    out_request->problem.op = "CONT";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "row_contiguous_src_to_contiguous_dst"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "cont", "ncols", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "cont", "nrows", ggml_backend_hrx_tensor_row_count(node));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "cont", "ne1", node->ne[1]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "cont", "ne2", node->ne[2]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "cont", "src_nb1", node->src[0]->nb[1] / sizeof(float));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "cont", "src_nb2", node->src[0]->nb[2] / sizeof(float));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "cont", "src_nb3", node->src[0]->nb[3] / sizeof(float));
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_cpy_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_CPY || !node->src[0]) {
        return false;
    }
    if (!ggml_is_contiguous(node->src[0]) || !ggml_is_contiguous(node)) {
        // A same-count copy out of a strided view into a flat buffer is a
        // flattening CONT; reuse that kernel rather than refusing the node,
        // which would strand the recurrent-state cache on another backend.
        if (ggml_backend_hrx_make_strided_flatten_request(
                device_context, node->src[0], node, out_request)) {
            return true;
        }
        ggml_backend_hrx_trace_event(device_context->reg_context, {
            {"event", "cpy_rejected"},
            {"reason", "non_contiguous"},
            {"node", ggml_get_name(node)},
            {"src0_contiguous", ggml_is_contiguous(node->src[0])},
            {"dst_contiguous", ggml_is_contiguous(node)},
            {"src0_type", ggml_type_name(node->src[0]->type)},
            {"dst_type", ggml_type_name(node->type)},
            {"src0_ne", {node->src[0]->ne[0], node->src[0]->ne[1], node->src[0]->ne[2], node->src[0]->ne[3]}},
            {"src0_nb", {node->src[0]->nb[0], node->src[0]->nb[1], node->src[0]->nb[2], node->src[0]->nb[3]}},
            {"dst_ne", {node->ne[0], node->ne[1], node->ne[2], node->ne[3]}},
            {"dst_nb", {node->nb[0], node->nb[1], node->nb[2], node->nb[3]}},
        });
        return false;
    }
    if (node->src[0]->type == GGML_TYPE_F32 && node->type == GGML_TYPE_Q8_1) {
        out_request->problem = {};
        out_request->problem.op = "QUANTIZE";
        out_request->problem.target_key = device_context->architecture;
        out_request->problem.supports = {
            {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
            {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
            {"layout", "contiguous"},
        };
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "q8_1", "blocks", node->src[0]->ne[0] / ggml_blck_size(GGML_TYPE_Q8_1));
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "q8_1", "ne1", node->src[0]->ne[1]);
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "q8_1", "z_count", node->src[0]->ne[2] * node->src[0]->ne[3]);
        out_request->tensors = {node->src[0], node};
        out_request->constants.clear();
        ggml_backend_hrx_append_constant<int32_t>(&out_request->constants, static_cast<int32_t>(node->src[0]->ne[0]));
        ggml_backend_hrx_append_constant<int32_t>(&out_request->constants, static_cast<int32_t>(node->src[0]->nb[1] / sizeof(float)));
        ggml_backend_hrx_append_constant<int32_t>(&out_request->constants, static_cast<int32_t>(node->src[0]->nb[2] / sizeof(float)));
        ggml_backend_hrx_append_constant<int32_t>(&out_request->constants, static_cast<int32_t>(node->src[0]->nb[3] / sizeof(float)));
        ggml_backend_hrx_append_constant<int32_t>(&out_request->constants, static_cast<int32_t>(node->ne[0]));
        ggml_backend_hrx_append_constant<int32_t>(&out_request->constants, static_cast<int32_t>(node->ne[1]));
        ggml_backend_hrx_append_constant<int32_t>(&out_request->constants, static_cast<int32_t>(node->ne[2]));
        ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
        ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
        return true;
    }
    // A same-type copy is a CONT by another name. The catalog carries only an
    // f32->f16 CPY route, so routing these to the CONT kernels is what lets the
    // recurrent-state write-back (cache_s_l0 <- new_state, 524288 f32) run on
    // HRX instead of aborting the scheduler.
    if (node->src[0]->type == node->type &&
        ggml_backend_hrx_make_strided_flatten_request(
            device_context, node->src[0], node, out_request)) {
        return true;
    }
    out_request->problem = {};
    out_request->problem.op = "CPY";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous_src_to_contiguous_dst"},
    };
    out_request->problem.shape["ncols"] = ggml_nelements(node);
    out_request->problem.shape["nrows"] = 1;
    out_request->problem.shape["copy.n"] = ggml_nelements(node);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_rope_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_ROPE ||
        !node->src[0] || !node->src[1] || node->src[0]->type != GGML_TYPE_F32 ||
        node->src[1]->type != GGML_TYPE_I32 || node->type != GGML_TYPE_F32 ||
        !ggml_backend_hrx_is_f32_dense(node->src[0]) ||
        !ggml_is_contiguous(node->src[1]) ||
        !ggml_backend_hrx_is_f32_dense(node)) {
        return false;
    }
    const ggml_tensor * freq = node->src[2];
    if (freq && (freq->type != GGML_TYPE_F32 || !ggml_is_contiguous(freq))) {
        return false;
    }
    int32_t n_dims = 0;
    int32_t mode = 0;
    float freq_base = 10000.0f;
    float freq_scale = 1.0f;
    float ext_factor = 0.0f;
    float attn_factor = 1.0f;
    std::memcpy(&n_dims, reinterpret_cast<const uint8_t *>(node->op_params) + sizeof(int32_t), sizeof(int32_t));
    std::memcpy(&mode, reinterpret_cast<const uint8_t *>(node->op_params) + 2 * sizeof(int32_t), sizeof(int32_t));
    std::memcpy(&freq_base, reinterpret_cast<const uint8_t *>(node->op_params) + 5 * sizeof(float), sizeof(float));
    std::memcpy(&freq_scale, reinterpret_cast<const uint8_t *>(node->op_params) + 6 * sizeof(float), sizeof(float));
    std::memcpy(&ext_factor, reinterpret_cast<const uint8_t *>(node->op_params) + 7 * sizeof(float), sizeof(float));
    std::memcpy(&attn_factor, reinterpret_cast<const uint8_t *>(node->op_params) + 8 * sizeof(float), sizeof(float));
    if (ext_factor != 0.0f) {
        return false;
    }
    // Multi-section rope (mrope / imrope) selects the position component per
    // sector instead of using a single position, so it needs its own kernel.
    const bool is_mrope  = (mode & GGML_ROPE_TYPE_MROPE) == GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = (mode & GGML_ROPE_TYPE_IMROPE) == GGML_ROPE_TYPE_IMROPE;
    if (is_mrope || is_imrope) {
        // GGML_ROPE_TYPE_VISION additionally halves n_dims per section; not
        // handled by the kernel below.
        if ((mode & GGML_ROPE_TYPE_VISION) == GGML_ROPE_TYPE_VISION || freq) {
            return false;
        }
        int32_t sections[4] = { 0, 0, 0, 0 };
        std::memcpy(sections, reinterpret_cast<const uint8_t *>(node->op_params) + 11 * sizeof(int32_t),
                    4 * sizeof(int32_t));
        if (sections[0] + sections[1] + sections[2] + sections[3] <= 0) {
            return false;
        }
        const ggml_tensor * src = node->src[0];
        out_request->problem = {};
        out_request->problem.op = "ROPE";
        out_request->problem.target_key = device_context->architecture;
        out_request->problem.supports = {
            {"src0_type", ggml_backend_hrx_catalog_type_name(src->type)},
            {"src1_type", ggml_backend_hrx_catalog_type_name(node->src[1]->type)},
            {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
            {"mode", is_imrope ? "IMROPE" : "MROPE"},
            {"freq_factor", "none"},
            {"ext_factor", "0"},
            {"layout", "multi_section_neox"},
        };
        auto put = [&](const char * key, int64_t value) {
            ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", key, value);
        };
        put("ne00", src->ne[0]);
        put("ne01", src->ne[1]);
        put("ne02", src->ne[2]);
        put("ne03", src->ne[3]);
        put("s01", static_cast<int64_t>(src->nb[1] / sizeof(float)));
        put("s02", static_cast<int64_t>(src->nb[2] / sizeof(float)));
        put("s03", static_cast<int64_t>(src->nb[3] / sizeof(float)));
        put("d1", static_cast<int64_t>(node->nb[1] / sizeof(float)));
        put("d2", static_cast<int64_t>(node->nb[2] / sizeof(float)));
        put("d3", static_cast<int64_t>(node->nb[3] / sizeof(float)));
        put("n_dims", n_dims);
        put("sec0", sections[0]);
        put("sec1", sections[1]);
        put("sec2", sections[2]);
        put("sec3", sections[3]);
        put("imrope", is_imrope ? 1 : 0);
        out_request->problem.shape["ncols"] = src->ne[0];
        out_request->problem.shape["nrows"] = src->ne[1] * src->ne[2] * src->ne[3];
        ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", src);
        ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
        out_request->tensors = {src, node->src[1], node};
        out_request->constants.clear();
        const float theta_scale = powf(freq_base, -2.0f / static_cast<float>(n_dims));
        ggml_backend_hrx_append_constant(&out_request->constants, theta_scale);
        ggml_backend_hrx_append_constant(&out_request->constants, freq_scale);
        ggml_backend_hrx_append_constant(&out_request->constants, attn_factor);
        return true;
    }
    if ((mode & ~(GGML_ROPE_TYPE_NEOX)) != 0) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = freq_scale == 1.0f ? "ROPE" : "ROPE_SCALE";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(node->src[1]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"mode", (mode & GGML_ROPE_TYPE_NEOX) ? "NEOX" : "NORMAL"},
        {"freq_factor", freq ? "src2" : "none"},
        {"ext_factor", "0"},
        {"layout", "element-strided-head-token-ne3-1"},
    };
    if (freq) {
        out_request->problem.supports["src2_type"] = ggml_backend_hrx_catalog_type_name(freq->type);
    }
    const int64_t ntokens = node->src[0]->ne[2];
    const int64_t nheads = node->src[0]->ne[1];
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", "ncols", node->src[0]->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", "n_dims", n_dims);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", "nheads", nheads);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", "ntokens", ntokens);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", "src0_head_stride", node->src[0]->nb[1] / sizeof(float));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", "src0_token_stride", node->src[0]->nb[2] / sizeof(float));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", "dst_head_stride", node->nb[1] / sizeof(float));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", "dst_token_stride", node->nb[2] / sizeof(float));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", "pos_token_stride", node->src[1]->nb[0] / sizeof(int32_t));
    out_request->problem.shape["ncols"] = node->src[0]->ne[0];
    out_request->problem.shape["nrows"] = nheads * ntokens;
    out_request->problem.shape["n_dims"] = n_dims;
    out_request->problem.shape["rows"] = nheads;
    out_request->problem.shape["cols"] = ntokens;
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", node->src[1]);
    if (freq) {
        ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src2", freq);
    }
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = freq ?
        std::vector<const ggml_tensor *>{node->src[0], node->src[1], freq, node} :
        std::vector<const ggml_tensor *>{node->src[0], node->src[1], node};
    out_request->constants.clear();
    // Loom ROPE kernels compute pow(theta_scale, pair), so pass the geometric base.
    ggml_backend_hrx_append_constant<float>(&out_request->constants, std::pow(freq_base, -2.0f / static_cast<float>(n_dims)));
    ggml_backend_hrx_append_constant<float>(&out_request->constants, freq_scale);
    ggml_backend_hrx_append_constant<float>(&out_request->constants, attn_factor);
    if (out_request->problem.op == "ROPE_SCALE") {
        ggml_backend_hrx_append_constant<float>(&out_request->constants, 1.0f);
    }
    return true;
}

static bool ggml_backend_hrx_make_mul_mat_q4_k_swiglu_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_GLU ||
        ggml_get_glu_op(node) != GGML_GLU_OP_SWIGLU ||
        !node->src[0] || !node->src[1] ||
        node->src[0]->op != GGML_OP_MUL_MAT ||
        node->src[1]->op != GGML_OP_MUL_MAT ||
        !node->src[0]->src[0] || !node->src[0]->src[1] ||
        !node->src[1]->src[0] || !node->src[1]->src[1] ||
        node->src[0]->src[1] != node->src[1]->src[1] ||
        node->src[0]->src[0]->type != GGML_TYPE_Q4_K ||
        node->src[1]->src[0]->type != GGML_TYPE_Q4_K ||
        node->src[0]->src[1]->type != GGML_TYPE_F32 ||
        node->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(node->src[0], node->src[1]) ||
        !ggml_are_same_shape(node->src[0], node) ||
        !ggml_is_contiguous(node->src[0]->src[0]) ||
        !ggml_is_contiguous(node->src[1]->src[0]) ||
        !ggml_is_contiguous(node->src[0]->src[1]) ||
        !ggml_is_contiguous(node)) {
        return false;
    }

    const ggml_tensor * x = node->src[0];
    const ggml_tensor * gate = node->src[1];
    const ggml_tensor * rhs = x->src[1];
    const int64_t k = x->src[0]->ne[0];
    const int64_t rows = x->src[0]->ne[1];
    const int64_t cols = rhs->ne[1];
    if (gate->src[0]->ne[0] != k || gate->src[0]->ne[1] != rows ||
        node->ne[0] != rows || ggml_backend_hrx_tensor_row_count(node) != cols) {
        return false;
    }

    ggml_backend_hrx_dispatch_request request = {};
    request.problem.op = "GLU";
    request.problem.target_key = device_context->architecture;
    request.problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(x->src[0]->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(gate->src[0]->type)},
        {"rhs_type", ggml_backend_hrx_catalog_type_name(rhs->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous"},
        {"fusion", "MUL_MAT_Q4_K + MUL_MAT_Q4_K + SWIGLU"},
    };
    request.problem.shape = {
        {"k", k},
        {"rows", rows},
        {"cols", cols},
    };
    ggml_backend_hrx_add_tensor_facts(&request.problem, "src0", x->src[0]);
    ggml_backend_hrx_add_tensor_facts(&request.problem, "src1", rhs);
    ggml_backend_hrx_add_tensor_facts(&request.problem, "gate", gate->src[0]);
    ggml_backend_hrx_add_tensor_facts(&request.problem, "dst", node);
    request.tensors = {x->src[0], gate->src[0], rhs, node};
    request.constants.clear();
    if (!ggml_backend_hrx_request_matches_loaded_route(device_context, request, "mul_mat_q4_k_swiglu_f32")) {
        return false;
    }
    *out_request = std::move(request);
    return true;
}

struct ggml_backend_hrx_mul_mat_id_swiglu_match {
    const ggml_tensor * gate = nullptr;
    const ggml_tensor * up = nullptr;
    const ggml_tensor * down = nullptr;
};

// Exact Qwen MoE gate/up -> SwiGLU topology used by the terminal epilogue.
// Gate remains materialized. The later up projection is computed
// by the GLU dispatch and writes the GLU destination directly, so only up may
// become dead.
static bool ggml_backend_hrx_match_current_mul_mat_id_swiglu(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_mul_mat_id_swiglu_match * out_match) {
    if (!device_context || !device_context->current_graph || !node || !out_match ||
        node->op != GGML_OP_GLU ||
        ggml_get_glu_op(node) != GGML_GLU_OP_SWIGLU ||
        !node->src[0] || !node->src[1] ||
        node->src[0]->op != GGML_OP_MUL_MAT_ID ||
        node->src[1]->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }

    // Split SWIGLU semantics are silu(src0) * src1.
    const ggml_tensor * gate = node->src[0];
    const ggml_tensor * up = node->src[1];
    if (!gate->src[0] || !gate->src[1] || !gate->src[2] ||
        !up->src[0] || !up->src[1] || !up->src[2] ||
        gate->src[1] != up->src[1] ||
        gate->src[2] != up->src[2] ||
        gate->src[0]->type != up->src[0]->type ||
        (up->src[0]->type != GGML_TYPE_Q4_K &&
         up->src[0]->type != GGML_TYPE_Q5_K) ||
        gate->src[1]->type != GGML_TYPE_F32 ||
        gate->src[2]->type != GGML_TYPE_I32 ||
        gate->type != GGML_TYPE_F32 ||
        up->type != GGML_TYPE_F32 ||
        node->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(gate->src[0]) ||
        !ggml_is_contiguous(up->src[0]) ||
        !ggml_is_contiguous(gate->src[1]) ||
        !ggml_is_contiguous(gate) ||
        !ggml_is_contiguous(up) ||
        !ggml_is_contiguous(node) ||
        !ggml_are_same_shape(gate, up) ||
        !ggml_are_same_shape(up, node)) {
        return false;
    }

    const ggml_tensor * rhs = up->src[1];
    const ggml_tensor * ids = up->src[2];
    // Restrict the production route to the measured PP512 shape. Besides
    // keeping the topology proof narrow, these stride checks
    // ensure the gate side-load uses the same dense output index as up.
    if (up->src[0]->ne[0] != 2048 || up->src[0]->ne[1] != 512 ||
        up->src[0]->ne[2] != 256 || up->src[0]->ne[3] != 1 ||
        rhs->ne[0] != 2048 || rhs->ne[1] != 1 ||
        rhs->ne[2] != 512 || rhs->ne[3] != 1 ||
        ids->ne[0] != 8 || ids->ne[1] != 512 ||
        ids->ne[2] != 1 || ids->ne[3] != 1 ||
        node->ne[0] != 512 || node->ne[1] != 8 ||
        node->ne[2] != 512 || node->ne[3] != 1 ||
        rhs->nb[0] != sizeof(float) ||
        rhs->nb[2] != 2048 * sizeof(float) ||
        ids->nb[0] != sizeof(int32_t) ||
        ids->nb[1] != 256 * sizeof(int32_t) ||
        node->nb[0] != sizeof(float) ||
        node->nb[2] != 4096 * sizeof(float)) {
        return false;
    }

    const ggml_cgraph * cgraph = device_context->current_graph;
    int gate_index = -1;
    int up_index = -1;
    int glu_index = -1;
    int down_index = -1;
    int gate_uses = 0;
    int up_uses = 0;
    int glu_uses = 0;
    const ggml_tensor * down = nullptr;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * candidate = cgraph->nodes[i];
        if (!candidate) {
            continue;
        }
        if (candidate == gate) {
            gate_index = i;
        } else if (candidate == up) {
            up_index = i;
        } else if (candidate == node) {
            glu_index = i;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (candidate->src[s] == gate) {
                ++gate_uses;
                if (candidate != node) {
                    return false;
                }
            }
            if (candidate->src[s] == up) {
                ++up_uses;
                if (candidate != node) {
                    return false;
                }
            }
            if (candidate->src[s] == node) {
                ++glu_uses;
                if (candidate->op != GGML_OP_MUL_MAT_ID ||
                    candidate->src[1] != node ||
                    candidate->src[2] != ids ||
                    down != nullptr) {
                    return false;
                }
                down = candidate;
                down_index = i;
            }
        }
    }
    if (gate_uses != 1 || up_uses != 1 || glu_uses != 1 || !down ||
        gate_index < 0 || up_index < 0 || glu_index < 0 || down_index < 0 ||
        !(gate_index < up_index && up_index < glu_index && glu_index < down_index)) {
        return false;
    }
    // The fused dispatch extends the hidden activation's lifetime from the
    // scheduler-visible up node to the later GLU node. Allow metadata between
    // them, but no intervening compute node that could reuse and overwrite the
    // hidden allocation. The route separately proves hidden/GLU-dst no-alias.
    for (int i = up_index + 1; i < glu_index; ++i) {
        const ggml_tensor * between = cgraph->nodes[i];
        if (between &&
            !ggml_backend_hrx_is_metadata_op(between) &&
            !ggml_backend_hrx_is_empty_op(between)) {
            return false;
        }
    }

    out_match->gate = gate;
    out_match->up = up;
    out_match->down = down;
    return true;
}

static bool ggml_backend_hrx_make_mul_mat_id_swiglu_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request,
        ggml_backend_hrx_mul_mat_id_swiglu_match * out_match = nullptr) {
    if (!device_context || !out_request) {
        return false;
    }
    ggml_backend_hrx_mul_mat_id_swiglu_match match = {};
    if (!ggml_backend_hrx_match_current_mul_mat_id_swiglu(
            device_context, node, &match)) {
        return false;
    }

    ggml_backend_hrx_dispatch_request request = {};
    if (!ggml_backend_hrx_make_mul_mat_id_request(
            device_context, match.up, &request) ||
        request.tensors.size() != 4) {
        return false;
    }
    const auto * terminal_qact_layer =
        ggml_backend_hrx_find_terminal_qact_layer(
            device_context->current_terminal_qact_plan, node);
    const bool terminal_qact =
        terminal_qact_layer &&
        terminal_qact_layer->glu == node &&
        terminal_qact_layer->down == match.down;
    const char * fusion_key = terminal_qact
        ? "MUL_MAT_ID_SWIGLU_Q5_DOWN_QACT_EPILOGUE"
        : "MUL_MAT_ID_SWIGLU_EPILOGUE";
    request.problem.supports["fusion"] = fusion_key;
    // Direct bindings precede all prepass outputs:
    //   up weight, hidden, ids, GLU dst, materialized gate,
    //   qact payload/scales/sums, MMID table.
    request.tensors[3] = node;
    request.tensors.push_back(match.gate);
    ggml_backend_hrx_add_tensor_facts(&request.problem, "dst", node);
    ggml_backend_hrx_add_tensor_facts(&request.problem, "gate", match.gate);

    ggml_backend_hrx_catalog_problem problem = request.problem;
    ggml_backend_hrx_add_tensor_overlap_facts(&problem, request.tensors);
    const auto * route =
        device_context->reg_context && device_context->reg_context->catalog
            ? ggml_backend_hrx_catalog_find_route(
                  *device_context->reg_context->catalog, problem)
            : nullptr;
    const auto fusion_it =
        route ? route->supports.find("fusion") : decltype(route->supports.find("fusion")){};
    const bool exact_terminal_qact_route =
        !terminal_qact ||
        (route &&
         route->id ==
             "mul_mat_id_q4_k_f32_mmq_gfx1151_wg256_pre_tbl_"
             "swiglu_q5_down_qact" &&
         route->source_id ==
             "mul_mat_id_q4_k_f32_mmqt_pre_swiglu_q5_down_qact" &&
         route->artifact_id ==
             "mul_mat_id_q4_k_f32_mmqt_pre_swiglu_"
             "q5_down_qact_loombc" &&
         route->root_symbol ==
             "@hrx2_mul_mat_id_q4_k_f32_mmqt_"
             "swiglu_q5_down_qact" &&
         route->export_name ==
             "hrx2_mul_mat_id_q4_k_f32_mmqt_"
             "swiglu_q5_down_qact" &&
         route->prepasses.size() == 2 &&
         route->prepasses[0].artifact_id ==
             "quant_act_q8_loombc" &&
         route->prepasses[1].artifact_id ==
             "mmid_table_loombc" &&
         route->prepasses[1].scratch_class == "mmid");
    if (!route || route->binding_count != 9 ||
        fusion_it == route->supports.end() ||
        fusion_it->second != request.problem.supports["fusion"] ||
        !exact_terminal_qact_route) {
        return false;
    }

    *out_request = std::move(request);
    if (out_match) {
        *out_match = match;
    }
    return true;
}

static bool ggml_backend_hrx_terminal_qact_producer_route_is_exact(
        const ggml_backend_hrx_catalog_route * route) {
    if (!route ||
        route->id !=
            "mul_mat_id_q4_k_f32_mmq_gfx1151_wg256_pre_tbl_"
            "swiglu_q5_down_qact" ||
        route->source_id !=
            "mul_mat_id_q4_k_f32_mmqt_pre_swiglu_q5_down_qact" ||
        route->artifact_id !=
            "mul_mat_id_q4_k_f32_mmqt_pre_swiglu_"
            "q5_down_qact_loombc" ||
        route->root_symbol !=
            "@hrx2_mul_mat_id_q4_k_f32_mmqt_"
            "swiglu_q5_down_qact" ||
        route->export_name !=
            "hrx2_mul_mat_id_q4_k_f32_mmqt_"
            "swiglu_q5_down_qact" ||
        route->binding_count != 9 ||
        route->parameter_count != 9 ||
        route->constant_byte_length != 0 ||
        route->prepasses.size() != 2) {
        return false;
    }
    const auto fusion = route->supports.find("fusion");
    const auto & qact = route->prepasses[0];
    const auto & table = route->prepasses[1];
    return fusion != route->supports.end() &&
           fusion->second ==
               "MUL_MAT_ID_SWIGLU_Q5_DOWN_QACT_EPILOGUE" &&
           qact.enabled &&
           qact.artifact_id == "quant_act_q8_loombc" &&
           qact.root_symbol == "@hrx2_quant_act_q8" &&
           qact.src_index == 1 &&
           qact.scratch_class.empty() &&
           table.enabled &&
           table.artifact_id == "mmid_table_loombc" &&
           table.root_symbol == "@hrx2_mmid_table" &&
           table.src_index == 2 &&
           table.bytes_per_element == 8 &&
           table.scratch_class == "mmid";
}

static bool ggml_backend_hrx_terminal_qact_consumer_route_is_exact(
        const ggml_backend_hrx_catalog_route * route) {
    if (!route ||
        route->id !=
            "mul_mat_id_q5_k_f32_mmq_gfx1151_wg256_tbl_"
            "down_group4_terminal_qact" ||
        route->source_id !=
            "mul_mat_id_q5_k_f32_mmqt_down_group4_terminal_qact" ||
        route->artifact_id !=
            "mul_mat_id_q5_k_f32_mmqt_down_group4_"
            "terminal_qact_loombc" ||
        route->root_symbol !=
            "@hrx2_mul_mat_id_q5_k_f32_mmqt_"
            "down_group4_terminal_qact" ||
        route->export_name !=
            "hrx2_mul_mat_id_q5_k_f32_mmqt_"
            "down_group4_terminal_qact" ||
        route->binding_count != 5 ||
        route->parameter_count != 5 ||
        route->constant_byte_length != 0 ||
        route->prepasses.size() != 1) {
        return false;
    }
    const auto fusion = route->supports.find("fusion");
    const auto & table = route->prepasses[0];
    return fusion != route->supports.end() &&
           fusion->second ==
               "MUL_MAT_ID_Q5_DOWN_TERMINAL_QACT" &&
           table.enabled &&
           table.artifact_id == "mmid_table_loombc" &&
           table.root_symbol == "@hrx2_mmid_table" &&
           table.src_index == 2 &&
           table.bytes_per_element == 8 &&
           table.scratch_class == "mmid";
}

static bool ggml_backend_hrx_make_terminal_qact_producer_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request,
        ggml_backend_hrx_mul_mat_id_swiglu_match * out_match) {
    ggml_backend_hrx_dispatch_request request = {};
    ggml_backend_hrx_mul_mat_id_swiglu_match match = {};
    if (!ggml_backend_hrx_make_mul_mat_id_swiglu_request(
            device_context, node, &request, &match)) {
        return false;
    }
    request.problem.supports["fusion"] =
        "MUL_MAT_ID_SWIGLU_Q5_DOWN_QACT_EPILOGUE";
    ggml_backend_hrx_catalog_problem problem = request.problem;
    ggml_backend_hrx_add_tensor_overlap_facts(
        &problem, request.tensors);
    const auto * route =
        device_context->reg_context &&
                device_context->reg_context->catalog
            ? ggml_backend_hrx_catalog_find_route(
                  *device_context->reg_context->catalog, problem)
            : nullptr;
    if (!ggml_backend_hrx_terminal_qact_producer_route_is_exact(
            route)) {
        return false;
    }
    *out_request = std::move(request);
    if (out_match) {
        *out_match = match;
    }
    return true;
}

static bool ggml_backend_hrx_make_terminal_qact_consumer_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    ggml_backend_hrx_dispatch_request request = {};
    if (!ggml_backend_hrx_make_mul_mat_id_request(
            device_context, node, &request) ||
        request.tensors.size() != 4) {
        return false;
    }
    request.problem.supports["fusion"] =
        "MUL_MAT_ID_Q5_DOWN_TERMINAL_QACT";
    ggml_backend_hrx_catalog_problem problem = request.problem;
    ggml_backend_hrx_add_tensor_overlap_facts(
        &problem, request.tensors);
    const auto * route =
        device_context->reg_context &&
                device_context->reg_context->catalog
            ? ggml_backend_hrx_catalog_find_route(
                  *device_context->reg_context->catalog, problem)
            : nullptr;
    if (!ggml_backend_hrx_terminal_qact_consumer_route_is_exact(
            route)) {
        return false;
    }
    *out_request = std::move(request);
    return true;
}

static bool ggml_backend_hrx_make_glu_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (ggml_backend_hrx_make_mul_mat_id_swiglu_request(
            device_context, node, out_request)) {
        return true;
    }
    if (ggml_backend_hrx_make_mul_mat_q4_k_swiglu_request(device_context, node, out_request)) {
        return true;
    }
    if (!device_context || !out_request || !node || node->op != GGML_OP_GLU || !node->src[0] ||
        ggml_get_glu_op(node) != GGML_GLU_OP_SWIGLU ||
        !ggml_backend_hrx_is_f32_dense(node->src[0]) ||
        !ggml_backend_hrx_is_f32_dense(node)) {
        return false;
    }
    const bool split = node->src[1] != nullptr;
    if (split && !ggml_backend_hrx_is_f32_dense(node->src[1])) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "GLU";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", split ? "contiguous_split_swiglu" : "packed_contiguous"},
        {"glu_op", "SWIGLU"},
    };
    if (split) {
        out_request->problem.supports["src1_type"] = ggml_backend_hrx_catalog_type_name(node->src[1]->type);
    } else {
        // Both halves live in one src0 row: ncols is src0->ne[0]/2, but the row
        // stride stays src0->ne[0], which is not 2*ncols when ne0 is odd. The
        // offsets carry ggml's swapped flag (silu operand first when unswapped).
        const int32_t swapped = ggml_get_op_params_i32(node, 1);
        out_request->problem.supports["swapped"] = swapped ? "true" : "false";
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "swiglu", "src_stride", node->src[0]->ne[0]);
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "swiglu", "x_offset", swapped ? node->ne[0] : 0);
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "swiglu", "g_offset", swapped ? 0 : node->ne[0]);
    }
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "swiglu", "ncols", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "swiglu", "nrows", ggml_backend_hrx_tensor_row_count(node));
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    if (split) {
        ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", node->src[1]);
    }
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = split ? std::vector<const ggml_tensor *>{node->src[0], node->src[1], node} :
        std::vector<const ggml_tensor *>{node->src[0], node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_dispatch_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !node || !out_request) {
        return false;
    }
    *out_request = {};
    switch (node->op) {
        case GGML_OP_ADD:
            return ggml_backend_hrx_make_add_request(device_context, node, out_request);
        case GGML_OP_MUL:
            return ggml_backend_hrx_make_mul_request(device_context, node, out_request);
        case GGML_OP_DIV:
            return ggml_backend_hrx_make_div_request(device_context, node, out_request);
        case GGML_OP_MUL_MAT:
            return ggml_backend_hrx_make_mul_mat_request(device_context, node, out_request);
        case GGML_OP_MUL_MAT_ID:
            return ggml_backend_hrx_make_mul_mat_id_request(device_context, node, out_request);
        case GGML_OP_SCALE:
            return ggml_backend_hrx_make_scale_request(device_context, node, out_request);
        case GGML_OP_CLAMP:
            return ggml_backend_hrx_make_clamp_request(device_context, node, out_request);
        case GGML_OP_RMS_NORM:
            return ggml_backend_hrx_make_rms_norm_request(device_context, node, out_request);
        case GGML_OP_SUM_ROWS:
            return ggml_backend_hrx_make_sum_rows_request(device_context, node, out_request);
        case GGML_OP_SOFT_MAX:
            return ggml_backend_hrx_make_soft_max_request(device_context, node, out_request);
        case GGML_OP_ARGSORT:
            return ggml_backend_hrx_make_argsort_request(device_context, node, out_request);
        case GGML_OP_GET_ROWS:
            return ggml_backend_hrx_make_get_rows_request(device_context, node, out_request);
        case GGML_OP_SET_ROWS:
            return ggml_backend_hrx_make_set_rows_request(device_context, node, out_request);
        case GGML_OP_CONT:
            return ggml_backend_hrx_make_cont_request(device_context, node, out_request);
        case GGML_OP_CPY:
            return ggml_backend_hrx_make_cpy_request(device_context, node, out_request);
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_backend_hrx_make_flash_attn_request(device_context, node, out_request);
        case GGML_OP_ROPE:
            return ggml_backend_hrx_make_rope_request(device_context, node, out_request);
        case GGML_OP_GLU:
            return ggml_backend_hrx_make_glu_request(device_context, node, out_request);
        case GGML_OP_UNARY:
            return ggml_backend_hrx_make_unary_request(device_context, node, out_request);
        case GGML_OP_L2_NORM:
            return ggml_backend_hrx_make_l2_norm_request(device_context, node, out_request);
        case GGML_OP_SSM_CONV: {
            if (const auto * layer =
                    ggml_backend_hrx_find_recurrent_cache_layer(
                        device_context, node)) {
                return ggml_backend_hrx_make_ssm_conv_state_cache_request(
                    device_context, node, *layer, out_request);
            }
            // The PP512 direct-window request is installed only by its exact
            // graph plan. Outside that plan, materialize the window and use the
            // ordinary convolution; selecting the direct-pieces route here is
            // unsafe when graph scheduling reuses either source span.
            return ggml_backend_hrx_make_ssm_conv_request(device_context, node, out_request);
        }
        case GGML_OP_CONCAT: {
            // Snapshot the carried state and the graph-proven overwritten x
            // prefix while writing the recurrent cache in the same dispatch.
            ggml_backend_hrx_dispatch_request tail;
            if (ggml_backend_hrx_make_concat_window_tail_request(
                    device_context, device_context->current_graph, node, &tail) &&
                ggml_backend_hrx_request_matches_loaded_route(
                    device_context, tail, "concat_window_tail")) {
                *out_request = std::move(tail);
                return true;
            }
            return ggml_backend_hrx_make_concat_request(device_context, node, out_request);
        }
        case GGML_OP_GATED_DELTA_NET:
            if (const auto * layer =
                    ggml_backend_hrx_find_gdn_qk_scale_layer(
                        device_context, node)) {
                return ggml_backend_hrx_make_gdn_qk_scale_request(
                    device_context, node, *layer, out_request);
            }
            if (const auto * layer =
                    ggml_backend_hrx_find_recurrent_cache_layer(
                        device_context, node)) {
                return ggml_backend_hrx_make_gdn_state_cache_request(
                    device_context, node, *layer, out_request);
            }
            return ggml_backend_hrx_make_gated_delta_net_request(device_context, node, out_request);
        default:
            return false;
    }
}

static inline void ggml_backend_hrx_hash_bytes(uint64_t * h, const void * data, size_t size) {
    const auto * p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < size; ++i) {
        *h = (*h ^ p[i]) * 1099511628211ull;
    }
}

static uint64_t ggml_backend_hrx_compiled_route_key(
        const ggml_backend_hrx_catalog_route & route,
        const std::vector<ggml_backend_hrx_catalog_binding> & bindings,
        const std::vector<int64_t> & workload_arguments,
        const std::string & target_key) {
    uint64_t h = 1469598103934665603ull;
    for (const std::string * part : {&target_key, &route.id, &route.artifact_id, &route.root_symbol}) {
        ggml_backend_hrx_hash_bytes(&h, part->data(), part->size());
        ggml_backend_hrx_hash_bytes(&h, "|", 1);
    }
    for (const auto & binding : bindings) {
        ggml_backend_hrx_hash_bytes(&h, binding.key.data(), binding.key.size());
        ggml_backend_hrx_hash_bytes(&h, "=", 1);
        ggml_backend_hrx_hash_bytes(&h, binding.value.data(), binding.value.size());
        ggml_backend_hrx_hash_bytes(&h, "|", 1);
    }
    for (int64_t value : workload_arguments) {
        ggml_backend_hrx_hash_bytes(&h, &value, sizeof(value));
    }
    return h;
}

static bool ggml_backend_hrx_resolve_workload_arguments(
        const ggml_backend_hrx_catalog_route & route,
        const ggml_backend_hrx_catalog_problem & problem,
        std::vector<int64_t> * out_arguments,
        std::string * out_error) {
    out_arguments->clear();
    for (const std::string & source : route.workload_argument_sources) {
        static constexpr const char * k_shape_prefix = "shape.";
        std::string shape_key = source;
        if (shape_key.compare(0, 6, k_shape_prefix) == 0) {
            shape_key = shape_key.substr(6);
        }
        const auto it = problem.shape.find(shape_key);
        if (it == problem.shape.end()) {
            if (out_error) {
                *out_error = "route " + route.id + " workload argument references missing shape value " + source;
            }
            return false;
        }
        out_arguments->push_back(it->second);
    }
    return true;
}

static ggml_backend_hrx_buffer_context * ggml_backend_hrx_tensor_buffer_context(const ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buffer || buffer->iface.get_base != ggml_backend_hrx_buffer_get_base) {
        return nullptr;
    }
    return ggml_backend_hrx_get_buffer_context(buffer);
}

static bool ggml_backend_hrx_make_tensor_binding(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * tensor,
        hrx_buffer_ref_t * out_ref) {
    auto * buffer_context = ggml_backend_hrx_tensor_buffer_context(tensor);
    if (!buffer_context || buffer_context->device_context != device_context || !buffer_context->buffer || !out_ref) {
        return false;
    }
    *out_ref = {
        /* .buffer = */ buffer_context->buffer,
        /* .offset = */ ggml_backend_hrx_tensor_offset(buffer_context, tensor),
        /* .length = */ ggml_nbytes(tensor),
    };
    return true;
}

static void ggml_backend_hrx_clear_quant_scratch_source(
        ggml_backend_hrx_device_context * device_context);

static bool ggml_backend_hrx_reserve_quant_scratch(
        ggml_backend_hrx_device_context * device_context,
        size_t bytes) {
    if (device_context->quant_scratch && device_context->quant_scratch_capacity >= bytes) {
        return true;
    }
    if (device_context->quant_scratch) {
        device_context->retired_quant_scratch.push_back(device_context->quant_scratch);
        device_context->quant_scratch = nullptr;
        device_context->quant_scratch_capacity = 0;
        ggml_backend_hrx_clear_quant_scratch_source(device_context);
    }
    // The floor covers every activation matrix the graph can present (the widest
    // is k*cols for a single MUL_MAT), so the buffer is allocated once and never
    // regrows mid-graph.
    const size_t capacity = ggml_backend_hrx_align_up(std::max<size_t>(bytes * 2, 64u << 20), GGML_HRX_ALIGNMENT);
    hrx_buffer_params_t params = {
        /* .type = */ HRX_MEMORY_TYPE_DEVICE_LOCAL,
        /* .access = */ HRX_MEMORY_ACCESS_ALL,
        /* .usage = */ HRX_BUFFER_USAGE_DEFAULT,
        /* .queue_affinity = */ 0,
    };
    if (!GGML_HRX_CHECK(hrx_allocator_allocate_buffer(
            hrx_device_allocator(device_context->device), params, capacity, &device_context->quant_scratch))) {
        device_context->quant_scratch = nullptr;
        return false;
    }
    device_context->quant_scratch_capacity = capacity;
    return true;
}

static void ggml_backend_hrx_clear_quant_scratch_source(
        ggml_backend_hrx_device_context * device_context) {
    device_context->quant_scratch_source = nullptr;
    device_context->quant_scratch_source_owner = nullptr;
    device_context->quant_scratch_source_buffer = nullptr;
    device_context->quant_scratch_source_offset = 0;
    device_context->quant_scratch_source_length = 0;
    device_context->quant_scratch_source_node = -1;
    device_context->quant_scratch_epoch = 0;
    device_context->quant_scratch_bytes = 0;
    device_context->quant_scratch_artifact.clear();
}

// Canonicalize only the exact alias used by the live graph: a contiguous,
// zero-offset RESHAPE covering its owner's complete storage. Other VIEW,
// PERMUTE and TRANSPOSE nodes retain tensor identity even if they happen to
// point at the same allocation.
static const ggml_tensor * ggml_backend_hrx_quant_scratch_owner(
        const ggml_tensor * tensor) {
    const ggml_tensor * owner = tensor;
    while (owner && owner->view_src &&
           owner->op == GGML_OP_RESHAPE &&
           owner->view_offs == 0 &&
           owner->type == owner->view_src->type &&
           ggml_is_contiguous(owner) &&
           ggml_is_contiguous(owner->view_src) &&
           ggml_nbytes(owner) == ggml_nbytes(owner->view_src)) {
        owner = owner->view_src;
    }
    return owner;
}

static bool ggml_backend_hrx_buffer_refs_overlap(
        const hrx_buffer_ref_t & lhs,
        const hrx_buffer_ref_t & rhs) {
    return lhs.buffer == rhs.buffer &&
           lhs.offset < rhs.offset + rhs.length &&
           rhs.offset < lhs.offset + lhs.length;
}

// Pointer-identical reuse preserves the backend's existing consecutive-node
// behavior. Crossing from a full-span RESHAPE to its owner is new, so prove
// once at the attempted reuse that no scheduled compute node between the two
// writes any byte of the source range.
static bool ggml_backend_hrx_quant_source_unchanged_between(
        ggml_backend_hrx_device_context * device_context,
        const hrx_buffer_ref_t & source_binding) {
    if (!device_context->current_graph ||
        device_context->quant_scratch_source_node < 0 ||
        device_context->current_node_index <=
            device_context->quant_scratch_source_node) {
        return false;
    }
    for (int i = device_context->quant_scratch_source_node + 1;
         i < device_context->current_node_index; ++i) {
        const ggml_tensor * between = device_context->current_graph->nodes[i];
        if (!between ||
            ggml_backend_hrx_is_metadata_op(between) ||
            ggml_backend_hrx_is_empty_op(between)) {
            continue;
        }
        hrx_buffer_ref_t destination = {};
        if (!ggml_backend_hrx_make_tensor_binding(
                device_context, between, &destination) ||
            ggml_backend_hrx_buffer_refs_overlap(
                source_binding, destination)) {
            return false;
        }
    }
    return true;
}

static bool ggml_backend_hrx_quant_scratch_matches(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * source,
        const hrx_buffer_ref_t & source_binding,
        size_t scratch_bytes,
        const std::string & artifact) {
    const bool key_matches =
        device_context->quant_scratch_source_owner ==
            ggml_backend_hrx_quant_scratch_owner(source) &&
        device_context->quant_scratch_source_buffer == source_binding.buffer &&
        device_context->quant_scratch_source_offset == source_binding.offset &&
        device_context->quant_scratch_source_length == source_binding.length &&
        device_context->quant_scratch_epoch == device_context->graph_epoch &&
        device_context->quant_scratch_bytes == scratch_bytes &&
        device_context->quant_scratch_artifact == artifact;
    if (!key_matches) {
        return false;
    }
    if (device_context->quant_scratch_source == source) {
        return true;
    }
    if (!ggml_backend_hrx_quant_source_unchanged_between(
            device_context, source_binding)) {
        return false;
    }
    // Future pointer-identical users of this same artifact need no rescan.
    device_context->quant_scratch_source = source;
    device_context->quant_scratch_source_node =
        device_context->current_node_index;
    return true;
}

static void ggml_backend_hrx_set_quant_scratch_source(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * source,
        const hrx_buffer_ref_t & source_binding,
        size_t scratch_bytes,
        const std::string & artifact) {
    device_context->quant_scratch_source = source;
    device_context->quant_scratch_source_owner =
        ggml_backend_hrx_quant_scratch_owner(source);
    device_context->quant_scratch_source_buffer = source_binding.buffer;
    device_context->quant_scratch_source_offset = source_binding.offset;
    device_context->quant_scratch_source_length = source_binding.length;
    device_context->quant_scratch_source_node =
        device_context->current_node_index;
    device_context->quant_scratch_epoch = device_context->graph_epoch;
    device_context->quant_scratch_bytes = scratch_bytes;
    device_context->quant_scratch_artifact = artifact;
}

// Only a tensor in a weights buffer is safe to cache persistently: its contents
// never change for the lifetime of the process. Compute buffers are reused
// across allocations, so a data pointer on its own can alias a different tensor.
// Scratch for a prepass that names its own class. Kept apart from the shared
// quantisation scratch so two prepasses on one route cannot alias.
static bool ggml_backend_hrx_reserve_class_scratch(
        ggml_backend_hrx_device_context * device_context,
        const std::string & cls,
        size_t bytes,
        hrx_buffer_t * out_buffer,
        bool * out_fresh) {
    hrx_buffer_t & buffer = device_context->class_scratch[cls];
    size_t & have = device_context->class_scratch_bytes[cls];
    *out_fresh = false;
    if (buffer && have >= bytes) {
        *out_buffer = buffer;
        return true;
    }
    // A reallocation leaves the contents undefined, so whatever the caller had
    // cached about them no longer holds.
    *out_fresh = true;
    if (buffer) {
        hrx_buffer_release(buffer);
        buffer = nullptr;
    }
    hrx_buffer_params_t params = {
        /* .type = */ HRX_MEMORY_TYPE_DEVICE_LOCAL,
        /* .access = */ HRX_MEMORY_ACCESS_ALL,
        /* .usage = */ HRX_BUFFER_USAGE_DEFAULT,
        /* .queue_affinity = */ 0,
    };
    if (!GGML_HRX_CHECK(hrx_allocator_allocate_buffer(
            hrx_device_allocator(device_context->device), params, bytes, &buffer))) {
        buffer = nullptr;
        have = 0;
        return false;
    }
    have = bytes;
    *out_buffer = buffer;
    return true;
}

static bool ggml_backend_hrx_prepass_can_persist(const ggml_tensor * tensor) {
    if (!tensor) {
        return false;
    }
    // Model weights commonly reach MUL_MAT as views. The view itself does not
    // necessarily carry the allocation's usage flag; its owner does.
    const ggml_tensor * owner = tensor;
    while (owner->view_src) {
        owner = owner->view_src;
    }
    ggml_backend_buffer_t buffer = owner->buffer;
    return buffer != nullptr &&
           ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static bool ggml_backend_hrx_reserve_weight_arena(
        ggml_backend_hrx_device_context * device_context,
        size_t bytes,
        size_t * out_offset) {
    const size_t offset = ggml_backend_hrx_align_up(device_context->weight_arena_used, GGML_HRX_ALIGNMENT);
    if (!device_context->weight_arena) {
        // Sized for the whole dense weight set expanded to f16; the arena is
        // allocated once and never grown, because growing it would strand the
        // offsets already handed out.
        const size_t capacity = size_t{6144} << 20;
        hrx_buffer_params_t params = {
            /* .type = */ HRX_MEMORY_TYPE_DEVICE_LOCAL,
            /* .access = */ HRX_MEMORY_ACCESS_ALL,
            /* .usage = */ HRX_BUFFER_USAGE_DEFAULT,
            /* .queue_affinity = */ 0,
        };
        if (!GGML_HRX_CHECK(hrx_allocator_allocate_buffer(
                hrx_device_allocator(device_context->device), params, capacity, &device_context->weight_arena))) {
            device_context->weight_arena = nullptr;
            return false;
        }
        device_context->weight_arena_capacity = capacity;
    }
    if (offset + bytes > device_context->weight_arena_capacity) {
        // Not fatal: the caller falls back to the per-dispatch scratch path.
        return false;
    }
    device_context->weight_arena_used = offset + bytes;
    *out_offset = offset;
    return true;
}

static ggml_backend_hrx_compiled_route * ggml_backend_hrx_get_compiled_route(
        ggml_backend_hrx_device_context * device_context,
        const ggml_backend_hrx_catalog_route & route,
        const ggml_backend_hrx_catalog_problem & problem,
        const std::vector<ggml_backend_hrx_catalog_binding> & resolved_bindings,
        const std::vector<int64_t> & workload_arguments) {
    if (!device_context || !device_context->jit || !device_context->reg_context || !device_context->reg_context->catalog) {
        return nullptr;
    }
    const uint64_t cache_key = ggml_backend_hrx_compiled_route_key(route, resolved_bindings, workload_arguments, problem.target_key);
    {
        std::lock_guard<std::mutex> lock(device_context->compiled_routes_mutex);
        const auto it = device_context->compiled_routes.find(cache_key);
        if (it != device_context->compiled_routes.end()) {
            ggml_backend_hrx_test_record_jit_cache_hit(route.id);
            ggml_backend_hrx_trace_event(device_context->reg_context, {
                {"event", "route_jit_cache_hit"},
                {"device", device_context->name},
                {"route_id", route.id},
                {"artifact_id", route.artifact_id},
                {"launch_workload_argument_count", it->second->launch_config.workload_argument_count},
                {"workgroup_count", {
                    it->second->launch_config.workgroup_count[0],
                    it->second->launch_config.workgroup_count[1],
                    it->second->launch_config.workgroup_count[2],
                }},
                {"workgroup_size", {
                    it->second->launch_config.workgroup_size[0],
                    it->second->launch_config.workgroup_size[1],
                    it->second->launch_config.workgroup_size[2],
                }},
            });
            return it->second.get();
        }
    }

    const auto * artifact = ggml_backend_hrx_catalog_find_artifact(*device_context->reg_context->catalog, route.artifact_id);
    if (!artifact || artifact->data.empty()) {
        GGML_LOG_ERROR("%s: route %s references missing artifact %s\n", __func__, route.id.c_str(), route.artifact_id.c_str());
        return nullptr;
    }

    std::vector<ggml_hrx_loom_jit_config_binding_t> jit_bindings;
    jit_bindings.reserve(resolved_bindings.size());
    for (const auto & binding : resolved_bindings) {
        jit_bindings.push_back({
            /* .key = */ binding.key.c_str(),
            /* .value = */ binding.value.c_str(),
        });
    }
    const std::string module_name = "ggml_hrx_" + route.id;
    ggml_hrx_loom_jit_compile_options_t compile_options = {
        /* .structure_size = */ sizeof(ggml_hrx_loom_jit_compile_options_t),
        /* .source_data = */ artifact->data.data(),
        /* .source_size = */ artifact->data.size(),
        /* .source_format = */ GGML_HRX_LOOM_JIT_SOURCE_FORMAT_BYTECODE,
        /* .source_identifier = */ artifact->path.c_str(),
        /* .root_symbol = */ route.root_symbol.c_str(),
        /* .module_name = */ module_name.c_str(),
        /* .artifact_identifier = */ route.id.c_str(),
        /* .config_bindings = */ jit_bindings.data(),
        /* .config_binding_count = */ jit_bindings.size(),
        /* .workload_arguments = */ workload_arguments.empty() ? nullptr : workload_arguments.data(),
        /* .workload_argument_count = */ workload_arguments.size(),
    };
    ggml_hrx_loom_jit_compile_result_t compile_result = {};
    if (!GGML_HRX_CHECK(ggml_hrx_loom_jit_amdgpu_compile(device_context->jit, &compile_options, &compile_result))) {
        ggml_hrx_loom_jit_compile_result_deinitialize(&compile_result);
        return nullptr;
    }
    ggml_backend_hrx_write_evidence_file(
        device_context, route.id + ".hsaco", compile_result.hsaco_data, compile_result.hsaco_size);
    ggml_backend_hrx_write_evidence_file(
        device_context, route.id + ".compile-report.json",
        compile_result.compile_report_json, compile_result.compile_report_json_size);

    hrx_executable_t executable = nullptr;
    // The "amdgpu" family routes target selection through the AMDGPU device
    // spec, which matches the artifact key ("gfx1151") against the advertised
    // device targets.
    if (!GGML_HRX_CHECK(hrx_executable_load_data(
            device_context->device,
            compile_result.hsaco_data,
            compile_result.hsaco_size,
            "amdgpu",
            device_context->architecture.c_str(),
            &executable))) {
        ggml_hrx_loom_jit_compile_result_deinitialize(&compile_result);
        return nullptr;
    }
    uint32_t export_ordinal = 0;
    const char * export_name = route.export_name.empty() ? nullptr : route.export_name.c_str();
    if (!export_name || !GGML_HRX_CHECK(hrx_executable_lookup_export_by_name(executable, export_name, &export_ordinal))) {
        GGML_LOG_ERROR("%s: failed to resolve export %s for route %s\n", __func__, export_name ? export_name : "<empty>", route.id.c_str());
        hrx_executable_release(executable);
        ggml_hrx_loom_jit_compile_result_deinitialize(&compile_result);
        return nullptr;
    }
    hrx_executable_export_info_t export_info = {};
    if (!GGML_HRX_CHECK(hrx_executable_export_info(executable, export_ordinal, &export_info))) {
        hrx_executable_release(executable);
        ggml_hrx_loom_jit_compile_result_deinitialize(&compile_result);
        return nullptr;
    }
    if (export_info.binding_count != route.binding_count ||
        export_info.parameter_count != route.parameter_count ||
        export_info.constant_byte_length != route.constant_byte_length) {
        GGML_LOG_ERROR(
            "%s: route %s JIT export ABI mismatch "
            "(bindings=%u expected=%u constants_size=%u expected_constants_size=%u "
            "parameters=%u expected_parameters=%u)\n",
            __func__,
            route.id.c_str(),
            export_info.binding_count,
            route.binding_count,
            export_info.constant_byte_length,
            route.constant_byte_length,
            export_info.parameter_count,
            route.parameter_count);
        hrx_executable_release(executable);
        ggml_hrx_loom_jit_compile_result_deinitialize(&compile_result);
        return nullptr;
    }

    ggml_backend_hrx_compiled_route_ptr compiled(new (std::nothrow) ggml_backend_hrx_compiled_route());
    if (!compiled) {
        hrx_executable_release(executable);
        ggml_hrx_loom_jit_compile_result_deinitialize(&compile_result);
        return nullptr;
    }
    compiled->route = &route;
    compiled->executable = executable;
    compiled->export_ordinal = export_ordinal;
    compiled->export_info = export_info;
    compiled->launch_config = compile_result.launch_config;

    ggml_backend_hrx_test_record_jit_compile(route.id);
    ggml_backend_hrx_trace_event(device_context->reg_context, {
        {"event", "route_jit_compiled"},
        {"device", device_context->name},
        {"route_id", route.id},
        {"artifact_id", route.artifact_id},
        {"root_symbol", route.root_symbol},
        {"launch_workload_argument_count", compile_result.launch_config.workload_argument_count},
        {"binding_count", export_info.binding_count},
        {"parameter_count", export_info.parameter_count},
        {"constant_byte_length", export_info.constant_byte_length},
        {"workgroup_count", {
            compile_result.launch_config.workgroup_count[0],
            compile_result.launch_config.workgroup_count[1],
            compile_result.launch_config.workgroup_count[2],
        }},
        {"workgroup_size", {
            compile_result.launch_config.workgroup_size[0],
            compile_result.launch_config.workgroup_size[1],
            compile_result.launch_config.workgroup_size[2],
        }},
    });
    ggml_hrx_loom_jit_compile_result_deinitialize(&compile_result);

    std::lock_guard<std::mutex> lock(device_context->compiled_routes_mutex);
    auto inserted = device_context->compiled_routes.emplace(cache_key, std::move(compiled));
    if (!inserted.second) {
        return inserted.first->second.get();
    }
    return inserted.first->second.get();
}

// Everything the route resolution depends on: the op, and each tensor's type,
// shape, strides and address. Two dispatches of a node with the same signature
// resolve identically.
static uint64_t ggml_backend_hrx_node_signature(const ggml_tensor * node) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    mix((uint64_t) node->op);
    const ggml_tensor * parts[7] = {node, node->src[0], node->src[1], node->src[2],
                                    node->src[3], node->src[4], node->src[5]};
    for (const ggml_tensor * t : parts) {
        if (!t) {
            mix(0xd15ea5eull);
            continue;
        }
        // The tensor's identity matters as well as its contents: the shortcut
        // reuses the memoised tensor list, and a different tensor object with
        // matching type, shape, strides and address would otherwise hash equal.
        mix((uint64_t) (uintptr_t) t);
        mix((uint64_t) t->type);
        mix((uint64_t) (uintptr_t) t->data);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            mix((uint64_t) t->ne[i]);
            mix((uint64_t) t->nb[i]);
        }
    }
    for (int i = 0; i < GGML_MAX_OP_PARAMS / (int) sizeof(int32_t); ++i) {
        mix((uint64_t) (uint32_t) node->op_params[i]);
    }
    return h;
}

static uint64_t
ggml_backend_hrx_ssm_conv_silu_request_fingerprint(
        const ggml_backend_hrx_ssm_conv_silu_layer & layer,
        const std::vector<const ggml_tensor *> & tensors) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) {
        h = (h ^ v) * 1099511628211ull;
    };
    mix(ggml_backend_hrx_node_signature(layer.ssm_conv));
    mix(static_cast<uint64_t>(layer.ssm_conv_index));
    mix(static_cast<uint64_t>(layer.silu_index));
    mix(layer.x_dst_alias ? 1 : 0);
    for (const ggml_tensor * tensor : tensors) {
        if (!tensor) {
            mix(0xd15ea5eull);
            continue;
        }
        mix(reinterpret_cast<uintptr_t>(tensor));
        mix(static_cast<uint64_t>(tensor->op));
        mix(static_cast<uint64_t>(tensor->type));
        mix(reinterpret_cast<uintptr_t>(tensor->data));
        mix(reinterpret_cast<uintptr_t>(tensor->buffer));
        mix(reinterpret_cast<uintptr_t>(tensor->view_src));
        mix(static_cast<uint64_t>(tensor->view_offs));
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            mix(static_cast<uint64_t>(tensor->ne[i]));
            mix(static_cast<uint64_t>(tensor->nb[i]));
        }
        for (int i = 0;
             i < GGML_MAX_OP_PARAMS / static_cast<int>(sizeof(int32_t));
             ++i) {
            mix(static_cast<uint64_t>(
                static_cast<uint32_t>(tensor->op_params[i])));
        }
    }
    return h;
}

static bool ggml_backend_hrx_catalog_binding_equal(
        const ggml_backend_hrx_catalog_binding & lhs,
        const ggml_backend_hrx_catalog_binding & rhs) {
    return lhs.key == rhs.key &&
           lhs.value == rhs.value &&
           lhs.source == rhs.source &&
           lhs.shape_source == rhs.shape_source;
}

static bool ggml_backend_hrx_prepass_descriptor_equal(
        const ggml_backend_hrx_catalog_prepass & lhs,
        const ggml_backend_hrx_catalog_prepass & rhs) {
    if (lhs.enabled != rhs.enabled ||
        lhs.artifact_id != rhs.artifact_id ||
        lhs.root_symbol != rhs.root_symbol ||
        lhs.export_name != rhs.export_name ||
        lhs.scratch_element_sources != rhs.scratch_element_sources ||
        lhs.src_index != rhs.src_index ||
        lhs.src_indices != rhs.src_indices ||
        lhs.dst_index != rhs.dst_index ||
        lhs.bytes_per_element != rhs.bytes_per_element ||
        lhs.element_binding_key != rhs.element_binding_key ||
        lhs.persistent != rhs.persistent ||
        lhs.scratch_class != rhs.scratch_class ||
        lhs.bindings.size() != rhs.bindings.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.bindings.size(); ++i) {
        if (!ggml_backend_hrx_catalog_binding_equal(
                lhs.bindings[i], rhs.bindings[i])) {
            return false;
        }
    }
    return true;
}

static bool ggml_backend_hrx_export_info_equal(
        const hrx_executable_export_info_t & current,
        const hrx_executable_export_info_t & saved,
        const std::string & saved_name) {
    const char * current_name = current.name ? current.name : "";
    return saved_name == current_name &&
           current.flags == saved.flags &&
           current.constant_byte_length == saved.constant_byte_length &&
           current.binding_count == saved.binding_count &&
           current.parameter_count == saved.parameter_count &&
           current.workgroup_size[0] == saved.workgroup_size[0] &&
           current.workgroup_size[1] == saved.workgroup_size[1] &&
           current.workgroup_size[2] == saved.workgroup_size[2];
}

static bool ggml_backend_hrx_launch_config_equal(
        const ggml_hrx_loom_jit_launch_config_t & lhs,
        const ggml_hrx_loom_jit_launch_config_t & rhs) {
    return lhs.workgroup_count[0] == rhs.workgroup_count[0] &&
           lhs.workgroup_count[1] == rhs.workgroup_count[1] &&
           lhs.workgroup_count[2] == rhs.workgroup_count[2] &&
           lhs.workgroup_size[0] == rhs.workgroup_size[0] &&
           lhs.workgroup_size[1] == rhs.workgroup_size[1] &&
           lhs.workgroup_size[2] == rhs.workgroup_size[2] &&
           lhs.subgroup_size == rhs.subgroup_size &&
           lhs.workgroup_storage_bytes == rhs.workgroup_storage_bytes &&
           lhs.workload_argument_count == rhs.workload_argument_count &&
           lhs.fields == rhs.fields;
}

static bool ggml_backend_hrx_dispatch_config_matches_launch(
        const hrx_dispatch_config_t & dispatch,
        const ggml_hrx_loom_jit_launch_config_t & launch) {
    return dispatch.workgroup_count[0] == launch.workgroup_count[0] &&
           dispatch.workgroup_count[1] == launch.workgroup_count[1] &&
           dispatch.workgroup_count[2] == launch.workgroup_count[2] &&
           dispatch.workgroup_size[0] == launch.workgroup_size[0] &&
           dispatch.workgroup_size[1] == launch.workgroup_size[1] &&
           dispatch.workgroup_size[2] == launch.workgroup_size[2] &&
           dispatch.subgroup_size == launch.subgroup_size;
}

static bool ggml_backend_hrx_buffer_ref_equal(
        const hrx_buffer_ref_t & lhs,
        const hrx_buffer_ref_t & rhs) {
    return lhs.buffer == rhs.buffer &&
           lhs.offset == rhs.offset &&
           lhs.length == rhs.length;
}

static bool ggml_backend_hrx_decimal_equal(
        const std::string & text,
        int64_t value) {
    std::array<char, 32> storage = {};
    const auto converted = std::to_chars(
        storage.data(), storage.data() + storage.size(), value);
    return converted.ec == std::errc() &&
           text.size() ==
               static_cast<size_t>(converted.ptr - storage.data()) &&
           std::memcmp(storage.data(), text.data(), text.size()) == 0;
}

static bool ggml_backend_hrx_resolved_prepass_bindings_are_consistent(
        const ggml_backend_hrx_device_context::resolved_prepass & cached,
        const std::unordered_map<std::string, int64_t> & shape) {
    const auto & descriptor = cached.descriptor;
    const size_t expected_count =
        descriptor.bindings.size() +
        (descriptor.element_binding_key.empty() ? 0 : 1);
    if (cached.bindings.size() != expected_count) {
        return false;
    }
    for (size_t i = 0; i < descriptor.bindings.size(); ++i) {
        const auto & source = descriptor.bindings[i];
        const auto & resolved = cached.bindings[i];
        if (resolved.key != source.key) {
            return false;
        }
        if (!source.value.empty()) {
            if (resolved.value != source.value) {
                return false;
            }
        } else {
            const auto value = shape.find(source.shape_source);
            if (value == shape.end() ||
                !ggml_backend_hrx_decimal_equal(
                    resolved.value, value->second)) {
                return false;
            }
        }
    }
    if (!descriptor.element_binding_key.empty()) {
        const auto & resolved = cached.bindings.back();
        if (resolved.key != descriptor.element_binding_key ||
            !ggml_backend_hrx_decimal_equal(
                resolved.value, cached.elements)) {
            return false;
        }
    }
    return true;
}

static bool ggml_backend_hrx_resolved_prepass_is_consistent(
        const ggml_backend_hrx_device_context::resolved_prepass & cached,
        const ggml_backend_hrx_catalog_prepass & descriptor,
        size_t descriptor_index,
        const std::unordered_map<std::string, int64_t> & shape) {
    if (cached.descriptor_index != descriptor_index ||
        cached.descriptor_identity != &descriptor ||
        !ggml_backend_hrx_prepass_descriptor_equal(
            cached.descriptor, descriptor)) {
        return false;
    }

    int64_t elements = 1;
    for (const auto & source : descriptor.scratch_element_sources) {
        const std::string key =
            source.compare(0, 6, "shape.") == 0 ?
            source.substr(6) : source;
        const auto value = shape.find(key);
        if (value == shape.end()) {
            return false;
        }
        elements *= value->second;
    }
    if (cached.elements != elements ||
        !ggml_backend_hrx_resolved_prepass_bindings_are_consistent(
            cached, shape) ||
        !cached.compiled ||
        !cached.executable ||
        cached.compiled->executable != cached.executable ||
        cached.compiled->export_ordinal != cached.export_ordinal ||
        !ggml_backend_hrx_export_info_equal(
            cached.compiled->export_info,
            cached.export_info,
            cached.export_name) ||
        !ggml_backend_hrx_launch_config_equal(
            cached.compiled->launch_config,
            cached.launch_config) ||
        !ggml_backend_hrx_dispatch_config_matches_launch(
            cached.dispatch_config,
            cached.launch_config) ||
        cached.export_info.binding_count !=
            descriptor.src_indices.size() + cached.output_count ||
        cached.export_info.parameter_count !=
            descriptor.src_indices.size() + cached.output_count ||
        cached.export_info.constant_byte_length != 0 ||
        cached.launch_config.workload_argument_count != 0) {
        return false;
    }

    if (descriptor.bytes_per_element != 0) {
        const size_t bytes =
            static_cast<size_t>(elements) * descriptor.bytes_per_element;
        return cached.output_count == 1 &&
               cached.output_offsets[0] == 0 &&
               cached.output_lengths[0] == bytes &&
               cached.output_offsets[1] == 0 &&
               cached.output_offsets[2] == 0 &&
               cached.output_lengths[1] == 0 &&
               cached.output_lengths[2] == 0 &&
               cached.scratch_bytes == bytes;
    }

    const size_t quant_bytes = static_cast<size_t>(elements);
    const size_t scale_offset =
        ggml_backend_hrx_align_up(quant_bytes, GGML_HRX_ALIGNMENT);
    const size_t scale_bytes =
        static_cast<size_t>(elements / 32) * sizeof(float);
    const size_t sum_offset = ggml_backend_hrx_align_up(
        scale_offset + scale_bytes, GGML_HRX_ALIGNMENT);
    const size_t sum_bytes =
        static_cast<size_t>(elements / 16) * sizeof(int32_t);
    return cached.output_count == 3 &&
           cached.output_offsets[0] == 0 &&
           cached.output_offsets[1] == scale_offset &&
           cached.output_offsets[2] == sum_offset &&
           cached.output_lengths[0] == quant_bytes &&
           cached.output_lengths[1] == scale_bytes &&
           cached.output_lengths[2] == sum_bytes &&
           cached.scratch_bytes == sum_offset + sum_bytes;
}

static bool ggml_backend_hrx_resolved_prepass_plan_matches(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        uint64_t node_signature,
        const ggml_backend_hrx_device_context::resolved_dispatch & memo,
        std::vector<hrx_buffer_ref_t> * out_direct_bindings) {
    if (!memo.prepass_plan ||
        !device_context->current_graph ||
        device_context->current_graph->uid == 0) {
        return false;
    }
    const auto & cached = *memo.prepass_plan;
    const ggml_cgraph * graph = device_context->current_graph;
    if (cached.graph_uid != graph->uid ||
        cached.node_index != device_context->current_node_index ||
        cached.node_count != graph->n_nodes ||
        cached.node_index < 0 ||
        cached.node_index >= graph->n_nodes ||
        graph->nodes[cached.node_index] != node ||
        memo.signature != node_signature ||
        !device_context->reg_context ||
        !device_context->reg_context->catalog ||
        cached.catalog != device_context->reg_context->catalog.get() ||
        !memo.route ||
        cached.route != memo.route ||
        cached.route_id != memo.route->id ||
        !memo.compiled ||
        cached.compiled != memo.compiled ||
        cached.executable != memo.compiled->executable ||
        cached.export_ordinal != memo.compiled->export_ordinal ||
        memo.compiled->route != memo.route ||
        !ggml_backend_hrx_export_info_equal(
            memo.compiled->export_info,
            cached.export_info,
            cached.export_name) ||
        !ggml_backend_hrx_launch_config_equal(
            memo.compiled->launch_config,
            cached.launch_config) ||
        cached.constants != memo.constants ||
        cached.workload != memo.workload ||
        cached.direct_bindings.size() != memo.tensors.size() ||
        cached.prepasses.size() != memo.route->prepasses.size()) {
        return false;
    }

    out_direct_bindings->assign(
        memo.tensors.size(), hrx_buffer_ref_t{});
    for (size_t i = 0; i < memo.tensors.size(); ++i) {
        if (!ggml_backend_hrx_make_tensor_binding(
                device_context, memo.tensors[i],
                &(*out_direct_bindings)[i])) {
            return false;
        }
    }
    for (size_t i = 0; i < memo.tensors.size(); ++i) {
        if (!ggml_backend_hrx_buffer_ref_equal(
                (*out_direct_bindings)[i],
                cached.direct_bindings[i])) {
            return false;
        }
    }
    for (size_t i = 0; i < cached.prepasses.size(); ++i) {
        if (memo.route->prepasses[i].src_indices.empty() ||
            std::any_of(
                memo.route->prepasses[i].src_indices.begin(),
                memo.route->prepasses[i].src_indices.end(),
                [&memo](uint32_t index) {
                    return index >= memo.tensors.size();
                }) ||
            (memo.route->prepasses[i].dst_index >= 0 &&
             static_cast<size_t>(
                 memo.route->prepasses[i].dst_index) >=
                 memo.tensors.size()) ||
            !ggml_backend_hrx_resolved_prepass_is_consistent(
                cached.prepasses[i],
                memo.route->prepasses[i],
                i,
                cached.shape)) {
            return false;
        }
    }
    return true;
}

static bool ggml_backend_hrx_resolve_prepass_executable(
        ggml_backend_hrx_device_context * device_context,
        const ggml_backend_hrx_catalog_route & route,
        const ggml_backend_hrx_catalog_prepass & prepass,
        const ggml_backend_hrx_catalog_problem & problem,
        int64_t elements,
        size_t output_count,
        std::vector<ggml_backend_hrx_catalog_binding> * out_bindings,
        ggml_backend_hrx_compiled_route ** out_compiled,
        hrx_dispatch_config_t * out_config) {
    ggml_backend_hrx_catalog_route pre_route;
    pre_route.id = route.id + "+prepass";
    pre_route.target_key = route.target_key;
    pre_route.artifact_id = prepass.artifact_id;
    pre_route.root_symbol = prepass.root_symbol;
    pre_route.export_name = prepass.export_name;
    pre_route.bindings = prepass.bindings;
    pre_route.binding_count = static_cast<uint32_t>(
        prepass.src_indices.size() + output_count);
    pre_route.parameter_count = pre_route.binding_count;

    std::string pre_error;
    if (!ggml_backend_hrx_catalog_make_config_bindings(
            pre_route, problem, out_bindings, &pre_error)) {
        GGML_LOG_ERROR("%s: %s\n", __func__, pre_error.c_str());
        return false;
    }
    if (!prepass.element_binding_key.empty()) {
        ggml_backend_hrx_catalog_binding computed;
        computed.key = prepass.element_binding_key;
        computed.value = std::to_string(elements);
        out_bindings->push_back(std::move(computed));
    }
    *out_compiled = ggml_backend_hrx_get_compiled_route(
        device_context, pre_route, problem, *out_bindings, {});
    if (!*out_compiled || !(*out_compiled)->executable) {
        return false;
    }
    *out_config = {
        /* .workgroup_count = */ {
            (*out_compiled)->launch_config.workgroup_count[0],
            (*out_compiled)->launch_config.workgroup_count[1],
            (*out_compiled)->launch_config.workgroup_count[2],
        },
        /* .workgroup_size = */ {
            (*out_compiled)->launch_config.workgroup_size[0],
            (*out_compiled)->launch_config.workgroup_size[1],
            (*out_compiled)->launch_config.workgroup_size[2],
        },
        /* .subgroup_size = */
            (*out_compiled)->launch_config.subgroup_size,
    };
    return true;
}

static void ggml_backend_hrx_set_resolved_prepass_executable(
        ggml_backend_hrx_device_context::resolved_prepass * resolved,
        std::vector<ggml_backend_hrx_catalog_binding> bindings,
        ggml_backend_hrx_compiled_route * compiled,
        const hrx_dispatch_config_t & config) {
    resolved->bindings = std::move(bindings);
    resolved->compiled = compiled;
    resolved->executable = compiled->executable;
    resolved->export_ordinal = compiled->export_ordinal;
    resolved->export_info = compiled->export_info;
    resolved->export_name =
        compiled->export_info.name ?
        compiled->export_info.name : "";
    resolved->launch_config = compiled->launch_config;
    resolved->dispatch_config = config;
}

static bool ggml_backend_hrx_dispatch_node(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node) {
    static thread_local ggml_backend_hrx_dispatch_request request;
    static thread_local std::vector<hrx_buffer_ref_t> bindings;
    request.tensors.clear();
    request.constants.clear();
    request.problem.shape.clear();
    request.problem.supports.clear();
    request.problem.facts.clear();
    bindings.clear();
    const uint64_t node_signature = ggml_backend_hrx_node_signature(node);
    auto memo_it = device_context->resolved_dispatches.find(node);
    // These requests depend on surrounding graph topology, which is not in
    // node_signature. In particular, warmup and PP512 can reuse tensor
    // addresses and identical direct shapes while only PP512 has the exact
    // fusion chain. Rebuild their request while a graph is active rather than
    // reusing an ordinary route resolved for a different graph.
    const bool possible_gdn_terminal_mul =
        node->op == GGML_OP_MUL &&
        node->src[0] && node->src[1] &&
        node->src[1]->op == GGML_OP_UNARY &&
        ggml_get_unary_op(node->src[1]) == GGML_UNARY_OP_SILU &&
        node->ne[0] == 128 &&
        ggml_backend_hrx_tensor_row_count(node) == 16384 &&
        node->src[0]->ne[0] == 128 &&
        ggml_backend_hrx_tensor_row_count(node->src[0]) == 16384 &&
        node->src[1]->ne[0] == 128 &&
        ggml_backend_hrx_tensor_row_count(node->src[1]) == 16384;
    const bool possible_decode_gdn_terminal_mul =
        node->op == GGML_OP_MUL &&
        node->src[0] && node->src[1] &&
        node->src[1]->op == GGML_OP_UNARY &&
        ggml_get_unary_op(node->src[1]) == GGML_UNARY_OP_SILU &&
        node->ne[0] == 128 && node->ne[1] == 32 &&
        node->ne[2] == 1 && node->ne[3] == 1 &&
        node->src[0]->ne[0] == 128 && node->src[0]->ne[1] == 32 &&
        node->src[0]->ne[2] == 1 && node->src[0]->ne[3] == 1 &&
        node->src[1]->ne[0] == 128 && node->src[1]->ne[1] == 32 &&
        node->src[1]->ne[2] == 1 && node->src[1]->ne[3] == 1;
    const bool possible_fa_terminal_mul =
        node->op == GGML_OP_MUL &&
        node->src[0] && node->src[1] &&
        node->src[0]->op == GGML_OP_RESHAPE &&
        node->src[0]->src[0] &&
        node->src[0]->src[0]->op == GGML_OP_FLASH_ATTN_EXT &&
        node->src[1]->op == GGML_OP_UNARY &&
        ggml_get_unary_op(node->src[1]) == GGML_UNARY_OP_SIGMOID &&
        node->src[1]->src[0] &&
        node->src[1]->src[0]->op == GGML_OP_CONT &&
        node->ne[0] == 4096 && node->ne[1] == 512 &&
        node->ne[2] == 1 && node->ne[3] == 1;
    const bool possible_recurrent_ssm =
        node->op == GGML_OP_SSM_CONV &&
        node->ne[0] == 8192 && node->ne[1] == 1 &&
        node->ne[2] == 1 && node->ne[3] == 1 &&
        node->src[1] &&
        node->src[1]->ne[0] == 4 &&
        node->src[1]->ne[1] == 8192 &&
        node->src[1]->ne[2] == 1 &&
        node->src[1]->ne[3] == 1;
    const bool possible_recurrent_gdn =
        node->op == GGML_OP_GATED_DELTA_NET &&
        node->src[2] &&
        node->src[2]->ne[0] == 128 &&
        node->src[2]->ne[1] == 32 &&
        node->src[2]->ne[2] == 1 &&
        node->src[2]->ne[3] == 1;
    const bool possible_recurrent_node =
        possible_recurrent_ssm || possible_recurrent_gdn;
    const bool possible_gdn_qk_scale =
        node->op == GGML_OP_GATED_DELTA_NET &&
        node->src[2] &&
        node->src[2]->ne[0] == 128 &&
        node->src[2]->ne[1] == 32 &&
        node->src[2]->ne[2] == 512 &&
        node->src[2]->ne[3] == 1;
    const auto * recurrent_layer =
        ggml_backend_hrx_find_recurrent_cache_layer(
            device_context, node);
    const auto * gdn_qk_scale_layer =
        ggml_backend_hrx_find_gdn_qk_scale_layer(
            device_context, node);
    const bool possible_moe_router_tail_terminal =
        node->op == GGML_OP_ADD &&
        node->ne[0] == 2048 &&
        (node->ne[1] == 1 || node->ne[1] == 512) &&
        node->ne[2] == 1 && node->ne[3] == 1;
    const bool possible_terminal_qact_down =
        node->op == GGML_OP_MUL_MAT_ID &&
        node->src[0] && node->src[1] && node->src[2] &&
        node->src[0]->type == GGML_TYPE_Q5_K &&
        node->src[0]->ne[0] == 512 &&
        node->src[0]->ne[1] == 2048 &&
        node->src[0]->ne[2] == 256 &&
        node->src[1]->ne[0] == 512 &&
        node->src[1]->ne[1] == 8 &&
        node->src[1]->ne[2] == 512 &&
        node->src[2]->ne[0] == 8 &&
        node->src[2]->ne[1] == 512;
    const bool graph_sensitive_fusion_node =
        device_context->current_graph &&
        (possible_recurrent_node ||
         possible_gdn_qk_scale ||
         possible_gdn_terminal_mul ||
         possible_decode_gdn_terminal_mul ||
         possible_fa_terminal_mul ||
         possible_moe_router_tail_terminal ||
         possible_terminal_qact_down ||
         node->op == GGML_OP_ARGSORT ||
         node->op == GGML_OP_DIV ||
         (node->op == GGML_OP_GLU &&
          node->src[0] && node->src[1] &&
          node->src[0]->op == GGML_OP_MUL_MAT_ID &&
          node->src[1]->op == GGML_OP_MUL_MAT_ID));
    const uint64_t graph_uid =
        device_context->current_graph ? device_context->current_graph->uid : 0;
    const bool graph_context_matches =
        !graph_sensitive_fusion_node ||
        (graph_uid != 0 && memo_it != device_context->resolved_dispatches.end() &&
         memo_it->second.graph_uid == graph_uid &&
         (!possible_recurrent_node || recurrent_layer != nullptr) &&
         (!possible_gdn_qk_scale || gdn_qk_scale_layer != nullptr));
    const bool memo_hit =
        memo_it != device_context->resolved_dispatches.end() &&
        memo_it->second.signature == node_signature &&
        memo_it->second.compiled != nullptr &&
        graph_context_matches;
    const bool prepass_plan_hit =
        memo_hit &&
        !memo_it->second.prepass_free &&
        ggml_backend_hrx_resolved_prepass_plan_matches(
            device_context,
            node,
            node_signature,
            memo_it->second,
            &bindings);
    const ggml_backend_hrx_catalog_route * route = nullptr;
    if (memo_hit &&
        (memo_it->second.prepass_free || prepass_plan_hit)) {
        request.tensors.assign(memo_it->second.tensors.begin(), memo_it->second.tensors.end());
        request.constants.assign(memo_it->second.constants.begin(), memo_it->second.constants.end());
        route = memo_it->second.route;
        if (prepass_plan_hit &&
            ggml_backend_hrx_trace_enabled(device_context->reg_context)) {
            request.problem.shape = memo_it->second.prepass_plan->shape;
        }
    } else {
        if (!ggml_backend_hrx_make_dispatch_request(device_context, node, &request) ||
            !device_context->reg_context || !device_context->reg_context->catalog) {
            return false;
        }
        ggml_backend_hrx_add_tensor_overlap_facts(&request.problem, request.tensors);
        route = ggml_backend_hrx_catalog_find_route(*device_context->reg_context->catalog, request.problem);
        if (!route) {
            return false;
        }
    }
    size_t prepass_bindings = 0;
    for (const auto & prepass : route->prepasses) {
        if (prepass.dst_index < 0) {
            prepass_bindings +=
                prepass.bytes_per_element != 0 ? 1 : 3;
        }
    }
    const size_t expected_bindings = request.tensors.size() + prepass_bindings;
    if (route->binding_count != expected_bindings || route->constant_byte_length != request.constants.size()) {
        ggml_backend_hrx_trace_event(device_context->reg_context, {
            {"event", "route_rejected"},
            {"reason", "request_abi_mismatch"},
            {"route_id", route->id},
            {"binding_count", route->binding_count},
            {"request_binding_count", expected_bindings},
            {"constant_byte_length", route->constant_byte_length},
            {"request_constant_byte_length", request.constants.size()},
        });
        return false;
    }

    std::vector<ggml_backend_hrx_catalog_binding> resolved_storage;
    std::vector<int64_t> workload_storage;
    const std::vector<ggml_backend_hrx_catalog_binding> * resolved_bindings_p = nullptr;
    const std::vector<int64_t> * workload_arguments_p = nullptr;
    ggml_backend_hrx_compiled_route * compiled = nullptr;
    if (memo_hit && memo_it->second.route == route) {
        resolved_bindings_p = &memo_it->second.bindings;
        workload_arguments_p = &memo_it->second.workload;
        compiled = memo_it->second.compiled;
    } else {
        std::string binding_error;
        if (!ggml_backend_hrx_catalog_make_config_bindings(*route, request.problem, &resolved_storage, &binding_error)) {
            GGML_LOG_ERROR("%s: %s\n", __func__, binding_error.c_str());
            return false;
        }
        std::string workload_error;
        if (!ggml_backend_hrx_resolve_workload_arguments(*route, request.problem, &workload_storage, &workload_error)) {
            GGML_LOG_ERROR("%s: %s\n", __func__, workload_error.c_str());
            return false;
        }
        compiled = ggml_backend_hrx_get_compiled_route(device_context, *route, request.problem, resolved_storage, workload_storage);
        if (!compiled || !compiled->executable) {
            return false;
        }
        auto & memo = device_context->resolved_dispatches[node];
        memo.signature = node_signature;
        memo.graph_uid = graph_sensitive_fusion_node ? graph_uid : 0;
        memo.route = route;
        memo.bindings = resolved_storage;
        memo.workload = workload_storage;
        memo.tensors = request.tensors;
        memo.constants = request.constants;
        memo.prepass_free = route->prepasses.empty();
        memo.compiled = compiled;
        memo.prepass_plan.reset();
        resolved_bindings_p = &memo.bindings;
        workload_arguments_p = &memo.workload;
    }
    const std::vector<ggml_backend_hrx_catalog_binding> & resolved_bindings = *resolved_bindings_p;
    const std::vector<int64_t> & workload_arguments = *workload_arguments_p;
    (void) resolved_bindings;
    (void) workload_arguments;

    if (!prepass_plan_hit) {
        bindings.assign(request.tensors.size(), hrx_buffer_ref_t{});
        for (size_t i = 0; i < request.tensors.size(); ++i) {
            if (!ggml_backend_hrx_make_tensor_binding(
                    device_context, request.tensors[i], &bindings[i])) {
                return false;
            }
        }
    }

    std::optional<ggml_backend_hrx_device_context::resolved_prepass_plan>
        pending_prepass_plan;
    if (!prepass_plan_hit &&
        !route->prepasses.empty() &&
        device_context->current_graph &&
        device_context->current_graph->uid != 0 &&
        device_context->current_node_index >= 0 &&
        device_context->current_node_index <
            device_context->current_graph->n_nodes &&
        device_context->current_graph->nodes[
            device_context->current_node_index] == node) {
        pending_prepass_plan.emplace();
        auto & pending = *pending_prepass_plan;
        pending.graph_uid = device_context->current_graph->uid;
        pending.node_index = device_context->current_node_index;
        pending.node_count = device_context->current_graph->n_nodes;
        pending.catalog = device_context->reg_context->catalog.get();
        pending.route = route;
        pending.route_id = route->id;
        pending.compiled = compiled;
        pending.executable = compiled->executable;
        pending.export_ordinal = compiled->export_ordinal;
        pending.export_info = compiled->export_info;
        pending.export_name =
            compiled->export_info.name ?
            compiled->export_info.name : "";
        pending.launch_config = compiled->launch_config;
        pending.constants = request.constants;
        pending.workload = workload_arguments;
        pending.direct_bindings = bindings;
        pending.shape = request.problem.shape;
        pending.prepasses.reserve(route->prepasses.size());
    }

    const auto * cached_prepass_plan =
        prepass_plan_hit ?
        &*memo_it->second.prepass_plan : nullptr;
    for (size_t prepass_index = 0;
         prepass_index < route->prepasses.size();
         ++prepass_index) {
        const auto & prepass = route->prepasses[prepass_index];
        if (prepass.src_indices.empty() ||
            std::any_of(
                prepass.src_indices.begin(),
                prepass.src_indices.end(),
                [](uint32_t index) {
                    return index >= request.tensors.size();
                }) ||
            (prepass.dst_index >= 0 &&
             static_cast<size_t>(prepass.dst_index) >=
                 request.tensors.size())) {
            GGML_LOG_ERROR(
                "%s: route %s prepass source or destination index out of range\n",
                __func__, route->id.c_str());
            return false;
        }
        const auto * cached_prepass =
            cached_prepass_plan ?
            &cached_prepass_plan->prepasses[prepass_index] : nullptr;
        int64_t elements = cached_prepass ? cached_prepass->elements : 1;
        if (!cached_prepass) {
            for (const auto & source : prepass.scratch_element_sources) {
                // Shape sources are written "shape.k" but the problem map is keyed "k".
                const std::string key = source.compare(0, 6, "shape.") == 0 ? source.substr(6) : source;
                const auto it = request.problem.shape.find(key);
                if (it == request.problem.shape.end()) {
                    GGML_LOG_ERROR("%s: route %s prepass scratch source %s missing from problem shape\n",
                                   __func__, route->id.c_str(), source.c_str());
                    return false;
                }
                elements *= it->second;
            }
        }

        size_t output_count = 0;
        std::array<size_t, 3> output_offsets = {};
        std::array<size_t, 3> output_lengths = {};
        size_t scratch_bytes = 0;
        if (cached_prepass) {
            output_count = cached_prepass->output_count;
            output_offsets = cached_prepass->output_offsets;
            output_lengths = cached_prepass->output_lengths;
            scratch_bytes = cached_prepass->scratch_bytes;
        } else if (prepass.bytes_per_element != 0) {
            output_count = 1;
            output_lengths[0] =
                static_cast<size_t>(elements) *
                prepass.bytes_per_element;
            scratch_bytes = output_lengths[0];
        } else {
            output_count = 3;
            output_lengths[0] = static_cast<size_t>(elements);
            output_offsets[1] = ggml_backend_hrx_align_up(
                output_lengths[0], GGML_HRX_ALIGNMENT);
            output_lengths[1] =
                static_cast<size_t>(elements / 32) * sizeof(float);
            output_offsets[2] = ggml_backend_hrx_align_up(
                output_offsets[1] + output_lengths[1],
                GGML_HRX_ALIGNMENT);
            output_lengths[2] =
                static_cast<size_t>(elements / 16) * sizeof(int32_t);
            scratch_bytes =
                output_offsets[2] + output_lengths[2];
        }

        ggml_backend_hrx_device_context::resolved_prepass *
            pending_prepass = nullptr;
        if (pending_prepass_plan) {
            pending_prepass_plan->prepasses.emplace_back();
            pending_prepass =
                &pending_prepass_plan->prepasses.back();
            pending_prepass->descriptor_index = prepass_index;
            pending_prepass->descriptor_identity = &prepass;
            pending_prepass->descriptor = prepass;
            pending_prepass->elements = elements;
            pending_prepass->output_count = output_count;
            pending_prepass->output_offsets = output_offsets;
            pending_prepass->output_lengths = output_lengths;
            pending_prepass->scratch_bytes = scratch_bytes;
        }

        std::vector<hrx_buffer_ref_t> outputs;
        outputs.reserve(output_count);
        bool already_done = false;
        const bool persist = prepass.persistent &&
                             ggml_backend_hrx_prepass_can_persist(request.tensors[prepass.src_index]);
        if (prepass.dst_index >= 0) {
            if (output_count != 1 ||
                prepass.bytes_per_element == 0) {
                return false;
            }
            hrx_buffer_ref_t output =
                bindings[static_cast<size_t>(prepass.dst_index)];
            const size_t bytes = output_lengths[0];
            if (output.length < bytes) {
                GGML_LOG_ERROR(
                    "%s: route %s direct prepass output is %zu bytes, needs %zu\n",
                    __func__, route->id.c_str(), output.length, bytes);
                return false;
            }
            output.length = bytes;
            outputs.push_back(output);
        } else if (prepass.bytes_per_element != 0) {
            const size_t bytes = output_lengths[0];
            if (!prepass.scratch_class.empty()) {
                // Its own region, and always re-run: the table depends on the
                // routing tensor's contents, which change every graph.
                hrx_buffer_t buffer = nullptr;
                bool fresh = false;
                const bool terminal_qact_owner =
                    prepass.scratch_class == "mmid" &&
                    device_context->current_terminal_qact_plan &&
                    device_context->current_terminal_qact_plan->ready;
                const auto fusion =
                    route->supports.find("fusion");
                const bool terminal_qact_binding =
                    terminal_qact_owner &&
                    fusion != route->supports.end() &&
                    (fusion->second ==
                         "MUL_MAT_ID_SWIGLU_Q5_DOWN_QACT_EPILOGUE" ||
                     fusion->second ==
                         "MUL_MAT_ID_Q5_DOWN_TERMINAL_QACT");
                const size_t reserve_bytes =
                    terminal_qact_owner
                        ? std::max(
                              bytes,
                              GGML_HRX_MMID_QACT_END)
                        : bytes;
                const size_t binding_bytes =
                    terminal_qact_binding
                        ? GGML_HRX_MMID_QACT_END
                        : bytes;
                if (!ggml_backend_hrx_reserve_class_scratch(
                        device_context, prepass.scratch_class,
                        reserve_bytes, &buffer, &fresh)) {
                    return false;
                }
                outputs.push_back({ buffer, 0, binding_bytes });
                // The gate, up and down projections of one layer route through
                // the same ids tensor, so the table only has to be built for
                // the first of them.
                const ggml_tensor * src = request.tensors[prepass.src_index];
                const ggml_tensor *& cached =
                    device_context->class_scratch_source[prepass.scratch_class];
                uint64_t & epoch = device_context->class_scratch_epoch[prepass.scratch_class];
                already_done = !fresh && cached == src && epoch == device_context->graph_epoch;
                if (!already_done) {
                    cached = src;
                    epoch = device_context->graph_epoch;
                }
            } else {
            bool in_arena = false;
            if (persist) {
                const auto key = std::make_pair(
                    bindings[prepass.src_index].buffer,
                    bindings[prepass.src_index].offset);
                const auto found = device_context->weight_arena_offsets.find(key);
                size_t offset = 0;
                if (found != device_context->weight_arena_offsets.end()) {
                    offset = found->second;
                    already_done = true;
                    in_arena = true;
                } else if (ggml_backend_hrx_reserve_weight_arena(device_context, bytes, &offset)) {
                    device_context->weight_arena_offsets[key] = offset;
                    in_arena = true;
                }
                if (in_arena) {
                    outputs.push_back({ device_context->weight_arena, offset, bytes });
                }
            }
            if (!in_arena) {
                if (prepass.persistent) {
                    static bool warned = false;
                    if (!warned) {
                        warned = true;
                        GGML_LOG_WARN("%s: persistent prepass %s fell back to the shared "
                                      "quant scratch (%zu MiB needed, arena %zu/%zu MiB, "
                                      "can_persist=%d); it will re-run on every dispatch\n",
                                      __func__, prepass.export_name.c_str(), bytes >> 20,
                                      device_context->weight_arena_used >> 20,
                                      device_context->weight_arena_capacity >> 20,
                                      (int) ggml_backend_hrx_prepass_can_persist(
                                          request.tensors[prepass.src_index]));
                    }
                }
                if (!ggml_backend_hrx_reserve_quant_scratch(device_context, bytes)) {
                    return false;
                }
                outputs.push_back({ device_context->quant_scratch, 0, bytes });
                already_done = ggml_backend_hrx_quant_scratch_matches(
                    device_context,
                    request.tensors[prepass.src_index],
                    bindings[prepass.src_index],
                    bytes,
                    prepass.artifact_id);
                if (!already_done) {
                    ggml_backend_hrx_set_quant_scratch_source(
                        device_context,
                        request.tensors[prepass.src_index],
                        bindings[prepass.src_index],
                        bytes,
                        prepass.artifact_id);
                }
            }
            }
        } else {
            // int8 payload, then one f32 scale per 32-element block, then one
            // i32 sum per 16. Three bindings so each kernel view starts at 0.
            const size_t quant_bytes = output_lengths[0];
            const size_t scale_offset = output_offsets[1];
            const size_t scale_bytes = output_lengths[1];
            const size_t sum_offset = output_offsets[2];
            const size_t sum_bytes = output_lengths[2];
            bool in_arena = false;
            bool in_class = false;
            if (persist) {
                const auto key = std::make_pair(
                    bindings[prepass.src_index].buffer,
                    bindings[prepass.src_index].offset);
                const auto found = device_context->weight_arena_offsets.find(key);
                size_t arena_base = 0;
                if (found != device_context->weight_arena_offsets.end()) {
                    arena_base = found->second;
                    already_done = true;
                    in_arena = true;
                } else if (ggml_backend_hrx_reserve_weight_arena(device_context, scratch_bytes, &arena_base)) {
                    device_context->weight_arena_offsets[key] = arena_base;
                    in_arena = true;
                }
                if (in_arena) {
                    outputs.push_back({ device_context->weight_arena, arena_base, quant_bytes });
                    outputs.push_back({ device_context->weight_arena, arena_base + scale_offset, scale_bytes });
                    outputs.push_back({ device_context->weight_arena, arena_base + sum_offset, sum_bytes });
                }
            }
            if (!in_arena && !prepass.scratch_class.empty()) {
                hrx_buffer_t buffer = nullptr;
                bool fresh = false;
                if (!ggml_backend_hrx_reserve_class_scratch(
                        device_context, prepass.scratch_class, scratch_bytes, &buffer, &fresh)) {
                    return false;
                }
                outputs.push_back({ buffer, 0, quant_bytes });
                outputs.push_back({ buffer, scale_offset, scale_bytes });
                outputs.push_back({ buffer, sum_offset, sum_bytes });
                in_class = true;
                const ggml_tensor * src = request.tensors[prepass.src_index];
                const ggml_tensor *& cached =
                    device_context->class_scratch_source[prepass.scratch_class];
                uint64_t & epoch = device_context->class_scratch_epoch[prepass.scratch_class];
                already_done = !fresh && cached == src && epoch == device_context->graph_epoch;
                if (!already_done) {
                    cached = src;
                    epoch = device_context->graph_epoch;
                }
            }
            if (!in_arena && !in_class) {
                if (!ggml_backend_hrx_reserve_quant_scratch(device_context, scratch_bytes)) {
                    return false;
                }
                outputs.push_back({ device_context->quant_scratch, 0, quant_bytes });
                outputs.push_back({ device_context->quant_scratch, scale_offset, scale_bytes });
                outputs.push_back({ device_context->quant_scratch, sum_offset, sum_bytes });
                already_done = ggml_backend_hrx_quant_scratch_matches(
                    device_context,
                    request.tensors[prepass.src_index],
                    bindings[prepass.src_index],
                    scratch_bytes,
                    prepass.artifact_id);
                if (!already_done) {
                    ggml_backend_hrx_set_quant_scratch_source(
                        device_context,
                        request.tensors[prepass.src_index],
                        bindings[prepass.src_index],
                        scratch_bytes,
                        prepass.artifact_id);
                }
            }
        }

        if (!already_done) {
            std::vector<ggml_backend_hrx_catalog_binding>
                pre_resolved;
            ggml_backend_hrx_compiled_route * pre_compiled =
                cached_prepass ? cached_prepass->compiled : nullptr;
            hrx_dispatch_config_t pre_config =
                cached_prepass ?
                cached_prepass->dispatch_config :
                hrx_dispatch_config_t{};
            if (!cached_prepass &&
                !ggml_backend_hrx_resolve_prepass_executable(
                    device_context,
                    *route,
                    prepass,
                    request.problem,
                    elements,
                    outputs.size(),
                    &pre_resolved,
                    &pre_compiled,
                    &pre_config)) {
                return false;
            }
            if (pending_prepass) {
                ggml_backend_hrx_set_resolved_prepass_executable(
                    pending_prepass,
                    std::move(pre_resolved),
                    pre_compiled,
                    pre_config);
            }
            std::vector<hrx_buffer_ref_t> pre_bindings;
            pre_bindings.reserve(
                prepass.src_indices.size() + outputs.size());
            for (uint32_t index : prepass.src_indices) {
                pre_bindings.push_back(bindings[index]);
            }
            for (const auto & out : outputs) {
                pre_bindings.push_back(out);
            }
            if (!GGML_HRX_CHECK(hrx_stream_dispatch(
                    device_context->active_stream, pre_compiled->executable, pre_compiled->export_ordinal,
                    &pre_config, nullptr, 0, pre_bindings.data(), pre_bindings.size(),
                    HRX_DISPATCH_FLAG_NONE))) {
                return false;
            }
            // Dispatches on a stream are otherwise free to overlap, and the
            // consumer reads every byte the prepass writes.
            if (!GGML_HRX_CHECK(hrx_stream_execution_barrier(device_context->active_stream))) {
                return false;
            }
        } else if (pending_prepass) {
            std::vector<ggml_backend_hrx_catalog_binding>
                pre_resolved;
            ggml_backend_hrx_compiled_route * pre_compiled =
                nullptr;
            hrx_dispatch_config_t pre_config = {};
            if (ggml_backend_hrx_resolve_prepass_executable(
                    device_context,
                    *route,
                    prepass,
                    request.problem,
                    elements,
                    outputs.size(),
                    &pre_resolved,
                    &pre_compiled,
                    &pre_config)) {
                ggml_backend_hrx_set_resolved_prepass_executable(
                    pending_prepass,
                    std::move(pre_resolved),
                    pre_compiled,
                    pre_config);
            } else {
                pending_prepass_plan.reset();
            }
        }
        if (prepass.dst_index < 0) {
            for (const auto & out : outputs) {
                bindings.push_back(out);
            }
        }
    }

    hrx_dispatch_config_t dispatch_config = {
        /* .workgroup_count = */ {
            compiled->launch_config.workgroup_count[0],
            compiled->launch_config.workgroup_count[1],
            compiled->launch_config.workgroup_count[2],
        },
        /* .workgroup_size = */ {
            compiled->launch_config.workgroup_size[0],
            compiled->launch_config.workgroup_size[1],
            compiled->launch_config.workgroup_size[2],
        },
        /* .subgroup_size = */ compiled->launch_config.subgroup_size,
    };

    if (ggml_backend_hrx_trace_enabled(device_context->reg_context)) {
        nlohmann::json binding_trace = nlohmann::json::array();
        for (size_t i = 0; i < request.tensors.size(); ++i) {
            const ggml_tensor * tensor = request.tensors[i];
            const ggml_tensor * owner = tensor;
            size_t view_depth = 0;
            while (owner && owner->view_src) {
                owner = owner->view_src;
                ++view_depth;
            }
            ggml_backend_buffer_t owner_buffer = owner ? owner->buffer : nullptr;
            binding_trace.push_back({
                {"index", i},
                {"name", tensor ? tensor->name : ""},
                {"buffer", reinterpret_cast<uintptr_t>(bindings[i].buffer)},
                {"offset", bindings[i].offset},
                {"length", bindings[i].length},
                {"view_offset", tensor ? tensor->view_offs : 0},
                {"view_depth", view_depth},
                {"buffer_usage", owner_buffer ?
                    static_cast<int>(ggml_backend_buffer_get_usage(owner_buffer)) : -1},
                {"op", tensor ? ggml_op_name(tensor->op) : ""},
            });
        }
        for (size_t i = request.tensors.size(); i < bindings.size(); ++i) {
            binding_trace.push_back({
                {"index", i},
                {"name", "prepass_scratch"},
                {"buffer", reinterpret_cast<uintptr_t>(bindings[i].buffer)},
                {"offset", bindings[i].offset},
                {"length", bindings[i].length},
            });
        }
        nlohmann::json constant_trace = nlohmann::json::array();
        for (size_t offset = 0; offset + sizeof(float) <= request.constants.size(); offset += sizeof(float)) {
            float value = 0.0f;
            std::memcpy(&value, request.constants.data() + offset, sizeof(float));
            constant_trace.push_back(value);
        }

    ggml_backend_hrx_trace_event(device_context->reg_context, {
        {"event", "route_dispatch"},
        {"device", device_context->name},
        {"route_id", route->id},
        {"shape", request.problem.shape},
        {"bindings", binding_trace},
        {"constants_f32", constant_trace},
        {"launch_workload_argument_count", compiled->launch_config.workload_argument_count},
        {"workgroup_count", {
            dispatch_config.workgroup_count[0],
            dispatch_config.workgroup_count[1],
            dispatch_config.workgroup_count[2],
        }},
        {"workgroup_size", {
            dispatch_config.workgroup_size[0],
            dispatch_config.workgroup_size[1],
            dispatch_config.workgroup_size[2],
        }},
    });
    }
    ggml_backend_hrx_test_record_dispatch(route->id);
    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
        device_context->active_stream,
        compiled->executable,
        compiled->export_ordinal,
        &dispatch_config,
        request.constants.empty() ? nullptr : request.constants.data(),
        request.constants.size(),
        bindings.data(),
        bindings.size(),
        HRX_DISPATCH_FLAG_NONE))) {
        return false;
    }

    if (pending_prepass_plan &&
        pending_prepass_plan->prepasses.size() ==
            route->prepasses.size()) {
        bool complete = true;
        for (size_t i = 0;
             i < pending_prepass_plan->prepasses.size();
             ++i) {
            if (!ggml_backend_hrx_resolved_prepass_is_consistent(
                    pending_prepass_plan->prepasses[i],
                    route->prepasses[i],
                    i,
                    pending_prepass_plan->shape)) {
                complete = false;
                break;
            }
        }
        const auto publish =
            device_context->resolved_dispatches.find(node);
        if (complete &&
            publish != device_context->resolved_dispatches.end() &&
            publish->second.signature == node_signature &&
            publish->second.route == route &&
            publish->second.compiled == compiled &&
            publish->second.constants ==
                pending_prepass_plan->constants &&
            publish->second.workload ==
                pending_prepass_plan->workload) {
            publish->second.prepass_plan =
                std::move(*pending_prepass_plan);
        }
    }
    return true;
}

static bool ggml_backend_hrx_graph_contains_node(
        const ggml_cgraph * cgraph,
        const ggml_tensor * tensor) {
    if (!cgraph || !tensor) {
        return false;
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        if (cgraph->nodes[i] == tensor) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx_has_only_expected_consumers(
        const ggml_cgraph * cgraph,
        const ggml_tensor * producer,
        const ggml_tensor * expected0,
        const ggml_tensor * expected1 = nullptr) {
    if (!cgraph || !producer || !expected0) {
        return false;
    }
    bool found0 = false;
    bool found1 = expected1 == nullptr;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * consumer = cgraph->nodes[i];
        if (!consumer) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (consumer->src[s] != producer) {
                continue;
            }
            if (consumer == expected0) {
                found0 = true;
            } else if (consumer == expected1) {
                found1 = true;
            } else {
                return false;
            }
        }
    }
    return found0 && found1;
}

static bool ggml_backend_hrx_topk_moe_graph_match_is_safe(
        const ggml_cgraph * cgraph,
        const ggml_backend_hrx_topk_moe_softmax_norm_match & match) {
    // Requiring every matched tensor to be an actual graph node avoids treating
    // a coincidental source chain outside this scheduled graph as dead.
    for (const ggml_tensor * tensor : {
            match.softmax,
            match.probs_reshape,
            match.argsort,
            match.ids_view,
            match.get_rows,
            match.weights_reshape,
            match.sum_rows,
            match.clamp,
            match.div}) {
        if (!ggml_backend_hrx_graph_contains_node(cgraph, tensor)) {
            return false;
        }
    }

    // Every materialized producer that the fusion removes must be private to
    // this chain. The ids VIEW is intentionally exempt: the expert GEMMs consume
    // it after the fused kernel writes its full top-8 result.
    const bool private_chain =
        ggml_backend_hrx_has_only_expected_consumers(
            cgraph, match.softmax, match.probs_reshape, match.argsort) &&
        ggml_backend_hrx_has_only_expected_consumers(
            cgraph, match.probs_reshape, match.get_rows) &&
        ggml_backend_hrx_has_only_expected_consumers(
            cgraph, match.argsort, match.ids_view) &&
        ggml_backend_hrx_has_only_expected_consumers(
            cgraph, match.get_rows, match.weights_reshape) &&
        ggml_backend_hrx_has_only_expected_consumers(
            cgraph, match.weights_reshape, match.sum_rows, match.div) &&
        ggml_backend_hrx_has_only_expected_consumers(
            cgraph, match.sum_rows, match.clamp) &&
        ggml_backend_hrx_has_only_expected_consumers(
            cgraph, match.clamp, match.div);
    if (!private_chain || match.logits->ne[1] != 1) {
        return private_chain;
    }

    // Decode delays the logits read until terminal DIV, where the one-wave
    // kernel also writes an overlapping weights output. The logits must have no
    // other consumer, and no non-absorbed dispatch between logits and DIV may
    // write any byte of their storage.
    if (!ggml_backend_hrx_has_only_expected_consumers(
            cgraph, match.logits, match.softmax)) {
        return false;
    }
    ggml_backend_buffer_t logits_buffer = nullptr;
    size_t logits_offset = 0;
    size_t logits_length = 0;
    if (!ggml_backend_hrx_tensor_storage_range(
            match.logits, &logits_buffer, &logits_offset, &logits_length)) {
        return false;
    }
    bool after_logits = false;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * between = cgraph->nodes[i];
        if (between == match.logits) {
            after_logits = true;
            continue;
        }
        if (!after_logits) {
            continue;
        }
        if (between == match.div) {
            return true;
        }
        if (!between ||
            between == match.softmax ||
            between == match.probs_reshape ||
            between == match.argsort ||
            between == match.ids_view ||
            between == match.get_rows ||
            between == match.weights_reshape ||
            between == match.sum_rows ||
            between == match.clamp ||
            ggml_backend_hrx_is_metadata_op(between) ||
            ggml_backend_hrx_is_empty_op(between)) {
            continue;
        }
        ggml_backend_buffer_t between_buffer = nullptr;
        size_t between_offset = 0;
        size_t between_length = 0;
        if (!ggml_backend_hrx_tensor_storage_range(
                between, &between_buffer, &between_offset, &between_length)) {
            return false;
        }
        if (between_buffer == logits_buffer &&
            std::max(logits_offset, between_offset) <
                std::min(logits_offset + logits_length,
                         between_offset + between_length)) {
            return false;
        }
    }
    return false;
}

static bool ggml_backend_hrx_topk_moe_match_absorbs(
        const ggml_backend_hrx_topk_moe_softmax_norm_match & match,
        const ggml_tensor * producer) {
    return producer == match.softmax ||
           (match.logits->ne[1] == 1 && producer == match.argsort) ||
           producer == match.get_rows ||
           producer == match.sum_rows ||
           producer == match.clamp;
}

static int ggml_backend_hrx_graph_node_index(
        const ggml_cgraph * cgraph,
        const ggml_tensor * tensor) {
    if (!cgraph || !tensor) {
        return -1;
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        if (cgraph->nodes[i] == tensor) {
            return i;
        }
    }
    return -1;
}

static bool ggml_backend_hrx_parse_exact_layer_name(
        const ggml_tensor * tensor,
        const char * prefix,
        int * out_layer) {
    if (!tensor || !prefix || !out_layer) {
        return false;
    }
    const char * name = ggml_get_name(tensor);
    const size_t prefix_length = std::strlen(prefix);
    if (!name || std::strncmp(name, prefix, prefix_length) != 0) {
        return false;
    }
    const char * first = name + prefix_length;
    const char * last = name + std::strlen(name);
    int layer = -1;
    const auto parsed = std::from_chars(first, last, layer);
    if (parsed.ec != std::errc() || parsed.ptr != last ||
        layer < 0 || layer >= 40) {
        return false;
    }
    *out_layer = layer;
    return true;
}

static bool ggml_backend_hrx_has_exact_layer_name(
        const ggml_tensor * tensor,
        const char * prefix,
        int expected_layer) {
    int layer = -1;
    return ggml_backend_hrx_parse_exact_layer_name(
               tensor, prefix, &layer) &&
           layer == expected_layer;
}

struct ggml_backend_hrx_storage_span {
    ggml_backend_buffer_t buffer = nullptr;
    size_t offset = 0;
    size_t length = 0;
};

static bool ggml_backend_hrx_get_storage_span(
        const ggml_tensor * tensor,
        ggml_backend_hrx_storage_span * out) {
    return out &&
           ggml_backend_hrx_tensor_storage_range(
               tensor, &out->buffer, &out->offset, &out->length);
}

static bool ggml_backend_hrx_storage_contains_at(
        const ggml_backend_hrx_storage_span & outer,
        const ggml_backend_hrx_storage_span & inner,
        size_t expected_offset) {
    return expected_offset <= outer.length &&
           outer.buffer == inner.buffer &&
           inner.offset == outer.offset + expected_offset &&
           inner.length <= outer.length - expected_offset;
}

static bool ggml_backend_hrx_route_has_zero_constraint(
        const ggml_backend_hrx_catalog_route & route,
        const char * source) {
    for (const auto & constraint : route.constraints) {
        if (constraint.source == source &&
            constraint.has_eq_value &&
            constraint.eq_value == 0) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx_moe_router_tail_route_is_exact(
        const ggml_backend_hrx_catalog_route * route) {
    if (!route ||
        route->id != "sum_slices_f32" ||
        route->family != "sum_slices_f32" ||
        route->op != "ADD" ||
        route->source_id != "sum_slices_f32" ||
        route->artifact_id != "sum_slices_f32_loombc" ||
        route->root_symbol != "@hrx2_sum_slices_f32" ||
        route->export_name != "hrx2_sum_slices_f32" ||
        route->binding_count != 4 ||
        route->parameter_count != 4 ||
        route->constant_byte_length != 0 ||
        route->prepasses.size() != 1 ||
        route->constraints.size() != 2 ||
        !ggml_backend_hrx_route_has_zero_constraint(
            *route, "tensor_overlap.0_1") ||
        !ggml_backend_hrx_route_has_zero_constraint(
            *route, "tensor_overlap.0_2")) {
        return false;
    }
    const auto & prepass = route->prepasses[0];
    return prepass.enabled &&
           prepass.artifact_id == "sum_slices_f32_loombc" &&
           prepass.root_symbol == "@hrx2_moe_router_snapshot_f32" &&
           prepass.export_name == "hrx2_moe_router_snapshot_f32" &&
           prepass.src_index == 2 &&
           prepass.bytes_per_element == 4 &&
           prepass.element_binding_key.empty() &&
           !prepass.persistent &&
           prepass.scratch_class == "moe_router_weight_snapshot" &&
           prepass.scratch_element_sources ==
               std::vector<std::string>({
                   "shape.sumslices.ntokens",
                   "shape.sumslices.nslices",
               });
}

static const ggml_backend_hrx_catalog_route *
ggml_backend_hrx_find_request_route(
        ggml_backend_hrx_device_context * device_context,
        const ggml_backend_hrx_dispatch_request & request) {
    if (!device_context || !device_context->reg_context ||
        !device_context->reg_context->catalog) {
        return nullptr;
    }
    ggml_backend_hrx_catalog_problem problem = request.problem;
    ggml_backend_hrx_add_tensor_overlap_facts(
        &problem, request.tensors);
    return ggml_backend_hrx_catalog_find_route(
        *device_context->reg_context->catalog, problem);
}

static bool ggml_backend_hrx_moe_router_tail_down_route_is_exact(
        ggml_backend_hrx_device_context * device_context,
        const ggml_backend_hrx_catalog_route * route,
        const ggml_tensor * down,
        enum ggml_type weight_type,
        int64_t ntokens) {
    if (!device_context || !route || !down ||
        (ntokens != 1 && ntokens != 512)) {
        return false;
    }
    if (weight_type == GGML_TYPE_Q5_K) {
        if (route->id ==
                (ntokens == 1
                     ? "mul_mat_id_q5_k_f32_mmq_gfx1151_wg256_tbl_"
                       "decode_table_down_group4"
                     : "mul_mat_id_q5_k_f32_mmq_gfx1151_wg256_tbl_"
                       "down_group4")) {
            return true;
        }
        const auto * terminal_layer =
            ntokens == 512
                ? ggml_backend_hrx_find_terminal_qact_layer(
                      device_context->current_terminal_qact_plan, down)
                : nullptr;
        return terminal_layer &&
               terminal_layer->down == down &&
               ggml_backend_hrx_terminal_qact_consumer_route_is_exact(
                   route);
    }
    return weight_type == GGML_TYPE_Q6_K &&
           route->id ==
               (ntokens == 1
                    ? "mul_mat_id_q6_k_f32_mmq_gfx1151_wg256_tbl_decode_table"
                    : "mul_mat_id_q6_k_f32_mmq_gfx1151_wg256_tbl");
}

static bool ggml_backend_hrx_moe_router_tail_topk_routes_are_exact(
        ggml_backend_hrx_device_context * device_context,
        const ggml_backend_hrx_topk_moe_softmax_norm_match & match) {
    if (match.logits->ne[1] == 512) {
        ggml_backend_hrx_dispatch_request stage = {};
        ggml_backend_hrx_dispatch_request copy = {};
        if (!ggml_backend_hrx_make_available_topk_moe_requests(
                device_context, match, &stage, &copy)) {
            return false;
        }
        const auto * stage_route =
            ggml_backend_hrx_find_request_route(device_context, stage);
        const auto * copy_route =
            ggml_backend_hrx_find_request_route(device_context, copy);
        return stage_route && copy_route &&
               stage_route->id ==
                   "topk_moe_softmax_norm_stage_f32_pp512_wg256" &&
               copy_route->id ==
                   "topk_moe_weights_copy_f32_pp512_wg256";
    }
    ggml_backend_hrx_dispatch_request decode = {};
    if (match.logits->ne[1] != 1 ||
        !ggml_backend_hrx_make_available_topk_moe_decode_request(
            device_context, match, &decode)) {
        return false;
    }
    const auto * decode_route =
        ggml_backend_hrx_find_request_route(device_context, decode);
    return decode_route &&
           decode_route->id ==
               "topk_moe_softmax_norm_decode_f32_tg1_wg32";
}

static bool ggml_backend_hrx_moe_router_tail_slice_is_exact(
        const ggml_tensor * slice,
        const ggml_tensor * selected,
        const ggml_tensor * dst,
        int64_t ntokens,
        int slot) {
    if (!slice || !selected || !dst || slot < 0 || slot >= 8 ||
        slice->type != GGML_TYPE_F32 ||
        !ggml_backend_hrx_has_exact_shape(slice, 2048, ntokens) ||
        slice->nb[0] != sizeof(float) ||
        slice->nb[1] != 16384 * sizeof(float) ||
        !ggml_backend_hrx_metadata_chain_reaches(slice, selected)) {
        return false;
    }
    ggml_backend_hrx_storage_span selected_span = {};
    ggml_backend_hrx_storage_span slice_span = {};
    if (!ggml_backend_hrx_get_storage_span(selected, &selected_span) ||
        !ggml_backend_hrx_get_storage_span(slice, &slice_span)) {
        return false;
    }
    const size_t expected_offset =
        static_cast<size_t>(slot) * 2048 * sizeof(float);
    return slice_span.buffer == selected_span.buffer &&
           slice_span.offset == selected_span.offset + expected_offset &&
           ggml_backend_hrx_disjoint_storage_spans(slice, dst);
}

static bool ggml_backend_hrx_moe_router_tail_consumers_are_exact(
        const ggml_cgraph * cgraph,
        const ggml_backend_hrx_moe_router_tail_layer_plan & layer,
        const ggml_tensor * router_weights,
        const ggml_tensor * router_value) {
    int down_uses = 0;
    int mul_uses = 0;
    int router_uses = 0;
    std::array<int, 6> add_uses = {};
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * consumer = cgraph->nodes[i];
        if (!consumer || ggml_backend_hrx_is_metadata_op(consumer) ||
            ggml_backend_hrx_is_empty_op(consumer)) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * source = consumer->src[s];
            if (!source) {
                continue;
            }
            if (source == layer.down) {
                if (consumer != layer.mul || s != 0) {
                    return false;
                }
                ++down_uses;
            }
            if (ggml_backend_hrx_metadata_chain_reaches(
                    source, layer.mul)) {
                bool expected = false;
                if (consumer == layer.adds[0]) {
                    expected =
                        (s == 0 && source == layer.adds[0]->src[0]) ||
                        (s == 1 && source == layer.adds[0]->src[1]);
                } else {
                    for (int add = 1; add < 7; ++add) {
                        if (consumer == layer.adds[add] &&
                            s == 1 &&
                            source == layer.adds[add]->src[1]) {
                            expected = true;
                        }
                    }
                }
                if (!expected) {
                    return false;
                }
                ++mul_uses;
            }
            if (ggml_backend_hrx_metadata_chain_reaches(
                    source, router_value)) {
                if (consumer != layer.mul || s != 1 ||
                    source != router_weights) {
                    return false;
                }
                ++router_uses;
            }
            for (int add = 0; add < 6; ++add) {
                if (source == layer.adds[add]) {
                    if (consumer != layer.adds[add + 1] || s != 0) {
                        return false;
                    }
                    ++add_uses[add];
                }
            }
        }
    }
    return down_uses == 1 &&
           mul_uses == 8 &&
           router_uses == 1 &&
           std::all_of(
               add_uses.begin(), add_uses.end(),
               [](int uses) { return uses == 1; });
}

static bool ggml_backend_hrx_moe_router_tail_source_survives(
        const ggml_cgraph * cgraph,
        const ggml_tensor * source_producer,
        const ggml_tensor * source,
        const ggml_backend_hrx_moe_router_tail_layer_plan & layer) {
    const int source_index =
        ggml_backend_hrx_graph_node_index(cgraph, source_producer);
    const int terminal_index = layer.add_indices[6];
    ggml_backend_hrx_storage_span source_span = {};
    if (source_index < 0 || terminal_index <= source_index ||
        !ggml_backend_hrx_get_storage_span(source, &source_span)) {
        return false;
    }
    for (int i = source_index + 1; i < terminal_index; ++i) {
        const ggml_tensor * writer = cgraph->nodes[i];
        if (!writer || ggml_backend_hrx_is_metadata_op(writer) ||
            ggml_backend_hrx_is_empty_op(writer)) {
            continue;
        }
        bool planned_skip = false;
        for (int add = 0; add < 6; ++add) {
            planned_skip = planned_skip || writer == layer.adds[add];
        }
        if (planned_skip) {
            continue;
        }
        ggml_backend_hrx_storage_span writer_span = {};
        if (!ggml_backend_hrx_get_storage_span(writer, &writer_span)) {
            return false;
        }
        if (writer_span.buffer == source_span.buffer &&
            std::max(writer_span.offset, source_span.offset) <
                std::min(
                    writer_span.offset + writer_span.length,
                    source_span.offset + source_span.length)) {
            return false;
        }
    }
    return true;
}

static bool ggml_backend_hrx_match_moe_router_tail_layer(
        ggml_backend_hrx_device_context * device_context,
        const ggml_cgraph * cgraph,
        int layer_number,
        const ggml_tensor * terminal,
        ggml_backend_hrx_moe_router_tail_layer_plan * out_layer) {
    if (!device_context || !cgraph || !terminal || !out_layer ||
        terminal->op != GGML_OP_ADD ||
        !ggml_backend_hrx_has_exact_layer_name(
            terminal, "ffn_moe_out-", layer_number)) {
        return false;
    }

    ggml_backend_hrx_moe_router_tail_layer_plan layer = {};
    layer.layer = layer_number;
    layer.add_indices.fill(-1);
    const ggml_tensor * cursor = terminal;
    for (int add = 6; add >= 0; --add) {
        if (!cursor || cursor->op != GGML_OP_ADD ||
            !cursor->src[0] || !cursor->src[1] ||
            !ggml_backend_hrx_has_exact_shape(
                cursor, 2048, terminal->ne[1]) ||
            !ggml_backend_hrx_is_f32_dense(cursor)) {
            return false;
        }
        layer.adds[add] = cursor;
        layer.add_indices[add] =
            ggml_backend_hrx_graph_node_index(cgraph, cursor);
        if (layer.add_indices[add] < 0) {
            return false;
        }
        cursor = cursor->src[0];
    }
    const ggml_tensor * first_slice = cursor;
    const ggml_tensor * mul = nullptr;
    if (first_slice && first_slice->view_src) {
        mul = first_slice->view_src;
    }
    if (!mul) {
        for (const ggml_tensor * cur = first_slice;
             cur && ggml_backend_hrx_is_metadata_op(cur);) {
            if (cur->src[0]) {
                cur = cur->src[0];
            } else if (cur->view_src) {
                cur = cur->view_src;
            } else {
                cur = nullptr;
            }
            if (cur && cur->op == GGML_OP_MUL) {
                mul = cur;
                break;
            }
        }
    }
    if (!mul || mul->op != GGML_OP_MUL ||
        !mul->src[0] || !mul->src[1]) {
        return false;
    }
    const ggml_tensor * down = mul->src[0];
    const ggml_tensor * router_weights = mul->src[1];
    if (!down || down->op != GGML_OP_MUL_MAT_ID ||
        !down->src[0] || !down->src[1] || !down->src[2] ||
        !ggml_backend_hrx_has_exact_layer_name(
            down, "ffn_moe_down-", layer_number) ||
        !ggml_backend_hrx_has_exact_layer_name(
            mul, "ffn_moe_weighted-", layer_number)) {
        return false;
    }

    const int64_t ntokens = terminal->ne[1];
    if ((ntokens != 1 && ntokens != 512) ||
        !ggml_backend_hrx_has_exact_shape(
            down->src[0], 512, 2048, 256) ||
        (down->src[0]->type != GGML_TYPE_Q5_K &&
         down->src[0]->type != GGML_TYPE_Q6_K) ||
        !ggml_backend_hrx_has_exact_shape(
            down->src[1], 512, 8, ntokens) ||
        !ggml_backend_hrx_is_f32_dense(down->src[1]) ||
        down->src[1]->nb[1] != 512 * sizeof(float) ||
        down->src[1]->nb[2] != 4096 * sizeof(float) ||
        !ggml_backend_hrx_has_exact_shape(
            down->src[2], 8, ntokens) ||
        down->src[2]->type != GGML_TYPE_I32 ||
        down->src[2]->nb[0] != sizeof(int32_t) ||
        down->src[2]->nb[1] != 256 * sizeof(int32_t) ||
        !ggml_backend_hrx_has_exact_shape(down, 2048, 8, ntokens) ||
        !ggml_backend_hrx_is_f32_dense(down) ||
        down->nb[1] != 2048 * sizeof(float) ||
        down->nb[2] != 16384 * sizeof(float) ||
        !ggml_backend_hrx_has_exact_shape(mul, 2048, 8, ntokens) ||
        !ggml_backend_hrx_is_f32_dense(mul) ||
        !ggml_backend_hrx_same_storage_span(down, mul) ||
        !ggml_backend_hrx_has_exact_shape(
            router_weights, 1, 8, ntokens) ||
        !ggml_backend_hrx_is_f32_dense(router_weights) ||
        !ggml_backend_hrx_has_exact_shape(terminal, 2048, ntokens) ||
        !ggml_backend_hrx_is_f32_dense(terminal)) {
        return false;
    }

    std::array<const ggml_tensor *, 8> slices = {};
    slices[0] = first_slice;
    slices[1] = layer.adds[0]->src[1];
    for (int slot = 2; slot < 8; ++slot) {
        if (layer.adds[slot - 1]->src[0] !=
            layer.adds[slot - 2]) {
            return false;
        }
        slices[slot] = layer.adds[slot - 1]->src[1];
    }
    for (int add = 0; add < 7; ++add) {
        if (!ggml_backend_hrx_same_storage_span(
                layer.adds[add], terminal)) {
            return false;
        }
    }
    for (int slot = 0; slot < 8; ++slot) {
        if (!ggml_backend_hrx_moe_router_tail_slice_is_exact(
                slices[slot], mul, terminal, ntokens, slot)) {
            return false;
        }
    }

    layer.down = down;
    layer.mul = mul;
    ggml_backend_hrx_topk_moe_softmax_norm_match topk = {};
    const ggml_tensor * router_value =
        ggml_backend_hrx_zero_offset_source_chain_target(
            router_weights, GGML_OP_DIV);
    if (!router_value ||
        !ggml_backend_hrx_match_topk_moe_early_softmax_norm(
            router_value, &topk) ||
        !ggml_backend_hrx_topk_moe_graph_match_is_safe(cgraph, topk) ||
        !ggml_backend_hrx_moe_router_tail_topk_routes_are_exact(
            device_context, topk) ||
        down->src[2] != topk.ids_view ||
        !ggml_backend_hrx_moe_router_tail_consumers_are_exact(
            cgraph, layer, router_weights, router_value)) {
        return false;
    }

    layer.selected = down;
    layer.dst = terminal;
    layer.ntokens = ntokens;
    layer.router_source =
        ntokens == 512 ? topk.argsort : topk.div;
    layer.router_source_stride = ntokens == 512 ? 256 : 8;
    layer.router_source_offset = ntokens == 512 ? 8 : 0;
    layer.down_index = ggml_backend_hrx_graph_node_index(cgraph, down);
    layer.mul_index = ggml_backend_hrx_graph_node_index(cgraph, mul);
    const int topk_index =
        ggml_backend_hrx_graph_node_index(cgraph, topk.div);
    if (topk_index < 0 || layer.down_index < 0 ||
        layer.mul_index < 0 ||
        !(topk_index < layer.down_index &&
          layer.down_index < layer.mul_index &&
          layer.mul_index < layer.add_indices[0])) {
        return false;
    }
    for (int add = 1; add < 7; ++add) {
        if (layer.add_indices[add - 1] >=
            layer.add_indices[add]) {
            return false;
        }
    }
    if (!ggml_backend_hrx_moe_router_tail_source_survives(
            cgraph, layer.router_source, layer.router_source, layer)) {
        return false;
    }

    ggml_backend_hrx_storage_span selected_span = {};
    ggml_backend_hrx_storage_span dst_span = {};
    ggml_backend_hrx_storage_span source_span = {};
    ggml_backend_hrx_storage_span router_span = {};
    if (!ggml_backend_hrx_get_storage_span(
            layer.selected, &selected_span) ||
        !ggml_backend_hrx_get_storage_span(layer.dst, &dst_span) ||
        !ggml_backend_hrx_get_storage_span(
            layer.router_source, &source_span) ||
        !ggml_backend_hrx_get_storage_span(topk.div, &router_span) ||
        selected_span.length !=
            static_cast<size_t>(ntokens) * 8 * 2048 * sizeof(float) ||
        dst_span.length !=
            static_cast<size_t>(ntokens) * 2048 * sizeof(float) ||
        !ggml_backend_hrx_disjoint_storage_spans(
            layer.selected, layer.dst) ||
        !ggml_backend_hrx_disjoint_storage_spans(
            layer.selected, layer.router_source) ||
        layer.router_source->view_src != nullptr) {
        return false;
    }
    if (ntokens == 512) {
        const size_t expected_source_offset =
            layer_number == 39 ? 0 : 524288;
        if (source_span.length != 524288 ||
            !ggml_backend_hrx_storage_contains_at(
                dst_span, source_span, expected_source_offset) ||
            ((layer_number < 39) !=
             ggml_backend_hrx_storage_contains_at(
                 dst_span, router_span, 0)) ||
            (layer_number == 39 &&
             !ggml_backend_hrx_disjoint_storage_spans(
                 topk.div, layer.dst))) {
            return false;
        }
    } else {
        const bool expected_alias = layer_number < 39;
        const bool aliases_at_admitted_offset =
            ggml_backend_hrx_storage_contains_at(
                dst_span, source_span, 0) ||
            ggml_backend_hrx_storage_contains_at(
                dst_span, source_span, 256);
        if (source_span.length != 32 ||
            (expected_alias != aliases_at_admitted_offset) ||
            (!expected_alias &&
             !ggml_backend_hrx_disjoint_storage_spans(
                 layer.router_source, layer.dst))) {
            return false;
        }
    }

    ggml_backend_hrx_dispatch_request down_request = {};
    if (!ggml_backend_hrx_make_mul_mat_id_request(
            device_context, down, &down_request) ||
        !ggml_backend_hrx_moe_router_tail_down_route_is_exact(
            device_context,
            ggml_backend_hrx_find_request_route(
                device_context, down_request),
            down,
            down->src[0]->type, ntokens)) {
        return false;
    }
    ggml_backend_hrx_dispatch_request tail_request = {};
    if (!ggml_backend_hrx_make_moe_router_tail_request(
            device_context, layer, &tail_request) ||
        !ggml_backend_hrx_moe_router_tail_route_is_exact(
            ggml_backend_hrx_find_request_route(
                device_context, tail_request))) {
        return false;
    }

    *out_layer = layer;
    return true;
}

static bool ggml_backend_hrx_build_moe_router_tail_graph_plan(
        ggml_backend_hrx_device_context * device_context,
        const ggml_cgraph * cgraph,
        ggml_backend_hrx_moe_router_tail_graph_plan * out_plan) {
    if (!device_context || !cgraph || !out_plan ||
        !device_context->reg_context ||
        !device_context->reg_context->catalog) {
        return false;
    }
    ggml_backend_hrx_moe_router_tail_graph_plan candidate = {};
    candidate.graph_uid = cgraph->uid;
    candidate.node_count = cgraph->n_nodes;
    candidate.catalog = device_context->reg_context->catalog.get();
    candidate.examined = true;
    candidate.add_skip_mask.assign(
        static_cast<size_t>(cgraph->n_nodes), 0);

    std::array<const ggml_tensor *, 40> terminals = {};
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        int layer = -1;
        if (!ggml_backend_hrx_parse_exact_layer_name(
                cgraph->nodes[i], "ffn_moe_out-", &layer)) {
            continue;
        }
        if (terminals[layer] != nullptr) {
            return false;
        }
        terminals[layer] = cgraph->nodes[i];
    }
    int q5_layers = 0;
    int q6_layers = 0;
    for (int layer = 0; layer < 40; ++layer) {
        if (!terminals[layer] ||
            !ggml_backend_hrx_match_moe_router_tail_layer(
                device_context, cgraph, layer, terminals[layer],
                &candidate.layers[layer])) {
            return false;
        }
        if (layer == 0) {
            candidate.ntokens = candidate.layers[layer].ntokens;
        } else if (candidate.layers[layer].ntokens !=
                   candidate.ntokens) {
            return false;
        }
        const enum ggml_type weight_type =
            candidate.layers[layer].down->src[0]->type;
        q5_layers += weight_type == GGML_TYPE_Q5_K ? 1 : 0;
        q6_layers += weight_type == GGML_TYPE_Q6_K ? 1 : 0;
        const bool expected_q6 =
            layer == 1 || layer == 34 ||
            layer == 38 || layer == 39;
        if (expected_q6 != (weight_type == GGML_TYPE_Q6_K)) {
            return false;
        }
        // ADD0..ADD5 are replaced by terminal ADD6. The router MUL is proved
        // dead by the same all-layer plan and skipped in the producer path.
        for (int add = 0; add < 6; ++add) {
            const int index =
                candidate.layers[layer].add_indices[add];
            if (index < 0 ||
                candidate.add_skip_mask[
                    static_cast<size_t>(index)] != 0) {
                return false;
            }
            candidate.add_skip_mask[
                static_cast<size_t>(index)] = 1;
        }
    }
    if (q5_layers != 36 || q6_layers != 4 ||
        std::count(
            candidate.add_skip_mask.begin(),
            candidate.add_skip_mask.end(),
            static_cast<uint8_t>(1)) != 240) {
        return false;
    }
    candidate.ready = true;
    *out_plan = std::move(candidate);
    return true;
}

static bool ggml_backend_hrx_moe_router_tail_mul_is_planned(
        const ggml_backend_hrx_moe_router_tail_graph_plan * plan,
        const ggml_tensor * node) {
    if (!plan || !plan->ready || !node) {
        return false;
    }
    for (const auto & layer : plan->layers) {
        if (layer.mul == node) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx_is_fused_producer_node(
        ggml_backend_hrx_device_context * device_context,
        const ggml_cgraph * cgraph,
        int node_index,
        const ggml_backend_hrx_context * owner) {
    if (!device_context || !cgraph || node_index < 0 || node_index >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * producer = cgraph->nodes[node_index];
    // PP512 SSM_CONV writes the sole SiLU consumer directly. Classify that
    // consumer through the existing graph-UID-scoped fusion mask instead of
    // adding a PP-only branch to the steady per-node dispatch loop. This
    // function runs only while rebuilding that mask; decode cache hits never
    // read the PP-only sidecar.
    const auto * ssm_conv_silu_plan =
        device_context->current_gdn_qk_scale_plan
            ? device_context->ssm_conv_silu_plan.get()
            : nullptr;
    const auto * active_catalog =
        device_context->reg_context &&
                device_context->reg_context->catalog
            ? device_context->reg_context->catalog.get()
            : nullptr;
    if (ssm_conv_silu_plan &&
        device_context->current_graph == cgraph &&
        ssm_conv_silu_plan->owner == owner &&
        ssm_conv_silu_plan->valid &&
        ssm_conv_silu_plan->ready &&
        ssm_conv_silu_plan->graph_uid == cgraph->uid &&
        ssm_conv_silu_plan->node_count == cgraph->n_nodes &&
        ssm_conv_silu_plan->catalog == active_catalog &&
        static_cast<size_t>(node_index) <
            ssm_conv_silu_plan->skip_mask.size() &&
        ssm_conv_silu_plan
                ->skip_mask[static_cast<size_t>(node_index)] != 0) {
        ggml_backend_hrx_trace_event(device_context->reg_context, {
            {"event", "fused_producer_skipped"},
            {"producer_op", ggml_op_desc(producer)},
            {"consumer_op", "SSM_CONV"},
            {"route_id",
             "ssm_conv_f32_chan_concat_silu_regblock_wg1024"},
        });
        return true;
    }
    const auto * shared_expert_plan =
        device_context->current_shared_expert_terminal_plan;
    const auto * gdn_rms_side_plan =
        device_context->current_gdn_rms_side_plan;
    if (gdn_rms_side_plan &&
        device_context->current_graph == cgraph &&
        gdn_rms_side_plan->examined &&
        gdn_rms_side_plan->ready &&
        gdn_rms_side_plan->graph_uid == cgraph->uid &&
        gdn_rms_side_plan->node_count == cgraph->n_nodes &&
        gdn_rms_side_plan->catalog == active_catalog &&
        static_cast<size_t>(node_index) <
            gdn_rms_side_plan->skip_mask.size() &&
        gdn_rms_side_plan
                ->skip_mask[static_cast<size_t>(node_index)] != 0) {
        return true;
    }
    if (shared_expert_plan &&
        device_context->current_graph == cgraph &&
        shared_expert_plan->examined &&
        shared_expert_plan->ready &&
        shared_expert_plan->graph_uid == cgraph->uid &&
        shared_expert_plan->node_count == cgraph->n_nodes &&
        shared_expert_plan->catalog == active_catalog &&
        static_cast<size_t>(node_index) <
            shared_expert_plan->skip_mask.size() &&
        shared_expert_plan
                ->skip_mask[static_cast<size_t>(node_index)] != 0) {
        return true;
    }
    if (ggml_backend_hrx_moe_router_tail_mul_is_planned(
            device_context->current_moe_router_tail_plan,
            producer)) {
        return true;
    }
    const bool possible_fa =
        producer && producer->op == GGML_OP_FLASH_ATTN_EXT &&
        producer->ne[0] == 256 && producer->ne[1] == 16 &&
        producer->ne[2] == 512 && producer->ne[3] == 1;
    const bool possible_fa_cont =
        producer && producer->op == GGML_OP_CONT &&
        producer->ne[0] == 4096 && producer->ne[1] == 512 &&
        producer->ne[2] == 1 && producer->ne[3] == 1;
    const bool possible_fa_sigmoid =
        producer && producer->op == GGML_OP_UNARY &&
        ggml_get_unary_op(producer) == GGML_UNARY_OP_SIGMOID &&
        producer->ne[0] == 4096 && producer->ne[1] == 512 &&
        producer->ne[2] == 1 && producer->ne[3] == 1;
    if (possible_fa || possible_fa_cont || possible_fa_sigmoid) {
        for (int i = node_index + 1; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * terminal = cgraph->nodes[i];
            if (!terminal || terminal->op != GGML_OP_MUL) {
                continue;
            }
            ggml_backend_hrx_dispatch_request fused = {};
            ggml_backend_hrx_fa_gate_epilogue_match match = {};
            if (!ggml_backend_hrx_make_fa_gate_epilogue_request(
                    device_context, terminal, &fused, &match) ||
                (match.flash_attn != producer &&
                 match.cont != producer &&
                 match.sigmoid != producer)) {
                continue;
            }
            ggml_backend_hrx_trace_event(device_context->reg_context, {
                {"event", "fused_producer_skipped"},
                {"producer_op", ggml_op_desc(producer)},
                {"consumer_op", ggml_op_desc(terminal)},
                {"route_id",
                 "flash_attn_ext_f32_f16_wmma_gate_epilogue"},
            });
            return true;
        }
    }
    // One-token GDN q8 GEMM -> SiLU -> in-place MUL. Skip exactly the Q8 and
    // SiLU nodes only after the terminal-MUL request has re-proved the complete
    // topology/alias contract and resolved the exact four-binding SCF route.
    const bool possible_decode_gdn_q8 =
        producer && producer->op == GGML_OP_MUL_MAT &&
        producer->src[0] &&
        producer->src[0]->type == GGML_TYPE_Q8_0 &&
        producer->src[0]->ne[0] == 2048 &&
        producer->src[0]->ne[1] == 4096 &&
        producer->ne[0] == 4096 &&
        producer->ne[1] == 1;
    const bool possible_decode_gdn_silu =
        producer && producer->op == GGML_OP_UNARY &&
        ggml_get_unary_op(producer) == GGML_UNARY_OP_SILU &&
        producer->ne[0] == 128 &&
        producer->ne[1] == 32 &&
        producer->ne[2] == 1 &&
        producer->ne[3] == 1;
    if (possible_decode_gdn_q8 || possible_decode_gdn_silu) {
        for (int i = node_index + 1; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * terminal = cgraph->nodes[i];
            if (!terminal || terminal->op != GGML_OP_MUL) {
                continue;
            }
            ggml_backend_hrx_dispatch_request fused = {};
            ggml_backend_hrx_decode_gdn_q8_silu_mul_match match = {};
            if (!ggml_backend_hrx_make_decode_gdn_q8_silu_mul_request(
                    device_context, terminal, &fused, &match) ||
                (match.q8_gemm != producer &&
                 match.silu != producer)) {
                continue;
            }
            ggml_backend_hrx_trace_event(
                device_context->reg_context, {
                    {"event", "fused_producer_skipped"},
                    {"producer_op", ggml_op_desc(producer)},
                    {"consumer_op", ggml_op_desc(terminal)},
                    {"route_id",
                     "mul_mat_q8_0_f32_packed_decode_k2048_"
                     "r4096_c1_wg256_scfunroll2_"
                     "gdn_silu_mul_epilogue"},
                });
            return true;
        }
    }
    // GDN q8 GEMM -> SiLU -> in-place MUL: the terminal fused route computes
    // both materialized producers directly. Re-run the complete topology,
    // alias, consumer and route gate before skipping either one.
    const bool possible_gdn_q8 =
        producer && producer->op == GGML_OP_MUL_MAT &&
        producer->src[0] && producer->src[0]->type == GGML_TYPE_Q8_0 &&
        producer->src[0]->ne[0] == 2048 &&
        producer->src[0]->ne[1] == 4096 &&
        producer->ne[0] == 4096 &&
        producer->ne[1] == 512;
    const bool possible_gdn_silu =
        producer && producer->op == GGML_OP_UNARY &&
        ggml_get_unary_op(producer) == GGML_UNARY_OP_SILU &&
        producer->ne[0] == 128 &&
        ggml_backend_hrx_tensor_row_count(producer) == 16384;
    if (possible_gdn_q8 || possible_gdn_silu) {
        for (int i = node_index + 1; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * terminal = cgraph->nodes[i];
            if (!terminal || terminal->op != GGML_OP_MUL) {
                continue;
            }
            ggml_backend_hrx_dispatch_request fused = {};
            ggml_backend_hrx_gdn_q8_silu_mul_match match = {};
            if (!ggml_backend_hrx_make_gdn_q8_silu_mul_request(
                    device_context, terminal, &fused, &match) ||
                (match.q8_gemm != producer && match.silu != producer)) {
                continue;
            }
            ggml_backend_hrx_trace_event(device_context->reg_context, {
                {"event", "fused_producer_skipped"},
                {"producer_op", ggml_op_desc(producer)},
                {"consumer_op", ggml_op_desc(terminal)},
                {"route_id",
                 "mul_mat_q8_0_f32_wmmai8_gdn_silu_mul_epilogue"},
            });
            return true;
        }
    }
    // Terminal up-MMID + SwiGLU: the GLU dispatch recomputes the later up
    // projection, side-loads the already materialized gate projection and
    // writes the GLU destination. Skip only that exact private up producer,
    // and only after the topology-sensitive fused route has resolved.
    if (producer && producer->op == GGML_OP_MUL_MAT_ID) {
        for (int i = node_index + 1; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * terminal = cgraph->nodes[i];
            if (!terminal || terminal->op != GGML_OP_GLU ||
                terminal->src[1] != producer) {
                continue;
            }
            ggml_backend_hrx_dispatch_request fused = {};
            ggml_backend_hrx_mul_mat_id_swiglu_match match = {};
            if (!ggml_backend_hrx_make_mul_mat_id_swiglu_request(
                    device_context, terminal, &fused, &match) ||
                match.up != producer) {
                continue;
            }
            const auto * terminal_qact_layer =
                ggml_backend_hrx_find_terminal_qact_layer(
                    device_context->current_terminal_qact_plan,
                    terminal);
            const char * route_id = terminal_qact_layer
                ? "mul_mat_id_q4_k_f32_mmq_gfx1151_wg256_"
                  "pre_tbl_swiglu_q5_down_qact"
                : (producer->src[0] &&
                           producer->src[0]->type == GGML_TYPE_Q5_K
                       ? "mul_mat_id_q5_k_f32_mmq_gfx1151_"
                         "wg256_pre_tbl_swiglu"
                       : "mul_mat_id_q4_k_f32_mmq_gfx1151_"
                         "wg256_pre_tbl_swiglu");
            ggml_backend_hrx_trace_event(device_context->reg_context, {
                {"event", "fused_producer_skipped"},
                {"producer_op", ggml_op_desc(producer)},
                {"consumer_op", ggml_op_desc(terminal)},
                {"route_id", route_id},
            });
            return true;
        }
    }
    // PP512 runs as an alias-safe ARGSORT stage plus DIV copy, so those two stay
    // live. Decode runs the entire tail at DIV after one wave has loaded all
    // logits, so ARGSORT is absorbed there as well.
    if (producer &&
        (producer->op == GGML_OP_SOFT_MAX ||
         producer->op == GGML_OP_ARGSORT ||
         producer->op == GGML_OP_GET_ROWS ||
         producer->op == GGML_OP_SUM_ROWS ||
         producer->op == GGML_OP_CLAMP)) {
        for (int i = node_index + 1; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * terminal = cgraph->nodes[i];
            if (!terminal || terminal->op != GGML_OP_DIV) {
                continue;
            }
            ggml_backend_hrx_topk_moe_softmax_norm_match match = {};
            if (!ggml_backend_hrx_match_topk_moe_early_softmax_norm(
                    terminal, &match) ||
                !ggml_backend_hrx_topk_moe_match_absorbs(match, producer) ||
                !ggml_backend_hrx_topk_moe_graph_match_is_safe(cgraph, match)) {
                continue;
            }
            const bool available = match.logits->ne[1] == 1 ?
                ggml_backend_hrx_make_available_topk_moe_decode_request(
                    device_context, match, nullptr) :
                ggml_backend_hrx_make_available_topk_moe_requests(
                    device_context, match, nullptr, nullptr);
            if (!available) {
                continue;
            }
            ggml_backend_hrx_trace_event(device_context->reg_context, {
                {"event", "fused_producer_skipped"},
                {"producer_op", ggml_op_desc(producer)},
                {"consumer_op", ggml_op_desc(terminal)},
                {"route_id", match.logits->ne[1] == 1
                    ? "topk_moe_softmax_norm_decode_f32"
                    : "topk_moe_softmax_norm_stage_f32+topk_moe_weights_copy_f32"},
            });
            return true;
        }
    }
    // The copy of the conv window's tail into the state cache, when the
    // graph-scoped sparse CONCAT has already written the cache itself.
    if (producer && (producer->op == GGML_OP_CPY || producer->op == GGML_OP_CONT)) {
        const ggml_tensor * src = producer->src[0];
        const ggml_tensor * win = src && src->view_src ? src->view_src : src;
        if (win && win->op == GGML_OP_CONCAT) {
            ggml_backend_hrx_dispatch_request tail;
            if (ggml_backend_hrx_make_concat_window_tail_request(
                    device_context, cgraph, win, &tail) &&
                tail.tensors.size() == 4 && tail.tensors[3] == producer &&
                ggml_backend_hrx_request_matches_loaded_route(
                    device_context, tail, "concat_window_tail")) {
                return true;
            }
        }
    }
    if (!producer ||
        (producer->op != GGML_OP_RMS_NORM &&
         producer->op != GGML_OP_ADD &&
         producer->op != GGML_OP_CONT &&
         producer->op != GGML_OP_ROPE &&
         producer->op != GGML_OP_MUL_MAT &&
         producer->op != GGML_OP_SOFT_MAX)) {
        return false;
    }
    for (int i = node_index + 1; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * consumer = cgraph->nodes[i];
        if (!consumer || !consumer->src[0]) {
            continue;
        }
        bool producer_consumed = false;
        const char * expected_family = nullptr;
        if (consumer->op == GGML_OP_MUL) {
            const bool rms_norm_mul_producer =
                producer->op == GGML_OP_RMS_NORM && consumer->src[0] == producer;
            const bool add_rms_norm_mul_producer =
                consumer->src[0]->op == GGML_OP_RMS_NORM &&
                consumer->src[0]->src[0] &&
                consumer->src[0]->src[0]->op == GGML_OP_ADD &&
                ((producer->op == GGML_OP_RMS_NORM && consumer->src[0] == producer) ||
                 (producer->op == GGML_OP_ADD && consumer->src[0]->src[0] == producer));
            producer_consumed = rms_norm_mul_producer || add_rms_norm_mul_producer;
            expected_family = add_rms_norm_mul_producer ? "add_rms_norm_mul_f32" :
                (rms_norm_mul_producer ? "rms_norm_mul_f32" : nullptr);
        } else if (consumer->op == GGML_OP_SET_ROWS) {
            if (producer->op == GGML_OP_CONT &&
                ggml_backend_hrx_zero_offset_source_chain_target(consumer->src[0], GGML_OP_CONT) == producer) {
                producer_consumed = true;
                expected_family = "cont_set_rows_f32";
            } else if (producer->op == GGML_OP_ROPE &&
                       ggml_backend_hrx_zero_offset_source_chain_target(consumer->src[0], GGML_OP_ROPE) == producer) {
                producer_consumed = true;
                expected_family = "rope_set_rows_f32";
            }
        } else if (consumer->op == GGML_OP_GLU) {
            if (producer->op == GGML_OP_MUL_MAT && (consumer->src[0] == producer || consumer->src[1] == producer)) {
                producer_consumed = true;
                expected_family = "mul_mat_q4_k_swiglu_f32";
            }
        } else if (consumer->op == GGML_OP_CONT) {
            if (producer->op == GGML_OP_MUL_MAT &&
                consumer->src[0] &&
                consumer->src[0]->op == GGML_OP_PERMUTE &&
                consumer->src[0]->src[0] == producer) {
                producer_consumed = true;
                expected_family = (producer->src[1] && producer->src[1]->op == GGML_OP_SOFT_MAX) ?
                    "softmax_kqv_f32_f16" : "mul_mat_f16_f32_batched_cont";
            } else if (producer->op == GGML_OP_SOFT_MAX &&
                       consumer->src[0] &&
                       consumer->src[0]->op == GGML_OP_PERMUTE &&
                       consumer->src[0]->src[0] &&
                       consumer->src[0]->src[0]->op == GGML_OP_MUL_MAT &&
                       consumer->src[0]->src[0]->src[1] == producer) {
                producer_consumed = true;
                expected_family = "softmax_kqv_f32_f16";
            }
        }
        if (!producer_consumed || !expected_family) {
            continue;
        }
        ggml_backend_hrx_dispatch_request request = {};
        if (!ggml_backend_hrx_make_dispatch_request(device_context, consumer, &request) ||
            !device_context->reg_context || !device_context->reg_context->catalog) {
            continue;
        }
        ggml_backend_hrx_add_tensor_overlap_facts(&request.problem, request.tensors);
        const auto * route = ggml_backend_hrx_catalog_find_route(*device_context->reg_context->catalog, request.problem);
        if (route && route->family == expected_family) {
            ggml_backend_hrx_trace_event(device_context->reg_context, {
                {"event", "fused_producer_skipped"},
                {"producer_op", ggml_op_desc(producer)},
                {"consumer_op", ggml_op_desc(consumer)},
                {"route_id", route->id},
            });
            return true;
        }
    }
    return false;
}

static constexpr std::array<int, 30>
    ggml_backend_hrx_recurrent_cache_layers = {
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12,
        13, 14, 16, 17, 18, 20, 21, 22, 24, 25,
        26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
    };

static bool ggml_backend_hrx_shape_is(
        const ggml_tensor * tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    return tensor &&
           tensor->ne[0] == ne0 &&
           tensor->ne[1] == ne1 &&
           tensor->ne[2] == ne2 &&
           tensor->ne[3] == ne3;
}

static bool ggml_backend_hrx_storage_subspan(
        const ggml_tensor * child,
        const ggml_tensor * owner,
        size_t relative_offset,
        size_t length) {
    ggml_backend_buffer_t child_buffer = nullptr;
    ggml_backend_buffer_t owner_buffer = nullptr;
    size_t child_offset = 0;
    size_t owner_offset = 0;
    size_t child_length = 0;
    size_t owner_length = 0;
    return ggml_backend_hrx_tensor_storage_range(
               child, &child_buffer, &child_offset, &child_length) &&
           ggml_backend_hrx_tensor_storage_range(
               owner, &owner_buffer, &owner_offset, &owner_length) &&
           relative_offset <= owner_length &&
           length <= owner_length - relative_offset &&
           child_buffer == owner_buffer &&
           child_offset == owner_offset + relative_offset &&
           child_length == length;
}

static bool ggml_backend_hrx_storage_spans_overlap(
        const ggml_tensor * lhs,
        const ggml_tensor * rhs) {
    ggml_backend_buffer_t lhs_buffer = nullptr;
    ggml_backend_buffer_t rhs_buffer = nullptr;
    size_t lhs_offset = 0;
    size_t rhs_offset = 0;
    size_t lhs_length = 0;
    size_t rhs_length = 0;
    if (!ggml_backend_hrx_tensor_storage_range(
            lhs, &lhs_buffer, &lhs_offset, &lhs_length) ||
        !ggml_backend_hrx_tensor_storage_range(
            rhs, &rhs_buffer, &rhs_offset, &rhs_length) ||
        lhs_buffer != rhs_buffer) {
        return false;
    }
    return lhs_offset < rhs_offset + rhs_length &&
           rhs_offset < lhs_offset + lhs_length;
}

static bool ggml_backend_hrx_storage_spans_adjacent(
        const ggml_tensor * lhs,
        const ggml_tensor * rhs) {
    ggml_backend_buffer_t lhs_buffer = nullptr;
    ggml_backend_buffer_t rhs_buffer = nullptr;
    size_t lhs_offset = 0;
    size_t rhs_offset = 0;
    size_t lhs_length = 0;
    size_t rhs_length = 0;
    return ggml_backend_hrx_tensor_storage_range(
               lhs, &lhs_buffer, &lhs_offset, &lhs_length) &&
           ggml_backend_hrx_tensor_storage_range(
               rhs, &rhs_buffer, &rhs_offset, &rhs_length) &&
           lhs_buffer == rhs_buffer &&
           lhs_offset + lhs_length == rhs_offset;
}

static bool ggml_backend_hrx_all_storage_spans_disjoint(
        const std::vector<const ggml_tensor *> & tensors) {
    for (size_t i = 0; i < tensors.size(); ++i) {
        for (size_t j = i + 1; j < tensors.size(); ++j) {
            if (!ggml_backend_hrx_disjoint_storage_spans(
                    tensors[i], tensors[j])) {
                return false;
            }
        }
    }
    return true;
}

static int ggml_backend_hrx_exact_cache_layer(
        const ggml_tensor * tensor,
        char cache_kind) {
    if (!tensor || (cache_kind != 'r' && cache_kind != 's')) {
        return -1;
    }
    for (int layer : ggml_backend_hrx_recurrent_cache_layers) {
        char expected[64];
        std::snprintf(
            expected, sizeof(expected),
            "cache_%c_l%d (reshaped)", cache_kind, layer);
        if (std::strcmp(ggml_get_name(tensor), expected) == 0) {
            return layer;
        }
    }
    return -1;
}

static std::vector<const ggml_tensor *>
ggml_backend_hrx_compute_consumers(
        const ggml_cgraph * cgraph,
        const ggml_tensor * producer) {
    std::vector<const ggml_tensor *> consumers;
    if (!cgraph || !producer) {
        return consumers;
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!node || node == producer ||
            ggml_backend_hrx_is_metadata_op(node) ||
            ggml_backend_hrx_is_empty_op(node)) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (ggml_backend_hrx_metadata_chain_reaches(
                    node->src[s], producer)) {
                consumers.push_back(node);
            }
        }
    }
    return consumers;
}

static bool ggml_backend_hrx_match_terminal_qact_layer(
        ggml_backend_hrx_device_context * device_context,
        const ggml_cgraph * cgraph,
        int layer_number,
        const ggml_tensor * glu,
        ggml_backend_hrx_terminal_qact_layer_plan * out_layer) {
    if (!device_context || !cgraph || !glu || !out_layer ||
        !ggml_backend_hrx_has_exact_layer_name(
            glu, "ffn_moe_swiglu-", layer_number)) {
        return false;
    }

    ggml_backend_hrx_dispatch_request producer_request = {};
    ggml_backend_hrx_mul_mat_id_swiglu_match match = {};
    if (!ggml_backend_hrx_make_terminal_qact_producer_request(
            device_context, glu, &producer_request, &match) ||
        !match.up || !match.down ||
        match.up->src[0]->type != GGML_TYPE_Q4_K ||
        match.down->op != GGML_OP_MUL_MAT_ID ||
        !match.down->src[0] || !match.down->src[1] ||
        !match.down->src[2] ||
        match.down->src[0]->type != GGML_TYPE_Q5_K ||
        match.down->src[1] != glu ||
        match.down->src[2] != match.up->src[2] ||
        !ggml_backend_hrx_is_q5_down_group4_tensor(
            match.down->src[0]) ||
        !ggml_backend_hrx_has_exact_layer_name(
            match.down, "ffn_moe_down-", layer_number) ||
        !ggml_backend_hrx_has_exact_layer_name(
            match.down->src[2], "ffn_moe_topk-", layer_number) ||
        match.down->src[0]->ne[0] != 512 ||
        match.down->src[0]->ne[1] != 2048 ||
        match.down->src[0]->ne[2] != 256 ||
        match.down->src[0]->ne[3] != 1 ||
        match.down->ne[0] != 2048 ||
        match.down->ne[1] != 8 ||
        match.down->ne[2] != 512 ||
        match.down->ne[3] != 1 ||
        match.down->nb[0] != sizeof(float) ||
        match.down->nb[2] != 16384 * sizeof(float) ||
        ggml_nbytes(match.down) != 32 * 1024 * 1024) {
        return false;
    }

    const int glu_index =
        ggml_backend_hrx_graph_node_index(cgraph, glu);
    const int down_index =
        ggml_backend_hrx_graph_node_index(cgraph, match.down);
    if (glu_index < 0 || down_index <= glu_index) {
        return false;
    }
    // The qact suffix is graph-local class scratch. Nothing may dispatch
    // between its producer and consumer and repurpose the shared MMID owner.
    for (int i = glu_index + 1; i < down_index; ++i) {
        const ggml_tensor * between = cgraph->nodes[i];
        if (between &&
            !ggml_backend_hrx_is_metadata_op(between) &&
            !ggml_backend_hrx_is_empty_op(between)) {
            return false;
        }
    }

    ggml_backend_hrx_dispatch_request consumer_request = {};
    if (!ggml_backend_hrx_make_terminal_qact_consumer_request(
            device_context, match.down, &consumer_request)) {
        return false;
    }

    *out_layer = {
        /* .layer = */ layer_number,
        /* .glu_index = */ glu_index,
        /* .down_index = */ down_index,
        /* .glu = */ glu,
        /* .down = */ match.down,
        /* .ids = */ match.down->src[2],
    };
    return true;
}

static bool ggml_backend_hrx_build_terminal_qact_graph_plan(
        ggml_backend_hrx_device_context * device_context,
        const ggml_cgraph * cgraph,
        ggml_backend_hrx_terminal_qact_graph_plan * out_plan) {
    if (!device_context || !cgraph || !out_plan ||
        cgraph->uid == 0 ||
        !device_context->reg_context ||
        !device_context->reg_context->catalog) {
        return false;
    }

    ggml_backend_hrx_terminal_qact_graph_plan candidate = {};
    candidate.graph_uid = cgraph->uid;
    candidate.node_count = cgraph->n_nodes;
    candidate.catalog =
        device_context->reg_context->catalog.get();
    candidate.examined = true;
    std::array<bool, 40> seen = {};
    size_t q5_layers = 0;
    size_t q6_layers = 0;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * glu = cgraph->nodes[i];
        int layer = -1;
        if (!ggml_backend_hrx_parse_exact_layer_name(
                glu, "ffn_moe_swiglu-", &layer)) {
            continue;
        }
        if (seen[static_cast<size_t>(layer)]) {
            return false;
        }
        seen[static_cast<size_t>(layer)] = true;

        ggml_backend_hrx_mul_mat_id_swiglu_match generic = {};
        if (!ggml_backend_hrx_match_current_mul_mat_id_swiglu(
                device_context, glu, &generic) ||
            !generic.down || !generic.down->src[0]) {
            return false;
        }
        if (generic.down->src[0]->type == GGML_TYPE_Q5_K) {
            ggml_backend_hrx_terminal_qact_layer_plan layer_plan = {};
            if (!ggml_backend_hrx_match_terminal_qact_layer(
                    device_context, cgraph, layer, glu,
                    &layer_plan)) {
                return false;
            }
            candidate.layers[static_cast<size_t>(layer)] =
                layer_plan;
            ++q5_layers;
        } else if (
            generic.down->src[0]->type == GGML_TYPE_Q6_K &&
            generic.down->src[0]->ne[0] == 512 &&
            generic.down->src[0]->ne[1] == 2048 &&
            generic.down->src[0]->ne[2] == 256 &&
            generic.down->src[0]->ne[3] == 1 &&
            ggml_backend_hrx_has_exact_layer_name(
                generic.down, "ffn_moe_down-", layer)) {
            ++q6_layers;
        } else {
            return false;
        }
    }

    if (q5_layers != 36 || q6_layers != 4 ||
        std::count(seen.begin(), seen.end(), true) != 40) {
        return false;
    }
    candidate.layer_count = q5_layers;
    candidate.ready = true;
    *out_plan = std::move(candidate);
    return true;
}

static bool ggml_backend_hrx_match_shared_expert_terminal_layer(
        ggml_backend_hrx_device_context * device_context,
        const ggml_cgraph * cgraph,
        int layer_number,
        const ggml_tensor * terminal,
        ggml_backend_hrx_shared_expert_terminal_layer_plan * out_layer) {
    if (!device_context || !cgraph || !terminal || !out_layer ||
        terminal->op != GGML_OP_ADD ||
        !terminal->src[0] || !terminal->src[1] ||
        !ggml_backend_hrx_has_exact_layer_name(
            terminal, "ffn_out-", layer_number)) {
        return false;
    }

    ggml_backend_hrx_shared_expert_terminal_layer_plan layer = {};
    layer.layer = layer_number;
    layer.terminal = terminal;
    layer.addend = terminal->src[0];
    layer.mul = terminal->src[1];
    if (!layer.mul || layer.mul->op != GGML_OP_MUL ||
        !layer.mul->src[0] || !layer.mul->src[1]) {
        return false;
    }
    layer.down = layer.mul->src[0];
    layer.sigmoid = layer.mul->src[1];
    if (!layer.down || layer.down->op != GGML_OP_MUL_MAT ||
        !layer.down->src[0] || !layer.down->src[1] ||
        !layer.sigmoid ||
        layer.sigmoid->op != GGML_OP_UNARY ||
        ggml_get_unary_op(layer.sigmoid) != GGML_UNARY_OP_SIGMOID ||
        !layer.sigmoid->src[0]) {
        return false;
    }
    layer.raw_gate = layer.sigmoid->src[0];
    if (!layer.raw_gate ||
        layer.raw_gate->op != GGML_OP_MUL_MAT ||
        !layer.raw_gate->src[0] || !layer.raw_gate->src[1]) {
        return false;
    }

    char down_weight_name[64];
    char raw_weight_name[64];
    std::snprintf(
        down_weight_name, sizeof(down_weight_name),
        "blk.%d.ffn_down_shexp.weight", layer_number);
    std::snprintf(
        raw_weight_name, sizeof(raw_weight_name),
        "blk.%d.ffn_gate_inp_shexp.weight", layer_number);
    if (!ggml_backend_hrx_has_exact_layer_name(
            layer.down, "ffn_shexp-", layer_number) ||
        !ggml_backend_hrx_has_exact_layer_name(
            layer.down->src[1], "ffn_swiglu-", layer_number) ||
        !ggml_backend_hrx_has_exact_layer_name(
            layer.raw_gate, "shared_expert_gate-", layer_number) ||
        !ggml_backend_hrx_has_exact_layer_name(
            layer.raw_gate->src[1], "attn_post_norm-", layer_number) ||
        !ggml_backend_hrx_has_exact_layer_name(
            layer.sigmoid, "shared_expert_gate_sigmoid-",
            layer_number) ||
        !ggml_backend_hrx_has_exact_layer_name(
            layer.mul, "ffn_shexp_gated-", layer_number) ||
        !ggml_backend_hrx_has_exact_layer_name(
            layer.addend, "ffn_moe_out-", layer_number) ||
        std::strcmp(
            ggml_get_name(layer.down->src[0]), down_weight_name) != 0 ||
        std::strcmp(
            ggml_get_name(layer.raw_gate->src[0]), raw_weight_name) != 0) {
        return false;
    }

    constexpr size_t matrix_bytes =
        2048ull * 512ull * sizeof(float);
    constexpr size_t activation_bytes =
        512ull * 512ull * sizeof(float);
    constexpr size_t gate_bytes = 512ull * sizeof(float);
    if (layer.down->src[0]->type != GGML_TYPE_Q8_0 ||
        layer.down->src[1]->type != GGML_TYPE_F32 ||
        layer.down->type != GGML_TYPE_F32 ||
        layer.raw_gate->src[0]->type != GGML_TYPE_F32 ||
        layer.raw_gate->src[1]->type != GGML_TYPE_F32 ||
        layer.raw_gate->type != GGML_TYPE_F32 ||
        layer.sigmoid->type != GGML_TYPE_F32 ||
        layer.mul->type != GGML_TYPE_F32 ||
        layer.addend->type != GGML_TYPE_F32 ||
        layer.terminal->type != GGML_TYPE_F32 ||
        !ggml_backend_hrx_shape_is(
            layer.down->src[0], 512, 2048, 1, 1) ||
        !ggml_backend_hrx_shape_is(
            layer.down->src[1], 512, 512, 1, 1) ||
        !ggml_backend_hrx_shape_is(
            layer.down, 2048, 512, 1, 1) ||
        !ggml_backend_hrx_shape_is(
            layer.raw_gate->src[0], 2048, 1, 1, 1) ||
        !ggml_backend_hrx_shape_is(
            layer.raw_gate->src[1], 2048, 512, 1, 1) ||
        !ggml_backend_hrx_shape_is(
            layer.raw_gate, 1, 512, 1, 1) ||
        !ggml_backend_hrx_shape_is(
            layer.sigmoid, 1, 512, 1, 1) ||
        !ggml_backend_hrx_shape_is(
            layer.mul, 2048, 512, 1, 1) ||
        !ggml_backend_hrx_shape_is(
            layer.addend, 2048, 512, 1, 1) ||
        !ggml_backend_hrx_shape_is(
            layer.terminal, 2048, 512, 1, 1) ||
        layer.down->src[0]->nb[1] != 544 ||
        layer.down->src[1]->nb[1] != 512 * sizeof(float) ||
        layer.down->nb[1] != 2048 * sizeof(float) ||
        layer.raw_gate->src[0]->nb[1] !=
            2048 * sizeof(float) ||
        layer.raw_gate->src[1]->nb[1] !=
            2048 * sizeof(float) ||
        layer.raw_gate->nb[1] != sizeof(float) ||
        ggml_nbytes(layer.down) != matrix_bytes ||
        ggml_nbytes(layer.down->src[1]) != activation_bytes ||
        ggml_nbytes(layer.raw_gate) != gate_bytes ||
        ggml_nbytes(layer.sigmoid) != gate_bytes ||
        ggml_nbytes(layer.mul) != matrix_bytes ||
        ggml_nbytes(layer.addend) != matrix_bytes ||
        ggml_nbytes(layer.terminal) != matrix_bytes ||
        !ggml_is_contiguous(layer.down->src[0]) ||
        !ggml_backend_hrx_is_f32_dense(layer.down->src[1]) ||
        !ggml_backend_hrx_is_f32_dense(layer.down) ||
        !ggml_backend_hrx_is_f32_dense(layer.raw_gate->src[0]) ||
        !ggml_backend_hrx_is_f32_dense(layer.raw_gate->src[1]) ||
        !ggml_backend_hrx_is_f32_dense(layer.raw_gate) ||
        !ggml_backend_hrx_is_f32_dense(layer.sigmoid) ||
        !ggml_backend_hrx_is_f32_dense(layer.mul) ||
        !ggml_backend_hrx_is_f32_dense(layer.addend) ||
        !ggml_backend_hrx_is_f32_dense(layer.terminal) ||
        !ggml_backend_hrx_same_storage_span(layer.down, layer.mul) ||
        !ggml_backend_hrx_same_storage_span(
            layer.raw_gate, layer.sigmoid) ||
        !ggml_backend_hrx_same_storage_span(
            layer.addend, layer.terminal) ||
        !ggml_backend_hrx_all_storage_spans_disjoint({
            layer.down->src[0],
            layer.down->src[1],
            layer.raw_gate,
            layer.addend,
        }) ||
        !ggml_backend_hrx_all_storage_spans_disjoint({
            layer.down,
            layer.down->src[0],
            layer.down->src[1],
            layer.raw_gate,
            layer.addend,
        }) ||
        (layer.down->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        (layer.sigmoid->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        (layer.mul->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    const auto sole_consumer_is = [cgraph](
            const ggml_tensor * producer,
            const ggml_tensor * expected) {
        const auto consumers =
            ggml_backend_hrx_compute_consumers(cgraph, producer);
        return consumers.size() == 1 && consumers[0] == expected;
    };
    if (!sole_consumer_is(layer.down, layer.mul) ||
        !sole_consumer_is(layer.raw_gate, layer.sigmoid) ||
        !sole_consumer_is(layer.sigmoid, layer.mul) ||
        !sole_consumer_is(layer.mul, layer.terminal) ||
        !sole_consumer_is(layer.addend, layer.terminal)) {
        return false;
    }

    layer.down_index =
        ggml_backend_hrx_graph_node_index(cgraph, layer.down);
    layer.raw_gate_index =
        ggml_backend_hrx_graph_node_index(cgraph, layer.raw_gate);
    layer.sigmoid_index =
        ggml_backend_hrx_graph_node_index(cgraph, layer.sigmoid);
    layer.mul_index =
        ggml_backend_hrx_graph_node_index(cgraph, layer.mul);
    layer.terminal_index =
        ggml_backend_hrx_graph_node_index(cgraph, layer.terminal);
    if (layer.down_index < 0 ||
        !(layer.down_index < layer.raw_gate_index &&
          layer.raw_gate_index < layer.sigmoid_index &&
          layer.sigmoid_index < layer.mul_index &&
          layer.mul_index < layer.terminal_index)) {
        return false;
    }
    for (int i = layer.down_index + 1;
         i < layer.terminal_index; ++i) {
        const ggml_tensor * between = cgraph->nodes[i];
        if (!between || between == layer.raw_gate ||
            between == layer.sigmoid || between == layer.mul ||
            ggml_backend_hrx_is_metadata_op(between) ||
            ggml_backend_hrx_is_empty_op(between)) {
            continue;
        }
        return false;
    }

    ggml_backend_hrx_dispatch_request request = {};
    if (!ggml_backend_hrx_make_shared_expert_terminal_request(
            device_context, layer, &request)) {
        return false;
    }
    *out_layer = layer;
    return true;
}

static bool ggml_backend_hrx_build_shared_expert_terminal_graph_plan(
        ggml_backend_hrx_device_context * device_context,
        const ggml_cgraph * cgraph,
        ggml_backend_hrx_shared_expert_terminal_graph_plan * out_plan) {
    if (!device_context || !cgraph || !out_plan ||
        !device_context->reg_context ||
        !device_context->reg_context->catalog) {
        return false;
    }
    ggml_backend_hrx_shared_expert_terminal_graph_plan candidate = {};
    candidate.graph_uid = cgraph->uid;
    candidate.node_count = cgraph->n_nodes;
    candidate.catalog = device_context->reg_context->catalog.get();
    candidate.examined = true;
    candidate.skip_mask.assign(
        static_cast<size_t>(cgraph->n_nodes), 0);

    std::array<const ggml_tensor *, 40> terminals = {};
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        int layer = -1;
        if (!ggml_backend_hrx_parse_exact_layer_name(
                cgraph->nodes[i], "ffn_out-", &layer)) {
            continue;
        }
        if (terminals[layer] != nullptr) {
            return false;
        }
        terminals[layer] = cgraph->nodes[i];
    }
    for (int layer = 0; layer < 40; ++layer) {
        if (!terminals[layer] ||
            !ggml_backend_hrx_match_shared_expert_terminal_layer(
                device_context, cgraph, layer, terminals[layer],
                &candidate.layers[layer])) {
            return false;
        }
        const std::array<int, 3> skipped = {
            candidate.layers[layer].down_index,
            candidate.layers[layer].sigmoid_index,
            candidate.layers[layer].mul_index,
        };
        for (const int index : skipped) {
            if (index < 0 ||
                candidate.skip_mask[static_cast<size_t>(index)] != 0) {
                return false;
            }
            candidate.skip_mask[static_cast<size_t>(index)] = 1;
        }
    }
    if (std::count(
            candidate.skip_mask.begin(),
            candidate.skip_mask.end(), uint8_t{1}) != 120) {
        return false;
    }
    candidate.ready = true;
    *out_plan = std::move(candidate);
    return true;
}

static bool ggml_backend_hrx_match_gdn_rms_side_layer(
        ggml_backend_hrx_device_context * device_context,
        const ggml_cgraph * cgraph,
        const ggml_tensor * terminal,
        ggml_backend_hrx_gdn_rms_side_layer_plan * out_layer) {
    if (!device_context || !cgraph || !terminal || !out_layer ||
        terminal->op != GGML_OP_MUL) {
        return false;
    }
    ggml_backend_hrx_gdn_q8_silu_mul_match q8_match = {};
    if (!ggml_backend_hrx_match_gdn_q8_silu_mul(
            device_context, terminal, &q8_match) ||
        !q8_match.side ||
        q8_match.side->op != GGML_OP_MUL ||
        !q8_match.side->src[0] ||
        !q8_match.side->src[1] ||
        q8_match.side->src[0]->op != GGML_OP_RMS_NORM ||
        !q8_match.side->src[0]->src[0]) {
        return false;
    }

    ggml_backend_hrx_gdn_rms_side_layer_plan layer = {};
    layer.raw = q8_match.side->src[0]->src[0];
    layer.rms_norm = q8_match.side->src[0];
    layer.norm_weight = q8_match.side->src[1];
    layer.side = q8_match.side;
    layer.q8_gemm = q8_match.q8_gemm;
    layer.silu = q8_match.silu;
    layer.terminal = terminal;
    if (!ggml_backend_hrx_parse_exact_layer_name(
            layer.raw, "attn_output-", &layer.layer)) {
        return false;
    }
    const auto * qk_scale_layer =
        ggml_backend_hrx_find_gdn_qk_scale_layer(
            device_context, layer.layer);
    if (!qk_scale_layer || !qk_scale_layer->gated_delta_net ||
        !ggml_backend_hrx_metadata_chain_reaches(
            layer.raw, qk_scale_layer->gated_delta_net)) {
        return false;
    }
    layer.gated_delta_net_index =
        qk_scale_layer->gated_delta_net_index;

    char norm_weight_name[64];
    char gate_weight_name[64];
    std::snprintf(
        norm_weight_name, sizeof(norm_weight_name),
        "blk.%d.ssm_norm.weight", layer.layer);
    std::snprintf(
        gate_weight_name, sizeof(gate_weight_name),
        "blk.%d.attn_gate.weight", layer.layer);
    if (std::strcmp(
            ggml_get_name(layer.norm_weight),
            norm_weight_name) != 0 ||
        std::strcmp(
            ggml_get_name(layer.q8_gemm->src[0]),
            gate_weight_name) != 0 ||
        !ggml_backend_hrx_has_exact_layer_name(
            layer.q8_gemm->src[1], "attn_norm-", layer.layer)) {
        return false;
    }

    uint32_t eps_bits = 0;
    std::memcpy(
        &eps_bits, layer.rms_norm->op_params, sizeof(eps_bits));
    constexpr size_t side_bytes =
        128ull * 16384ull * sizeof(float);
    constexpr size_t weight_bytes = 128ull * sizeof(float);
    if (eps_bits != 0x358637bdu ||
        layer.raw->op != GGML_OP_VIEW ||
        layer.raw->type != GGML_TYPE_F32 ||
        layer.rms_norm->type != GGML_TYPE_F32 ||
        layer.norm_weight->type != GGML_TYPE_F32 ||
        layer.side->type != GGML_TYPE_F32 ||
        layer.raw->ne[0] != 128 ||
        ggml_backend_hrx_tensor_row_count(layer.raw) != 16384 ||
        layer.rms_norm->ne[0] != 128 ||
        ggml_backend_hrx_tensor_row_count(layer.rms_norm) !=
            16384 ||
        layer.norm_weight->ne[0] != 128 ||
        ggml_backend_hrx_tensor_row_count(layer.norm_weight) != 1 ||
        layer.side->ne[0] != 128 ||
        ggml_backend_hrx_tensor_row_count(layer.side) != 16384 ||
        ggml_nbytes(layer.raw) != side_bytes ||
        ggml_nbytes(layer.rms_norm) != side_bytes ||
        ggml_nbytes(layer.norm_weight) != weight_bytes ||
        ggml_nbytes(layer.side) != side_bytes ||
        !ggml_are_same_shape(layer.raw, layer.rms_norm) ||
        !ggml_are_same_shape(layer.rms_norm, layer.side) ||
        !ggml_backend_hrx_is_f32_dense(layer.raw) ||
        !ggml_backend_hrx_is_f32_dense(layer.rms_norm) ||
        !ggml_backend_hrx_is_f32_dense(layer.norm_weight) ||
        !ggml_backend_hrx_is_f32_dense(layer.side) ||
        !ggml_backend_hrx_all_storage_spans_disjoint({
            layer.q8_gemm->src[0],
            layer.q8_gemm->src[1],
            layer.raw,
            layer.norm_weight,
            layer.terminal,
        }) ||
        (layer.rms_norm->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        (layer.side->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const auto sole_consumer_is = [cgraph](
            const ggml_tensor * producer,
            const ggml_tensor * expected) {
        const auto consumers =
            ggml_backend_hrx_compute_consumers(cgraph, producer);
        return consumers.size() == 1 && consumers[0] == expected;
    };
    if (!sole_consumer_is(layer.rms_norm, layer.side) ||
        !sole_consumer_is(layer.side, layer.terminal)) {
        return false;
    }

    layer.rms_norm_index =
        ggml_backend_hrx_graph_node_index(cgraph, layer.rms_norm);
    layer.side_index =
        ggml_backend_hrx_graph_node_index(cgraph, layer.side);
    layer.q8_index =
        ggml_backend_hrx_graph_node_index(cgraph, layer.q8_gemm);
    layer.silu_index =
        ggml_backend_hrx_graph_node_index(cgraph, layer.silu);
    layer.terminal_index =
        ggml_backend_hrx_graph_node_index(cgraph, layer.terminal);
    if (layer.gated_delta_net_index < 0 ||
        layer.rms_norm_index < 0 ||
        !(layer.gated_delta_net_index < layer.rms_norm_index &&
          layer.rms_norm_index < layer.side_index &&
          layer.side_index < layer.q8_index &&
          layer.q8_index < layer.silu_index &&
          layer.silu_index < layer.terminal_index)) {
        return false;
    }
    // The raw side was consumed at RMS_NORM before fusion. It remains safe to
    // read at the terminal only when no unrelated compute can recycle or
    // overwrite its allocation across the absorbed interval.
    for (int i = layer.rms_norm_index + 1;
         i < layer.terminal_index; ++i) {
        const ggml_tensor * between = cgraph->nodes[i];
        if (!between || between == layer.side ||
            between == layer.q8_gemm ||
            between == layer.silu ||
            ggml_backend_hrx_is_metadata_op(between) ||
            ggml_backend_hrx_is_empty_op(between)) {
            continue;
        }
        return false;
    }

    ggml_backend_hrx_dispatch_request request = {};
    if (!ggml_backend_hrx_make_gdn_rms_side_request(
            device_context, layer, &request)) {
        return false;
    }
    *out_layer = layer;
    return true;
}

static bool ggml_backend_hrx_build_gdn_rms_side_graph_plan(
        ggml_backend_hrx_device_context * device_context,
        const ggml_cgraph * cgraph,
        ggml_backend_hrx_gdn_rms_side_graph_plan * out_plan) {
    if (!device_context || !cgraph || !out_plan ||
        !device_context->reg_context ||
        !device_context->reg_context->catalog) {
        return false;
    }
    ggml_backend_hrx_gdn_rms_side_graph_plan candidate = {};
    candidate.graph_uid = cgraph->uid;
    candidate.node_count = cgraph->n_nodes;
    candidate.catalog = device_context->reg_context->catalog.get();
    candidate.examined = true;
    candidate.skip_mask.assign(
        static_cast<size_t>(cgraph->n_nodes), 0);

    size_t matched = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * terminal = cgraph->nodes[i];
        if (!terminal || terminal->op != GGML_OP_MUL) {
            continue;
        }
        ggml_backend_hrx_gdn_rms_side_layer_plan layer = {};
        if (!ggml_backend_hrx_match_gdn_rms_side_layer(
                device_context, cgraph, terminal, &layer)) {
            continue;
        }
        const auto slot_it = std::find(
            ggml_backend_hrx_recurrent_cache_layers.begin(),
            ggml_backend_hrx_recurrent_cache_layers.end(),
            layer.layer);
        if (slot_it ==
            ggml_backend_hrx_recurrent_cache_layers.end()) {
            return false;
        }
        const size_t slot = static_cast<size_t>(
            std::distance(
                ggml_backend_hrx_recurrent_cache_layers.begin(),
                slot_it));
        if (candidate.layers[slot].terminal) {
            return false;
        }
        candidate.layers[slot] = layer;
        ++matched;
    }
    if (matched != candidate.layers.size()) {
        return false;
    }
    for (const auto & layer : candidate.layers) {
        const std::array<int, 4> skipped = {
            layer.rms_norm_index,
            layer.side_index,
            layer.q8_index,
            layer.silu_index,
        };
        for (const int index : skipped) {
            if (index < 0 ||
                static_cast<size_t>(index) >=
                    candidate.skip_mask.size() ||
                candidate.skip_mask[static_cast<size_t>(index)] != 0) {
                return false;
            }
            candidate.skip_mask[static_cast<size_t>(index)] = 1;
        }
    }
    if (std::count(
            candidate.skip_mask.begin(),
            candidate.skip_mask.end(), uint8_t{1}) != 120) {
        return false;
    }
    candidate.ready = true;
    *out_plan = std::move(candidate);
    return true;
}

static const ggml_tensor * ggml_backend_hrx_find_inplace_scale(
        const ggml_cgraph * cgraph,
        const ggml_tensor * owner) {
    const ggml_tensor * found = nullptr;
    for (int i = 0; cgraph && i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!node || node->op != GGML_OP_SCALE || !node->src[0] ||
            !ggml_backend_hrx_same_storage_span(node, owner) ||
            !ggml_backend_hrx_same_storage_span(node->src[0], owner)) {
            continue;
        }
        if (found) {
            return nullptr;
        }
        found = node;
    }
    return found;
}

static const ggml_tensor * ggml_backend_hrx_find_cache_write(
        const ggml_cgraph * cgraph,
        const ggml_tensor * owner,
        const ggml_tensor * source_owner,
        size_t source_offset,
        size_t length) {
    const ggml_tensor * found = nullptr;
    for (int i = 0; cgraph && i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!node || node->op != GGML_OP_CPY || !node->src[0] ||
            !ggml_backend_hrx_same_storage_span(node, owner) ||
            !ggml_backend_hrx_storage_subspan(
                node->src[0], source_owner, source_offset, length)) {
            continue;
        }
        if (found) {
            return nullptr;
        }
        found = node;
    }
    return found;
}

static const ggml_tensor * ggml_backend_hrx_find_attention_view(
        const ggml_cgraph * cgraph,
        const ggml_tensor * gated_delta_net,
        const ggml_tensor ** out_consumer) {
    const ggml_tensor * found = nullptr;
    const ggml_tensor * consumer = nullptr;
    for (int i = 0; cgraph && i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!node || node == gated_delta_net ||
            ggml_backend_hrx_is_metadata_op(node) ||
            ggml_backend_hrx_is_empty_op(node)) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * source = node->src[s];
            if (!source ||
                !ggml_backend_hrx_metadata_chain_reaches(
                    source, gated_delta_net) ||
                !ggml_backend_hrx_storage_subspan(
                    source, gated_delta_net, 0, 16384)) {
                continue;
            }
            if (found) {
                return nullptr;
            }
            found = source;
            consumer = node;
        }
    }
    if (out_consumer) {
        *out_consumer = consumer;
    }
    return found;
}

static bool ggml_backend_hrx_cache_owner_has_only_expected_touches(
        const ggml_cgraph * cgraph,
        const ggml_tensor * owner,
        const ggml_tensor * scale,
        const ggml_tensor * get_rows,
        const ggml_tensor * write) {
    for (int i = 0; cgraph && i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!node ||
            ggml_backend_hrx_is_metadata_op(node) ||
            ggml_backend_hrx_is_empty_op(node)) {
            continue;
        }
        bool touches = ggml_backend_hrx_storage_spans_overlap(node, owner);
        for (int s = 0; s < GGML_MAX_SRC && !touches; ++s) {
            touches =
                ggml_backend_hrx_storage_spans_overlap(node->src[s], owner);
        }
        if (!touches) {
            continue;
        }
        if (node == scale) {
            if (!node->src[0] ||
                !ggml_backend_hrx_same_storage_span(node, owner) ||
                !ggml_backend_hrx_same_storage_span(
                    node->src[0], owner)) {
                return false;
            }
            continue;
        }
        if (node == get_rows) {
            if (!node->src[0] ||
                !ggml_backend_hrx_same_storage_span(
                    node->src[0], owner) ||
                !ggml_backend_hrx_disjoint_storage_spans(node, owner)) {
                return false;
            }
            continue;
        }
        if (node == write) {
            if (!node->src[0] ||
                !ggml_backend_hrx_same_storage_span(node, owner) ||
                !ggml_backend_hrx_disjoint_storage_spans(
                    node->src[0], owner)) {
                return false;
            }
            continue;
        }
        return false;
    }
    return true;
}

static const ggml_backend_hrx_catalog_route *
ggml_backend_hrx_route_for_node(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        int node_index) {
    if (!device_context || !node ||
        !device_context->reg_context ||
        !device_context->reg_context->catalog) {
        return nullptr;
    }
    const int saved_node_index = device_context->current_node_index;
    device_context->current_node_index = node_index;
    ggml_backend_hrx_dispatch_request request = {};
    const bool made = ggml_backend_hrx_make_dispatch_request(
        device_context, node, &request);
    device_context->current_node_index = saved_node_index;
    if (!made) {
        return nullptr;
    }
    ggml_backend_hrx_add_tensor_overlap_facts(
        &request.problem, request.tensors);
    return ggml_backend_hrx_catalog_find_route(
        *device_context->reg_context->catalog, request.problem);
}

static const ggml_backend_hrx_catalog_route *
ggml_backend_hrx_route_for_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_backend_hrx_dispatch_request & request) {
    if (!device_context ||
        !device_context->reg_context ||
        !device_context->reg_context->catalog) {
        return nullptr;
    }
    ggml_backend_hrx_catalog_problem problem = request.problem;
    ggml_backend_hrx_add_tensor_overlap_facts(
        &problem, request.tensors);
    return ggml_backend_hrx_catalog_find_route(
        *device_context->reg_context->catalog, problem);
}

static bool ggml_backend_hrx_gdn_qk_scale_route_is_exact(
        const ggml_backend_hrx_catalog_route * route) {
    if (!route ||
        route->id !=
            "gated_delta_net_f32_sv128_qk_l2_full_head_wg256" ||
        route->family != "gated_delta_net_f32" ||
        route->op != "GATED_DELTA_NET" ||
        route->source_id !=
            "gated_delta_net_f32_qk_l2_full_head" ||
        route->artifact_id !=
            "gated_delta_net_f32_qk_l2_full_head_loombc" ||
        route->root_symbol !=
            "@hrx2_gated_delta_net_f32_sv128_qk_l2_full_head" ||
        route->export_name !=
            "hrx2_gated_delta_net_f32_sv128_qk_l2_full_head" ||
        route->binding_count != 7 ||
        route->parameter_count != 8 ||
        route->constant_byte_length != sizeof(float) ||
        !route->prepasses.empty() ||
        route->constraints.size() != 18) {
        return false;
    }
    const auto layout = route->supports.find("layout");
    const auto fusion = route->supports.find("fusion");
    if (layout == route->supports.end() ||
        layout->second !=
            "state_transposed_column_per_wave_qk_l2_full_head" ||
        fusion == route->supports.end() ||
        fusion->second !=
            "GATED_DELTA_NET_QK_L2_FULL_HEAD") {
        return false;
    }
    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = i + 1; j < 7; ++j) {
            if (i < 3 && j < 3) {
                continue;
            }
            const std::string source =
                "tensor_overlap." + std::to_string(i) +
                "_" + std::to_string(j);
            if (!ggml_backend_hrx_route_has_zero_constraint(
                    *route, source.c_str())) {
                return false;
            }
        }
    }

    return true;
}

static bool ggml_backend_hrx_build_gdn_qk_scale_plan(
        ggml_backend_hrx_device_context * device_context,
        const ggml_cgraph * cgraph,
        ggml_backend_hrx_gdn_qk_scale_plan * out_plan,
        std::string * out_error) {
    if (!device_context || !cgraph || !out_plan ||
        !device_context->reg_context ||
        !device_context->reg_context->catalog) {
        if (out_error) {
            *out_error = "missing graph, device context, or catalog";
        }
        return false;
    }
    *out_plan = {};
    out_plan->graph_uid = cgraph->uid;
    out_plan->node_count = cgraph->n_nodes;
    out_plan->catalog = device_context->reg_context->catalog.get();
    out_plan->skip_mask.assign(
        static_cast<size_t>(cgraph->n_nodes), 0);
    // Operator tests and other direct callers may submit a production-shaped
    // GDN as a one-node graph after materializing q/k themselves. That graph
    // has no producers to fuse; resolve the ordinary GDN route instead of
    // treating the absence of model-specific tensor names as a malformed
    // whole-model fusion topology.
    if (cgraph->n_nodes == 1 &&
        cgraph->nodes[0] &&
        cgraph->nodes[0]->op == GGML_OP_GATED_DELTA_NET) {
        out_plan->valid = true;
        return true;
    }

    auto reject = [&](const std::string & message) {
        if (out_error) {
            *out_error = message;
        }
        return false;
    };
    auto expected_slot = [](int layer) -> int {
        for (size_t i = 0;
             i < ggml_backend_hrx_recurrent_cache_layers.size();
             ++i) {
            if (ggml_backend_hrx_recurrent_cache_layers[i] == layer) {
                return static_cast<int>(i);
            }
        }
        return -1;
    };
    auto exact_f32_strides = [](
            const ggml_tensor * tensor,
            size_t nb1,
            size_t nb2,
            size_t nb3) {
        return tensor && tensor->type == GGML_TYPE_F32 &&
               tensor->nb[0] == sizeof(float) &&
               tensor->nb[1] == nb1 &&
               tensor->nb[2] == nb2 &&
               tensor->nb[3] == nb3;
    };

    std::array<std::optional<ggml_backend_hrx_gdn_qk_scale_layer>, 30>
        matches;
    int pp_gdn_count = 0;
    const ggml_backend_hrx_catalog_route * accepted_route = nullptr;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!node || node->op != GGML_OP_GATED_DELTA_NET ||
            !node->src[0] || !node->src[1] ||
            !node->src[2] || !node->src[3] ||
            !node->src[4] || !node->src[5] ||
            !ggml_backend_hrx_shape_is(
                node->src[2], 128, 32, 512, 1)) {
            continue;
        }
        ++pp_gdn_count;
        const ggml_tensor * q_norm = node->src[0];
        const ggml_tensor * k_norm = node->src[1];
        const ggml_tensor * raw_q = q_norm->src[0];
        const ggml_tensor * raw_k = k_norm->src[0];
        const ggml_tensor * v = node->src[2];
        int layer = -1;
        if (!ggml_backend_hrx_parse_exact_layer_name(
                q_norm, "q_conv_predelta-", &layer) ||
            !ggml_backend_hrx_has_exact_layer_name(
                k_norm, "k_conv_predelta-", layer) ||
            !ggml_backend_hrx_has_exact_layer_name(
                raw_q, "q_conv-", layer) ||
            !ggml_backend_hrx_has_exact_layer_name(
                raw_k, "k_conv-", layer)) {
            return reject(
                "PP512 GDN q/k normalization names are not exact");
        }
        const int slot = expected_slot(layer);
        if (slot < 0 || matches[static_cast<size_t>(slot)]) {
            return reject(
                "PP512 GDN has an unexpected or duplicate layer");
        }
        if (q_norm->op != GGML_OP_L2_NORM ||
            k_norm->op != GGML_OP_L2_NORM ||
            !raw_q || !raw_k ||
            !ggml_backend_hrx_shape_is(
                q_norm, 128, 16, 512, 1) ||
            !ggml_backend_hrx_shape_is(
                k_norm, 128, 16, 512, 1) ||
            !ggml_backend_hrx_shape_is(
                raw_q, 128, 16, 512, 1) ||
            !ggml_backend_hrx_shape_is(
                raw_k, 128, 16, 512, 1) ||
            !ggml_backend_hrx_shape_is(
                node->src[3], 1, 32, 512, 1) ||
            !ggml_backend_hrx_shape_is(
                node->src[4], 1, 32, 512, 1) ||
            !ggml_backend_hrx_shape_is(
                node->src[5], 128, 128, 32, 1) ||
            ggml_get_op_params_i32(node, 0) != 1 ||
            !exact_f32_strides(
                q_norm, 512, 8192, 4194304) ||
            !exact_f32_strides(
                k_norm, 512, 8192, 4194304) ||
            !exact_f32_strides(
                raw_q, 512, 32768, 16777216) ||
            !exact_f32_strides(
                raw_k, 512, 32768, 16777216) ||
            !exact_f32_strides(
                v, 512, 32768, 16777216) ||
            !exact_f32_strides(
                node->src[3], 4, 128, 65536) ||
            !exact_f32_strides(
                node->src[4], 4, 128, 65536) ||
            !exact_f32_strides(
                node->src[5], 512, 65536, 2097152) ||
            q_norm->type != GGML_TYPE_F32 ||
            k_norm->type != GGML_TYPE_F32 ||
            node->type != GGML_TYPE_F32 ||
            ggml_nbytes(q_norm) != 4194304 ||
            ggml_nbytes(k_norm) != 4194304 ||
            ggml_nbytes(node->src[3]) != 65536 ||
            ggml_nbytes(node->src[4]) != 65536 ||
            ggml_nbytes(node->src[5]) != 2097152 ||
            ggml_nbytes(node) != 10485760) {
            return reject(
                "PP512 GDN has a noncanonical shape, stride, or byte span");
        }
        uint32_t q_eps_bits = 0;
        uint32_t k_eps_bits = 0;
        std::memcpy(
            &q_eps_bits, q_norm->op_params,
            sizeof(q_eps_bits));
        std::memcpy(
            &k_eps_bits, k_norm->op_params,
            sizeof(k_eps_bits));
        if (q_eps_bits != 0x358637bdu ||
            k_eps_bits != 0x358637bdu) {
            return reject(
                "PP512 GDN q/k epsilon is not exactly 1e-6");
        }

        const int q_norm_index =
            ggml_backend_hrx_graph_node_index(cgraph, q_norm);
        const int k_norm_index =
            ggml_backend_hrx_graph_node_index(cgraph, k_norm);
        if (q_norm_index < 0 || k_norm_index < 0 ||
            !(q_norm_index < i && k_norm_index < i)) {
            return reject(
                "PP512 GDN q/k normalization order is not exact");
        }
        const auto q_norm_consumers =
            ggml_backend_hrx_compute_consumers(cgraph, q_norm);
        const auto k_norm_consumers =
            ggml_backend_hrx_compute_consumers(cgraph, k_norm);
        const auto raw_q_consumers =
            ggml_backend_hrx_compute_consumers(cgraph, raw_q);
        const auto raw_k_consumers =
            ggml_backend_hrx_compute_consumers(cgraph, raw_k);
        if (q_norm_consumers.size() != 1 ||
            q_norm_consumers[0] != node ||
            k_norm_consumers.size() != 1 ||
            k_norm_consumers[0] != node ||
            raw_q_consumers.size() != 1 ||
            raw_q_consumers[0] != q_norm ||
            raw_k_consumers.size() != 1 ||
            raw_k_consumers[0] != k_norm) {
            return reject(
                "PP512 GDN q/k normalization is not private");
        }

        const ggml_tensor * packed = raw_q->view_src;
        if (!packed ||
            raw_k->view_src != packed ||
            v->view_src != packed ||
            raw_q->view_offs != 0 ||
            raw_k->view_offs != 8192 ||
            v->view_offs != 16384 ||
            ggml_nbytes(packed) != 16777216 ||
            !ggml_backend_hrx_storage_subspan(
                raw_q, packed, 0, 16752640) ||
            !ggml_backend_hrx_storage_subspan(
                raw_k, packed, 8192, 16752640) ||
            !ggml_backend_hrx_storage_subspan(
                v, packed, 16384, 16760832)) {
            return reject(
                "PP512 GDN q/k/v packed projection layout is not exact");
        }
        if (!ggml_backend_hrx_all_storage_spans_disjoint({
                packed,
                q_norm,
                k_norm,
                node->src[3],
                node->src[4],
                node->src[5],
                node}) ||
            !ggml_backend_hrx_all_storage_spans_disjoint({
                q_norm,
                k_norm,
                v,
                node->src[3],
                node->src[4],
                node->src[5],
                node})) {
            return reject(
                "PP512 GDN violates the direct prepass alias contract");
        }

        ggml_backend_hrx_gdn_qk_scale_layer match = {};
        match.layer = layer;
        match.q_norm_index = q_norm_index;
        match.k_norm_index = k_norm_index;
        match.gated_delta_net_index = i;
        match.q_norm = q_norm;
        match.k_norm = k_norm;
        match.gated_delta_net = node;
        match.raw_q = raw_q;
        match.raw_k = raw_k;

        ggml_backend_hrx_dispatch_request request = {};
        if (!ggml_backend_hrx_make_gdn_qk_scale_request(
                device_context, node, match, &request)) {
            return reject(
                "failed to form PP512 GDN q/k scale request");
        }
        const auto * route =
            ggml_backend_hrx_route_for_request(
                device_context, request);
        if (!ggml_backend_hrx_gdn_qk_scale_route_is_exact(route)) {
            return reject(
                "PP512 GDN q/k scale route is absent or has the wrong ABI");
        }
        if (accepted_route && route != accepted_route) {
            return reject(
                "PP512 GDN layers do not resolve one exact q/k scale route");
        }
        accepted_route = route;
        matches[static_cast<size_t>(slot)] = match;
    }
    if (pp_gdn_count == 0) {
        out_plan->valid = true;
        return true;
    }
    if (pp_gdn_count !=
        static_cast<int>(
            ggml_backend_hrx_recurrent_cache_layers.size())) {
        return reject(
            "PP512 graph has a partial GDN q/k scale topology");
    }

    out_plan->layers.reserve(matches.size());
    size_t skipped = 0;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (!matches[i] ||
            matches[i]->layer !=
                ggml_backend_hrx_recurrent_cache_layers[i]) {
            return reject(
                "PP512 GDN q/k scale layers do not match the exact 30-layer order");
        }
        for (int index : {
                 matches[i]->q_norm_index,
                 matches[i]->k_norm_index}) {
            if (index < 0 || index >= cgraph->n_nodes ||
                out_plan->skip_mask[
                    static_cast<size_t>(index)] != 0) {
                return reject(
                    "PP512 GDN q/k scale skip set is not one-to-one");
            }
            out_plan->skip_mask[static_cast<size_t>(index)] = 1;
            ++skipped;
        }
        out_plan->layers.push_back(*matches[i]);
    }
    if (skipped != 60) {
        return reject(
            "PP512 GDN q/k scale skip set is not exactly 60 passes");
    }
    out_plan->valid = true;
    out_plan->ready = true;
    return true;
}

static bool
ggml_backend_hrx_ssm_conv_silu_route_is_exact(
        const ggml_backend_hrx_catalog_route * route) {
    if (!route ||
        route->id !=
            "ssm_conv_f32_chan_concat_silu_regblock_wg1024" ||
        route->family !=
            "ssm_conv_f32_chan_concat" ||
        route->op != "SSM_CONV" ||
        route->source_id !=
            "ssm_conv_f32_chan_concat_silu_regblock_wg1024" ||
        route->artifact_id !=
            "ssm_conv_f32_chan_concat_silu_regblock_wg1024_loombc" ||
        route->root_symbol !=
            "@hrx2_ssm_conv_f32_chan_concat_silu_regblock_wg1024" ||
        route->export_name !=
            "hrx2_ssm_conv_f32_chan_concat_silu_regblock_wg1024" ||
        route->binding_count != 4 ||
        route->parameter_count != 4 ||
        route->constant_byte_length != 0 ||
        !route->prepasses.empty() ||
        route->constraints.size() != 5 ||
        route->bindings.size() != 6) {
        return false;
    }
    const auto layout = route->supports.find("layout");
    const auto fusion = route->supports.find("fusion");
    if (layout == route->supports.end() ||
        layout->second !=
            "conv_window_channels_first_concat_silu_regblock_wg1024" ||
        fusion == route->supports.end() ||
        fusion->second !=
            "SSM_CONV_SILU_REGBLOCK_WG1024") {
        return false;
    }
    for (const char * source : {
             "tensor_overlap.0_1",
             "tensor_overlap.0_2",
             "tensor_overlap.0_3",
             "tensor_overlap.1_2",
             "tensor_overlap.2_3"}) {
        if (!ggml_backend_hrx_route_has_zero_constraint(
                *route, source)) {
            return false;
        }
    }
    const std::array<std::pair<const char *, const char *>, 6>
        expected_bindings = {{
            {"@hrx2.shape.ssm_conv.d_conv",
             "ssm_conv.d_conv"},
            {"@hrx2.shape.ssm_conv.d_inner",
             "ssm_conv.d_inner"},
            {"@hrx2.shape.ssm_conv.n_t",
             "ssm_conv.n_t"},
            {"@hrx2.shape.ssm_conv.n_s",
             "ssm_conv.n_s"},
            {"@hrx2.shape.ssm_conv.state_row_stride",
             "ssm_conv.state_row_stride"},
            {"@hrx2.shape.ssm_conv.x_row_stride",
             "ssm_conv.x_row_stride"},
        }};
    for (size_t i = 0; i < expected_bindings.size(); ++i) {
        if (route->bindings[i].key != expected_bindings[i].first ||
            route->bindings[i].shape_source !=
                expected_bindings[i].second ||
            !route->bindings[i].value.empty()) {
            return false;
        }
    }
    const std::array<std::pair<const char *, int64_t>, 10>
        exact_shapes = {{
            {"ncols", 8192},
            {"nrows", 512},
            {"d_conv", 4},
            {"n_t", 512},
            {"ssm_conv.d_conv", 4},
            {"ssm_conv.d_inner", 8192},
            {"ssm_conv.n_t", 512},
            {"ssm_conv.n_s", 1},
            {"ssm_conv.state_row_stride", 8192},
            {"ssm_conv.x_row_stride", 8192},
        }};
    for (const auto & [name, value] : exact_shapes) {
        const auto minimum = route->shape_min.find(name);
        const auto maximum = route->shape_max.find(name);
        if (minimum == route->shape_min.end() ||
            maximum == route->shape_max.end() ||
            minimum->second != value ||
            maximum->second != value) {
            return false;
        }
    }
    return true;
}

static bool
ggml_backend_hrx_build_ssm_conv_silu_plan(
        ggml_backend_hrx_device_context * device_context,
        const ggml_cgraph * cgraph,
        ggml_backend_hrx_ssm_conv_silu_plan * out_plan,
        std::string * out_error) {
    if (!device_context || !cgraph || !out_plan ||
        !device_context->reg_context ||
        !device_context->reg_context->catalog) {
        if (out_error) {
            *out_error = "missing graph, device context, or catalog";
        }
        return false;
    }
    *out_plan = {};
    out_plan->graph_uid = cgraph->uid;
    out_plan->node_count = cgraph->n_nodes;
    out_plan->catalog = device_context->reg_context->catalog.get();
    out_plan->skip_mask.assign(
        static_cast<size_t>(cgraph->n_nodes), 0);

    auto reject = [&](const std::string & message) {
        if (out_error) {
            *out_error = message;
        }
        return false;
    };
    auto expected_slot = [](int layer) -> int {
        for (size_t i = 0;
             i < ggml_backend_hrx_recurrent_cache_layers.size();
             ++i) {
            if (ggml_backend_hrx_recurrent_cache_layers[i] == layer) {
                return static_cast<int>(i);
            }
        }
        return -1;
    };
    auto exact_f32_strides = [](
            const ggml_tensor * tensor,
            size_t nb1,
            size_t nb2,
            size_t nb3) {
        return tensor && tensor->type == GGML_TYPE_F32 &&
               tensor->nb[0] == sizeof(float) &&
               tensor->nb[1] == nb1 &&
               tensor->nb[2] == nb2 &&
               tensor->nb[3] == nb3;
    };

    std::array<std::optional<ggml_backend_hrx_ssm_conv_silu_layer>, 30>
        matches;
    int pp_ssm_count = 0;
    const ggml_backend_hrx_catalog_route * accepted_route = nullptr;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!node || node->op != GGML_OP_SSM_CONV ||
            !ggml_backend_hrx_shape_is(node, 8192, 512, 1, 1) ||
            !node->src[1] ||
            !ggml_backend_hrx_shape_is(
                node->src[1], 4, 8192, 1, 1)) {
            continue;
        }
        ++pp_ssm_count;
        int layer = -1;
        if (!ggml_backend_hrx_parse_exact_layer_name(
                node, "conv_output_raw-", &layer)) {
            return reject(
                "PP512 SSM convolution output name is not exact");
        }
        const int slot = expected_slot(layer);
        if (slot < 0 || matches[static_cast<size_t>(slot)]) {
            return reject(
                "PP512 SSM convolution has an unexpected or duplicate layer");
        }

        const ggml_tensor * state = nullptr;
        const ggml_tensor * x = nullptr;
        if (!ggml_backend_hrx_ssm_conv_window_pieces(
                node, &state, &x) ||
            !state || !x || !node->src[0] ||
            node->src[0]->op != GGML_OP_CONCAT ||
            !ggml_backend_hrx_has_exact_layer_name(
                state, "conv_states_reshaped-", layer) ||
            !ggml_backend_hrx_has_exact_layer_name(
                x, "linear_attn_qkv_mixed-", layer)) {
            return reject(
                "PP512 SSM convolution window pieces are not exact");
        }
        char expected_filter[64];
        std::snprintf(
            expected_filter, sizeof(expected_filter),
            "blk.%d.ssm_conv1d.weight", layer);
        if (std::strcmp(
                ggml_get_name(node->src[1]),
                expected_filter) != 0 ||
            !ggml_backend_hrx_shape_is(
                state, 8192, 3, 1, 1) ||
            !ggml_backend_hrx_shape_is(
                x, 8192, 512, 1, 1) ||
            !ggml_backend_hrx_shape_is(
                node->src[0], 8192, 515, 1, 1) ||
            !exact_f32_strides(
                state, 32768, 98304, 98304) ||
            !exact_f32_strides(
                x, 32768, 16777216, 16777216) ||
            !exact_f32_strides(
                node->src[0], 32768, 16875520, 16875520) ||
            !exact_f32_strides(
                node->src[1], 16, 131072, 131072) ||
            !exact_f32_strides(
                node, 32768, 16777216, 16777216) ||
            ggml_nbytes(state) != 98304 ||
            ggml_nbytes(x) != 16777216 ||
            ggml_nbytes(node->src[0]) != 16875520 ||
            ggml_nbytes(node->src[1]) != 131072 ||
            ggml_nbytes(node) != 16777216) {
            return reject(
                "PP512 SSM convolution has a noncanonical shape, stride, or byte span");
        }

        const ggml_tensor * window = node->src[0];
        const int window_index =
            ggml_backend_hrx_graph_node_index(cgraph, window);
        ggml_backend_buffer_t x_buffer = nullptr;
        size_t x_offset = 0;
        size_t x_length = 0;
        if (window_index < 0 || window_index >= i ||
            !ggml_backend_hrx_tensor_storage_range(
                x, &x_buffer, &x_offset, &x_length) ||
            x->nb[1] == 0 ||
            GGML_HRX_SSM_X_SNAPSHOT_ROWS >
                x_length / x->nb[1]) {
            return reject(
                "PP512 sparse SSM window has an invalid graph order or x span");
        }
        const size_t snapshot_bytes =
            GGML_HRX_SSM_X_SNAPSHOT_ROWS * x->nb[1];
        if (x_offset >
                std::numeric_limits<size_t>::max() - snapshot_bytes ||
            x_offset + snapshot_bytes >
                std::numeric_limits<size_t>::max() -
                    (x_length - snapshot_bytes)) {
            return reject(
                "PP512 sparse SSM x suffix range overflows");
        }
        const size_t x_suffix_begin = x_offset + snapshot_bytes;
        const size_t x_end = x_offset + x_length;
        for (int j = window_index + 1; j < i; ++j) {
            const ggml_tensor * writer = cgraph->nodes[j];
            if (!writer ||
                ggml_backend_hrx_is_metadata_op(writer) ||
                ggml_backend_hrx_is_empty_op(writer)) {
                continue;
            }
            ggml_backend_buffer_t writer_buffer = nullptr;
            size_t writer_offset = 0;
            size_t writer_length = 0;
            if (!ggml_backend_hrx_tensor_storage_range(
                    writer,
                    &writer_buffer,
                    &writer_offset,
                    &writer_length) ||
                writer_offset >
                    std::numeric_limits<size_t>::max() - writer_length) {
                return reject(
                    "PP512 sparse SSM cannot prove an intervening writer span");
            }
            const size_t writer_end = writer_offset + writer_length;
            if (writer_buffer == x_buffer &&
                std::max(writer_offset, x_suffix_begin) <
                    std::min(writer_end, x_end)) {
                return reject(
                    "PP512 sparse SSM has an intervening write beyond the snapshotted x prefix");
            }
        }

        const auto consumers =
            ggml_backend_hrx_compute_consumers(cgraph, node);
        if (consumers.size() != 1) {
            return reject(
                "PP512 SSM convolution does not have one sole consumer");
        }
        const ggml_tensor * silu = consumers[0];
        const int silu_index =
            ggml_backend_hrx_graph_node_index(cgraph, silu);
        if (!silu || silu->op != GGML_OP_UNARY ||
            ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU ||
            silu->src[0] != node ||
            silu_index <= i ||
            !ggml_backend_hrx_has_exact_layer_name(
                silu, "conv_output_silu-", layer) ||
            !ggml_backend_hrx_shape_is(
                silu, 8192, 512, 1, 1) ||
            !exact_f32_strides(
                silu, 32768, 16777216, 16777216) ||
            ggml_nbytes(silu) != 16777216 ||
            !ggml_backend_hrx_same_storage_span(node, silu)) {
            return reject(
                "PP512 SSM sole consumer is not the exact in-place SiLU");
        }

        const bool x_dst_alias =
            ggml_backend_hrx_same_storage_span(x, silu);
        if ((!x_dst_alias &&
             !ggml_backend_hrx_disjoint_storage_spans(x, silu)) ||
            !ggml_backend_hrx_all_storage_spans_disjoint({
                window, x, node->src[1]}) ||
            !ggml_backend_hrx_disjoint_storage_spans(window, silu) ||
            !ggml_backend_hrx_disjoint_storage_spans(
                node->src[1], silu)) {
            return reject(
                "PP512 SSM/SiLU violates the exact-or-disjoint alias contract");
        }

        ggml_backend_hrx_ssm_conv_silu_layer match = {};
        match.layer = layer;
        match.ssm_conv_index = i;
        match.silu_index = silu_index;
        match.x_dst_alias = x_dst_alias;
        match.ssm_conv = node;
        match.silu = silu;
        match.window = window;
        match.state = state;
        match.x = x;
        match.filter = node->src[1];
        match.dst = silu;
        match.node_signature =
            ggml_backend_hrx_node_signature(match.ssm_conv);

        ggml_backend_hrx_dispatch_request request = {};
        if (!ggml_backend_hrx_make_ssm_conv_silu_request(
                device_context, node, match, &request)) {
            return reject(
                "failed to form PP512 register-blocked SSM/SiLU request");
        }
        match.request_fingerprint =
            ggml_backend_hrx_ssm_conv_silu_request_fingerprint(
                match, request.tensors);
        const auto * route =
            ggml_backend_hrx_route_for_request(
                device_context, request);
        if (!ggml_backend_hrx_ssm_conv_silu_route_is_exact(route)) {
            return reject(
                "PP512 register-blocked SSM/SiLU route is absent or has the wrong ABI");
        }
        if (accepted_route && route != accepted_route) {
            return reject(
                "PP512 SSM/SiLU layers do not resolve one exact route");
        }
        accepted_route = route;
        matches[static_cast<size_t>(slot)] = match;
    }
    if (pp_ssm_count == 0) {
        out_plan->valid = true;
        return true;
    }
    if (pp_ssm_count !=
        static_cast<int>(
            ggml_backend_hrx_recurrent_cache_layers.size())) {
        return reject(
            "PP512 graph has a partial SSM_CONV-to-SiLU topology");
    }

    out_plan->layers.reserve(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
        if (!matches[i] ||
            matches[i]->layer !=
                ggml_backend_hrx_recurrent_cache_layers[i]) {
            return reject(
                "PP512 SSM/SiLU layers do not match the exact 30-layer order");
        }
        const int skip = matches[i]->silu_index;
        if (skip < 0 || skip >= cgraph->n_nodes ||
            out_plan->skip_mask[static_cast<size_t>(skip)] != 0) {
            return reject(
                "PP512 SSM/SiLU skip set is not one-to-one");
        }
        out_plan->skip_mask[static_cast<size_t>(skip)] = 1;
        out_plan->alias_layers +=
            matches[i]->x_dst_alias ? 1 : 0;
        out_plan->disjoint_layers +=
            matches[i]->x_dst_alias ? 0 : 1;
        out_plan->layers.push_back(*matches[i]);
    }
    if (out_plan->layers.size() != 30 ||
        out_plan->alias_layers != 19 ||
        out_plan->disjoint_layers != 11 ||
        std::count(
            out_plan->skip_mask.begin(),
            out_plan->skip_mask.end(),
            static_cast<uint8_t>(1)) != 30) {
        return reject(
            "PP512 SSM/SiLU requires exactly 30 layers, 19 aliases, 11 disjoint destinations, and 30 skips");
    }
    out_plan->valid = true;
    out_plan->ready = true;
    return true;
}

static bool ggml_backend_hrx_build_recurrent_cache_plan(
        ggml_backend_hrx_device_context * device_context,
        const ggml_cgraph * cgraph,
        ggml_backend_hrx_recurrent_cache_plan * out_plan,
        std::string * out_error) {
    if (!device_context || !cgraph || !out_plan ||
        !device_context->reg_context ||
        !device_context->reg_context->catalog) {
        if (out_error) {
            *out_error = "missing graph, device context, or catalog";
        }
        return false;
    }
    *out_plan = {};
    out_plan->graph_uid = cgraph->uid;
    out_plan->node_count = cgraph->n_nodes;
    out_plan->catalog = device_context->reg_context->catalog.get();
    out_plan->skip_mask.assign(
        static_cast<size_t>(cgraph->n_nodes), 0);

    auto reject = [&](const std::string & message) {
        if (out_error) {
            *out_error = message;
        }
        return false;
    };
    auto expected_slot = [](int layer) -> int {
        for (size_t i = 0;
             i < ggml_backend_hrx_recurrent_cache_layers.size();
             ++i) {
            if (ggml_backend_hrx_recurrent_cache_layers[i] == layer) {
                return static_cast<int>(i);
            }
        }
        return -1;
    };

    std::array<std::optional<ggml_backend_hrx_recurrent_cache_layer>, 30>
        matches;
    int decode_ssm_count = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!node || node->op != GGML_OP_SSM_CONV ||
            !ggml_backend_hrx_shape_is(node, 8192, 1, 1, 1) ||
            !node->src[1] ||
            !ggml_backend_hrx_shape_is(
                node->src[1], 4, 8192, 1, 1)) {
            continue;
        }
        ++decode_ssm_count;
        const ggml_tensor * concat = node->src[0];
        if (!concat || concat->op != GGML_OP_CONCAT ||
            ggml_get_op_params_i32(concat, 0) != 1 ||
            !ggml_backend_hrx_shape_is(concat, 8192, 4, 1, 1) ||
            !concat->src[0] || !concat->src[1]) {
            return reject("decode SSM convolution lacks the exact dim-1 window CONCAT");
        }
        const ggml_tensor * state = concat->src[0];
        const ggml_tensor * x = concat->src[1];
        const ggml_tensor * cache_r_get =
            ggml_backend_hrx_zero_offset_source_chain_target(
                state, GGML_OP_GET_ROWS);
        if (!cache_r_get || !cache_r_get->src[0] ||
            !cache_r_get->src[1] ||
            !ggml_backend_hrx_shape_is(state, 8192, 3, 1, 1) ||
            !ggml_backend_hrx_shape_is(x, 8192, 1, 1, 1) ||
            !ggml_backend_hrx_shape_is(
                cache_r_get, 24576, 1, 1, 1) ||
            cache_r_get->src[1]->type != GGML_TYPE_I32 ||
            ggml_nelements(cache_r_get->src[1]) != 1 ||
            !ggml_backend_hrx_same_storage_span(
                state, cache_r_get) ||
            ggml_nbytes(cache_r_get->src[0]) != 98304 ||
            ggml_nbytes(state) != 98304 ||
            ggml_nbytes(x) != 32768 ||
            ggml_nbytes(concat) != 131072 ||
            ggml_nbytes(node->src[1]) != 131072 ||
            ggml_nbytes(node) != 32768 ||
            state->nb[0] != sizeof(float) ||
            state->nb[1] != 8192 * sizeof(float) ||
            x->nb[0] != sizeof(float) ||
            x->nb[1] != 8192 * sizeof(float) ||
            node->src[1]->nb[0] != sizeof(float) ||
            node->src[1]->nb[1] != 4 * sizeof(float)) {
            return reject("decode SSM convolution has a noncanonical cache, input, or filter shape");
        }
        const int layer =
            ggml_backend_hrx_exact_cache_layer(
                cache_r_get->src[0], 'r');
        const int slot = expected_slot(layer);
        if (slot < 0 || matches[static_cast<size_t>(slot)]) {
            return reject("decode SSM cache-r owner has an unexpected or duplicate layer");
        }
        const ggml_tensor * cache_r_write =
            ggml_backend_hrx_find_cache_write(
                cgraph, cache_r_get->src[0], concat, 32768, 98304);
        const ggml_tensor * scale =
            ggml_backend_hrx_find_inplace_scale(
                cgraph, cache_r_get->src[0]);
        const int cache_r_get_index =
            ggml_backend_hrx_graph_node_index(cgraph, cache_r_get);
        const int concat_index =
            ggml_backend_hrx_graph_node_index(cgraph, concat);
        const int cache_r_write_index =
            ggml_backend_hrx_graph_node_index(cgraph, cache_r_write);
        const int scale_index =
            ggml_backend_hrx_graph_node_index(cgraph, scale);
        if (!cache_r_write ||
            cache_r_get_index < 0 || concat_index < 0 ||
            cache_r_write_index < 0 ||
            (scale && scale_index < 0) ||
            !((!scale || scale_index < cache_r_get_index) &&
              cache_r_get_index < concat_index &&
              concat_index < cache_r_write_index &&
              cache_r_write_index < i)) {
            return reject("decode cache-r load, CONCAT, update, and SSM order is not exact");
        }
        const auto get_consumers =
            ggml_backend_hrx_compute_consumers(cgraph, cache_r_get);
        const auto concat_consumers =
            ggml_backend_hrx_compute_consumers(cgraph, concat);
        if (get_consumers.size() != 1 ||
            get_consumers[0] != concat ||
            concat_consumers.size() != 2 ||
            std::find(
                concat_consumers.begin(), concat_consumers.end(),
                cache_r_write) == concat_consumers.end() ||
            std::find(
                concat_consumers.begin(), concat_consumers.end(),
                node) == concat_consumers.end()) {
            return reject("decode cache-r materialization has an extra consumer");
        }
        const auto conv_consumers =
            ggml_backend_hrx_compute_consumers(cgraph, node);
        if (conv_consumers.size() != 1) {
            return reject(
                "decode SSM convolution does not have one sole consumer");
        }
        const ggml_tensor * silu = conv_consumers[0];
        const int silu_index =
            ggml_backend_hrx_graph_node_index(cgraph, silu);
        if (!silu || silu->op != GGML_OP_UNARY ||
            ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU ||
            silu->src[0] != node ||
            silu_index <= i ||
            !ggml_backend_hrx_has_exact_layer_name(
                node, "conv_output_raw-", layer) ||
            !ggml_backend_hrx_has_exact_layer_name(
                silu, "conv_output_silu-", layer) ||
            !ggml_backend_hrx_shape_is(
                silu, 8192, 1, 1, 1) ||
            node->type != GGML_TYPE_F32 ||
            silu->type != GGML_TYPE_F32 ||
            silu->nb[0] != sizeof(float) ||
            silu->nb[1] != 8192 * sizeof(float) ||
            ggml_nbytes(silu) != 32768 ||
            !ggml_backend_hrx_same_storage_span(node, silu)) {
            return reject(
                "decode SSM sole consumer is not the exact in-place SiLU");
        }
        const bool x_dst_alias =
            ggml_backend_hrx_same_storage_span(x, silu);
        if ((!x_dst_alias &&
             !ggml_backend_hrx_disjoint_storage_spans(x, silu)) ||
            !ggml_backend_hrx_all_storage_spans_disjoint({
                cache_r_get->src[0], x, node->src[1]}) ||
            !ggml_backend_hrx_disjoint_storage_spans(
                cache_r_get->src[0], silu) ||
            !ggml_backend_hrx_disjoint_storage_spans(
                node->src[1], silu) ||
            !ggml_backend_hrx_disjoint_storage_spans(
                cache_r_get->src[0], state) ||
            !ggml_backend_hrx_cache_owner_has_only_expected_touches(
                cgraph, cache_r_get->src[0], scale,
                cache_r_get, cache_r_write)) {
            return reject("decode cache-r owner violates the direct convolution alias contract");
        }

        ggml_backend_hrx_recurrent_cache_layer match = {};
        match.layer = layer;
        match.cache_r_get_index = cache_r_get_index;
        match.concat_index = concat_index;
        match.cache_r_write_index = cache_r_write_index;
        match.ssm_conv_index = i;
        match.silu_index = silu_index;
        match.cache_r_get = cache_r_get;
        match.concat = concat;
        match.cache_r_write = cache_r_write;
        match.ssm_conv = node;
        match.silu = silu;
        match.cache_r_owner = cache_r_get->src[0];
        match.conv_x = x;
        match.conv_filter = node->src[1];
        match.conv_dst = node;
        matches[static_cast<size_t>(slot)] = match;
    }
    if (decode_ssm_count == 0) {
        out_plan->valid = true;
        return true;
    }
    if (decode_ssm_count !=
        static_cast<int>(ggml_backend_hrx_recurrent_cache_layers.size())) {
        return reject("decode graph has a partial recurrent SSM topology");
    }
    for (const auto & match : matches) {
        if (!match) {
            return reject("decode graph is missing an expected recurrent SSM layer");
        }
    }

    int decode_gdn_count = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!node || node->op != GGML_OP_GATED_DELTA_NET ||
            !node->src[0] || !node->src[1] ||
            !node->src[2] || !node->src[3] ||
            !node->src[4] || !node->src[5] ||
            !ggml_backend_hrx_shape_is(
                node->src[2], 128, 32, 1, 1)) {
            continue;
        }
        ++decode_gdn_count;
        const ggml_tensor * cache_s_get =
            ggml_backend_hrx_zero_offset_source_chain_target(
                node->src[5], GGML_OP_GET_ROWS);
        if (!cache_s_get || !cache_s_get->src[0] ||
            !cache_s_get->src[1]) {
            return reject("decode GDN lacks the exact cache-s GET_ROWS ancestry");
        }
        const int layer =
            ggml_backend_hrx_exact_cache_layer(
                cache_s_get->src[0], 's');
        const int slot = expected_slot(layer);
        if (slot < 0 ||
            !matches[static_cast<size_t>(slot)] ||
            matches[static_cast<size_t>(slot)]->gated_delta_net) {
            return reject("decode GDN cache-s owner has an unexpected or duplicate layer");
        }
        auto & match = *matches[static_cast<size_t>(slot)];
        const ggml_tensor * cache_s_write =
            ggml_backend_hrx_find_cache_write(
                cgraph, cache_s_get->src[0], node,
                16384, 2097152);
        const ggml_tensor * scale =
            ggml_backend_hrx_find_inplace_scale(
                cgraph, cache_s_get->src[0]);
        const ggml_tensor * attention_consumer = nullptr;
        const ggml_tensor * attention_dst =
            ggml_backend_hrx_find_attention_view(
                cgraph, node, &attention_consumer);
        const int cache_s_get_index =
            ggml_backend_hrx_graph_node_index(cgraph, cache_s_get);
        const int cache_s_write_index =
            ggml_backend_hrx_graph_node_index(cgraph, cache_s_write);
        const int scale_index =
            ggml_backend_hrx_graph_node_index(cgraph, scale);
        const int attention_consumer_index =
            ggml_backend_hrx_graph_node_index(
                cgraph, attention_consumer);
        if (!cache_s_write || !attention_dst ||
            cache_s_get_index < 0 || cache_s_write_index < 0 ||
            (scale && scale_index < 0) ||
            attention_consumer_index < 0 ||
            !(match.cache_r_write_index <
                  (scale ? scale_index : cache_s_get_index) &&
              (!scale || scale_index < cache_s_get_index) &&
              cache_s_get_index < match.ssm_conv_index &&
              match.ssm_conv_index < match.silu_index &&
              match.silu_index < i &&
              i < cache_s_write_index &&
              cache_s_write_index < attention_consumer_index)) {
            return reject("decode cache-s load, GDN, update, and attention order is not exact");
        }
        if (!ggml_backend_hrx_shape_is(
                node->src[0], 128, 16, 1, 1) ||
            !ggml_backend_hrx_shape_is(
                node->src[1], 128, 16, 1, 1) ||
            !ggml_backend_hrx_shape_is(
                node->src[3], 1, 32, 1, 1) ||
            !ggml_backend_hrx_shape_is(
                node->src[4], 1, 32, 1, 1) ||
            !ggml_backend_hrx_shape_is(
                node->src[5], 128, 128, 32, 1) ||
            !ggml_backend_hrx_shape_is(
                attention_dst, 128, 32, 1, 1) ||
            !ggml_backend_hrx_same_storage_span(
                node->src[5], cache_s_get) ||
            ggml_get_op_params_i32(node, 0) != 1 ||
            ggml_nbytes(cache_s_get->src[0]) != 2097152 ||
            ggml_nbytes(node->src[5]) != 2097152 ||
            ggml_nbytes(node) != 2113536 ||
            ggml_nbytes(attention_dst) != 16384 ||
            node->src[0]->nb[1] != 128 * sizeof(float) ||
            node->src[0]->nb[2] != 2048 * sizeof(float) ||
            node->src[0]->nb[3] != 2048 * sizeof(float) ||
            node->src[1]->nb[1] != 128 * sizeof(float) ||
            node->src[1]->nb[2] != 2048 * sizeof(float) ||
            node->src[1]->nb[3] != 2048 * sizeof(float) ||
            node->src[2]->nb[1] != 128 * sizeof(float) ||
            node->src[2]->nb[2] != 8192 * sizeof(float) ||
            node->src[2]->nb[3] != 8192 * sizeof(float) ||
            node->src[4]->nb[1] != sizeof(float) ||
            node->src[4]->nb[2] != 32 * sizeof(float) ||
            node->src[4]->nb[3] != 32 * sizeof(float)) {
            return reject("decode GDN has a noncanonical shape or stride");
        }
        const auto state_consumers =
            ggml_backend_hrx_compute_consumers(cgraph, cache_s_get);
        const auto gdn_consumers =
            ggml_backend_hrx_compute_consumers(cgraph, node);
        if (state_consumers.size() != 1 ||
            state_consumers[0] != node ||
            gdn_consumers.size() != 2 ||
            std::find(
                gdn_consumers.begin(), gdn_consumers.end(),
                cache_s_write) == gdn_consumers.end() ||
            std::find(
                gdn_consumers.begin(), gdn_consumers.end(),
                attention_consumer) == gdn_consumers.end()) {
            return reject("decode cache-s materialization has an extra consumer");
        }
        std::vector<const ggml_tensor *> direct_gdn = {
            node->src[0], node->src[1], node->src[2],
            node->src[3], node->src[4],
            cache_s_get->src[0], attention_dst,
        };
        if (!ggml_backend_hrx_all_storage_spans_disjoint(direct_gdn) ||
            !ggml_backend_hrx_disjoint_storage_spans(
                cache_s_get->src[0], node->src[5]) ||
            !ggml_backend_hrx_storage_spans_adjacent(
                match.cache_r_owner, cache_s_get->src[0]) ||
            !ggml_backend_hrx_cache_owner_has_only_expected_touches(
                cgraph, cache_s_get->src[0], scale,
                cache_s_get, cache_s_write)) {
            return reject("decode cache-s owner violates the direct GDN alias contract");
        }

        match.cache_s_get_index = cache_s_get_index;
        match.gated_delta_net_index = i;
        match.cache_s_write_index = cache_s_write_index;
        match.cache_s_get = cache_s_get;
        match.gated_delta_net = node;
        match.cache_s_write = cache_s_write;
        match.cache_s_owner = cache_s_get->src[0];
        match.attention_dst = attention_dst;
    }
    if (decode_gdn_count !=
        static_cast<int>(ggml_backend_hrx_recurrent_cache_layers.size())) {
        return reject("decode graph has a partial recurrent GDN topology");
    }

    out_plan->layers.reserve(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
        if (!matches[i] ||
            !matches[i]->gated_delta_net ||
            matches[i]->layer !=
                ggml_backend_hrx_recurrent_cache_layers[i]) {
            return reject("decode recurrent layers do not match the exact 30-layer order");
        }
        if (i > 0 &&
            !ggml_backend_hrx_storage_spans_adjacent(
                matches[i - 1]->cache_s_owner,
                matches[i]->cache_r_owner)) {
            return reject("decode recurrent cache slots are not packed consecutively");
        }
        out_plan->layers.push_back(*matches[i]);
    }

    const auto & probe = out_plan->layers.front();
    std::vector<std::pair<const ggml_tensor *, const char *>>
        reference_routes = {
            {probe.cache_r_get, "get_rows_wide_f32_wg256"},
            {probe.concat, "concat_f32_dim0_wide_wg256"},
            {probe.cache_r_write, "cont_strided_f32_wide_wg256"},
            {probe.cache_s_get, "get_rows_wide_f32_wg256"},
            // Outside the validated recurrent plan, materialize the CONCAT and
            // use the alias-safe ordinary convolution. The plan below replaces
            // it with the direct cache-r route before graph execution.
            {probe.ssm_conv, "ssm_conv_f32_chan_wg256"},
            {probe.silu, "silu_f32_wide_wg256"},
            {probe.gated_delta_net, "gated_delta_net_f32_sv128_wg128"},
            {probe.cache_s_write, "cont_strided_f32_wide_wg256"},
        };
    for (const ggml_tensor * scale : {
             ggml_backend_hrx_find_inplace_scale(
                 cgraph, probe.cache_r_owner),
             ggml_backend_hrx_find_inplace_scale(
                 cgraph, probe.cache_s_owner)}) {
        if (scale) {
            reference_routes.emplace_back(
                scale, "scale_f32_generic_wg256");
        }
    }
    for (const auto & [node, route_id] : reference_routes) {
        const int index =
            ggml_backend_hrx_graph_node_index(cgraph, node);
        const auto * route =
            ggml_backend_hrx_route_for_node(
                device_context, node, index);
        if (!route || route->id != route_id) {
            return reject(
                std::string("decode reference route mismatch: ") +
                route_id);
        }
    }

    ggml_backend_hrx_dispatch_request ssm_request = {};
    if (!ggml_backend_hrx_make_ssm_conv_state_cache_request(
            device_context, probe.ssm_conv, probe, &ssm_request)) {
        return reject("failed to form direct cache-r SSM request");
    }
    const auto * ssm_route =
        ggml_backend_hrx_route_for_request(
            device_context, ssm_request);
    if (!ssm_route ||
        ssm_route->id !=
            "ssm_conv_f32_state_cache_decode_wg256" ||
        ssm_route->family !=
            "ssm_conv_f32_chan_concat" ||
        ssm_route->op != "SSM_CONV" ||
        ssm_route->source_id !=
            "ssm_conv_f32_state_cache_decode" ||
        ssm_route->artifact_id !=
            "ssm_conv_f32_state_cache_decode_loombc" ||
        ssm_route->root_symbol !=
            "@hrx2_ssm_conv_f32_state_cache_decode" ||
        ssm_route->binding_count != 4 ||
        ssm_route->parameter_count != 4 ||
        ssm_route->constant_byte_length != 0) {
        return reject("direct cache-r SSM route is absent or has the wrong ABI");
    }

    ggml_backend_hrx_dispatch_request gdn_request = {};
    if (!ggml_backend_hrx_make_gdn_state_cache_request(
            device_context, probe.gated_delta_net,
            probe, &gdn_request)) {
        return reject("failed to form direct cache-s GDN request");
    }
    const auto * gdn_route =
        ggml_backend_hrx_route_for_request(
            device_context, gdn_request);
    if (!gdn_route ||
        gdn_route->id !=
            "gated_delta_net_f32_state_cache_decode_wg64" ||
        gdn_route->family !=
            "gated_delta_net_f32" ||
        gdn_route->op != "GATED_DELTA_NET" ||
        gdn_route->source_id !=
            "gated_delta_net_f32_state_cache_decode" ||
        gdn_route->artifact_id !=
            "gated_delta_net_f32_state_cache_decode_loombc" ||
        gdn_route->root_symbol !=
            "@hrx2_gated_delta_net_f32_sv128_state_cache_decode" ||
        gdn_route->binding_count != 7 ||
        gdn_route->parameter_count != 8 ||
        gdn_route->constant_byte_length != 4) {
        return reject("direct cache-s GDN route is absent or has the wrong ABI");
    }

    for (const auto & layer : out_plan->layers) {
        ggml_backend_hrx_dispatch_request layer_ssm_request = {};
        if (!ggml_backend_hrx_make_ssm_conv_state_cache_request(
                device_context, layer.ssm_conv,
                layer, &layer_ssm_request) ||
            ggml_backend_hrx_route_for_request(
                device_context, layer_ssm_request) != ssm_route) {
            return reject(
                "not every recurrent layer resolves the direct cache-r SSM route");
        }
        ggml_backend_hrx_dispatch_request layer_gdn_request = {};
        if (!ggml_backend_hrx_make_gdn_state_cache_request(
                device_context, layer.gated_delta_net,
                layer, &layer_gdn_request) ||
            ggml_backend_hrx_route_for_request(
                device_context, layer_gdn_request) != gdn_route) {
            return reject(
                "not every recurrent layer resolves the direct cache-s GDN route");
        }
    }

    size_t skipped = 0;
    for (const auto & layer : out_plan->layers) {
        for (int index : {
                 layer.cache_r_get_index,
                 layer.concat_index,
                 layer.cache_r_write_index,
                 layer.cache_s_get_index,
                 layer.silu_index,
                 layer.cache_s_write_index}) {
            if (index < 0 || index >= cgraph->n_nodes ||
                out_plan->skip_mask[static_cast<size_t>(index)] != 0) {
                return reject("decode recurrent skip set is not one-to-one");
            }
            out_plan->skip_mask[static_cast<size_t>(index)] = 1;
            ++skipped;
        }
    }
    if (skipped != 180) {
        return reject("decode recurrent skip set is not exactly 180 passes");
    }
    out_plan->valid = true;
    out_plan->ready = true;
    return true;
}

static bool
ggml_backend_hrx_ssm_conv_silu_memo_is_owned(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_backend_hrx_ssm_conv_silu_plan & plan,
        const ggml_backend_hrx_ssm_conv_silu_layer & layer) {
    if (!device_context || !layer.ssm_conv || !layer.window) {
        return false;
    }
    const auto memo_it =
        device_context->resolved_dispatches.find(layer.ssm_conv);
    if (memo_it == device_context->resolved_dispatches.end()) {
        return false;
    }
    const auto & memo = memo_it->second;
    return
        memo.signature == layer.node_signature &&
        memo.graph_uid == plan.graph_uid &&
        memo.tensors.size() == 4 &&
        memo.tensors[0] == layer.window &&
        memo.tensors[1] == layer.x &&
        memo.tensors[2] == layer.filter &&
        memo.tensors[3] == layer.dst;
}

static bool
ggml_backend_hrx_ssm_conv_silu_memo_matches(
        ggml_backend_hrx_device_context * device_context,
        const ggml_cgraph * cgraph,
        const ggml_backend_hrx_ssm_conv_silu_plan & plan,
        const ggml_backend_hrx_ssm_conv_silu_layer & layer) {
    ggml_backend_hrx_dispatch_request request = {};
    if (!cgraph ||
        layer.ssm_conv_index < 0 ||
        layer.ssm_conv_index >= cgraph->n_nodes ||
        layer.silu_index < 0 ||
        layer.silu_index >= cgraph->n_nodes ||
        cgraph->nodes[layer.ssm_conv_index] != layer.ssm_conv ||
        cgraph->nodes[layer.silu_index] != layer.silu ||
        !ggml_backend_hrx_ssm_conv_silu_memo_is_owned(
            device_context, plan, layer) ||
        ggml_backend_hrx_node_signature(layer.ssm_conv) !=
            layer.node_signature ||
        !ggml_backend_hrx_make_ssm_conv_silu_request(
            device_context,
            layer.ssm_conv,
            layer,
            &request) ||
        ggml_backend_hrx_ssm_conv_silu_request_fingerprint(
            layer, request.tensors) !=
            layer.request_fingerprint) {
        return false;
    }
    const auto & memo =
        device_context->resolved_dispatches.find(layer.ssm_conv)->second;
    return
        ggml_backend_hrx_ssm_conv_silu_route_is_exact(memo.route) &&
        memo.route->prepasses.empty() &&
        memo.prepass_free &&
        memo.compiled &&
        memo.compiled->executable &&
        memo.compiled->route == memo.route &&
        !memo.prepass_plan &&
        memo.constants.empty() &&
        memo.route->binding_count == memo.tensors.size() &&
        memo.route->constant_byte_length == 0;
}

static bool
ggml_backend_hrx_ssm_conv_silu_seed_set_matches(
        ggml_backend_hrx_device_context * device_context,
        const ggml_cgraph * cgraph,
        const ggml_backend_hrx_ssm_conv_silu_plan & plan) {
    if (!plan.valid || !plan.ready || plan.layers.size() != 30) {
        return false;
    }
    for (const auto & layer : plan.layers) {
        if (!ggml_backend_hrx_ssm_conv_silu_memo_matches(
                device_context, cgraph, plan, layer)) {
            return false;
        }
    }
    return true;
}

static void
ggml_backend_hrx_clear_ssm_conv_silu_plan(
        ggml_backend_hrx_device_context * device_context) {
    if (!device_context) {
        return;
    }
    if (!device_context->ssm_conv_silu_plan) {
        return;
    }
    for (const auto & layer : device_context->ssm_conv_silu_plan->layers) {
        const auto window_memo =
            device_context->resolved_dispatches.find(layer.window);
        if (window_memo != device_context->resolved_dispatches.end() &&
            window_memo->second.graph_uid ==
                device_context->ssm_conv_silu_plan->graph_uid) {
            // The sparse CONCAT route depends on this exact graph topology.
            // Invalidate it with the plan so no hot-path graph predicate is
            // needed for ordinary decode dispatches.
            window_memo->second.compiled = nullptr;
        }
        if (ggml_backend_hrx_ssm_conv_silu_memo_is_owned(
                device_context,
                *device_context->ssm_conv_silu_plan,
                layer)) {
            // Invalidate in place instead of erasing the map value. Besides
            // preserving capacity, this avoids pulling resolved_dispatch
            // destruction into the hot translation-unit layout. A null
            // compiled pointer can never satisfy memo_hit.
            device_context->resolved_dispatches.find(layer.ssm_conv)
                ->second.compiled = nullptr;
        }
    }
    device_context->ssm_conv_silu_plan.reset();
}

static bool
ggml_backend_hrx_graph_has_pp_ssm_conv_silu_candidate(
        const ggml_cgraph * cgraph) {
    for (int i = 0; cgraph && i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node && node->op == GGML_OP_SSM_CONV &&
            ggml_backend_hrx_shape_is(node, 8192, 512, 1, 1) &&
            node->src[1] &&
            ggml_backend_hrx_shape_is(
                node->src[1], 4, 8192, 1, 1)) {
            return true;
        }
    }
    return false;
}

static bool
ggml_backend_hrx_seed_ssm_conv_silu_dispatches(
        ggml_backend_hrx_context * context,
        const ggml_cgraph * cgraph,
        const ggml_backend_hrx_ssm_conv_silu_plan & plan,
        std::string * out_error) {
    auto reject = [&](const std::string & message) {
        if (out_error) {
            *out_error = message;
        }
        return false;
    };
    if (!context || !context->device_context || !cgraph ||
        !plan.valid || !plan.ready || plan.layers.size() != 30 ||
        !context->device_context->reg_context ||
        !context->device_context->reg_context->catalog) {
        return reject("cannot seed an incomplete SSM_CONV/SiLU plan");
    }
    std::vector<ggml_backend_hrx_ssm_conv_silu_dispatch_seed> pending;
    pending.reserve(plan.layers.size());
    const ggml_backend_hrx_catalog_route * accepted_route = nullptr;
    ggml_backend_hrx_compiled_route * accepted_compiled = nullptr;
    for (const auto & layer : plan.layers) {
        ggml_backend_hrx_dispatch_request request = {};
        if (layer.ssm_conv_index < 0 ||
            layer.ssm_conv_index >= cgraph->n_nodes ||
            layer.silu_index < 0 ||
            layer.silu_index >= cgraph->n_nodes ||
            cgraph->nodes[layer.ssm_conv_index] != layer.ssm_conv ||
            cgraph->nodes[layer.silu_index] != layer.silu ||
            ggml_backend_hrx_node_signature(layer.ssm_conv) !=
                layer.node_signature ||
            !ggml_backend_hrx_make_ssm_conv_silu_request(
                context->device_context,
                layer.ssm_conv,
                layer,
                &request)) {
            return reject(
                "failed to rebuild a validated SSM_CONV/SiLU request");
        }
        if (ggml_backend_hrx_ssm_conv_silu_request_fingerprint(
                layer, request.tensors) !=
            layer.request_fingerprint) {
            return reject(
                "validated SSM_CONV/SiLU request fingerprint changed");
        }
        ggml_backend_hrx_add_tensor_overlap_facts(
            &request.problem, request.tensors);
        const auto * route =
            ggml_backend_hrx_catalog_find_route(
                *context->device_context->reg_context->catalog,
                request.problem);
        if (!ggml_backend_hrx_ssm_conv_silu_route_is_exact(route) ||
            !route->prepasses.empty() ||
            !route->workload_argument_sources.empty() ||
            request.tensors.size() != 4 ||
            !request.constants.empty() ||
            route->binding_count != request.tensors.size() ||
            route->constant_byte_length != request.constants.size()) {
            return reject(
                "validated SSM_CONV/SiLU request no longer resolves its exact prepass-free route");
        }

        std::vector<ggml_backend_hrx_catalog_binding> bindings;
        std::string binding_error;
        if (!ggml_backend_hrx_catalog_make_config_bindings(
                *route,
                request.problem,
                &bindings,
                &binding_error)) {
            return reject(binding_error);
        }
        std::vector<int64_t> workload;
        std::string workload_error;
        if (!ggml_backend_hrx_resolve_workload_arguments(
                *route,
                request.problem,
                &workload,
                &workload_error)) {
            return reject(workload_error);
        }
        auto * compiled =
            ggml_backend_hrx_get_compiled_route(
                context->device_context,
                *route,
                request.problem,
                bindings,
                workload);
        if (!compiled || !compiled->executable) {
            return reject(
                "failed to compile the validated SSM_CONV/SiLU route");
        }
        if (compiled->route != route ||
            compiled->export_info.binding_count != 4 ||
            compiled->export_info.parameter_count != 4 ||
            compiled->export_info.constant_byte_length != 0 ||
            compiled->launch_config.workload_argument_count != 0 ||
            !workload.empty() ||
            (accepted_route && accepted_route != route) ||
            (accepted_compiled && accepted_compiled != compiled)) {
            return reject(
                "validated SSM_CONV/SiLU route has inconsistent compiled ABI or workload");
        }
        accepted_route = route;
        accepted_compiled = compiled;

        ggml_backend_hrx_ssm_conv_silu_dispatch_seed seed = {};
        seed.node = layer.ssm_conv;
        seed.signature = layer.node_signature;
        seed.graph_uid = cgraph->uid;
        seed.route = route;
        seed.bindings = std::move(bindings);
        seed.workload = std::move(workload);
        seed.tensors = std::move(request.tensors);
        seed.constants = std::move(request.constants);
        seed.compiled = compiled;
        pending.emplace_back(std::move(seed));
    }
    if (pending.size() != 30) {
        return reject(
            "SSM_CONV/SiLU dispatch seed set is not exactly 30 entries");
    }
    for (const auto & seed : pending) {
        const auto existing =
            context->device_context->resolved_dispatches.find(seed.node);
        if (existing !=
                context->device_context->resolved_dispatches.end() &&
            existing->second.prepass_plan.has_value()) {
            return reject(
                "SSM_CONV/SiLU dispatch seed would replace a prepass plan");
        }
    }
    for (auto & seed : pending) {
        auto & memo =
            context->device_context->resolved_dispatches[seed.node];
        memo.signature = seed.signature;
        memo.graph_uid = seed.graph_uid;
        memo.route = seed.route;
        memo.bindings = std::move(seed.bindings);
        memo.workload = std::move(seed.workload);
        memo.tensors = std::move(seed.tensors);
        memo.constants = std::move(seed.constants);
        memo.prepass_free = true;
        memo.compiled = seed.compiled;
    }
    return true;
}

static bool
ggml_backend_hrx_prepare_ssm_conv_silu_plan_miss(
        ggml_backend_hrx_context * context,
        const ggml_cgraph * cgraph,
        const char * caller) {
    ggml_backend_hrx_clear_ssm_conv_silu_plan(
        context ? context->device_context : nullptr);
    std::unique_ptr<ggml_backend_hrx_ssm_conv_silu_plan> candidate(
        new (std::nothrow) ggml_backend_hrx_ssm_conv_silu_plan);
    std::string ssm_conv_silu_error;
    if (!candidate ||
        !ggml_backend_hrx_build_ssm_conv_silu_plan(
            context->device_context,
            cgraph,
            candidate.get(),
            &ssm_conv_silu_error) ||
        !candidate->ready ||
        !ggml_backend_hrx_seed_ssm_conv_silu_dispatches(
            context,
            cgraph,
            *candidate,
            &ssm_conv_silu_error)) {
        ggml_backend_hrx_trace_event(
            context->device_context->reg_context, {
                {"event", "ssm_conv_silu_plan_rejected"},
                {"reason", ssm_conv_silu_error},
            });
        GGML_LOG_ERROR(
            "%s: SSM_CONV/SiLU graph plan rejected: %s\n",
            caller, ssm_conv_silu_error.c_str());
        context->device_context->current_recurrent_cache_plan =
            nullptr;
        context->device_context->current_gdn_qk_scale_plan =
            nullptr;
        context->device_context->current_graph = nullptr;
        context->device_context->current_node_index = -1;
        return false;
    }
    candidate->owner = context;
    context->device_context->ssm_conv_silu_plan = std::move(candidate);
    ggml_backend_hrx_trace_event(
        context->device_context->reg_context, {
            {"event", "ssm_conv_silu_plan_ready"},
            {"layer_count",
             context->device_context->ssm_conv_silu_plan->layers.size()},
            {"alias_layers",
             context->device_context->ssm_conv_silu_plan->alias_layers},
            {"disjoint_layers",
             context->device_context->ssm_conv_silu_plan->disjoint_layers},
            {"skipped_materializations", 30},
            {"seeded_dispatches", 30},
        });
    return true;
}

static bool
ggml_backend_hrx_prepare_ssm_conv_silu_for_pp(
        ggml_backend_hrx_context * context,
        const ggml_cgraph * cgraph) {
    if (!context || !context->device_context || !cgraph) {
        return false;
    }
    const auto * active_catalog =
        context->device_context->reg_context &&
                context->device_context->reg_context->catalog
            ? context->device_context->reg_context->catalog.get()
            : nullptr;
    const auto * plan =
        context->device_context->ssm_conv_silu_plan.get();
    const bool plan_hit =
        cgraph->uid != 0 &&
        plan &&
        plan->owner == context &&
        plan->valid &&
        plan->graph_uid == cgraph->uid &&
        plan->node_count == cgraph->n_nodes &&
        plan->catalog == active_catalog;
    if (plan_hit &&
        !ggml_backend_hrx_ssm_conv_silu_seed_set_matches(
            context->device_context, cgraph, *plan)) {
        std::string error;
        if (!ggml_backend_hrx_seed_ssm_conv_silu_dispatches(
                context, cgraph, *plan, &error)) {
            GGML_LOG_ERROR(
                "%s: failed to restore PP512 SSM_CONV/SiLU dispatch seeds: %s\n",
                __func__, error.c_str());
            ggml_backend_hrx_clear_ssm_conv_silu_plan(
                context->device_context);
            return false;
        }
    }
    if (!plan_hit) {
        if (plan) {
            ggml_backend_hrx_clear_ssm_conv_silu_plan(
                context->device_context);
        }
        if (cgraph->uid != 0 &&
            ggml_backend_hrx_graph_has_pp_ssm_conv_silu_candidate(
                cgraph) &&
            !ggml_backend_hrx_prepare_ssm_conv_silu_plan_miss(
                context, cgraph, __func__)) {
            return false;
        }
    }
    return true;
}

static enum ggml_status ggml_backend_hrx_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    if (context->device_context->options && context->device_context->options->trace_graph) {
        ggml_backend_hrx_trace_event(context->device_context->reg_context, {
            {"event", "graph_compute_begin"},
            {"device", context->device_context->name},
            {"node_count", cgraph ? cgraph->n_nodes : 0},
        });
    }
    if (!ggml_backend_hrx_sync_graph_entry_streams(context->device_context, context->stream)) {
        return GGML_STATUS_FAILED;
    }
    {
        std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
        context->device_context->active_stream = context->stream;
    }
    // A tensor pointer only identifies one set of contents within a single graph
    // evaluation, so the prepass result cannot be carried across calls.
    ggml_backend_hrx_clear_quant_scratch_source(context->device_context);
    context->device_context->class_scratch_source.clear();

    context->device_context->graph_epoch++;
    context->device_context->current_graph = cgraph;
    context->device_context->current_recurrent_cache_plan = nullptr;
    context->device_context->current_gdn_qk_scale_plan = nullptr;
    context->device_context->current_terminal_qact_plan = nullptr;
    context->device_context->current_shared_expert_terminal_plan =
        nullptr;
    context->device_context->current_gdn_rms_side_plan = nullptr;
    const bool recurrent_cache_plan_hit =
        cgraph && cgraph->uid != 0 &&
        context->recurrent_cache_plan.valid &&
        context->recurrent_cache_plan.graph_uid == cgraph->uid &&
        context->recurrent_cache_plan.node_count == cgraph->n_nodes &&
        context->recurrent_cache_plan.catalog ==
            (context->device_context->reg_context &&
                     context->device_context->reg_context->catalog
                 ? context->device_context->reg_context->catalog.get()
                 : nullptr);
    if (!recurrent_cache_plan_hit && cgraph) {
        std::string recurrent_cache_error;
        if (!ggml_backend_hrx_build_recurrent_cache_plan(
                context->device_context,
                cgraph,
                &context->recurrent_cache_plan,
                &recurrent_cache_error)) {
            ggml_backend_hrx_trace_event(
                context->device_context->reg_context, {
                    {"event", "recurrent_cache_plan_rejected"},
                    {"reason", recurrent_cache_error},
                });
            GGML_LOG_ERROR(
                "%s: recurrent-cache graph plan rejected: %s\n",
                __func__, recurrent_cache_error.c_str());
            context->recurrent_cache_plan = {};
            context->device_context->current_graph = nullptr;
            context->device_context->current_node_index = -1;
            return GGML_STATUS_FAILED;
        }
    }
    if (cgraph && context->recurrent_cache_plan.ready) {
        context->device_context->current_recurrent_cache_plan =
            &context->recurrent_cache_plan;
        if (!recurrent_cache_plan_hit) {
            ggml_backend_hrx_trace_event(
                context->device_context->reg_context, {
                    {"event", "recurrent_cache_plan_ready"},
                    {"layer_count",
                     context->recurrent_cache_plan.layers.size()},
                    {"skipped_materializations", 180},
                    {"producer_dispatches", 60},
            });
        }
    }
    const bool gdn_qk_scale_plan_hit =
        cgraph && cgraph->uid != 0 &&
        context->gdn_qk_scale_plan.valid &&
        context->gdn_qk_scale_plan.graph_uid == cgraph->uid &&
        context->gdn_qk_scale_plan.node_count == cgraph->n_nodes &&
        context->gdn_qk_scale_plan.catalog ==
            (context->device_context->reg_context &&
                     context->device_context->reg_context->catalog
                 ? context->device_context->reg_context->catalog.get()
                 : nullptr);
    if (!gdn_qk_scale_plan_hit && cgraph) {
        if (context->device_context->ssm_conv_silu_plan) {
            ggml_backend_hrx_clear_ssm_conv_silu_plan(
                context->device_context);
        }
        std::string gdn_qk_scale_error;
        if (!ggml_backend_hrx_build_gdn_qk_scale_plan(
                context->device_context,
                cgraph,
                &context->gdn_qk_scale_plan,
                &gdn_qk_scale_error)) {
            ggml_backend_hrx_trace_event(
                context->device_context->reg_context, {
                    {"event", "gdn_qk_scale_plan_rejected"},
                    {"reason", gdn_qk_scale_error},
                });
            GGML_LOG_ERROR(
                "%s: GDN q/k scale graph plan rejected: %s\n",
                __func__, gdn_qk_scale_error.c_str());
            context->gdn_qk_scale_plan = {};
            context->device_context->current_recurrent_cache_plan =
                nullptr;
            context->device_context->current_graph = nullptr;
            context->device_context->current_node_index = -1;
            return GGML_STATUS_FAILED;
        }
    }
    if (cgraph && context->gdn_qk_scale_plan.ready) {
        context->device_context->current_gdn_qk_scale_plan =
            &context->gdn_qk_scale_plan;
        if (!ggml_backend_hrx_prepare_ssm_conv_silu_for_pp(
                context, cgraph)) {
            context->device_context->current_recurrent_cache_plan =
                nullptr;
            context->device_context->current_gdn_qk_scale_plan =
                nullptr;
            context->device_context->current_graph = nullptr;
            context->device_context->current_node_index = -1;
            return GGML_STATUS_FAILED;
        }
        if (!gdn_qk_scale_plan_hit) {
            ggml_backend_hrx_trace_event(
                context->device_context->reg_context, {
                    {"event", "gdn_qk_scale_plan_ready"},
                    {"layer_count",
                     context->gdn_qk_scale_plan.layers.size()},
                    {"skipped_materializations", 60},
            });
        }
    }
    const bool gdn_rms_side_cache_hit =
        cgraph && cgraph->uid != 0 &&
        context->gdn_rms_side_plan.examined &&
        context->gdn_rms_side_plan.graph_uid == cgraph->uid &&
        context->gdn_rms_side_plan.node_count == cgraph->n_nodes &&
        context->gdn_rms_side_plan.catalog ==
            (context->device_context->reg_context
                 ? context->device_context->reg_context->catalog.get()
                 : nullptr);
    if (!gdn_rms_side_cache_hit) {
        // Publish the terminal route and producer mask only when all thirty
        // recurrent layers resolve the exact standalone-scale route.
        context->gdn_rms_side_plan = {};
        context->gdn_rms_side_plan.graph_uid =
            cgraph ? cgraph->uid : 0;
        context->gdn_rms_side_plan.node_count =
            cgraph ? cgraph->n_nodes : 0;
        context->gdn_rms_side_plan.catalog =
            context->device_context->reg_context
                ? context->device_context->reg_context->catalog.get()
                : nullptr;
        context->gdn_rms_side_plan.examined = true;
        ggml_backend_hrx_gdn_rms_side_graph_plan candidate = {};
        if (ggml_backend_hrx_build_gdn_rms_side_graph_plan(
                context->device_context, cgraph, &candidate)) {
            context->gdn_rms_side_plan = std::move(candidate);
        }
    }
    if (context->device_context->current_gdn_qk_scale_plan &&
        !context->gdn_rms_side_plan.ready) {
        // A ready q/k-scale plan identifies the exact PP512 recurrent graph.
        // Do not silently run its old RMS materialization path when the
        // admitted terminal route or any one of the thirty layer contracts
        // failed to resolve.
        context->device_context->current_recurrent_cache_plan = nullptr;
        context->device_context->current_gdn_qk_scale_plan = nullptr;
        context->device_context->current_graph = nullptr;
        context->device_context->current_node_index = -1;
        return GGML_STATUS_FAILED;
    }
    if (context->gdn_rms_side_plan.ready) {
        context->device_context->current_gdn_rms_side_plan =
            &context->gdn_rms_side_plan;
    }
    const bool terminal_qact_cache_hit =
        cgraph && cgraph->uid != 0 &&
        context->terminal_qact_plan.examined &&
        context->terminal_qact_plan.graph_uid == cgraph->uid &&
        context->terminal_qact_plan.node_count == cgraph->n_nodes &&
        context->terminal_qact_plan.catalog ==
            (context->device_context->reg_context
                 ? context->device_context->reg_context->catalog.get()
                 : nullptr);
    if (!terminal_qact_cache_hit) {
        // Publish only the complete 36-Q5/4-Q6 PP topology. Producer and
        // consumer routes are selected together, so a partial catalog can
        // never expose qact scratch without its exact reader.
        context->terminal_qact_plan = {};
        context->terminal_qact_plan.graph_uid =
            cgraph ? cgraph->uid : 0;
        context->terminal_qact_plan.node_count =
            cgraph ? cgraph->n_nodes : 0;
        context->terminal_qact_plan.catalog =
            context->device_context->reg_context
                ? context->device_context->reg_context->catalog.get()
                : nullptr;
        context->terminal_qact_plan.examined = true;
        ggml_backend_hrx_terminal_qact_graph_plan candidate = {};
        if (ggml_backend_hrx_build_terminal_qact_graph_plan(
                context->device_context, cgraph, &candidate)) {
            context->terminal_qact_plan = std::move(candidate);
        }
    }
    if (context->device_context->current_gdn_qk_scale_plan &&
        !context->terminal_qact_plan.ready) {
        // The GDN q/k plan is the established exact-PP512 discriminator.
        // Never fall back to the old Q5 activation path on that graph.
        context->device_context->current_recurrent_cache_plan = nullptr;
        context->device_context->current_gdn_qk_scale_plan = nullptr;
        context->device_context->current_gdn_rms_side_plan = nullptr;
        context->device_context->current_graph = nullptr;
        context->device_context->current_node_index = -1;
        return GGML_STATUS_FAILED;
    }
    if (context->terminal_qact_plan.ready) {
        context->device_context->current_terminal_qact_plan =
            &context->terminal_qact_plan;
        if (!terminal_qact_cache_hit) {
            ggml_backend_hrx_trace_event(
                context->device_context->reg_context, {
                    {"event", "terminal_qact_plan_ready"},
                    {"layer_count",
                     context->terminal_qact_plan.layer_count},
                    {"scratch_bytes",
                     GGML_HRX_MMID_QACT_END},
            });
        }
    }
    context->device_context->current_moe_router_tail_plan = nullptr;
    const bool moe_router_tail_cache_hit =
        cgraph && cgraph->uid != 0 &&
        context->moe_router_tail_plan.examined &&
        context->moe_router_tail_plan.graph_uid == cgraph->uid &&
        context->moe_router_tail_plan.node_count == cgraph->n_nodes &&
        context->moe_router_tail_plan.catalog ==
            (context->device_context->reg_context
                 ? context->device_context->reg_context->catalog.get()
                 : nullptr);
    if (!moe_router_tail_cache_hit) {
        // Publish only a completely validated 40-layer candidate. Remember a
        // rejection only for this exact nonzero graph UID/catalog pair, so the
        // ordinary fusion mask can still be reused without ever turning a
        // rejected partial match into a ready plan.
        context->moe_router_tail_plan = {};
        context->moe_router_tail_plan.graph_uid =
            cgraph ? cgraph->uid : 0;
        context->moe_router_tail_plan.node_count =
            cgraph ? cgraph->n_nodes : 0;
        context->moe_router_tail_plan.catalog =
            context->device_context->reg_context
                ? context->device_context->reg_context->catalog.get()
                : nullptr;
        context->moe_router_tail_plan.examined = true;
        ggml_backend_hrx_moe_router_tail_graph_plan candidate = {};
        if (ggml_backend_hrx_build_moe_router_tail_graph_plan(
                context->device_context, cgraph, &candidate)) {
            context->moe_router_tail_plan = std::move(candidate);
        }
    }
    if (context->moe_router_tail_plan.ready) {
        context->device_context->current_moe_router_tail_plan =
            &context->moe_router_tail_plan;
    }
    const bool shared_expert_terminal_cache_hit =
        cgraph && cgraph->uid != 0 &&
        context->shared_expert_terminal_plan.examined &&
        context->shared_expert_terminal_plan.graph_uid == cgraph->uid &&
        context->shared_expert_terminal_plan.node_count ==
            cgraph->n_nodes &&
        context->shared_expert_terminal_plan.catalog ==
            (context->device_context->reg_context
                 ? context->device_context->reg_context->catalog.get()
                 : nullptr);
    if (!shared_expert_terminal_cache_hit) {
        // Publish producer suppression only after every one of the forty
        // layers resolves the exact fused route and alias contract.
        context->shared_expert_terminal_plan = {};
        context->shared_expert_terminal_plan.graph_uid =
            cgraph ? cgraph->uid : 0;
        context->shared_expert_terminal_plan.node_count =
            cgraph ? cgraph->n_nodes : 0;
        context->shared_expert_terminal_plan.catalog =
            context->device_context->reg_context
                ? context->device_context->reg_context->catalog.get()
                : nullptr;
        context->shared_expert_terminal_plan.examined = true;
        ggml_backend_hrx_shared_expert_terminal_graph_plan candidate = {};
        if (ggml_backend_hrx_build_shared_expert_terminal_graph_plan(
                context->device_context, cgraph, &candidate)) {
            context->shared_expert_terminal_plan =
                std::move(candidate);
        }
    }
    if (context->shared_expert_terminal_plan.ready) {
        context->device_context->current_shared_expert_terminal_plan =
            &context->shared_expert_terminal_plan;
    }
    const bool fusion_cache_hit =
        cgraph && cgraph->uid != 0 &&
        recurrent_cache_plan_hit &&
        gdn_qk_scale_plan_hit &&
        gdn_rms_side_cache_hit &&
        terminal_qact_cache_hit &&
        moe_router_tail_cache_hit &&
        shared_expert_terminal_cache_hit &&
        context->fusion_graph_uid == cgraph->uid &&
        context->fusion_producer_mask.size() ==
            static_cast<size_t>(cgraph->n_nodes);
    if (!fusion_cache_hit) {
        // Invalidate before recomputing so an early dispatch failure cannot
        // associate a partially updated mask with the previous graph UID.
        context->fusion_graph_uid = 0;
        context->fusion_producer_mask.assign(
            static_cast<size_t>(cgraph ? cgraph->n_nodes : 0), 0);
    }
    for (int i = 0; cgraph && i < cgraph->n_nodes; ++i) {
        context->device_context->current_node_index = i;
        const ggml_tensor * node = cgraph->nodes[i];
        if (!ggml_backend_hrx_is_metadata_op(node) && !ggml_backend_hrx_is_empty_op(node)) {
            if (context->device_context->current_gdn_qk_scale_plan &&
                context->device_context->current_gdn_qk_scale_plan
                        ->skip_mask[static_cast<size_t>(i)] != 0) {
                ggml_backend_hrx_trace_event(
                    context->device_context->reg_context, {
                        {"event", "fused_producer_skipped"},
                        {"producer_op", ggml_op_desc(node)},
                        {"consumer_op", "GATED_DELTA_NET"},
                        {"route_id",
                         "gated_delta_net_f32_sv128_qk_l2_full_head_wg256"},
                    });
                continue;
            }
            if (context->device_context->current_recurrent_cache_plan) {
                if (context->device_context->current_recurrent_cache_plan
                            ->skip_mask[static_cast<size_t>(i)] != 0) {
                    bool ssm_materialization = false;
                    for (const auto & layer :
                         context->device_context
                             ->current_recurrent_cache_plan->layers) {
                        if (i == layer.cache_r_get_index ||
                            i == layer.concat_index ||
                            i == layer.cache_r_write_index ||
                            i == layer.silu_index) {
                            ssm_materialization = true;
                            break;
                        }
                    }
                    ggml_backend_hrx_trace_event(
                        context->device_context->reg_context, {
                            {"event", "fused_producer_skipped"},
                            {"producer_op", ggml_op_desc(node)},
                            {"consumer_op", ssm_materialization
                                 ? "SSM_CONV" : "GATED_DELTA_NET"},
                            {"route_id", ssm_materialization
                                 ? "ssm_conv_f32_state_cache_decode_wg256"
                                 : "gated_delta_net_f32_state_cache_decode_wg64"},
                        });
                    continue;
                }
            }
            if (context->device_context->current_moe_router_tail_plan &&
                context->device_context->current_moe_router_tail_plan
                    ->add_skip_mask[static_cast<size_t>(i)] != 0) {
                continue;
            }
            const bool fused_producer = fusion_cache_hit ?
                context->fusion_producer_mask[static_cast<size_t>(i)] != 0 :
                ggml_backend_hrx_is_fused_producer_node(
                    context->device_context, cgraph, i, context);
            if (!fusion_cache_hit) {
                context->fusion_producer_mask[static_cast<size_t>(i)] =
                    fused_producer ? 1 : 0;
            }
            if (fused_producer) {
                continue;
            }
            if (ggml_backend_hrx_dispatch_node(context->device_context, node)) {
                continue;
            }
            if (context->device_context->options && context->device_context->options->trace_graph) {
                ggml_backend_hrx_trace_event(context->device_context->reg_context, {
                    {"event", "unsupported_compute_node"},
                    {"device", context->device_context->name},
                    {"op", ggml_op_desc(node)},
                    {"node", ggml_get_name(node)},
                });
            }
            GGML_LOG_ERROR(
                "%s: HRX3 backend has no matching compute route; unsupported op %s node=%s\n",
                __func__, ggml_op_desc(node), ggml_get_name(node));
            context->device_context->current_recurrent_cache_plan = nullptr;
            context->device_context->current_gdn_qk_scale_plan =
                nullptr;
            context->device_context->current_terminal_qact_plan =
                nullptr;
            context->device_context->current_moe_router_tail_plan = nullptr;
            context->device_context
                ->current_shared_expert_terminal_plan = nullptr;
            context->device_context->current_gdn_rms_side_plan =
                nullptr;
            context->device_context->current_graph = nullptr;
            context->device_context->current_node_index = -1;
            if (cgraph && cgraph->uid == 0) {
                ggml_backend_hrx_clear_ssm_conv_silu_plan(
                    context->device_context);
            }
            return GGML_STATUS_FAILED;
        }
    }

    if (!fusion_cache_hit && cgraph && cgraph->uid != 0) {
        context->fusion_graph_uid = cgraph->uid;
    }
    context->device_context->current_recurrent_cache_plan = nullptr;
    context->device_context->current_gdn_qk_scale_plan = nullptr;
    context->device_context->current_terminal_qact_plan = nullptr;
    context->device_context->current_moe_router_tail_plan = nullptr;
    context->device_context->current_shared_expert_terminal_plan =
        nullptr;
    context->device_context->current_gdn_rms_side_plan = nullptr;
    context->device_context->current_graph = nullptr;
    context->device_context->current_node_index = -1;
    if (cgraph && cgraph->uid == 0) {
        ggml_backend_hrx_clear_ssm_conv_silu_plan(
            context->device_context);
    }
    ggml_backend_hrx_synchronize(backend);
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_hrx_i = {
    /* .get_name           = */ ggml_backend_hrx_get_name,
    /* .free               = */ ggml_backend_hrx_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ ggml_backend_hrx_synchronize,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ ggml_backend_hrx_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

static const char * ggml_backend_hrx_device_get_name(ggml_backend_dev_t dev) {
    return ggml_backend_hrx_get_device_context(dev)->name.c_str();
}

static const char * ggml_backend_hrx_device_get_description(ggml_backend_dev_t dev) {
    return ggml_backend_hrx_get_device_context(dev)->description.c_str();
}

static void ggml_backend_hrx_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * context = ggml_backend_hrx_get_device_context(dev);
    *free = context->memory_total;
    *total = context->memory_total;
}

static enum ggml_backend_dev_type ggml_backend_hrx_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_hrx_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = ggml_backend_hrx_device_get_name(dev);
    props->description = ggml_backend_hrx_device_get_description(dev);
    props->type = ggml_backend_hrx_device_get_type(dev);
    ggml_backend_hrx_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = nullptr;
    props->caps = {
        /* .async = */ true,
        /* .host_buffer = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events = */ false,
    };
}

static ggml_backend_t ggml_backend_hrx_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    auto * device_context = ggml_backend_hrx_get_device_context(dev);
    hrx_stream_t stream = nullptr;
    if (!GGML_HRX_CHECK(hrx_stream_create(device_context->device, 0, &stream))) {
        return nullptr;
    }

    auto * context = new (std::nothrow) ggml_backend_hrx_context {
        /* .device_context = */ device_context,
        /* .stream         = */ stream,
        /* .name           = */ device_context->name,
        /* .fusion_graph_uid = */ 0,
        /* .fusion_producer_mask = */ {},
        /* .moe_router_tail_plan = */ {},
        /* .terminal_qact_plan = */ {},
        /* .shared_expert_terminal_plan = */ {},
        /* .gdn_rms_side_plan = */ {},
        /* .recurrent_cache_plan = */ {},
        /* .gdn_qk_scale_plan = */ {},
    };
    if (!context) {
        hrx_stream_release(stream);
        return nullptr;
    }

    ggml_backend_t backend = new (std::nothrow) ggml_backend {
        /* .guid    = */ ggml_backend_hrx_guid(),
        /* .iface   = */ ggml_backend_hrx_i,
        /* .device  = */ dev,
        /* .context = */ context,
    };
    if (!backend) {
        hrx_stream_release(stream);
        delete context;
        return nullptr;
    }

    ggml_backend_hrx_register_stream(device_context, stream);
    return backend;
}

// True when `op` writes its result through a view that is a *whole-tensor* alias of
// its own src[0] (ggml_clamp and friends build their result with ggml_view_tensor).
// Such a view has the same type, the same ne[]/nb[] and a zero view offset, so it is
// exactly as row-contiguous as the tensor it aliases and an elementwise kernel that
// writes dst[i] from src0[i] is safe to run in place.
static bool ggml_backend_hrx_is_full_inplace_alias(const ggml_tensor * op) {
    const ggml_tensor * src = op->src[0];
    if (!op->view_src || !src) {
        return false;
    }
    // The alias must be of src[0] itself (either directly, or of the same storage
    // when src[0] is itself a view) and must cover the whole tensor.
    const ggml_tensor * op_base  = op->view_src;
    const ggml_tensor * src_base = src->view_src ? src->view_src : src;
    if (op_base != src_base || op->view_offs != (src->view_src ? src->view_offs : 0)) {
        return false;
    }
    if (op->type != src->type) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (op->ne[i] != src->ne[i] || op->nb[i] != src->nb[i]) {
            return false;
        }
    }
    return true;
}

static bool ggml_backend_hrx_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    // Metadata ops only reinterpret an existing allocation, and graph_compute
    // skips them outright, so claiming them is free. They have to be claimed
    // before the view rejection below, which would otherwise reject every one
    // of them: a VIEW/RESHAPE/PERMUTE/TRANSPOSE node has view_src set by
    // construction and its ne[] differs from src[0], so it can never be a full
    // in-place alias. Refusing them aborts the scheduler outright
    // ("pre-allocated tensor in a buffer that cannot run the operation") as
    // soon as a tensor living in an HRX buffer is sliced, which is what the
    // recurrent-state cache does every layer.
    if (ggml_backend_hrx_is_metadata_op(op) || ggml_backend_hrx_is_empty_op(op)) {
        return true;
    }
    // Reject compute ops whose output is a non-contiguous view - these tensors may be
    // used by SCALE/SET_ROWS later, which require row-contiguous data. Two cases are
    // exempt. A whole-tensor in-place alias has the same layout as the tensor it
    // aliases, so it cannot introduce a non-contiguous consumer; without that
    // exemption an op like CLAMP is rejected here, but a view can never be reassigned
    // to another backend ("views are always on the same backend as the source"), so
    // the CPU ends up executing it directly on an HRX device pointer and faults.
    // A contiguous partial view is equally safe: HRX kernels address dst by row and
    // element index, which such a view satisfies exactly as an ordinary tensor does,
    // with the view offset already folded into tensor->data. This is what lets the
    // recurrent-state cache live in an HRX buffer -- its per-layer slices are written
    // by CPY into contiguous views, and rejecting those aborts the scheduler.
    if (op->view_src != nullptr &&
        !ggml_backend_hrx_is_full_inplace_alias(op) &&
        !ggml_is_contiguous(op)) {
        auto * dc = ggml_backend_hrx_get_device_context(dev);
        ggml_backend_hrx_trace_event(dc ? dc->reg_context : nullptr, {
            {"event", "supports_op_rejected"},
            {"reason", "non_contiguous_view_output"},
            {"op", ggml_op_desc(op)},
            {"node", ggml_get_name(op)},
            {"dst_type", ggml_type_name(op->type)},
            {"dst_ne", {op->ne[0], op->ne[1], op->ne[2], op->ne[3]}},
            {"dst_nb", {op->nb[0], op->nb[1], op->nb[2], op->nb[3]}},
        });
        return false;
    }
    auto * device_context = ggml_backend_hrx_get_device_context(dev);
    const uint64_t support_signature = ggml_backend_hrx_node_signature(op);
    {
        const auto cached = device_context->supports_answers.find(support_signature);
        if (cached != device_context->supports_answers.end()) {
            return cached->second;
        }
    }
    ggml_backend_hrx_dispatch_request request;
    if (!ggml_backend_hrx_make_dispatch_request(device_context, op, &request)) {
        if (ggml_backend_hrx_trace_enabled(device_context->reg_context)) {
            nlohmann::json srcs = nlohmann::json::array();
            for (int i = 0; i < GGML_MAX_SRC && op->src[i]; ++i) {
                const ggml_tensor * s = op->src[i];
                srcs.push_back({
                    {"type", ggml_type_name(s->type)},
                    {"ne", {s->ne[0], s->ne[1], s->ne[2], s->ne[3]}},
                    {"nb", {s->nb[0], s->nb[1], s->nb[2], s->nb[3]}},
                    {"dense", ggml_is_contiguous(s)},
                });
            }
            ggml_backend_hrx_trace_event(device_context->reg_context, {
                {"event", "supports_op_rejected"},
                {"reason", "request_builder"},
                {"op", ggml_op_desc(op)},
                {"node", ggml_get_name(op)},
                {"dst_type", ggml_type_name(op->type)},
                {"dst_ne", {op->ne[0], op->ne[1], op->ne[2], op->ne[3]}},
                {"dst_nb", {op->nb[0], op->nb[1], op->nb[2], op->nb[3]}},
                {"dst_dense", ggml_is_contiguous(op)},
                {"srcs", srcs},
            });
        }
        return false;
    }
    ggml_backend_hrx_add_tensor_overlap_facts(&request.problem, request.tensors);
    const ggml_backend_hrx_catalog_route * route =
        device_context->reg_context && device_context->reg_context->catalog
            ? ggml_backend_hrx_catalog_find_route(*device_context->reg_context->catalog, request.problem)
            : nullptr;
    bool supported = route != nullptr;
    // Having a route is not enough: routes specialise on the shape, and a
    // specialisation can fail to compile even though its siblings are fine. The
    // scheduler has already committed the node by the time dispatch discovers
    // that, and the destination is left holding whatever was in the buffer, so
    // compile it here. The result is cached under the same key dispatch uses,
    // so a node that is claimed costs nothing extra later.
    // Memoised on the route and shape: supports_op runs for every node of every
    // graph, and building the compiled-route cache key allocates a string per
    // call, which is enough to show up in prompt throughput.
    uint64_t probe_key = 1469598103934665603ull ^ (uint64_t) (uintptr_t) route;
    for (const auto & entry : request.problem.shape) {
        probe_key = (probe_key ^ (uint64_t) entry.second) * 1099511628211ull;
    }
    const bool probe_seen = device_context->supports_probe_ok.count(probe_key) != 0;
    if (device_context->supports_probe_bad.count(probe_key) != 0) {
        return false;
    }
    if (!probe_seen && supported && route->constant_byte_length == 0) {
        std::vector<ggml_backend_hrx_catalog_binding> bindings;
        std::vector<int64_t> workload;
        std::string ignored;
        if (ggml_backend_hrx_catalog_make_config_bindings(*route, request.problem, &bindings, &ignored) &&
            ggml_backend_hrx_resolve_workload_arguments(*route, request.problem, &workload, &ignored) &&
            ggml_backend_hrx_get_compiled_route(device_context, *route, request.problem, bindings, workload) == nullptr) {
            ggml_backend_hrx_trace_event(device_context->reg_context, {
                {"event", "supports_op_rejected"},
                {"reason", "specialisation_compile_failed"},
                {"op", request.problem.op},
                {"route_id", route->id},
                {"shape", request.problem.shape},
            });
            device_context->supports_probe_bad.insert(probe_key);
            supported = false;
        } else {
            device_context->supports_probe_ok.insert(probe_key);
        }
    }
    if (!supported && route == nullptr) {
        ggml_backend_hrx_trace_event(device_context->reg_context, {
            {"event", "supports_op_rejected"},
            {"reason", "no_catalog_route"},
            {"op", request.problem.op},
            {"supports", request.problem.supports},
            {"shape", request.problem.shape},
        });
    }
    device_context->supports_answers[support_signature] = supported;
    return supported;
}

static bool ggml_backend_hrx_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_hrx_buffer_type_get_name) {
        return false;
    }
    return buft->device == dev;
}

static const ggml_backend_device_i ggml_backend_hrx_device_i = {
    /* .get_name             = */ ggml_backend_hrx_device_get_name,
    /* .get_description      = */ ggml_backend_hrx_device_get_description,
    /* .get_memory           = */ ggml_backend_hrx_device_get_memory,
    /* .get_type             = */ ggml_backend_hrx_device_get_type,
    /* .get_props            = */ ggml_backend_hrx_device_get_props,
    /* .init_backend         = */ ggml_backend_hrx_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_hrx_device_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_hrx_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hrx_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static ggml_backend_hrx_reg_context * ggml_backend_hrx_get_reg_context(ggml_backend_reg_t reg) {
    return static_cast<ggml_backend_hrx_reg_context *>(reg->context);
}

static const char * ggml_backend_hrx_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_HRX_NAME;
}

static size_t ggml_backend_hrx_reg_get_device_count(ggml_backend_reg_t reg) {
    return ggml_backend_hrx_get_reg_context(reg)->devices.size();
}

static ggml_backend_dev_t ggml_backend_hrx_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * context = ggml_backend_hrx_get_reg_context(reg);
    GGML_ASSERT(index < context->devices.size());
    return &context->devices[index];
}

static void * ggml_backend_hrx_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_hrx_reg_i = {
    /* .get_name         = */ ggml_backend_hrx_reg_get_name,
    /* .get_device_count = */ ggml_backend_hrx_reg_get_device_count,
    /* .get_device       = */ ggml_backend_hrx_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hrx_reg_get_proc_address,
};

ggml_backend_hrx_reg_context::~ggml_backend_hrx_reg_context() {
    for (auto & device_context : device_contexts) {
        if (device_context) {
            ggml_backend_hrx_sync_streams(device_context.get());
            std::lock_guard<std::mutex> lock(device_context->compiled_routes_mutex);
            device_context->compiled_routes.clear();
        }
        if (device_context && device_context->transfer_stream) {
            hrx_status_t status = hrx_stream_synchronize(device_context->transfer_stream);
            if (!hrx_status_is_ok(status)) {
                hrx_status_ignore(status);
            }
            ggml_backend_hrx_unregister_stream(device_context.get(), device_context->transfer_stream);
            hrx_stream_release(device_context->transfer_stream);
            device_context->transfer_stream = nullptr;
        }
        if (device_context && device_context->jit) {
            ggml_hrx_loom_jit_amdgpu_release(device_context->jit);
            device_context->jit = nullptr;
        }
        if (device_context && device_context->device) {
            hrx_device_release(device_context->device);
            device_context->device = nullptr;
        }
    }
    if (gpu_initialized) {
        hrx_status_t status = hrx_gpu_shutdown();
        if (!hrx_status_is_ok(status)) {
            hrx_status_ignore(status);
        }
    }
}

static std::unique_ptr<ggml_backend_hrx_reg_context> ggml_backend_hrx_create_reg_context() {
    auto context = std::make_unique<ggml_backend_hrx_reg_context>();
    context->options = ggml_backend_hrx_parse_options();

    if (!context->options.trace_jsonl_path.empty()) {
        context->trace_jsonl.open(context->options.trace_jsonl_path, std::ios::out | std::ios::app);
        if (!context->trace_jsonl) {
            GGML_LOG_ERROR(
                "%s: failed to open GGML_HRX_TRACE_JSONL path %s\n",
                __func__, context->options.trace_jsonl_path.c_str());
        }
    }

    ggml_backend_hrx_trace_event(context.get(), {
        {"event", "backend_init"},
        {"catalog_dir", context->options.catalog_dir},
        {"evidence_dir", context->options.evidence_dir},
        {"trace_routes", context->options.trace_routes},
        {"trace_graph", context->options.trace_graph},
        {"staging_arena_size", context->options.staging_arena_size},
    });

    std::string catalog_error;
    context->catalog = ggml_backend_hrx_load_catalog(
        ggml_backend_hrx_optional_c_str(context->options.catalog_dir),
        &catalog_error);
    if (!context->catalog) {
        GGML_LOG_ERROR("%s: %s\n", __func__, catalog_error.c_str());
        ggml_backend_hrx_trace_event(context.get(), {
            {"event", "catalog_error"},
            {"error", catalog_error},
        });
        return context;
    }
    ggml_backend_hrx_trace_event(context.get(), {
        {"event", "catalog_loaded"},
        {"catalog_id", context->catalog->catalog_id},
        {"source", context->catalog->source},
        {"sources", context->catalog->source_count},
        {"artifacts", context->catalog->artifact_count},
        {"families", context->catalog->family_count},
        {"routes", context->catalog->route_count},
        {"fusions", context->catalog->fusion_count},
    });

    hrx_status_t status = hrx_gpu_initialize(0);
    if (hrx_status_is_ok(status)) {
        context->gpu_initialized = true;
    } else if (hrx_status_code(status) == HRX_STATUS_ALREADY_EXISTS) {
        hrx_status_ignore(status);
    } else {
        hrx_status_ignore(status);
        return context;
    }

    int device_count = 0;
    if (!GGML_HRX_CHECK(hrx_gpu_device_count(&device_count)) || device_count <= 0) {
        return context;
    }

    context->device_contexts.reserve(device_count);
    context->devices.reserve(device_count);

    for (int i = 0; i < device_count; ++i) {
        hrx_device_t device = nullptr;
        if (!GGML_HRX_CHECK(hrx_gpu_device_get(i, &device)) || !device) {
            continue;
        }
        hrx_device_retain(device);

        auto device_context = std::make_unique<ggml_backend_hrx_device_context>();
        device_context->reg_context = context.get();
        device_context->options = &context->options;
        device_context->device = device;
        device_context->name = std::string(GGML_HRX_NAME) + std::to_string(i);
        device_context->description = ggml_backend_hrx_device_description(device);
        device_context->architecture = ggml_backend_hrx_device_architecture(device);
        device_context->memory_total = ggml_backend_hrx_total_memory(device);
        ggml_hrx_loom_jit_amdgpu_options_t jit_options = {
            /* .structure_size      = */ sizeof(ggml_hrx_loom_jit_amdgpu_options_t),
            /* .processor           = */ device_context->architecture.c_str(),
            /* .identifier          = */ device_context->name.c_str(),
            /* .sanitizer           = */ ggml_backend_hrx_optional_c_str(context->options.loom_sanitizer),
            /* .sanitizer_reporting = */ ggml_backend_hrx_optional_c_str(context->options.loom_sanitizer_reporting),
        };
        if (!GGML_HRX_CHECK(ggml_hrx_loom_jit_amdgpu_create(&jit_options, &device_context->jit))) {
            device_context->jit = nullptr;
        }
        if (!GGML_HRX_CHECK(hrx_stream_create(device_context->device, 0, &device_context->transfer_stream))) {
            if (device_context->jit) {
                ggml_hrx_loom_jit_amdgpu_release(device_context->jit);
                device_context->jit = nullptr;
            }
            hrx_device_release(device);
            continue;
        }
        ggml_backend_hrx_register_stream(device_context.get(), device_context->transfer_stream);

        ggml_backend_hrx_trace_event(context.get(), {
            {"event", "device_initialized"},
            {"device", device_context->name},
            {"description", device_context->description},
            {"architecture", device_context->architecture},
            {"memory_total", device_context->memory_total},
            {"jit_available", device_context->jit != nullptr},
        });

        context->device_contexts.emplace_back(std::move(device_context));
        context->devices.push_back({
            /* .iface   = */ ggml_backend_hrx_device_i,
            /* .reg     = */ nullptr,
            /* .context = */ context->device_contexts.back().get(),
        });
    }

    return context;
}

} // namespace

ggml_backend_t ggml_backend_hrx_init(size_t dev_num) {
    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (!reg || dev_num >= ggml_backend_reg_dev_count(reg)) {
        GGML_LOG_ERROR("%s: invalid HRX device index %zu\n", __func__, dev_num);
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, dev_num), nullptr);
}

bool ggml_backend_is_hrx(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_hrx_guid());
}

int ggml_backend_hrx_get_device_count(void) {
    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    return reg ? static_cast<int>(ggml_backend_reg_dev_count(reg)) : 0;
}

void ggml_backend_hrx_get_device_description(int device, char * description, size_t description_size) {
    if (!description || description_size == 0) {
        return;
    }

    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (!reg || device < 0 || static_cast<size_t>(device) >= ggml_backend_reg_dev_count(reg)) {
        description[0] = '\0';
        return;
    }

    const char * value = ggml_backend_dev_description(
        ggml_backend_reg_dev_get(reg, static_cast<size_t>(device)));
    std::snprintf(description, description_size, "%s", value ? value : "");
}

void ggml_backend_hrx_get_device_memory(int device, size_t * free, size_t * total) {
    if (free) {
        *free = 0;
    }
    if (total) {
        *total = 0;
    }

    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (!reg || device < 0 || static_cast<size_t>(device) >= ggml_backend_reg_dev_count(reg)) {
        return;
    }

    ggml_backend_dev_memory(
        ggml_backend_reg_dev_get(reg, static_cast<size_t>(device)), free, total);
}

ggml_backend_buffer_type_t ggml_backend_hrx_buffer_type(size_t dev_num) {
    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (!reg || dev_num >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, dev_num));
}

ggml_backend_reg_t ggml_backend_hrx_reg(void) {
    static std::unique_ptr<ggml_backend_hrx_reg_context> context =
        ggml_backend_hrx_create_reg_context();

    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_hrx_reg_i,
        /* .context     = */ context.get(),
    };

    if (context) {
        for (auto & device : context->devices) {
            device.reg = &reg;
        }
    }

    return &reg;
}

std::vector<ggml_backend_hrx_test_case> ggml_backend_hrx_test_cases(
        ggml_backend_dev_t dev,
        const std::string & family) {
    std::vector<ggml_backend_hrx_test_case> out;
    if (!dev) {
        return out;
    }
    auto * device_context = ggml_backend_hrx_get_device_context(dev);
    if (!device_context || !device_context->reg_context || !device_context->reg_context->catalog) {
        return out;
    }
    const auto & catalog = *device_context->reg_context->catalog;
    std::vector<size_t> case_indices;
    if (family.empty()) {
        for (size_t i = 0; i < catalog.test_cases.size(); ++i) {
            if (catalog.test_cases[i].target_key == device_context->architecture) {
                case_indices.push_back(i);
            }
        }
    } else {
        const auto it = catalog.test_cases_by_target_family.find(
            ggml_backend_hrx_test_case_index_key(device_context->architecture, family));
        if (it == catalog.test_cases_by_target_family.end()) {
            return out;
        }
        case_indices = it->second;
    }
    out.reserve(case_indices.size());
    for (const size_t test_case_index : case_indices) {
        if (test_case_index >= catalog.test_cases.size()) {
            continue;
        }
        const auto & catalog_case = catalog.test_cases[test_case_index];
        ggml_backend_hrx_test_case test_case;
        test_case.id = catalog_case.id;
        test_case.op = catalog_case.op;
        test_case.family = catalog_case.family;
        test_case.scenario = catalog_case.scenario;
        test_case.expected_route_id = catalog_case.expected_route_id;
        test_case.supports = catalog_case.supports;
        test_case.shape = catalog_case.shape;
        test_case.tolerance = catalog_case.tolerance;
        test_case.repeat = catalog_case.repeat;
        out.push_back(std::move(test_case));
    }
    return out;
}

void ggml_backend_hrx_test_reset_dispatch_record(void) {
    std::lock_guard<std::mutex> lock(g_ggml_backend_hrx_test_dispatch_recorder.mutex);
    g_ggml_backend_hrx_test_dispatch_recorder.enabled = true;
    g_ggml_backend_hrx_test_dispatch_recorder.routes.clear();
}

ggml_backend_hrx_test_route_record ggml_backend_hrx_test_get_route_record(
        const std::string & route_id) {
    ggml_backend_hrx_test_route_record record;
    if (route_id.empty()) {
        return record;
    }
    std::lock_guard<std::mutex> lock(g_ggml_backend_hrx_test_dispatch_recorder.mutex);
    const auto it = g_ggml_backend_hrx_test_dispatch_recorder.routes.find(route_id);
    if (it == g_ggml_backend_hrx_test_dispatch_recorder.routes.end()) {
        return record;
    }
    return it->second;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hrx_reg)

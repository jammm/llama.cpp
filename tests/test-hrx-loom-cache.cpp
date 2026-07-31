#include "loom-catalog/ggml-hrx-loom-catalog-runtime-internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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

static void expect_true(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::abort();
    }
}

int main() {
    static const unsigned char source_a[] = "kernel.def @a() { }";
    static const unsigned char source_b[] = "kernel.def @b() { }";

    ggml_backend_hrx_loom_catalog_entry entry_a = make_entry(source_a, sizeof(source_a) - 1);
    ggml_backend_hrx_loom_catalog_entry entry_b = make_entry(source_b, sizeof(source_b) - 1);
    ggml_backend_hrx_loom_kernel_plan   plan_a0 = make_plan(&entry_a, "257", "256");
    ggml_backend_hrx_loom_kernel_plan   plan_a1 = make_plan(&entry_a, "257", "256");
    ggml_backend_hrx_loom_kernel_plan   plan_b0 = make_plan(&entry_a, "2048", "256");
    ggml_backend_hrx_loom_kernel_plan   plan_c0 = make_plan(&entry_b, "257", "256");

    ggml_backend_hrx_loom_route_fingerprint fingerprint;
    expect_true(ggml_backend_hrx_loom_route_fingerprint_init(&fingerprint, "gfx1100", &plan_a0),
                "valid plan must initialize a route fingerprint");
    expect_true(ggml_backend_hrx_loom_route_fingerprint_matches(fingerprint, "gfx1100", &plan_a1),
                "identical plans must match");
    expect_true(!ggml_backend_hrx_loom_route_fingerprint_matches(fingerprint, "gfx1100", &plan_b0),
                "config changes must not match");
    expect_true(!ggml_backend_hrx_loom_route_fingerprint_matches(fingerprint, "gfx1100", &plan_c0),
                "source identity changes must not match");
    expect_true(!ggml_backend_hrx_loom_route_fingerprint_matches(fingerprint, "gfx1151", &plan_a1),
                "target changes must not match");

    plan_a1.config_bindings[0].type = "index";
    expect_true(!ggml_backend_hrx_loom_route_fingerprint_matches(fingerprint, "gfx1100", &plan_a1),
                "config type changes must not match");
    plan_a1.config_bindings[0].type = "i64";

    entry_a.symbol = "hrx_add_f32_v2";
    expect_true(!ggml_backend_hrx_loom_route_fingerprint_matches(fingerprint, "gfx1100", &plan_a1),
                "entry changes must not match");
    entry_a.symbol = "hrx_add_f32";

    expect_true(!ggml_backend_hrx_loom_route_fingerprint_init(nullptr, "gfx1100", &plan_a0),
                "null fingerprint must be rejected");
    expect_true(!ggml_backend_hrx_loom_route_fingerprint_init(&fingerprint, "gfx1100", nullptr),
                "null plan must be rejected");

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
    return 0;
}

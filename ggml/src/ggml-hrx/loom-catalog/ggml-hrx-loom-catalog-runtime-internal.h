#pragma once

#include "ggml-hrx-loom-catalog-runtime.h"
#include "ggml-hrx-loom-catalog.h"
#include "ggml-hrx-runtime-util.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

static constexpr size_t GGML_BACKEND_HRX_LOOM_MAX_BINDINGS        = 8;
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

struct ggml_backend_hrx_loom_kernel_plan {
    const ggml_backend_hrx_loom_catalog_entry * entry                                                      = nullptr;
    hrx_dispatch_config_t                       dispatch                                                   = {};
    hrx_buffer_ref_t                            bindings[GGML_BACKEND_HRX_LOOM_MAX_BINDINGS]               = {};
    uint8_t                                     binding_scratch_index[GGML_BACKEND_HRX_LOOM_MAX_BINDINGS]  = {};
    size_t                                      binding_scratch_offset[GGML_BACKEND_HRX_LOOM_MAX_BINDINGS] = {};
    size_t                                      binding_scratch_length[GGML_BACKEND_HRX_LOOM_MAX_BINDINGS] = {};
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
};

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

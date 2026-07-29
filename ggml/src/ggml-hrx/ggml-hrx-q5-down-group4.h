#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

// Physical storage contract for the capacity-neutral Q5_K down-expert layout.
//
// GGML keeps the logical tensor shape and byte count unchanged. Only tensors
// matching the exact Qwen3.6 target name/type/shape contract are permuted from
//
//   [expert][row][block2][field11][byte16]
//
// to
//
//   [expert][group4][block2][component11][row4][byte16].
//
// The component order groups each qh half with the four qs packets that consume
// it. Every semantic 16-byte unit has exactly one source and destination, so
// the representation has no padding or persistent duplicate.
namespace ggml_hrx_q5_down_group4 {

static constexpr size_t kRows = 2048;
static constexpr size_t kBlocks = 2;
static constexpr size_t kFields = 11;
static constexpr size_t kUnitBytes = 16;
static constexpr size_t kGroupRows = 4;
static constexpr size_t kGroups = kRows / kGroupRows;
static constexpr size_t kExperts = 256;
static constexpr size_t kBlockBytes = kFields * kUnitBytes;
static constexpr size_t kExpertBytes =
    kRows * kBlocks * kBlockBytes;
static constexpr size_t kTensorBytes =
    kExperts * kExpertBytes;

static_assert(kBlockBytes == 176, "Q5_K block size changed");
static_assert(kExpertBytes == 720896, "Q5_K expert size changed");
static_assert(kTensorBytes == 184549376, "Q5_K tensor size changed");

// Packed component -> canonical field:
// metadata, qh0, qs0/2/4/6, qh1, qs1/3/5/7.
static constexpr std::array<size_t, kFields> kFieldOrder = {
    0, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10,
};

enum class range_state {
    none,
    exact,
    overlap,
};

// The model loader asks supports_op about a tensor attached to a direct,
// zero-byte backend buffer before allocating or uploading it. That exact
// capability probe may select the packed route without registered storage.
// Every live request still needs the exact registered range; aliases and
// non-classified tensors may never consume packed bytes.
inline bool request_storage_allowed(
        bool exact_tensor,
        range_state state,
        bool exact_zero_size_capability_probe) {
    if (!exact_tensor) {
        return state == range_state::none;
    }
    return state == range_state::exact ||
           (state == range_state::none &&
            exact_zero_size_capability_probe);
}

inline bool has_exact_weight_name(const char * name) {
    if (!name) {
        return false;
    }
    static constexpr char kPrefix[] = "blk.";
    static constexpr char kSuffix[] = ".ffn_down_exps.weight";
    if (std::strncmp(name, kPrefix, sizeof(kPrefix) - 1) != 0) {
        return false;
    }

    const char * cursor = name + sizeof(kPrefix) - 1;
    if (*cursor < '0' || *cursor > '9') {
        return false;
    }
    // The target has exactly forty layers. Reject alternate spellings such as
    // blk.00 so the classifier remains a single canonical grammar.
    if (*cursor == '0' && cursor[1] >= '0' && cursor[1] <= '9') {
        return false;
    }
    unsigned layer = 0;
    do {
        layer = layer * 10u + static_cast<unsigned>(*cursor - '0');
        if (layer >= 40u) {
            return false;
        }
        ++cursor;
    } while (*cursor >= '0' && *cursor <= '9');
    return std::strcmp(cursor, kSuffix) == 0;
}

inline bool is_exact_tensor(
        bool is_q5_k,
        bool is_contiguous,
        const int64_t ne[4],
        size_t nbytes,
        const char * name) {
    return is_q5_k &&
           is_contiguous &&
           ne &&
           ne[0] == 512 &&
           ne[1] == static_cast<int64_t>(kRows) &&
           ne[2] == static_cast<int64_t>(kExperts) &&
           ne[3] == 1 &&
           nbytes == kTensorBytes &&
           has_exact_weight_name(name);
}

inline size_t canonical_unit(
        size_t expert,
        size_t row,
        size_t block,
        size_t field) {
    return ((expert * kRows + row) * kBlocks + block) * kFields + field;
}

inline size_t packed_unit(
        size_t expert,
        size_t row,
        size_t block,
        size_t field) {
    size_t component = 0;
    while (component < kFields && kFieldOrder[component] != field) {
        ++component;
    }
    if (component == kFields) {
        return kTensorBytes / kUnitBytes;
    }
    const size_t group = row / kGroupRows;
    const size_t row4 = row % kGroupRows;
    return ((((expert * kGroups + group) * kBlocks + block) * kFields +
             component) *
            kGroupRows) +
           row4;
}

inline bool pack(
        const void * canonical_data,
        size_t canonical_size,
        void * packed_data,
        size_t packed_size) {
    if (!canonical_data || !packed_data ||
        canonical_data == packed_data ||
        canonical_size != kTensorBytes ||
        packed_size != kTensorBytes) {
        return false;
    }
    const auto * canonical =
        static_cast<const uint8_t *>(canonical_data);
    auto * packed = static_cast<uint8_t *>(packed_data);
    size_t packed_offset = 0;
    for (size_t expert = 0; expert < kExperts; ++expert) {
        for (size_t group = 0; group < kGroups; ++group) {
            for (size_t block = 0; block < kBlocks; ++block) {
                for (size_t component = 0; component < kFields;
                     ++component) {
                    const size_t field = kFieldOrder[component];
                    for (size_t row4 = 0; row4 < kGroupRows; ++row4) {
                        const size_t row = group * kGroupRows + row4;
                        const size_t source_offset =
                            canonical_unit(expert, row, block, field) *
                            kUnitBytes;
                        std::memcpy(
                            packed + packed_offset,
                            canonical + source_offset,
                            kUnitBytes);
                        packed_offset += kUnitBytes;
                    }
                }
            }
        }
    }
    return packed_offset == kTensorBytes;
}

inline bool unpack(
        const void * packed_data,
        size_t packed_size,
        void * canonical_data,
        size_t canonical_size) {
    if (!packed_data || !canonical_data ||
        packed_data == canonical_data ||
        packed_size != kTensorBytes ||
        canonical_size != kTensorBytes) {
        return false;
    }
    const auto * packed = static_cast<const uint8_t *>(packed_data);
    auto * canonical = static_cast<uint8_t *>(canonical_data);
    size_t packed_offset = 0;
    for (size_t expert = 0; expert < kExperts; ++expert) {
        for (size_t group = 0; group < kGroups; ++group) {
            for (size_t block = 0; block < kBlocks; ++block) {
                for (size_t component = 0; component < kFields;
                     ++component) {
                    const size_t field = kFieldOrder[component];
                    for (size_t row4 = 0; row4 < kGroupRows; ++row4) {
                        const size_t row = group * kGroupRows + row4;
                        const size_t destination_offset =
                            canonical_unit(expert, row, block, field) *
                            kUnitBytes;
                        std::memcpy(
                            canonical + destination_offset,
                            packed + packed_offset,
                            kUnitBytes);
                        packed_offset += kUnitBytes;
                    }
                }
            }
        }
    }
    return packed_offset == kTensorBytes;
}

}  // namespace ggml_hrx_q5_down_group4

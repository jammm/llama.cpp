#!/usr/bin/env python3

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from shutil import copytree

import generate_loom_catalog as catalog
import generate_loom_route_impl as route_impl
import validate_loom_routes as loom
from utils import hrx_route_emit as route_emit
from utils import hrx_route_schema as route_schema


TOOLS_DIR = Path(__file__).resolve().parent
CATALOG_ROOT = TOOLS_DIR.parent / "loom-catalog"
METADATA_PATH = Path("metadata.json")
ROUTE_PATH = Path("routes/generic/add/f32/contiguous.json")
SUM_ROWS_ROUTE_PATH = Path("routes/generic/sum_rows/f32/contiguous_4d.json")
FUSION_ROUTE_PATH = Path("routes/generic/rms_norm_mul/f32/contiguous_4d.json")
Q8_PREPASS_ROUTE_PATH = Path(
    "routes/gfx1151/mul_mat/q8_0_f32/mul_mat_q8_0_f32_wmmai8.json"
)
Q8_PACKED_LOOP_ROUTE_PATH = Path(
    "routes/gfx1151/mul_mat/q8_0_f32/"
    "mul_mat_q8_0_f32_packed_loop_k32_8192_c1_16_wg64.json"
)
Q8_DECODE_ROUTE_CASES = (
    (
        Path(
            "routes/gfx1151/mul_mat/q8_0_f32/"
            "mul_mat_q8_0_f32_packed_decode_k2048_wg256_scfunroll2.json"
        ),
        2048,
        2,
    ),
    (
        Path(
            "routes/gfx1151/mul_mat/q8_0_f32/"
            "mul_mat_q8_0_f32_packed_decode_k4096_wg256_scfunroll4.json"
        ),
        4096,
        4,
    ),
)
Q5_DOWN_GROUP4_ROUTE_PATH = Path(
    "routes/gfx1151/mul_mat_id/q5_k_f32/"
    "mul_mat_id_q5_k_f32_mmq_gfx1151_wg256_tbl_down_group4.json"
)
FLASH_ATTN_WMMA_ROUTE_PATH = Path(
    "routes/gfx1151/flash_attn_ext/f32_f16/wmma_f32.json"
)
FLASH_ATTN_WAVE_ROUTE_PATH = Path(
    "routes/gfx1151/flash_attn_ext/f32_f16/wave256.json"
)
TEST_TARGET_A = "__test_target_a"
TEST_TARGET_B = "__test_target_b"
TEST_TARGET_MISSING = "__test_missing_target"
TEST_UNARY_OP_SIGMOID = 7
TEST_UNARY_OP_SOFTPLUS = 15

EXTENDED_OP_CASES = {
    "GGML_OP_CONCAT": {
        "tensors": {"src0": "F32", "src1": "F32", "dst": "F32"},
        "attributes": {"dim": "i32"},
    },
    "GGML_OP_CONT": {
        "tensors": {"src0": "F32", "dst": "F32"},
        "attributes": {},
    },
    "GGML_OP_FLASH_ATTN_EXT": {
        "tensors": {
            "src0": "F32",
            "src1": "F16",
            "src2": "F16",
            "src3": "F16",
            "src4": "F32",
            "dst": "F32",
        },
        "optional": {"src4"},
        "attributes": {
            "scale": "f32",
            "max_bias": "f32",
            "logit_softcap": "f32",
            "precision": "i32",
        },
    },
    "GGML_OP_GATED_DELTA_NET": {
        "tensors": {
            "src0": "F32",
            "src1": "F32",
            "src2": "F32",
            "src3": "F32",
            "src4": "F32",
            "src5": "F32",
            "dst": "F32",
        },
        "attributes": {"K": "i32"},
    },
    "GGML_OP_MUL_MAT": {
        "tensors": {"src0": "Q5_K", "src1": "F32", "dst": "F32"},
        "attributes": {},
    },
    "GGML_OP_MUL_MAT_ID": {
        "tensors": {"src0": "Q4_K", "src1": "F32", "src2": "I32", "dst": "F32"},
        "attributes": {},
    },
    "GGML_OP_SCALE": {
        "tensors": {"src0": "F32", "dst": "F32"},
        "attributes": {"scale": "f32", "bias": "f32"},
    },
    "GGML_OP_SSM_CONV": {
        "tensors": {"src0": "F32", "src1": "F32", "dst": "F32"},
        "attributes": {},
    },
    "GGML_OP_UNARY": {
        "tensors": {"src0": "F32", "dst": "F32"},
        "attributes": {"unary_op": "i32"},
    },
}


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_route_and_definition(source_root, route_path):
    route_full_path = source_root / route_path
    route = read_json(route_full_path)
    definition = read_json((route_full_path.parent / route["definition"]).resolve())
    return route, definition


def write_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def copy_catalog(tmpdir):
    copied = Path(tmpdir) / "loom-catalog"
    copytree(CATALOG_ROOT, copied)
    mutate_metadata(copied, lambda metadata: metadata.update({"routes": [str(ROUTE_PATH)]}))
    return copied


def mutate_route(source_root, mutator):
    mutate_route_at(source_root, ROUTE_PATH, mutator)


def mutate_route_at(source_root, route_path, mutator):
    path = source_root / route_path
    route = read_json(path)
    mutator(route)
    write_json(path, route)


def mutate_metadata(source_root, mutator):
    path = source_root / METADATA_PATH
    metadata = read_json(path)
    mutator(metadata)
    write_json(path, metadata)


def set_metadata_targets(source_root, targets):
    mutate_metadata(source_root, lambda metadata: metadata.update({"targets": targets}))


def set_metadata_routes(source_root, routes):
    mutate_metadata(source_root, lambda metadata: metadata.update({"routes": routes}))


def set_route_architectures(source_root, architectures):
    mutate_route(source_root, lambda route: route.update({"architectures": architectures}))


def set_targets_and_architectures(source_root, targets, architectures):
    set_metadata_targets(source_root, targets)
    set_metadata_routes(source_root, [str(ROUTE_PATH)])
    set_route_architectures(source_root, architectures)


def expect_valid(source_root):
    loom.validate_catalog(source_root)


def expect_valid_mutation(name, mutator):
    with tempfile.TemporaryDirectory(prefix=f"{name}-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        mutate_route(source_root, mutator)
        loom.validate_catalog(source_root)


def expect_valid_catalog_mutation(name, mutator):
    with tempfile.TemporaryDirectory(prefix=f"{name}-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        mutator(source_root)
        loom.validate_catalog(source_root)


def expect_invalid(name, mutator, expected):
    with tempfile.TemporaryDirectory(prefix=f"{name}-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        mutate_route(source_root, mutator)
        try:
            loom.validate_catalog(source_root)
        except ValueError as err:
            message = str(err)
            if expected in message:
                return
            raise AssertionError(f"{name}: expected error containing {expected!r}, got {message!r}") from err
        raise AssertionError(f"{name}: validator accepted invalid catalog")


def expect_invalid_catalog_mutation(name, mutator, expected):
    with tempfile.TemporaryDirectory(prefix=f"{name}-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        mutator(source_root)
        try:
            loom.validate_catalog(source_root)
        except ValueError as err:
            message = str(err)
            if expected in message:
                return
            raise AssertionError(f"{name}: expected error containing {expected!r}, got {message!r}") from err
        raise AssertionError(f"{name}: validator accepted invalid catalog")


def expect_invalid_route(name, route_path, mutator, expected):
    with tempfile.TemporaryDirectory(prefix=f"{name}-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        set_metadata_routes(source_root, [str(route_path)])
        mutate_route_at(source_root, route_path, mutator)
        try:
            loom.validate_catalog(source_root)
        except ValueError as err:
            message = str(err)
            if expected in message:
                return
            raise AssertionError(
                f"{name}: expected error containing {expected!r}, got {message!r}"
            ) from err
        raise AssertionError(f"{name}: validator accepted invalid catalog")


def expect_generation_valid(name, mutator, targets, expected_entries):
    with tempfile.TemporaryDirectory(prefix=f"{name}-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        mutator(source_root)
        loom.validate_catalog(source_root)
        entries = catalog.build_entries(source_root, targets)
        if len(entries) != len(expected_entries):
            raise AssertionError(f"{name}: expected {len(expected_entries)} entries, got {len(entries)}")
        actual_entries = [{"id": entry["id"], "target": entry["target"]} for entry in entries]
        if actual_entries != expected_entries:
            raise AssertionError(f"{name}: expected generated entries {expected_entries}, got {actual_entries}")


def expect_generation_invalid(name, mutator, targets, expected):
    with tempfile.TemporaryDirectory(prefix=f"{name}-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        mutator(source_root)
        loom.validate_catalog(source_root)
        try:
            catalog.build_entries(source_root, targets)
        except ValueError as err:
            message = str(err)
            if expected in message:
                return
            raise AssertionError(f"{name}: expected error containing {expected!r}, got {message!r}") from err
        raise AssertionError(f"{name}: generator accepted invalid catalog")


def expect_extended_op_schema():
    for dtype in ("Q5_K", "Q6_K", "Q8_0"):
        route_schema.validate_literal(dtype, "dtype", f"test dtype {dtype}")
        if route_emit.CPP_DTYPE_NAMES.get(dtype) != f"GGML_TYPE_{dtype}":
            raise AssertionError(f"missing C++ dtype mapping for {dtype}")

    for op, case in EXTENDED_OP_CASES.items():
        tensors = {
            role: {
                "type": dtype,
                **({"optional": True} if role in case.get("optional", set()) else {}),
            }
            for role, dtype in case["tensors"].items()
        }
        attributes = {
            name: {"type": scalar_type}
            for name, scalar_type in case["attributes"].items()
        }
        match = {"tensors": tensors, "attributes": attributes}
        route = {
            "schema": "ggml-hrx-loom-route-v1",
            "match": {"op": op, **match},
        }
        rule = route_schema.OP_RULES[op]
        route_schema.validate_tensors(route, f"test {op}", rule)
        route_schema.validate_attributes(match, f"test {op}", rule)

        route_emit.validate_attribute_indices(
            route,
            f"test {op}",
            route_schema,
            route_emit.ATTRIBUTE_INDICES,
        )
        lines = []
        route_emit.emit_tensor_setup(
            lines,
            route,
            rule,
            route_schema,
            lambda reason: f"return {reason};",
            "unsupported",
        )
        route_emit.emit_attributes(
            lines,
            route,
            route_schema,
            route_emit.ATTRIBUTE_INDICES,
        )
        for role in tensors:
            expected = f"    const ggml_tensor * {role} = {route_emit.role_expr(role)};"
            if expected not in lines:
                raise AssertionError(f"{op}: missing generated tensor role {role}")
        for name, scalar_type in case["attributes"].items():
            index = route_emit.ATTRIBUTE_INDICES[op][name]
            getter = route_emit.ATTRIBUTE_GETTERS[scalar_type]
            cpp_type = route_emit.CPP_SCALAR_TYPES[scalar_type]
            expected = f"    const {cpp_type} attribute_{name} = {getter}(node, {index});"
            if expected not in lines:
                raise AssertionError(f"{op}: missing generated attribute {name}")

    required_optional_input_route = {
        "schema": "ggml-hrx-loom-route-v1",
        "match": {
            "op": "GGML_OP_FLASH_ATTN_EXT",
            "tensors": {
                "src0": {"type": "F32"},
                "src1": {"type": "F16"},
                "src2": {"type": "F16"},
                "src3": {"type": "F16"},
                "dst": {"type": "F32"},
            },
            "attributes": {},
        }
    }
    required_optional_input_lines = []
    route_emit.emit_tensor_setup(
        required_optional_input_lines,
        required_optional_input_route,
        route_schema.OP_RULES["GGML_OP_FLASH_ATTN_EXT"],
        route_schema,
        lambda reason: f"return {reason};",
        "unsupported",
    )
    if "    if (!dst || !src0 || !src1 || !src2 || !src3) {" not in required_optional_input_lines:
        raise AssertionError("declared FLASH_ATTN_EXT mask was not emitted as a required input")

    if route_emit.role_expr("src5") != "node->src[5]":
        raise AssertionError("src5 tensor role was not emitted generically")

    unary_case = EXTENDED_OP_CASES["GGML_OP_UNARY"]
    unary_tensors = {
        role: {"type": dtype}
        for role, dtype in unary_case["tensors"].items()
    }
    unary_attributes = {
        name: {"type": scalar_type}
        for name, scalar_type in unary_case["attributes"].items()
    }
    unary_context = route_schema.RouteContext(
        "test GGML_OP_UNARY",
        route_schema.OP_RULES["GGML_OP_UNARY"],
        unary_tensors,
        unary_attributes,
        {},
    )
    for unary_op in (TEST_UNARY_OP_SIGMOID, TEST_UNARY_OP_SOFTPLUS):
        route_schema.validate_predicates(
            [{"field": "attribute.unary_op", "equals": unary_op}],
            "test GGML_OP_UNARY",
            unary_context,
        )


def expect_prepass_schema_and_generation():
    definitions = loom.load_definitions(CATALOG_ROOT)
    route_path = CATALOG_ROOT / Q8_PREPASS_ROUTE_PATH
    route = read_json(route_path)
    definition_path = (
        route_path.parent /
        route_schema.require_string(route, "definition", route_path)
    ).resolve()
    generated = route_impl.generate_route_impl(
        route_path,
        route,
        definitions[definition_path],
    )
    cols_maxima = [
        predicate["max"]
        for predicate in route["match"]["predicates"]
        if predicate.get("field") == "shape.src1.cols" and "max" in predicate
    ]
    if cols_maxima != [16384]:
        raise AssertionError(
            f"Q8 prepass route must stay within quant_act_q8's cols range, got {cols_maxima}"
        )
    expected_fragments = [
        'ggml_backend_hrx_loom_find_entry(catalog, "quant_act_q8")',
        'plan->scratch[0].name = "qact";',
        "plan->scratch[0].minimum_capacity = 67108864;",
        "if (route_scratch_qact_qs_end > SIZE_MAX - 255) {",
        "plan->main.binding_scratch_index[3] = 1;",
        "plan->main.binding_scratch_offset[4] = route_scratch_qact_ds_offset;",
        "plan->prepasses[0].kernel.binding_scratch_index[3] = 1;",
        "plan->prepasses[0].cache_scratch_index = 1;",
        "plan->prepasses[0].cache_source = src1;",
        "plan->prepass_count = 1;",
    ]
    for fragment in expected_fragments:
        if fragment not in generated:
            raise AssertionError(f"prepass generator missing {fragment!r}")

    with tempfile.TemporaryDirectory(prefix="prepass-dedup-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        original_path = source_root / Q8_PREPASS_ROUTE_PATH
        duplicate_path = original_path.with_name("mul_mat_q8_0_f32_wmmai8_alias.json")
        duplicate = read_json(original_path)
        duplicate["id"] = "mul_mat_q8_0_f32_wmmai8_alias"
        write_json(duplicate_path, duplicate)
        mutate_metadata(
            source_root,
            lambda metadata: metadata["routes"].append(
                str(Q8_PREPASS_ROUTE_PATH.with_name(duplicate_path.name))
            ),
        )
        loom.validate_catalog(source_root)
        entries = catalog.build_entries(source_root, ["gfx1151"])
        quant_entries = [
            entry
            for entry in entries
            if entry["id"] == "quant_act_q8" and entry["target"] == "gfx1151"
        ]
        if len(quant_entries) != 1:
            raise AssertionError(
                f"expected one deduplicated quant_act_q8 entry, got {len(quant_entries)}"
            )


def expect_q8_decode_small_column_coverage():
    definitions = loom.load_definitions(CATALOG_ROOT)
    expected_predicates = [
        {
            "field": "shape.src1.cols",
            "min": 1,
        },
        {
            "field": "shape.src1.cols",
            "max": 16,
        },
    ]
    for relative_path, expected_k, expected_unroll in Q8_DECODE_ROUTE_CASES:
        route_path = CATALOG_ROOT / relative_path
        route = read_json(route_path)
        definition_path = (
            route_path.parent /
            route_schema.require_string(route, "definition", route_path)
        ).resolve()
        generated = route_impl.generate_route_impl(
            route_path,
            route,
            definitions[definition_path],
        )
        predicates = route["match"]["predicates"]
        cols_predicates = [
            predicate
            for predicate in predicates
            if predicate.get("field") == "shape.src1.cols"
        ]
        if cols_predicates != expected_predicates:
            raise AssertionError(
                "Q8 decode route must cover the frozen catalog's 1..16 "
                f"small-column domain, got {cols_predicates}"
            )
        k_equals = [
            predicate["equals"]
            for predicate in predicates
            if predicate.get("field") == "shape.src0.k" and
            isinstance(predicate.get("equals"), int)
        ]
        if k_equals != [expected_k]:
            raise AssertionError(
                f"Q8 decode route must specialize k={expected_k}, got {k_equals}"
            )
        unroll_values = [
            binding["value"]
            for binding in route["config"]["bindings"]
            if binding["name"] == "hrx2_tuning_q8_0_f32_unroll_factor"
        ]
        if unroll_values != [expected_unroll]:
            raise AssertionError(
                "Q8 decode route must preserve the frozen specialization's "
                f"unroll={expected_unroll}, got {unroll_values}"
            )
        for fragment in (
            f"if (!(shape_src0_k == {expected_k})) {{",
            "if (!(shape_src1_cols >= 1)) {",
            "if (!(shape_src1_cols <= 16)) {",
        ):
            if fragment not in generated:
                raise AssertionError(
                    f"Q8 decode generator missing guard {fragment!r}"
                )


def expect_q8_packed_loop_coverage():
    definitions = loom.load_definitions(CATALOG_ROOT)
    route_path = CATALOG_ROOT / Q8_PACKED_LOOP_ROUTE_PATH
    route = read_json(route_path)
    definition_path = (
        route_path.parent /
        route_schema.require_string(route, "definition", route_path)
    ).resolve()
    definition = definitions[definition_path]
    generated = route_impl.generate_route_impl(
        route_path,
        route,
        definition,
    )
    if definition["symbol"] != "hrx2_mul_mat_q8_0_f32_static_packed":
        raise AssertionError("Q8 packed-loop route must retain the frozen kernel")
    if definition["workgroup_size"] != [64, 1, 1]:
        raise AssertionError("Q8 packed-loop route must retain WG64")
    if route["priority"] >= 141:
        raise AssertionError(
            "generic Q8 packed-loop route must remain below specialized decode routes"
        )
    expected_predicates = (
        {"field": "shape.src0.k", "min": 32},
        {"field": "shape.src0.k", "max": 32768},
        {"field": "shape.src0.k", "multiple_of": 32},
        {"field": "shape.src1.cols", "min": 1},
        {"field": "shape.src1.cols", "max": 16},
    )
    predicates = route["match"]["predicates"]
    for predicate in expected_predicates:
        if predicate not in predicates:
            raise AssertionError(
                f"Q8 packed-loop route missing coverage predicate {predicate}"
            )
    for fragment in (
        "if (!(shape_src0_k >= 32)) {",
        "if (!(shape_src0_k <= 32768)) {",
        "if (!(shape_src0_k % 32 == 0)) {",
        "if (!(shape_src1_cols >= 1)) {",
        "if (!(shape_src1_cols <= 16)) {",
        "const int64_t main_config_hrx2_tuning_workgroup_size_value = 64;",
        "const int64_t main_dispatch_workgroups[3] = "
        "{shape_src0_rows, shape_src1_cols, 1};",
    ):
        if fragment not in generated:
            raise AssertionError(
                f"Q8 packed-loop generator missing {fragment!r}"
            )


def expect_q5_down_group4_selected_route():
    definitions = loom.load_definitions(CATALOG_ROOT)
    route_path = CATALOG_ROOT / Q5_DOWN_GROUP4_ROUTE_PATH
    route = read_json(route_path)
    definition_path = (
        route_path.parent /
        route_schema.require_string(route, "definition", route_path)
    ).resolve()
    definition = definitions[definition_path]
    generated = route_impl.generate_route_impl(
        route_path,
        route,
        definition,
    )
    if route["match"]["tensors"]["src0"].get("storage") != "q5_k_expert_down_group4":
        raise AssertionError("Q5 down group4 route must require transformed storage")
    expected_selected_predicate = {
        "field": "shape.src1.selected_rows",
        "equals": "shape.src2.nselected",
    }
    if expected_selected_predicate not in route["match"]["predicates"]:
        raise AssertionError(
            "Q5 down group4 route must cover selected-expert activations"
        )
    if [item["name"] for item in route["scratch"]] != ["mmid"]:
        raise AssertionError("ordinary Q5 down route must allocate only its MMID table")
    if [item["id"] for item in route["prepasses"]] != ["mmid_table"]:
        raise AssertionError("ordinary Q5 down route must build only its MMID table")
    if definition["symbol"] != "hrx2_mul_mat_id_q5_k_f32_mmqt_down_group4_static":
        raise AssertionError("Q5 down group4 route must retain the frozen kernel")
    for fragment in (
        'ggml_backend_hrx_loom_storage_layout_matches(request, src0, '
        '"q5_k_expert_down_group4")',
        "if (!(shape_src1_selected_rows == shape_src2_nselected)) {",
        'plan->scratch[0].name = "mmid";',
        "plan->main.binding_scratch_index[4] = 1;",
        "plan->prepass_count = 1;",
    ):
        if fragment not in generated:
            raise AssertionError(
                f"Q5 down group4 generator missing {fragment!r}"
            )


def expect_flash_attn_wave_warmup_coverage():
    definitions = loom.load_definitions(CATALOG_ROOT)

    def load_route(relative_path):
        route_path = CATALOG_ROOT / relative_path
        route = read_json(route_path)
        definition_path = (
            route_path.parent /
            route_schema.require_string(route, "definition", route_path)
        ).resolve()
        return route_path, route, definitions[definition_path]

    wave_path, wave, wave_definition = load_route(FLASH_ATTN_WAVE_ROUTE_PATH)
    wmma_path, wmma, wmma_definition = load_route(FLASH_ATTN_WMMA_ROUTE_PATH)
    generated = route_impl.generate_route_impl(
        wave_path,
        wave,
        wave_definition,
    )

    if wave_definition["symbol"] != "hrx2_flash_attn_ext_f32_f16_wave":
        raise AssertionError("flash-attention wave route must retain the frozen kernel")
    if wave_definition["abi"] != {
        "binding_count": 5,
        "parameter_count": 6,
        "constant_byte_length": 4,
    }:
        raise AssertionError("flash-attention wave route must retain the frozen ABI")
    if wave_definition["workgroup_size"] != [32, 1, 1]:
        raise AssertionError("flash-attention wave route must retain its single-wave WG32")
    source_path = CATALOG_ROOT / wave_definition["source"]
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha256 != "35ae0f5e4c0008e69d51945bb0acb0b83b0dee5f6844445783bdfeb179c34a3c":
        raise AssertionError(
            "flash-attention wave source differs from the frozen kernel "
            "(apart from catalog config-name normalization)"
        )

    if wave["priority"] >= wmma["priority"]:
        raise AssertionError(
            "flash-attention wave fallback must remain below the specialized WMMA route"
        )
    router = route_impl.generate_op_router_impl(
        [
            (wmma_path, wmma, wmma_definition),
            (wave_path, wave, wave_definition),
        ],
        "GGML_OP_FLASH_ATTN_EXT",
    )
    wmma_call = "ggml_backend_hrx_loom_route_flash_attn_ext_f32_f16_wmma("
    wave_call = "ggml_backend_hrx_loom_route_flash_attn_ext_f32_f16_wave256("
    if router.rfind(wmma_call) >= router.rfind(wave_call):
        raise AssertionError(
            "flash-attention op router must try specialized WMMA before wave256"
        )

    predicates = wave["match"]["predicates"]
    expected_predicates = (
        {"field": "shape.src0.d", "equals": 256},
        {"field": "shape.src0.ntokens", "min": 1},
        {"field": "shape.src0.ntokens", "max": 1048576},
        {"field": "shape.src1.nkv", "min": 1},
        {"field": "shape.src1.nkv", "max": 1048576},
    )
    for predicate in expected_predicates:
        if predicate not in predicates:
            raise AssertionError(
                f"flash-attention wave route missing domain predicate {predicate}"
            )
    for predicate in predicates:
        if (
            predicate.get("field") in {
                "shape.src0.ntokens",
                "shape.src1.nkv",
            } and
            "multiple_of" in predicate
        ):
            raise AssertionError(
                "flash-attention wave route must cover non-WMMA token/KV tails"
            )

    if wave["derived"]["query_tiles"] != {
        "type": "i64",
        "ceil_div": ["shape.src0.ntokens", 4],
    }:
        raise AssertionError("flash-attention wave route must retain its four-query tile")
    dispatch = wave["invocation"]["dispatch"]
    if dispatch != {
        "workgroups": ["derived.query_tiles", "shape.src0.nheads"],
        "workgroup_size": [32, 1, 1],
    }:
        raise AssertionError("flash-attention wave dispatch does not match the frozen route")
    warmup_workgroups = [
        (2 + 4 - 1) // 4,
        16,
        1,
    ]
    if warmup_workgroups != [1, 16, 1]:
        raise AssertionError("flash-attention warmup signature dispatch changed")

    expected_config_names = {
        "hrx2_shape_fa_ntokens",
        "hrx2_shape_fa_nheads",
        "hrx2_shape_fa_nkv",
        "hrx2_shape_fa_gqa",
        "hrx2_shape_fa_q_stride_token",
        "hrx2_shape_fa_q_stride_head",
        "hrx2_shape_fa_k_stride_pos",
        "hrx2_shape_fa_k_stride_head",
        "hrx2_shape_fa_v_stride_pos",
        "hrx2_shape_fa_v_stride_head",
        "hrx2_shape_fa_mask_stride_token",
        "hrx2_shape_fa_dst_stride_head",
        "hrx2_shape_fa_dst_stride_token",
    }
    actual_config_names = {
        binding["name"] for binding in wave["config"]["bindings"]
    }
    if actual_config_names != expected_config_names:
        raise AssertionError("flash-attention wave compile-time config contract changed")
    for fragment in (
        "if (!(shape_src0_ntokens >= 1)) {",
        "if (!(shape_src1_nkv >= 1)) {",
        "const int64_t derived_query_tiles = "
        "static_cast<int64_t>((shape_src0_ntokens + 4 - 1) / 4);",
        "const int64_t main_dispatch_workgroups[3] = "
        "{derived_query_tiles, shape_src0_nheads, 1};",
    ):
        if fragment not in generated:
            raise AssertionError(
                f"flash-attention wave generator missing {fragment!r}"
            )


def main():
    expect_valid(CATALOG_ROOT)
    sum_rows_route, sum_rows_definition = read_route_and_definition(CATALOG_ROOT, SUM_ROWS_ROUTE_PATH)
    sum_rows_impl = route_impl.generate_route_impl(str(SUM_ROWS_ROUTE_PATH), sum_rows_route, sum_rows_definition)
    if "shape_dst_d0 != shape_src0_d0" in sum_rows_impl:
        raise AssertionError("v1 shape captures must not imply cross-tensor equality")

    fusion_route, fusion_definition = read_route_and_definition(CATALOG_ROOT, FUSION_ROUTE_PATH)
    fusion_impl = route_impl.generate_route_impl(str(FUSION_ROUTE_PATH), fusion_route, fusion_definition)
    if "shape_rms_out_d0 != shape_x_d0" not in fusion_impl:
        raise AssertionError("v2 structural shape declarations must enforce cross-tensor equality")
    for fragment in (
        "plan->main.entry = entry;",
        "plan->consumed_node_count = 2;",
        "plan->consumed_node_indices[0] = request->node_index + 0;",
        "plan->consumed_node_indices[1] = request->node_index + 1;",
    ):
        if fragment not in fusion_impl:
            raise AssertionError(f"fusion-v2 execution plan missing {fragment!r}")

    expect_extended_op_schema()
    expect_prepass_schema_and_generation()
    expect_q8_packed_loop_coverage()
    expect_q8_decode_small_column_coverage()
    expect_q5_down_group4_selected_route()
    expect_flash_attn_wave_warmup_coverage()
    expect_valid_mutation(
        "workgroups-dispatch",
        lambda route: route["invocation"].update({
            "dispatch": {"workgroups": ["derived.total_size"], "workgroup_size": [256, 1, 1]}
        }),
    )

    cases = [
        (
            "missing-architectures",
            lambda route: route.pop("architectures"),
            "expected array field architectures",
        ),
        (
            "empty-architectures",
            lambda route: route.update({"architectures": []}),
            "architectures must not be empty",
        ),
        (
            "scalar-architectures",
            lambda route: route.update({"architectures": "gfx1100"}),
            "expected array field architectures",
        ),
        (
            "empty-architecture",
            lambda route: route.update({"architectures": [""]}),
            "expected non-empty string",
        ),
        (
            "non-string-architecture",
            lambda route: route.update({"architectures": [7]}),
            "expected non-empty string",
        ),
        (
            "duplicate-architectures",
            lambda route: route.update({"architectures": ["gfx1100", "gfx1100"]}),
            "duplicate architecture gfx1100",
        ),
        (
            "mixed-any-architecture",
            lambda route: route.update({"architectures": ["*", "gfx1100"]}),
            "architecture * cannot be combined with explicit architectures",
        ),
        (
            "unknown-architecture",
            lambda route: route.update({"architectures": [TEST_TARGET_MISSING]}),
            f"architecture {TEST_TARGET_MISSING} is not listed in metadata targets",
        ),
        (
            "ambiguous-dispatch",
            lambda route: route["invocation"]["dispatch"].update({"workgroups": ["derived.total_size"]}),
            "expects exactly one of work_items or workgroups",
        ),
        (
            "scalar-dispatch",
            lambda route: route["invocation"]["dispatch"].update({"work_items": "derived.total_size"}),
            "expected an array with 1 to 3 integer values",
        ),
        (
            "too-many-dispatch-axes",
            lambda route: route["invocation"]["dispatch"].update({"work_items": [1, 1, 1, 1]}),
            "expected an array with 1 to 3 integer values",
        ),
        (
            "missing-config",
            lambda route: route.pop("config"),
            "expected object field config",
        ),
        (
            "duplicate-config-names",
            lambda route: route["config"]["bindings"][1].update({"name": "shape_pointwise_total_size"}),
            "duplicate config binding name shape_pointwise_total_size",
        ),
        (
            "unresolved-config-source",
            lambda route: route["config"]["bindings"][0].update({"source": "derived.missing"}),
            "derived value missing is not available",
        ),
        (
            "source-value-conflict",
            lambda route: route["config"]["bindings"][0].update({"value": 7}),
            "expected exactly one of source or value",
        ),
        (
            "forbidden-routing-field",
            lambda route: route.update({"routing": {}}),
            "unsupported field routing",
        ),
        (
            "forbidden-launch-field",
            lambda route: route.update({"launch": {}}),
            "unsupported field launch",
        ),
    ]

    for name, mutator, expected in cases:
        expect_invalid(name, mutator, expected)

    expect_invalid_route(
        "fusion-unknown-transient",
        FUSION_ROUTE_PATH,
        lambda route: route["match"]["predicates"][2].update({"transients": ["missing"]}),
        "tensor missing is not declared in tensors",
    )
    expect_invalid_route(
        "fusion-no-overlap-arity",
        FUSION_ROUTE_PATH,
        lambda route: route["match"]["predicates"][3].update({"no_overlap": ["x"]}),
        "expected two tensor names",
    )
    prepass_cases = [
        (
            "scratch-buffer-source-conflict",
            lambda route: route["invocation"]["buffers"][3].update(
                {"tensor": "src1"}
            ),
            "expected exactly one of tensor or scratch",
        ),
        (
            "unknown-scratch-segment",
            lambda route: route["invocation"]["buffers"][3].update(
                {"scratch": "qact.missing"}
            ),
            "unknown scratch segment qact.missing",
        ),
        (
            "nonpositive-scratch-length",
            lambda route: route["scratch"][0]["segments"][0].update(
                {"length": 0}
            ),
            "expected positive integer literal or integer source",
        ),
        (
            "prepass-scratch-access-mismatch",
            lambda route: route["prepasses"][0]["invocation"]["buffers"][1].update(
                {"kind": "input"}
            ),
            "kind input does not match binding access write",
        ),
        (
            "unsupported-prepass-cache-scope",
            lambda route: route["prepasses"][0]["cache"].update(
                {"scope": "graph"}
            ),
            "cache.scope must be execution",
        ),
    ]
    for name, mutator, expected in prepass_cases:
        expect_invalid_route(
            name,
            Q8_PREPASS_ROUTE_PATH,
            mutator,
            expected,
        )

    expect_invalid_catalog_mutation(
        "missing-metadata-targets",
        lambda source_root: mutate_metadata(source_root, lambda metadata: metadata.pop("targets")),
        "expected non-empty list field targets",
    )
    expect_invalid_catalog_mutation(
        "scalar-metadata-targets",
        lambda source_root: mutate_metadata(source_root, lambda metadata: metadata.update({"targets": TEST_TARGET_A})),
        "expected non-empty list field targets",
    )
    expect_invalid_catalog_mutation(
        "empty-metadata-targets",
        lambda source_root: mutate_metadata(source_root, lambda metadata: metadata.update({"targets": []})),
        "expected non-empty list field targets",
    )
    expect_invalid_catalog_mutation(
        "non-string-metadata-target",
        lambda source_root: set_metadata_targets(source_root, ["gfx1100", 7]),
        "targets must contain non-empty strings",
    )
    expect_invalid_catalog_mutation(
        "duplicate-metadata-targets",
        lambda source_root: set_metadata_targets(source_root, [TEST_TARGET_A, TEST_TARGET_A]),
        f"duplicate target {TEST_TARGET_A}",
    )
    expect_invalid_catalog_mutation(
        "empty-metadata-target",
        lambda source_root: set_metadata_targets(source_root, ["gfx1100", ""]),
        "targets must contain non-empty strings",
    )

    expect_invalid_catalog_mutation(
        "route-architecture-subset",
        lambda source_root: set_targets_and_architectures(
            source_root,
            [TEST_TARGET_A, TEST_TARGET_B],
            [TEST_TARGET_A],
        ),
        f"metadata target {TEST_TARGET_B} is not covered by any route architecture",
    )
    expect_valid_catalog_mutation(
        "route-architecture-any",
        lambda source_root: set_targets_and_architectures(
            source_root,
            [TEST_TARGET_A, TEST_TARGET_B],
            ["*"],
        ),
    )
    expect_generation_valid(
        "generate-compatible-subset",
        lambda source_root: set_targets_and_architectures(
            source_root,
            [TEST_TARGET_A, TEST_TARGET_B],
            [TEST_TARGET_A, TEST_TARGET_B],
        ),
        [TEST_TARGET_A],
        [{"id": "add_f32_contiguous", "target": TEST_TARGET_A}],
    )
    expect_generation_valid(
        "generate-any-architecture",
        lambda source_root: set_targets_and_architectures(
            source_root,
            [TEST_TARGET_A, TEST_TARGET_B],
            ["*"],
        ),
        [TEST_TARGET_A, TEST_TARGET_B],
        [
            {"id": "add_f32_contiguous", "target": TEST_TARGET_A},
            {"id": "add_f32_contiguous", "target": TEST_TARGET_B},
        ],
    )
    expect_generation_invalid(
        "generate-unknown-target",
        lambda source_root: set_targets_and_architectures(source_root, [TEST_TARGET_A], [TEST_TARGET_A]),
        [TEST_TARGET_B],
        f"selected target {TEST_TARGET_B} is not listed in metadata targets",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

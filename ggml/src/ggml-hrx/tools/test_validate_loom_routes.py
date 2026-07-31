#!/usr/bin/env python3

import hashlib
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from shutil import copytree

import generate_loom_catalog as catalog
import generate_loom_route_impl as route_generator
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
Q5_EXPERT_PLANES_ROUTE_PATH = Path(
    "routes/generic/mul_mat_id/q5_k_f32/expert_planes_4d.json"
)
Q5_EXPERT_PLANES_BROADCAST_ROUTE_PATH = Path(
    "routes/generic/mul_mat_id/q5_k_f32/"
    "expert_planes_src1_broadcast_4d.json"
)
Q5_PRE_TBL_ROUTE_PATH = Path(
    "routes/gfx1151/mul_mat_id/q5_k_f32/"
    "mul_mat_id_q5_k_f32_mmq_gfx1151_wg256_pre_tbl.json"
)
Q5_SWIGLU_ROUTE_PATH = Path(
    "routes/gfx1151/mul_mat_id/q5_k_f32/"
    "mul_mat_id_q5_k_f32_swiglu.json"
)
Q5_DOWN_GROUP4_ROUTE_PATH = Path(
    "routes/gfx1151/mul_mat_id/q5_k_f32/"
    "mul_mat_id_q5_k_f32_mmq_gfx1151_wg256_tbl_down_group4.json"
)
Q4_SWIGLU_Q5_DOWN_ROUTE_PATH = Path(
    "routes/gfx1151/mul_mat_id/q4_k_f32/"
    "mul_mat_id_q4_k_swiglu_q5_down_qact.json"
)
Q4_TOPK_QACT_ROUTE_PATH = Path(
    "routes/gfx1151/mul_mat_id/q4_k_f32/"
    "mul_mat_id_q4_k_f32_mmq_gfx1151_wg256_pre_tbl_"
    "decode_topk_qact_table.json"
)
FLASH_ATTN_WMMA_ROUTE_PATH = Path(
    "routes/gfx1151/flash_attn_ext/f32_f16/wmma_f32.json"
)
FLASH_ATTN_WAVE_ROUTE_PATH = Path(
    "routes/gfx1151/flash_attn_ext/f32_f16/wave256.json"
)
FLASH_ATTN_GATE_EPILOGUE_ROUTE_PATH = Path(
    "routes/gfx1151/flash_attn_ext/f32_f16/wmma_gate_epilogue.json"
)
SSM_CONCAT_ROUTE_PATH = Path(
    "routes/gfx1151/concat/f32/window_tail_ssm_silu_pp512.json"
)
SSM_COMPUTE_ROUTE_PATH = Path(
    "routes/gfx1151/ssm_conv/f32/"
    "channels_first_concat_silu_regblock_wg1024.json"
)
SUM_SLICES_ROUTE_PATH = Path(
    "routes/gfx1151/mul/f32/sum_slices_pp512.json"
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
    "GGML_OP_CPY": {
        "tensors": {"src0": "F32", "src1": "F32", "dst": "F32"},
        "optional": {"src1"},
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
        "attributes": {"precision": "i32", "hint": "i32"},
    },
    "GGML_OP_MUL_MAT_ID": {
        "tensors": {"src0": "Q4_K", "src1": "F32", "src2": "I32", "dst": "F32"},
        "attributes": {},
    },
    "GGML_OP_RESHAPE": {
        "tensors": {"src0": "F32", "dst": "F32"},
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
    "GGML_OP_VIEW": {
        "tensors": {"src0": "F32", "dst": "F32"},
        "attributes": {},
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


def expect_invalid_full_catalog_mutation(name, route_path, mutator, expected):
    with tempfile.TemporaryDirectory(prefix=f"{name}-") as tmpdir:
        source_root = Path(tmpdir) / "loom-catalog"
        copytree(CATALOG_ROOT, source_root)
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
    if unary_context.resolve_source(
        "tensor.src0.view_offset", "test tensor view_offset"
    ) != "i64":
        raise AssertionError("ordinary tensor view_offset must resolve as i64")
    if route_emit.source_expr(
        "tensor.src0.view_offset"
    ) != "static_cast<int64_t>(src0->view_offs)":
        raise AssertionError("tensor view_offset must emit tensor->view_offs")
    for unary_op in (TEST_UNARY_OP_SIGMOID, TEST_UNARY_OP_SOFTPLUS):
        route_schema.validate_predicates(
            [{"field": "attribute.unary_op", "equals": unary_op}],
            "test GGML_OP_UNARY",
            unary_context,
        )


def expect_fusion_tensor_dag_schema_and_generation():
    route, definition = read_route_and_definition(
        CATALOG_ROOT, FUSION_ROUTE_PATH
    )

    def validate_and_generate(candidate, source):
        tensors, attributes, predicates = route_schema.validate_fusion_match(
            candidate, source, definition
        )
        context = route_schema.FusionRouteContext(
            source,
            tensors,
            attributes,
            candidate["derived"],
        )
        route_schema.validate_predicates(predicates, source, context)
        return route_generator.generate_route_impl(
            FUSION_ROUTE_PATH, candidate, definition
        )

    default_impl = route_generator.generate_route_impl(
        FUSION_ROUTE_PATH, route, definition
    )
    for fragment in (
        "request->cgraph->nodes[request->node_index + 1]",
        "if (rms_out != mul_op->src[0]) {",
        "plan->consumed_node_count = 2;",
        "plan->consumed_node_indices[1] = request->node_index + 1;",
    ):
        if fragment not in default_impl:
            raise AssertionError(
                f"default consecutive fusion behavior lost {fragment!r}"
            )
    if "for (int candidate_index = 0;" in default_impl:
        raise AssertionError(
            "default fusion unexpectedly enabled tensor-DAG traversal"
        )
    if "dispatch_owner_node_index" in default_impl:
        raise AssertionError(
            "fusion without dispatch_owner unexpectedly changed scheduling"
        )
    if "deferred_read" in default_impl:
        raise AssertionError(
            "fusion without dispatch_owner unexpectedly tracked staged accesses"
        )
    if "ggml_backend_hrx_loom_storage_layout_matches(" in default_impl:
        raise AssertionError(
            "fusion without a storage transform unexpectedly emitted a storage guard"
        )

    dag_route = deepcopy(route)
    dag_route["match"]["traversal"] = "tensor_dag"
    dag_route["tensors"]["context_out"] = {"type": "F32"}
    rms_match = dag_route["match"]["ops"]["rms"]
    mul_match = dag_route["match"]["ops"]["mul"]
    dag_route["match"]["ops"] = {
        "rms": rms_match,
        "context": {
            "op": "GGML_OP_ADD",
            "tensors": {
                "src0": "x",
                "src1": "weight",
                "dst": "context_out",
            },
            "attributes": {},
        },
        "mul": mul_match,
    }
    dag_route["match"]["consumed_ops"] = ["rms", "mul"]
    tensors, attributes, predicates = route_schema.validate_fusion_match(
        dag_route, "test tensor_dag fusion", definition
    )
    context = route_schema.FusionRouteContext(
        "test tensor_dag fusion",
        tensors,
        attributes,
        dag_route["derived"],
    )
    route_schema.validate_predicates(
        predicates, "test tensor_dag fusion", context
    )
    dag_impl = route_generator.generate_route_impl(
        FUSION_ROUTE_PATH, dag_route, definition
    )
    for fragment in (
        "for (int candidate_index = 0; "
        "candidate_index < request->cgraph->n_nodes; ++candidate_index) {",
        "ggml_backend_hrx_loom_tensor_dag_edge_matches("
        "candidate->src[0], rms_out)",
        "const int consumed_node_indices[consumed_node_count] = "
        "{rms_op_index, mul_op_index};",
        "ggml_backend_hrx_loom_tensor_is_transient_at_indices("
        "request, rms_out, consumed_node_indices, consumed_node_count)",
        "plan->consumed_node_count = 2;",
        "plan->consumed_node_indices[0] = rms_op_index;",
        "plan->consumed_node_indices[1] = mul_op_index;",
    ):
        if fragment not in dag_impl:
            raise AssertionError(
                f"tensor-DAG fusion generator missing {fragment!r}"
            )

    deferred_route = deepcopy(dag_route)
    deferred_route["match"]["dispatch_owner"] = "mul"
    deferred_impl = validate_and_generate(
        deferred_route,
        "test deferred fusion dispatch",
    )
    if (
        "plan->dispatch_owner_node_index = mul_op_index;"
        not in deferred_impl
    ):
        raise AssertionError(
            "fusion dispatch owner was not materialized in the plan"
        )
    for fragment in (
        "plan->main.deferred_read_tensors[0] = x;",
        "plan->main.deferred_read_tensors[1] = weight;",
        "plan->main.deferred_read_binding_mask = 3;",
    ):
        if fragment not in deferred_impl:
            raise AssertionError(
                "fusion dispatch owner did not track a deferred read: "
                f"{fragment!r}"
            )

    witness_route = deepcopy(route)
    witness_route["match"]["traversal"] = "tensor_dag"
    witness_route["match"]["dispatch_owner"] = "mul"
    witness_route["match"]["consumed_ops"] = ["rms", "mul"]
    witness_route["tensors"]["consumer_out"] = {"type": "F32"}
    witness_route["tensors"]["context_out"] = {"type": "F32"}
    consumer_match = {
        "op": "GGML_OP_UNARY",
        "tensors": {"src0": "y", "dst": "consumer_out"},
        "attributes": {
            "unary_op": {"type": "i32", "source": "op_param.0"},
        },
    }
    context_match = {
        "op": "GGML_OP_ADD",
        "tensors": {
            "src0": "x",
            "src1": "weight",
            "dst": "context_out",
        },
        "attributes": {},
    }
    base_ops = dict(witness_route["match"]["ops"])
    witness_route["match"]["ops"] = {
        **base_ops,
        "context": context_match,
        "consumer": consumer_match,
    }
    reordered_witness_route = deepcopy(witness_route)
    reordered_witness_route["match"]["ops"] = {
        **base_ops,
        "consumer": consumer_match,
        "context": context_match,
    }
    witness_impls = (
        validate_and_generate(
            witness_route,
            "test deferred-owner structural witness",
        ),
        validate_and_generate(
            reordered_witness_route,
            "test reordered deferred-owner structural witness",
        ),
    )
    for fragment in (
        "bool candidate_has_consumer_witness = false;",
        "for (int consumer_witness_index = 0;",
        "consumer_candidate->op != GGML_OP_UNARY",
        "ggml_backend_hrx_loom_tensor_dag_edge_matches("
        "consumer_candidate->src[0], candidate)",
        "if (!candidate_has_consumer_witness) {",
    ):
        if any(fragment not in impl for impl in witness_impls):
            raise AssertionError(
                "order-invariant deferred-owner witness missing " +
                repr(fragment)
            )

    early_witness_route = deepcopy(witness_route)
    early_witness_route["match"]["ops"] = {
        "rms": base_ops["rms"],
        "consumer": consumer_match,
        "context": context_match,
        "mul": base_ops["mul"],
    }
    early_witness_impl = validate_and_generate(
        early_witness_route,
        "test early-declared structural witness",
    )
    if (
        "ggml_backend_hrx_loom_tensor_dag_edge_matches("
        "consumer_op->src[0], candidate)"
        not in early_witness_impl
    ):
        raise AssertionError(
            "a structural witness declared before its producer did not "
            "constrain that producer"
        )

    deferred_router = route_generator.generate_op_router_impl(
        [(FUSION_ROUTE_PATH, deferred_route, definition)],
        "GGML_OP_RMS_NORM",
    )
    selection_guard = (
        "!ggml_backend_hrx_loom_plan_can_be_selected(request, plan)"
    )
    response_return = (
        "if (response.result != GGML_BACKEND_HRX_LOOM_UNSUPPORTED)"
    )
    if (
        selection_guard not in deferred_router or
        deferred_router.index(selection_guard) >=
        deferred_router.index(response_return)
    ):
        raise AssertionError(
            "op router must reject a conflicting deferred candidate before "
            "returning it"
        )

    external_transient_route = deepcopy(route)
    external_transient_route["match"]["traversal"] = "tensor_dag"
    next(
        predicate
        for predicate in external_transient_route["match"]["predicates"]
        if "transients" in predicate
    )["transients"].insert(0, "x")
    external_transient_route["match"]["predicates"].append({
        "field": "tensor.x.view_offset",
        "equals": 0,
    })
    external_transient_impl = validate_and_generate(
        external_transient_route,
        "test external fusion transient",
    )
    for fragment in (
        "ggml_backend_hrx_loom_tensor_is_transient_at_indices("
        "request, x, consumed_node_indices, consumed_node_count)",
        "if (!(static_cast<int64_t>(x->view_offs) == 0)) {",
    ):
        if fragment not in external_transient_impl:
            raise AssertionError(
                f"external fusion transient generator missing {fragment!r}"
            )

    external_transient_router = route_generator.generate_op_router_impl(
        [
            (
                FUSION_ROUTE_PATH,
                external_transient_route,
                definition,
            )
        ],
        "GGML_OP_RMS_NORM",
    )
    for fragment in (
        "case GGML_OP_NONE:",
        "value = value->src[0] ? value->src[0] : value->view_src;",
        "ggml_backend_hrx_loom_tensor_is_transient_at_indices_cached(",
    ):
        if fragment not in external_transient_router:
            raise AssertionError(
                f"fusion output-safety generator missing {fragment!r}"
            )
    for obsolete in (
        "ggml_backend_hrx_loom_tensor_is_graph_output(",
        "ggml_backend_hrx_loom_tensor_consumers_are_consumed(",
    ):
        if obsolete in external_transient_router:
            raise AssertionError(
                "fusion router retained an O(n_nodes) safety scan "
                f"helper {obsolete!r}"
            )

    metadata_route = deepcopy(route)
    metadata_route["match"]["traversal"] = "tensor_dag"
    metadata_route["tensors"]["reshape_out"] = {"type": "F32"}
    metadata_route["tensors"]["view_out"] = {"type": "F32"}
    rms_match = metadata_route["match"]["ops"]["rms"]
    mul_match = deepcopy(metadata_route["match"]["ops"]["mul"])
    mul_match["tensors"]["src0"] = "view_out"
    metadata_route["match"]["ops"] = {
        "rms": rms_match,
        "reshape": {
            "op": "GGML_OP_RESHAPE",
            "tensors": {"src0": "rms_out", "dst": "reshape_out"},
            "attributes": {},
        },
        "view": {
            "op": "GGML_OP_VIEW",
            "tensors": {"src0": "reshape_out", "dst": "view_out"},
            "attributes": {},
        },
        "mul": mul_match,
    }
    metadata_route["match"]["consumed_ops"] = ["rms", "mul"]
    next(
        predicate
        for predicate in metadata_route["match"]["predicates"]
        if "transients" in predicate
    )["transients"] = ["rms_out", "reshape_out", "view_out"]
    metadata_impl = validate_and_generate(
        metadata_route,
        "test fusion metadata ops",
    )
    for fragment in (
        "candidate->op != GGML_OP_RESHAPE",
        "candidate->op != GGML_OP_VIEW",
        "const ggml_tensor * reshape_out = reshape_op;",
        "const ggml_tensor * view_out = view_op;",
        "const int consumed_node_indices[consumed_node_count] = "
        "{rms_op_index, mul_op_index};",
    ):
        if fragment not in metadata_impl:
            raise AssertionError(
                f"fusion metadata-op generator missing {fragment!r}"
            )

    bounded_owner_route = deepcopy(metadata_route)
    bounded_owner_route["match"]["predicates"].append({
        "consumers_through_view": {
            "owner": "rms_out",
            "view": "view_out",
        },
    })
    bounded_owner_impl = validate_and_generate(
        bounded_owner_route,
        "test fusion consumers through view",
    )
    bounded_owner_router = route_generator.generate_op_router_impl(
        [(FUSION_ROUTE_PATH, bounded_owner_route, definition)],
        "GGML_OP_RMS_NORM",
    )
    bounded_owner_generated = bounded_owner_impl + bounded_owner_router
    for fragment in (
        "ggml_backend_hrx_loom_tensor_consumers_through_view("
        "request, rms_out, view_out)",
        "ggml_backend_hrx_loom_tensor_consumers_through_view_cached(",
    ):
        if fragment not in bounded_owner_generated:
            raise AssertionError(
                "fusion consumers-through-view generator missing "
                f"{fragment!r}"
            )

    invalid_bounded_owner_cases = (
        (
            "missing owner",
            {"view": "view_out"},
            "expected non-empty string field owner",
        ),
        (
            "missing view",
            {"owner": "rms_out"},
            "expected non-empty string field view",
        ),
        (
            "unknown owner",
            {"owner": "missing", "view": "view_out"},
            "tensor missing is not declared in tensors",
        ),
        (
            "unknown view",
            {"owner": "rms_out", "view": "missing"},
            "tensor missing is not declared in tensors",
        ),
        (
            "same owner and view",
            {"owner": "rms_out", "view": "rms_out"},
            "owner and view must be distinct",
        ),
    )
    for name, value, expected in invalid_bounded_owner_cases:
        candidate = deepcopy(metadata_route)
        candidate["match"]["predicates"].append({
            "consumers_through_view": value,
        })
        try:
            validate_and_generate(candidate, f"test {name}")
        except ValueError as err:
            if expected in str(err):
                continue
            raise AssertionError(
                f"{name}: expected {expected!r}, got {str(err)!r}"
            ) from err
        raise AssertionError(
            f"{name}: fusion validator accepted invalid "
            "consumers_through_view predicate"
        )

    invalid_cases = (
        (
            "unknown traversal",
            lambda candidate: candidate["match"].update(
                {"traversal": "graph_v1"}
            ),
            "expected one of consecutive, tensor_dag",
        ),
        (
            "empty consumed ops",
            lambda candidate: candidate["match"].update(
                {"consumed_ops": []}
            ),
            "expected non-empty operation array",
        ),
        (
            "unknown consumed op",
            lambda candidate: candidate["match"].update(
                {"consumed_ops": ["rms", "missing"]}
            ),
            "operation missing is not declared in match.ops",
        ),
        (
            "duplicate consumed op",
            lambda candidate: candidate["match"].update(
                {"consumed_ops": ["rms", "rms"]}
            ),
            "duplicate consumed operation rms",
        ),
        (
            "unconsumed anchor",
            lambda candidate: candidate["match"].update(
                {"consumed_ops": ["mul"]}
            ),
            "must include first anchor rms",
        ),
        (
            "non-string dispatch owner",
            lambda candidate: candidate["match"].update(
                {"dispatch_owner": 7}
            ),
            "expected non-empty operation name",
        ),
        (
            "unknown dispatch owner",
            lambda candidate: candidate["match"].update(
                {"dispatch_owner": "missing"}
            ),
            "operation missing is not declared in match.ops",
        ),
        (
            "unconsumed dispatch owner",
            lambda candidate: candidate["match"].update({
                "dispatch_owner": "mul",
                "consumed_ops": ["rms"],
            }),
            "operation mul must be consumed",
        ),
        (
            "disconnected tensor DAG",
            lambda candidate: (
                candidate["match"].update({"traversal": "tensor_dag"}),
                candidate["match"]["ops"]["mul"]["tensors"].update(
                    {"src0": "weight", "src1": "weight"}
                ),
            ),
            "is not connected to the tensor_dag rooted at anchor rms",
        ),
        (
            "unconsumed transient",
            lambda candidate: next(
                predicate
                for predicate in candidate["match"]["predicates"]
                if "transients" in predicate
            ).update({"transients": ["y"]}),
            "tensor y is not consumed inside the fusion",
        ),
    )
    for name, mutator, expected in invalid_cases:
        candidate = deepcopy(route)
        mutator(candidate)
        try:
            route_schema.validate_fusion_match(
                candidate, f"test {name}", definition
            )
        except ValueError as err:
            if expected in str(err):
                continue
            raise AssertionError(
                f"{name}: expected {expected!r}, got {str(err)!r}"
            ) from err
        raise AssertionError(f"{name}: fusion validator accepted invalid route")


def expect_fusion_storage_transform_generation():
    route_path = CATALOG_ROOT / Q4_SWIGLU_Q5_DOWN_ROUTE_PATH
    definitions = loom.load_definitions(CATALOG_ROOT)
    loom.validate_route(
        route_path,
        definitions,
        loom.load_metadata_targets(CATALOG_ROOT),
        loom.load_storage_transform_ids(CATALOG_ROOT),
    )
    route, definition = read_route_and_definition(
        CATALOG_ROOT, Q4_SWIGLU_Q5_DOWN_ROUTE_PATH
    )
    generated = route_generator.generate_route_impl(
        route_path, route, definition
    )
    expected_guard = (
        "    if (!ggml_backend_hrx_loom_storage_layout_matches("
        'request, down_weight, "q5_k_expert_down_group4")) {\n'
        "        return ggml_backend_hrx_loom_unsupported("
        "GGML_BACKEND_HRX_LOOM_UNSUPPORTED_LAYOUT);\n"
        "    }"
    )
    if generated.count(expected_guard) != 1:
        raise AssertionError(
            "Q4->Q5 fusion must emit exactly one down-weight storage guard"
        )

    no_storage_route = deepcopy(route)
    no_storage_route["tensors"]["down_weight"].pop("storage")
    no_storage_impl = route_generator.generate_route_impl(
        route_path, no_storage_route, definition
    )
    if "ggml_backend_hrx_loom_storage_layout_matches(" in no_storage_impl:
        raise AssertionError(
            "fusion storage guards must remain opt-in per tensor"
        )


def expect_q4_topk_topology_disambiguation():
    route_path = CATALOG_ROOT / Q4_TOPK_QACT_ROUTE_PATH
    route, definition = read_route_and_definition(
        CATALOG_ROOT, Q4_TOPK_QACT_ROUTE_PATH
    )
    match = route["match"]
    if match["ops"]["swiglu"]["tensors"].get("src0") != "main_dst":
        raise AssertionError(
            "Q4 top-k route must identify its projection through GLU.src0"
        )
    if (
        match["ops"]["swiglu"]["tensors"].get("src1") !=
        "main_partner_dst"
    ):
        raise AssertionError(
            "Q4 top-k route must identify its sibling projection through "
            "GLU.src1"
        )
    if "swiglu" in match["consumed_ops"]:
        raise AssertionError(
            "Q4 top-k route must match, but not consume, the downstream GLU"
        )
    if match["consumed_ops"][-1] != "main_partner":
        raise AssertionError(
            "Q4 top-k route must consume the sibling projection"
        )
    if match.get("dispatch_owner") != "main_partner":
        raise AssertionError(
            "Q4 top-k route must defer until the sibling projection"
        )
    if [prepass["id"] for prepass in route["prepasses"]] != [
        "topk_qact_mmid_table_decode",
        "mul_mat_id_q4_k_f32_mmq_gfx1151_wg256_pre_tbl",
    ]:
        raise AssertionError(
            "Q4 top-k route must produce shared scratch before dispatching "
            "the first projection"
        )
    generated = route_generator.generate_route_impl(
        route_path, route, definition
    )
    for fragment in (
        "bool candidate_has_swiglu_witness = false;",
        "for (int swiglu_witness_index = 0;",
        "ggml_backend_hrx_loom_tensor_dag_edge_matches("
        "swiglu_candidate->src[0], candidate)",
        "if (!candidate_has_swiglu_witness) {",
        "main_op = candidate;",
        "main_partner_op = candidate;",
        "if (!main_partner_op) {",
        "const int consumed_node_indices[consumed_node_count] = "
        "{softmax_op_index, argsort_op_index, get_rows_op_index, "
        "sum_rows_op_index, clamp_op_index, div_op_index, main_op_index, "
        "main_partner_op_index};",
        "plan->consumed_node_count = 8;",
        "plan->dispatch_owner_node_index = main_partner_op_index;",
        "plan->prepasses[0].kernel.deferred_read_tensors[0] = logits;",
        "plan->prepasses[0].kernel.deferred_read_tensors[3] = main_input;",
        "plan->prepasses[0].kernel.deferred_read_binding_mask = 9;",
        "plan->prepasses[1].kernel.deferred_read_tensors[0] = main_weight;",
        "plan->prepasses[1].kernel.deferred_read_tensors[1] = main_input;",
        "plan->prepasses[1].kernel.deferred_read_binding_mask = 3;",
        "plan->main.deferred_read_binding_mask = 3;",
        "plan->prepass_count = 2;",
    ):
        if fragment not in generated:
            raise AssertionError(
                "Q4 top-k topology matcher missing " + repr(fragment)
            )
    if generated.count("bool candidate_has_swiglu_witness = false;") != 2:
        raise AssertionError(
            "Q4 top-k route must disambiguate both Q4 projections with the "
            "GLU witness"
        )
    if (
        "ggml_backend_hrx_loom_bind_tensor(request, argsort_dst, "
        "&plan->main.bindings[2])" not in generated or
        "ggml_backend_hrx_loom_bind_tensor(request, argsort_dst, "
        "&plan->prepasses[1].kernel.bindings[2])" not in generated or
        "plan->main.deferred_read_tensors[2]" in generated or
        "plan->prepasses[1].kernel.deferred_read_tensors[2]" in generated
    ):
        raise AssertionError(
            "Both Q4 projections must bind the exact guaranteed argsort "
            "prepass output without treating it as an external delayed read"
        )


def expect_preexisting_tensor_dag_matchers_unchanged():
    definitions = loom.load_definitions(CATALOG_ROOT)
    checked = 0
    metadata_ops = {
        "GGML_OP_RESHAPE",
        "GGML_OP_VIEW",
        "GGML_OP_PERMUTE",
        "GGML_OP_TRANSPOSE",
    }
    for route_path in sorted((CATALOG_ROOT / "routes").rglob("*.json")):
        route = read_json(route_path)
        match = route.get("match", {})
        if match.get("traversal") != "tensor_dag":
            continue
        definition_path = (
            route_path.parent /
            route_schema.require_string(route, "definition", route_path)
        ).resolve()
        generated = route_generator.generate_route_impl(
            route_path,
            route,
            definitions[definition_path],
        )
        if "matched_assignment_count" in generated:
            raise AssertionError(
                f"{route_path}: global tensor-DAG backtracking returned"
            )
        ops = match["ops"]
        consumed = set(match.get("consumed_ops", ops))
        has_declared_compute_witness = any(
            producer_name in consumed and
            consumer_name not in consumed and
            consumer["op"] not in metadata_ops and
            producer.get("tensors", {}).get("dst") in {
                tensor_name
                for role, tensor_name in consumer.get("tensors", {}).items()
                if role != "dst"
            }
            for producer_name, producer in ops.items()
            for consumer_name, consumer in ops.items()
        )
        checked += 1
        if (
            "candidate_has_" in generated and
            not has_declared_compute_witness
        ):
            raise AssertionError(
                f"{route_path}: greedy matcher changed without a declared "
                "outbound edge to an unconsumed compute op"
            )
    if checked == 0:
        raise AssertionError("no pre-existing tensor-DAG routes checked")


def expect_prepass_schema_and_generation():
    definitions = loom.load_definitions(CATALOG_ROOT)
    route_path = CATALOG_ROOT / Q8_PREPASS_ROUTE_PATH
    route = read_json(route_path)
    definition_path = (
        route_path.parent /
        route_schema.require_string(route, "definition", route_path)
    ).resolve()
    generated = route_generator.generate_route_impl(
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

    uncached = deepcopy(route)
    uncached["prepasses"][0].pop("cache")
    op_rule, tensors, attributes, predicates = route_schema.validate_match(
        uncached, "test uncached prepass", definitions[definition_path]
    )
    context = route_schema.RouteContext(
        "test uncached prepass",
        op_rule,
        tensors,
        attributes,
        uncached["derived"],
    )
    route_schema.validate_derived(uncached, "test uncached prepass", context)
    route_schema.validate_predicates(
        predicates, "test uncached prepass", context
    )
    scratch = route_schema.validate_scratch(
        uncached, "test uncached prepass", context
    )
    loom.validate_prepasses(
        uncached,
        route_path,
        definitions,
        context,
        scratch,
    )
    uncached_generated = route_generator.generate_route_impl(
        route_path,
        uncached,
        definitions[definition_path],
    )
    for fragment in (
        'ggml_backend_hrx_loom_find_entry(catalog, "quant_act_q8")',
        "plan->prepasses[0].kernel.binding_scratch_index[3] = 1;",
        "plan->prepass_count = 1;",
    ):
        if fragment not in uncached_generated:
            raise AssertionError(
                f"uncached prepass generator missing {fragment!r}"
            )
    for fragment in (
        "plan->prepasses[0].cache_scratch_index",
        "plan->prepasses[0].cache_source",
    ):
        if fragment in uncached_generated:
            raise AssertionError(
                f"uncached prepass unexpectedly emitted {fragment!r}"
            )

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
        generated = route_generator.generate_route_impl(
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
    generated = route_generator.generate_route_impl(
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


def expect_q5_expert_plane_routes():
    definitions = loom.load_definitions(CATALOG_ROOT)
    targets = loom.load_metadata_targets(CATALOG_ROOT)
    transforms = loom.load_storage_transform_ids(CATALOG_ROOT)
    route_paths = (
        Q5_EXPERT_PLANES_ROUTE_PATH,
        Q5_EXPERT_PLANES_BROADCAST_ROUTE_PATH,
    )
    routes = []
    route_definitions = []
    generated_routes = []
    for relative_path in route_paths:
        route_path = CATALOG_ROOT / relative_path
        loom.validate_route(route_path, definitions, targets, transforms)
        route, definition = read_route_and_definition(
            CATALOG_ROOT, relative_path
        )
        routes.append(route)
        route_definitions.append(definition)
        generated_routes.append(
            route_generator.generate_route_impl(
                route_path, route, definition
            )
        )

    ordinary, broadcast = routes
    ordinary_definition, broadcast_definition = route_definitions
    ordinary_generated, broadcast_generated = generated_routes
    optimized = read_json(CATALOG_ROOT / Q5_PRE_TBL_ROUTE_PATH)
    fusion = read_json(CATALOG_ROOT / Q5_SWIGLU_ROUTE_PATH)
    if ordinary["priority"] != 104 or broadcast["priority"] != 105:
        raise AssertionError(
            "generic Q5 expert-plane priorities must remain ordinary=104, "
            "broadcast=105"
        )
    if not (
        fusion["priority"] > optimized["priority"] >
        broadcast["priority"] > ordinary["priority"]
    ):
        raise AssertionError(
            "Q5 route priorities must preserve fusion, optimized, broadcast, "
            "ordinary order"
        )
    ordered_route_paths = (
        Q5_SWIGLU_ROUTE_PATH,
        Q5_PRE_TBL_ROUTE_PATH,
        Q5_EXPERT_PLANES_BROADCAST_ROUTE_PATH,
        Q5_EXPERT_PLANES_ROUTE_PATH,
    )
    router_inputs = []
    ordered_route_ids = []
    for relative_path in ordered_route_paths:
        route, definition = read_route_and_definition(
            CATALOG_ROOT, relative_path
        )
        router_inputs.append((
            CATALOG_ROOT / relative_path,
            route,
            definition,
        ))
        ordered_route_ids.append(route["id"])
    generated_router = route_generator.generate_op_router_impl(
        router_inputs, "GGML_OP_MUL_MAT_ID"
    )
    route_call_positions = [
        generated_router.index(
            f"ggml_backend_hrx_loom_route_{route_id}"
            "(catalog, request, plan);"
        )
        for route_id in ordered_route_ids
    ]
    if route_call_positions != sorted(route_call_positions):
        raise AssertionError(
            "generated Q5 MUL_MAT_ID router must try fusion, optimized, "
            "broadcast, then ordinary routes"
        )

    expected_source = (
        "sources/generic/mul_mat_id/q5_k_f32/expert_planes.loom"
    )
    for route, definition in zip(routes, route_definitions):
        src0 = route["match"]["tensors"]["src0"]
        if src0.get("type") != "Q5_K" or src0.get("storage") != "canonical":
            raise AssertionError(
                "generic Q5 expert-plane routes must require canonical Q5_K"
            )
        if definition["source"] != expected_source:
            raise AssertionError(
                "generic Q5 expert-plane definitions must share the "
                "canonical scalar source"
            )
        if definition["symbol"] != "mul_mat_id_q5_k_f32_static":
            raise AssertionError(
                "generic Q5 expert-plane definitions must share one symbol"
            )
        if definition["abi"]["binding_count"] != 4:
            raise AssertionError(
                "generic Q5 expert-plane definitions must retain four buffers"
            )
        if definition["workgroup_size"] != [256, 1, 1]:
            raise AssertionError(
                "generic Q5 expert-plane definitions must retain WG256"
            )

    ordinary_predicates = ordinary["match"]["predicates"]
    required_ordinary_predicates = (
        {"field": "shape.src0.d0", "min": 256},
        {"field": "shape.src0.d0", "max": 32768},
        {"field": "shape.src0.d0", "multiple_of": 256},
        {"field": "shape.dst.d0", "min": 1},
        {"field": "shape.dst.d0", "max": 32768},
        {"field": "shape.src0.d2", "min": 1},
        {"field": "shape.src0.d2", "max": 1024},
        {"field": "shape.dst.d1", "min": 1},
        {"field": "shape.dst.d1", "max": 128},
        {"field": "shape.dst.d2", "min": 1},
        {"field": "shape.dst.d2", "max": 1048576},
        {"field": "shape.dst.d1", "equals": "shape.src1.d1"},
        {"field": "shape.dst.d1", "equals": "shape.src2.d0"},
        {"field": "shape.dst.d2", "equals": "shape.src1.d2"},
        {"field": "shape.dst.d2", "equals": "shape.src2.d1"},
        {"field": "tensor.src1.element_strides.2", "min": 1},
        {"field": "tensor.src1.element_strides.2", "max": 1048576},
        {"field": "tensor.src2.element_strides.0", "equals": 1},
        {"field": "tensor.src2.element_strides.1", "min": 1},
        {"field": "tensor.src2.element_strides.1", "max": 1048576},
        {"field": "tensor.dst.element_strides.2", "min": 1},
        {"field": "tensor.dst.element_strides.2", "max": 1048576},
        {"contiguous": "src0"},
        {"contiguous": "src1"},
        {"contiguous": "dst"},
    )
    for predicate in required_ordinary_predicates:
        if predicate not in ordinary_predicates:
            raise AssertionError(
                f"ordinary Q5 expert-plane route missing {predicate!r}"
            )
    for k, rows in ((512, 2048), (2048, 512)):
        if not (256 <= k <= 32768 and k % 256 == 0):
            raise AssertionError(
                f"Q5 loader-probe k={k} fell outside the route domain"
            )
        if not (1 <= rows <= 32768):
            raise AssertionError(
                f"Q5 loader-probe rows={rows} fell outside the route domain"
            )
    if not (1 <= 256 <= 1024 and 1 <= 8 <= 128 and 1 <= 512 <= 1048576):
        raise AssertionError(
            "Q5 loader-probe expert, selected, or token extent fell outside "
            "the route domain"
        )

    ordinary_stride = next(
        binding
        for binding in ordinary["config"]["bindings"]
        if binding["name"] == "shape_mul_mat_id_src1_selected_stride"
    )
    broadcast_stride = next(
        binding
        for binding in broadcast["config"]["bindings"]
        if binding["name"] == "shape_mul_mat_id_src1_selected_stride"
    )
    if ordinary_stride != {
        "name": "shape_mul_mat_id_src1_selected_stride",
        "type": "i64",
        "source": "tensor.src1.element_strides.1",
    }:
        raise AssertionError(
            "ordinary Q5 expert-plane route must retain selected-plane stride"
        )
    if broadcast_stride != {
        "name": "shape_mul_mat_id_src1_selected_stride",
        "type": "i64",
        "value": 0,
    }:
        raise AssertionError(
            "broadcast Q5 expert-plane route must zero selected-plane stride"
        )
    if {
        "field": "shape.src1.d1",
        "equals": 1,
    } not in broadcast["match"]["predicates"]:
        raise AssertionError(
            "broadcast Q5 expert-plane route must require one src1 plane"
        )
    for route in routes:
        for predicate in (
            {"contiguous": "src0"},
            {"contiguous": "src1"},
            {"contiguous": "dst"},
        ):
            if predicate not in route["match"]["predicates"]:
                raise AssertionError(
                    "generic Q5 expert-plane route must require the kernel's "
                    f"contiguous layout: missing {predicate!r}"
                )
        if {"contiguous": "src2"} in route["match"]["predicates"]:
            raise AssertionError(
                "generic Q5 expert-plane route must allow a padded src2 "
                "token stride"
            )
        if {
            "field": "tensor.src2.element_strides.0",
            "equals": 1,
        } not in route["match"]["predicates"]:
            raise AssertionError(
                "generic Q5 expert-plane route must require packed src2 "
                "selected indices"
            )

    for generated in generated_routes:
        for fragment in (
            "src0->type != GGML_TYPE_Q5_K",
            'ggml_backend_hrx_loom_storage_layout_matches('
            'request, src0, "canonical")',
            "const int64_t main_dispatch_workgroups[3] = "
            "{shape_dst_d0, shape_dst_d1, shape_dst_d2};",
        ):
            if fragment not in generated:
                raise AssertionError(
                    f"Q5 expert-plane generator missing {fragment!r}"
                )
    if (
        "main_config_shape_mul_mat_id_src1_selected_stride_value = "
        "static_cast<int64_t>((src1->nb[1] / "
        "ggml_type_size(src1->type)))"
    ) not in ordinary_generated:
        raise AssertionError(
            "ordinary Q5 expert-plane generator lost selected-plane stride"
        )
    if (
        "main_config_shape_mul_mat_id_src1_selected_stride_value = 0;"
    ) not in broadcast_generated:
        raise AssertionError(
            "broadcast Q5 expert-plane generator lost zero selected stride"
        )

    source = (
        CATALOG_ROOT / ordinary_definition["source"]
    ).read_text(encoding="utf-8")
    for fragment in (
        "%block_bytes = index.constant 176 : index",
        "%block_f16s = index.constant 88 : index",
        "%scale_offset = index.constant 4 : index",
        "%qh_offset = index.constant 16 : index",
        "%qs_offset = index.constant 48 : index",
        "%expert_nonnegative = scalar.cmpi sge, %expert_i32, "
        "%zero_i32 : i32",
        "%expert_below_nexperts = scalar.cmpi slt, %expert_i32, "
        "%nexperts_i32 : i32",
        "%expert_valid = scalar.andi %expert_nonnegative, "
        "%expert_below_nexperts : i1",
        "%expert_nonnegative_mask = scalar.xori %expert_sign_mask, "
        "%allones_i32 : i32",
        "%expert_valid_mask = scalar.andi %expert_nonnegative_mask, "
        "%expert_below_mask : i32",
        "%expert_safe_i32 = scalar.andi %expert_i32, "
        "%expert_valid_mask : i32",
        "%q_i8 = view.load %src0_i8_view[%q_index] : "
        "view<[%src0_i8_count]xi8, #dense> -> i8",
        "%qh_i8 = view.load %src0_i8_view[%qh_index0] : "
        "view<[%src0_i8_count]xi8, #dense> -> i8",
        "scf.if %expert_valid {",
    ):
        if fragment not in source:
            raise AssertionError(
                f"generic Q5 expert-plane source missing {fragment!r}"
            )
    if "vector.bitfield.extractu" in source:
        raise AssertionError(
            "generic Q5 expert-plane source must use bounded byte loads"
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
    generated = route_generator.generate_route_impl(
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
    expected_storage_guard = (
        "    if (!ggml_backend_hrx_loom_storage_layout_matches("
        'request, src0, "q5_k_expert_down_group4")) {\n'
        "        return ggml_backend_hrx_loom_unsupported("
        "GGML_BACKEND_HRX_LOOM_UNSUPPORTED_LAYOUT);\n"
        "    }"
    )
    if generated.count(expected_storage_guard) != 1:
        raise AssertionError(
            "ordinary Q5 down route must retain exactly one storage guard"
        )
    for fragment in (
        "if (!(shape_src1_selected_rows == shape_src2_nselected)) {",
        'plan->scratch[0].name = "mmid";',
        "plan->main.binding_scratch_index[4] = 1;",
        "plan->prepass_count = 1;",
    ):
        if fragment not in generated:
            raise AssertionError(
                f"Q5 down group4 generator missing {fragment!r}"
            )


def expect_flash_attn_gate_epilogue_safety():
    route_path = CATALOG_ROOT / FLASH_ATTN_GATE_EPILOGUE_ROUTE_PATH
    definitions = loom.load_definitions(CATALOG_ROOT)
    loom.validate_route(
        route_path,
        definitions,
        loom.load_metadata_targets(CATALOG_ROOT),
        loom.load_storage_transform_ids(CATALOG_ROOT),
    )
    route, definition = read_route_and_definition(
        CATALOG_ROOT, FLASH_ATTN_GATE_EPILOGUE_ROUTE_PATH
    )
    predicates = route["match"]["predicates"]
    transient_tensors = next(
        predicate["transients"]
        for predicate in predicates
        if "transients" in predicate
    )
    if "q" not in transient_tensors:
        raise AssertionError(
            "full flash-attention fusion must prove its external q input transient"
        )
    if {
        "same_or_disjoint_storage": ["q", "terminal_dst"]
    } not in predicates:
        raise AssertionError(
            "full flash-attention fusion must reject partial q/output overlap"
        )

    generated = route_generator.generate_route_impl(
        route_path, route, definition
    )
    for fragment in (
        "ggml_backend_hrx_loom_tensor_is_transient_at_indices("
        "request, q, consumed_node_indices, consumed_node_count)",
        "ggml_backend_hrx_loom_tensors_are_same_or_disjoint("
        "q, terminal_dst)",
    ):
        if fragment not in generated:
            raise AssertionError(
                f"full flash-attention safety generator missing {fragment!r}"
            )


def expect_sum_slices_broadcast_weights_layout():
    route_path = CATALOG_ROOT / SUM_SLICES_ROUTE_PATH
    definitions = loom.load_definitions(CATALOG_ROOT)
    loom.validate_route(
        route_path,
        definitions,
        loom.load_metadata_targets(CATALOG_ROOT),
        loom.load_storage_transform_ids(CATALOG_ROOT),
    )
    route, definition = read_route_and_definition(
        CATALOG_ROOT, SUM_SLICES_ROUTE_PATH
    )
    weights = route["tensors"]["weights_dst"]
    if weights["shape"] != [
        "weights_one",
        "weights_selected",
        "weights_tokens",
        "weights_batch",
    ]:
        raise AssertionError(
            "sum-slices must model MUL's [1, selected, tokens, batch] "
            "broadcast-weight input"
        )
    predicates = route["match"]["predicates"]
    for predicate in (
        {"field": "shape.weights_dst.weights_one", "equals": 1},
        {"field": "shape.weights_dst.weights_selected", "equals": 8},
        {"field": "shape.weights_dst.weights_tokens", "in": [1, 512]},
        {
            "field": "shape.weights_dst.weights_tokens",
            "equals": "shape.selected.ntokens",
        },
        {"field": "shape.weights_dst.weights_batch", "equals": 1},
        {"field": "tensor.weights_dst.element_strides.0", "equals": 1},
        {"field": "tensor.weights_dst.element_strides.1", "equals": 1},
        {"field": "tensor.weights_dst.element_strides.2", "equals": 8},
    ):
        if predicate not in predicates:
            raise AssertionError(
                f"sum-slices broadcast-weight route missing {predicate!r}"
            )

    generated = route_generator.generate_route_impl(
        route_path, route, definition
    )
    for fragment in (
        "if (!(shape_weights_dst_weights_one == 1)) {",
        "if (!(shape_weights_dst_weights_selected == 8)) {",
        "if (!((weights_dst->nb[1] / "
        "ggml_type_size(weights_dst->type)) == 1)) {",
        "if (!((weights_dst->nb[2] / "
        "ggml_type_size(weights_dst->type)) == 8)) {",
        "main_config_hrx2_shape_sumslices_router_source_stride_value = "
        "static_cast<int64_t>((weights_dst->nb[2] / "
        "ggml_type_size(weights_dst->type)));",
    ):
        if fragment not in generated:
            raise AssertionError(
                f"sum-slices generator missing broadcast-weight guard "
                f"{fragment!r}"
            )


def expect_ssm_structural_fusion():
    definitions = loom.load_definitions(CATALOG_ROOT)
    targets = loom.load_metadata_targets(CATALOG_ROOT)
    transforms = loom.load_storage_transform_ids(CATALOG_ROOT)
    atomic_path = CATALOG_ROOT / SSM_CONCAT_ROUTE_PATH
    fallback_path = CATALOG_ROOT / SSM_COMPUTE_ROUTE_PATH
    for route_path in (atomic_path, fallback_path):
        loom.validate_route(route_path, definitions, targets, transforms)

    atomic_route = read_json(atomic_path)
    fallback_route = read_json(fallback_path)
    atomic_definition = definitions[
        (
            atomic_path.parent /
            route_schema.require_string(
                atomic_route, "definition", atomic_path
            )
        ).resolve()
    ]
    fallback_definition = definitions[
        (
            fallback_path.parent /
            route_schema.require_string(
                fallback_route, "definition", fallback_path
            )
        ).resolve()
    ]
    prepass = atomic_route["prepasses"][0]
    prepass_definition = definitions[
        (
            atomic_path.parent /
            route_schema.require_string(
                prepass, "definition", atomic_path
            )
        ).resolve()
    ]
    prepass_source = (
        CATALOG_ROOT / prepass_definition["source"]
    ).read_text(encoding="utf-8")
    for fragment in (
        "%new_state = view.load %x_view[%xi]",
        "view.store %new_state, %c_view[%ci]",
    ):
        if fragment not in prepass_source:
            raise AssertionError(
                "atomic SSM prepass no longer performs the CPY state update: "
                f"missing {fragment!r}"
            )
    if atomic_definition["op"] != "GGML_OP_CONCAT":
        raise AssertionError("atomic SSM route must retain its CONCAT owner")
    if prepass_definition["op"] != "GGML_OP_CONCAT":
        raise AssertionError("atomic SSM prepass must retain its CONCAT owner")
    if fallback_definition["op"] != "GGML_OP_SSM_CONV":
        raise AssertionError("SSM fallback must retain its native SSM_CONV owner")

    atomic_match = atomic_route["match"]
    if atomic_match["traversal"] != "tensor_dag":
        raise AssertionError("atomic SSM route must match the full tensor DAG")
    if list(atomic_match["ops"]) != [
        "concat", "ssm", "silu", "state_view", "copy", "cache_view"
    ]:
        raise AssertionError(
            "atomic SSM route must match exact CONCAT/SSM/SiLU/cache-update DAG"
        )
    if atomic_match["consumed_ops"] != ["concat", "copy", "ssm", "silu"]:
        raise AssertionError(
            "atomic SSM route must consume all four compute ops, not metadata views"
        )
    if atomic_match["ops"]["copy"]["tensors"].get("src1") != "cache_target":
        raise AssertionError(
            "atomic SSM route must prove the CPY destination view it writes"
        )
    predicates = atomic_match["predicates"]
    for predicate in (
        {"field": "tensor.state_tail.view_offset", "equals": 16777216},
        {
            "field": "tensor.cache.view_offset",
            "equals": "tensor.cache_target.view_offset",
        },
        {"same_shape": ["cache_target", "cache"]},
        {"same_layout": ["cache_target", "cache"]},
        {"transients": ["window", "state_tail", "ssm_dst"]},
        {"same_or_disjoint_storage": ["ssm_dst", "dst"]},
        {"same_or_disjoint_storage": ["x", "dst"]},
    ):
        if predicate not in predicates:
            raise AssertionError(
                f"atomic SSM route missing safety proof {predicate}"
            )
    for pair in (
        ["state", "cache"],
        ["x", "cache"],
        ["window", "cache"],
        ["cache", "filter"],
        ["cache", "dst"],
    ):
        if {"no_overlap": pair} not in predicates:
            raise AssertionError(
                f"atomic SSM route missing cache alias gate {pair}"
            )
    if len(atomic_route["prepasses"]) != 1 or "cache" in prepass:
        raise AssertionError(
            "atomic SSM route must use one deliberately uncached prepass"
        )
    if prepass["id"] != "concat_window_tail_ssm_prepass":
        raise AssertionError("atomic SSM route changed its exact prepass id")
    cache_stride_bindings = [
        binding
        for binding in prepass["config"]["bindings"]
        if binding["name"] == "hrx2_shape_concat_tail_cache_row_stride"
    ]
    if cache_stride_bindings != [{
        "name": "hrx2_shape_concat_tail_cache_row_stride",
        "type": "i64",
        "source": "shape.x.channels",
    }]:
        raise AssertionError(
            "atomic SSM prepass cache rows must stay packed by channel count"
        )

    fallback_match = fallback_route["match"]
    if fallback_match["traversal"] != "tensor_dag":
        raise AssertionError(
            "SSM fallback must capture its earlier CONCAT producer through tensor_dag"
        )
    if list(fallback_match["ops"]) != ["ssm", "silu", "concat"]:
        raise AssertionError(
            "SSM fallback must capture only the original x producer"
        )
    if fallback_match["consumed_ops"] != ["ssm", "silu"]:
        raise AssertionError(
            "SSM fallback must not consume or redispatch the earlier CONCAT"
        )
    fallback_predicates = fallback_match["predicates"]
    if {"transients": ["ssm_dst"]} not in fallback_predicates:
        raise AssertionError(
            "SSM fallback must prove only its SSM intermediate private"
        )
    for predicate in (
        {"same_or_disjoint_storage": ["ssm_dst", "dst"]},
        {"same_or_disjoint_storage": ["x", "dst"]},
    ):
        if predicate not in fallback_predicates:
            raise AssertionError(
                f"SSM fallback missing alias safety predicate {predicate}"
            )

    atomic_generated = route_generator.generate_route_impl(
        atomic_path, atomic_route, atomic_definition
    )
    for fragment in (
        "const int consumed_node_indices[consumed_node_count] = "
        "{concat_op_index, copy_op_index, ssm_op_index, silu_op_index};",
        "candidate->op != GGML_OP_SSM_CONV",
        "candidate->op != GGML_OP_CPY",
        "const ggml_tensor * cache_target = copy_op->src[1];",
        "ggml_backend_hrx_loom_tensor_dag_edge_matches("
        "cache_target, cache_view_op)",
        "ggml_backend_hrx_loom_tensor_is_transient_at_indices("
        "request, window, consumed_node_indices, consumed_node_count)",
        "ggml_backend_hrx_loom_tensors_are_same_or_disjoint("
        "ssm_dst, dst)",
        "const int64_t main_dispatch_workgroups[3] = "
        "{derived_ssm_workgroups, shape_x_sequences, 1};",
        "ggml_backend_hrx_loom_bind_tensor(request, window, "
        "&plan->main.bindings[0])",
        "ggml_backend_hrx_loom_bind_tensor(request, cache, "
        "&plan->prepasses[0].kernel.bindings[3])",
        "prepass_0_config_hrx2_shape_concat_tail_cache_row_stride_value = "
        "static_cast<int64_t>(shape_x_channels);",
        "const int64_t prepass_0_dispatch_workgroups[3] = "
        "{derived_concat_snapshot_workgroups, 1, 1};",
        "plan->consumed_node_count = 4;",
        "plan->prepass_count = 1;",
    ):
        if fragment not in atomic_generated:
            raise AssertionError(
                f"atomic SSM generator missing {fragment!r}"
            )
    if "cache_scratch_index" in atomic_generated:
        raise AssertionError(
            "uncached atomic SSM prepass unexpectedly emitted scratch caching"
        )
    if "dispatch_owner_node_index" in atomic_generated:
        raise AssertionError(
            "atomic CONCAT-owned SSM route must dispatch immediately"
        )

    fallback_generated = route_generator.generate_route_impl(
        fallback_path, fallback_route, fallback_definition
    )
    for fragment in (
        "candidate->op != GGML_OP_CONCAT",
        "ggml_backend_hrx_loom_tensor_dag_edge_matches(candidate, window)",
        "const int consumed_node_indices[consumed_node_count] = "
        "{ssm_op_index, silu_op_index};",
        "ggml_backend_hrx_loom_tensor_is_transient_at_indices("
        "request, ssm_dst, consumed_node_indices, consumed_node_count)",
        "ggml_backend_hrx_loom_bind_tensor(request, x, "
        "&plan->main.bindings[1])",
        "const int64_t main_dispatch_workgroups[3] = {256, 1, 1};",
        "plan->consumed_node_count = 2;",
        "plan->prepass_count = 0;",
    ):
        if fragment not in fallback_generated:
            raise AssertionError(
                f"SSM fallback generator missing {fragment!r}"
            )

    with tempfile.TemporaryDirectory(prefix="ssm-structural-catalog-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        set_metadata_targets(source_root, ["gfx1151"])
        set_metadata_routes(
            source_root,
            [str(SSM_CONCAT_ROUTE_PATH), str(SSM_COMPUTE_ROUTE_PATH)],
        )
        loom.validate_catalog(source_root)
        entry_ids = {
            entry["id"]
            for entry in catalog.build_entries(source_root, ["gfx1151"])
        }
        if entry_ids != {
            "concat_window_tail_ssm_silu_pp512_wg1024",
            "concat_window_tail_ssm_prepass",
            "ssm_conv_f32_chan_concat_silu_regblock_wg1024",
        }:
            raise AssertionError(
                f"atomic and fallback SSM routes must package exact kernels, got {entry_ids}"
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
    generated = route_generator.generate_route_impl(
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
    router = route_generator.generate_op_router_impl(
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
        "hrx2_shape_fa_d",
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
    sum_rows_impl = route_generator.generate_route_impl(str(SUM_ROWS_ROUTE_PATH), sum_rows_route, sum_rows_definition)
    if "shape_dst_d0 != shape_src0_d0" in sum_rows_impl:
        raise AssertionError("v1 shape captures must not imply cross-tensor equality")

    fusion_route, fusion_definition = read_route_and_definition(CATALOG_ROOT, FUSION_ROUTE_PATH)
    fusion_impl = route_generator.generate_route_impl(str(FUSION_ROUTE_PATH), fusion_route, fusion_definition)
    if "shape_rms_out_d0 != shape_x_d0" not in fusion_impl:
        raise AssertionError("v2 structural shape declarations must enforce cross-tensor equality")

    expect_extended_op_schema()
    expect_fusion_tensor_dag_schema_and_generation()
    expect_fusion_storage_transform_generation()
    expect_q4_topk_topology_disambiguation()
    expect_preexisting_tensor_dag_matchers_unchanged()
    expect_prepass_schema_and_generation()
    expect_q8_packed_loop_coverage()
    expect_q8_decode_small_column_coverage()
    expect_q5_expert_plane_routes()
    expect_q5_down_group4_selected_route()
    expect_flash_attn_gate_epilogue_safety()
    expect_sum_slices_broadcast_weights_layout()
    expect_ssm_structural_fusion()
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

    expect_invalid_full_catalog_mutation(
        "fusion-unknown-transient",
        FUSION_ROUTE_PATH,
        lambda route: route["match"]["predicates"][2].update({"transients": ["missing"]}),
        "tensor missing is not declared in tensors",
    )
    expect_invalid_full_catalog_mutation(
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
        expect_invalid_full_catalog_mutation(
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

import argparse
from pathlib import Path

import onnx
from onnx import helper, numpy_helper
import numpy as np


def replace_first_prelu_with_relu_form(model):
    nodes = list(model.graph.node)
    for idx, node in enumerate(nodes):
        if node.op_type == "PRelu":
            x, slope = node.input
            y = node.output[0]
            neg = y + "_neg"
            relu_pos = y + "_relu_pos"
            relu_neg = y + "_relu_neg"
            scaled_neg = y + "_scaled_neg"
            new_nodes = [
                helper.make_node("Neg", [x], [neg], name=node.name + "_Neg"),
                helper.make_node("Relu", [x], [relu_pos], name=node.name + "_ReluPos"),
                helper.make_node("Relu", [neg], [relu_neg], name=node.name + "_ReluNeg"),
                helper.make_node("Mul", [relu_neg, slope], [scaled_neg], name=node.name + "_ScaleNeg"),
                helper.make_node("Sub", [relu_pos, scaled_neg], [y], name=node.name + "_Sub"),
            ]
            nodes[idx:idx + 1] = new_nodes
            del model.graph.node[:]
            model.graph.node.extend(nodes)
            return node.name
    raise RuntimeError("No PRelu node found")


def reshape_first_prelu_slope(model):
    for node in model.graph.node:
        if node.op_type == "PRelu":
            slope_name = node.input[1]
            for init in model.graph.initializer:
                if init.name == slope_name:
                    arr = numpy_helper.to_array(init).astype(np.float32).reshape(1, -1, 1, 1)
                    init.CopyFrom(numpy_helper.from_array(arr, slope_name))
                    return node.name
            # Slope may be produced by Constant. Rewrite that tensor if found.
            for n in model.graph.node:
                if n.op_type == "Constant" and n.output and n.output[0] == slope_name:
                    for attr in n.attribute:
                        if attr.name == "value":
                            arr = numpy_helper.to_array(attr.t).astype(np.float32).reshape(1, -1, 1, 1)
                            attr.t.CopyFrom(numpy_helper.from_array(arr))
                            return node.name
            raise RuntimeError(f"Slope tensor {slope_name!r} not found")
    raise RuntimeError("No PRelu node found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("relu-form", "reshape-slope"))
    args = parser.parse_args()

    model = onnx.load(args.input)
    if args.mode == "relu-form":
        changed = replace_first_prelu_with_relu_form(model)
    else:
        changed = reshape_first_prelu_slope(model)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"{args.mode}: rewrote {changed}, saved {args.output}")


if __name__ == "__main__":
    main()

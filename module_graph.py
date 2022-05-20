import traceback
from typing import Any, NamedTuple, Optional

import graphviz
import torch
import torch.fx


class TensorMetadata(NamedTuple):
    # TensorMetadata is a structure containing pertinent information
    # about a tensor within a PyTorch program.

    # General Tensor metadata
    shape: torch.Size
    dtype: torch.dtype
    requires_grad: bool
    memory_format: Optional[torch.memory_format]

    def __repr__(self):
        return "×".join(map(str, self.shape))


def _extract_tensor_metadata(result: torch.Tensor) -> TensorMetadata:
    """
    Extract a TensorMetadata NamedTuple describing `result`.
    """
    shape = result.shape
    dtype = result.dtype
    requires_grad = result.requires_grad

    memory_formats = {
        torch.contiguous_format,
        torch.channels_last,
        torch.channels_last_3d,
    }

    memory_format = None

    for query_format in memory_formats:
        if result.is_contiguous(memory_format=query_format):
            memory_format = query_format
            break

    return TensorMetadata(shape, dtype, requires_grad, memory_format)


class ResultProbe(torch.fx.Interpreter):
    def run_node(self, n: torch.fx.Node) -> Any:
        try:
            result = super().run_node(n)
        except Exception:
            traceback.print_exc()
            raise RuntimeError(
                f"ShapeProp error for: node={n.format_node()} with " f"meta={n.meta}"
            )
        find_tensor_in_result = False

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                nonlocal find_tensor_in_result
                find_tensor_in_result = True
                return _extract_tensor_metadata(obj)
            else:
                return obj

        n.meta["result"] = torch.fx.node.map_aggregate(result, extract_tensor_meta)
        n.meta["find_tensor_in_result"] = find_tensor_in_result
        return result


def html_table(*content, **kwargs):
    kwargs_pairs = [f'{k}="{v}"' for k, v in kwargs.items()]
    return f'<table {" ".join(kwargs_pairs)}>' + "\n".join(content) + "</table>"


def html_tr(*content, **kwargs):
    kwargs_pairs = [f'{k}="{v}"' for k, v in kwargs.items()]
    return f'<tr {" ".join(kwargs_pairs)}>' + "\n".join(content) + "</tr>"


def html_td(content, **kwargs):
    kwargs_pairs = [f'{k}="{v}"' for k, v in kwargs.items()]
    return f'<td {" ".join(kwargs_pairs)}>' + str(content) + "</td>"


def call_module_table_html(model, node):
    name = node._pretty_print_target(node.target)
    result = node.meta["result"]

    head = name
    cols = [[html_td(result)]]

    if node.op == "call_module":
        module = model.get_submodule(node.target)
        head = str(module)
        cols[0] = [html_td(name, rowspan=len(cols)), *cols[0]]
    elif node.op == "call_method":
        head = f".{head}"

    head_kwargs = dict(colspan=len(cols[0]))
    if not node.meta["find_tensor_in_result"]:
        head_kwargs["bgcolor"] = "lightgray"

    html = html_table(
        html_tr(html_td(head, **head_kwargs)),
        *[html_tr(*c) for c in cols],
        border=0,
        cellborder=1,
        cellspacing=0,
    )
    return f"<{html}>"


def single_node(model: torch.nn.Module, graph: graphviz.Digraph, node: torch.fx.Node):
    node_label = call_module_table_html(model, node)
    node_kwargs = dict(shape="plaintext")
    graph.node(node.name, node_label, **node_kwargs)
    for in_node in node.all_input_nodes:
        # graph.edge(f"{in_node.name}:name", f"{node.name}:name")
        edge_kwargs = dict()
        if (
            not node.meta["find_tensor_in_result"]
            or not in_node.meta["find_tensor_in_result"]
        ):
            edge_kwargs.update(dict(style="dashed", color="lightgrey"))
        graph.edge(in_node.name, node.name, **edge_kwargs)


def model_graph(model: torch.nn.Module, *args, **kwargs) -> graphviz.Digraph:
    symbolic_traced: torch.fx.GraphModule = torch.fx.symbolic_trace(model)
    ResultProbe(symbolic_traced).run(*args, **kwargs)
    graph = graphviz.Digraph("model", format="svg", node_attr={"shape": "plaintext"})
    for node in symbolic_traced.graph.nodes:
        single_node(model, graph, node)
    return graph


def _test():
    import networks
    from torchvision import models

    model = models.squeezenet1_0()
    graph = model_graph(model, torch.randn(1, 3, 256, 256))
    graph.render(directory="test", view=True)


if __name__ == "__main__":
    _test()

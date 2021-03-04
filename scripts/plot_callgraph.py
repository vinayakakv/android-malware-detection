import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from androguard.misc import AnalyzeAPK

plt.figure(figsize=(10, 5))


def plot_call_graph(cg: nx.classes.multidigraph.MultiDiGraph):
    layout = nx.drawing.nx_agraph.graphviz_layout(cg, prog='dot')
    labels, cm = {}, []
    legend = ''
    node_list = []
    for i, node in enumerate(nx.topological_sort(cg)):
        node_list.append(node)
        labels[node] = i
        cm.append('yellow' if node.is_external() else 'blue')
        legend += '%d, \\texttt{%s %s}\n' % (i, node.class_name.replace('$', '\\$'), node.name)
    plt.axis('off')
    nx.draw_networkx(cg, pos=layout, nodelist=node_list, node_color=cm, labels=labels, alpha=0.6, node_size=500,
                     font_family='serif')
    with open("cg.table", "w") as f:
        f.write(legend)
    plt.tight_layout()
    plt.savefig("cg.pdf", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw FCG of small APKs')
    parser.add_argument(
        '-s', '--source-file',
        help='The APK file to analyze and draw',
        required=True
    )
    args = parser.parse_args()
    if not Path(args.source_file).exists():
        raise FileNotFoundError(f"{args.source_file} doesn't exist")
    a, d, dx = AnalyzeAPK(args.source_file)
    plot_call_graph(dx.get_call_graph())

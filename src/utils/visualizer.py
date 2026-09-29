import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from network.link import CONGESTED, NORMAL, SATURATED


SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#dcdcd6"
NEUTRAL_NODE = "#e8e8e2"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

STATUS_COLORS = {NORMAL: "#0ca30c", CONGESTED: "#fab219", SATURATED: "#d03b3b"}

SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue",
    ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def _style_axes(axis):

    axis.set_facecolor(SURFACE)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=8)

    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(GRID_COLOR)


def _save(figure, path):
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)


def _qpu_color(qpu, num_qpus):
    return SERIES[qpu] if num_qpus <= len(SERIES) else NEUTRAL_NODE


def _qubit_map(partition):
    return {q: qpu for qpu, qubits in partition.items() for q in qubits}


class Visualizer:

    def topology_positions(self, topology):

        if topology.model in ("ring", "mesh", "star") or len(topology.qpu_capacities) <= 3:
            return nx.circular_layout(topology.graph)

        if topology.model == "line":
            return {qpu: (qpu, 0.0) for qpu in topology.qpu_ids}

        if topology.model == "grid":
            columns = math.ceil(math.sqrt(len(topology.qpu_capacities)))
            return {qpu: (qpu % columns, -(qpu // columns)) for qpu in topology.qpu_ids}

        return nx.kamada_kawai_layout(topology.graph)

    def communication_graph(self, graph, path, subtitle):

        figure, axis = plt.subplots(figsize=(9, 7.5), facecolor=SURFACE)
        positions = nx.kamada_kawai_layout(graph) if graph.number_of_edges() else nx.circular_layout(graph)

        weights = [data["weight"] for _, _, data in graph.edges(data=True)]
        peak = max(weights, default=1)

        nx.draw_networkx_edges(
            graph, positions, ax=axis, edge_color=SERIES[0],
            width=[0.6 + 3.4 * w / peak for w in weights], alpha=0.55,
        )
        nx.draw_networkx_nodes(
            graph, positions, ax=axis, node_size=360, node_color=NEUTRAL_NODE,
            edgecolors=TEXT_PRIMARY, linewidths=0.8,
        )
        nx.draw_networkx_labels(graph, positions, ax=axis, font_size=8, font_color=TEXT_PRIMARY)

        if graph.number_of_edges() <= 70:
            nx.draw_networkx_edge_labels(
                graph, positions, ax=axis,
                edge_labels={(u, v): d["weight"] for u, v, d in graph.edges(data=True)},
                font_size=6, font_color=TEXT_SECONDARY,
                bbox={"boxstyle": "round,pad=0.1", "fc": SURFACE, "ec": "none", "alpha": 0.8},
            )

        axis.set_title(
            f"Logical communication graph\n{subtitle}", color=TEXT_PRIMARY, fontsize=11
        )
        axis.axis("off")
        _save(figure, path)

    def partition(self, graph, partition, topology, path, title, info_lines=None,
                  highlight_qubits=None, previous_partition=None):

        highlight = set(highlight_qubits or [])
        centers = self.topology_positions(topology)
        qubit_map = _qubit_map(partition)
        num_qpus = len(partition)

        points = list(centers.values())
        spacing = min(
            (math.dist(points[i], points[j])
             for i in range(len(points)) for j in range(i + 1, len(points))),
            default=1.0,
        )
        radius = spacing * 0.3

        positions = {}

        for qpu, qubits in partition.items():
            cx, cy = centers[qpu]
            for index, qubit in enumerate(sorted(qubits)):
                angle = 2 * math.pi * index / max(len(qubits), 1)
                positions[qubit] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

        figure, axis = plt.subplots(figsize=(10, 8), facecolor=SURFACE)

        nx.draw_networkx_edges(
            topology.graph, centers, ax=axis, edge_color=GRID_COLOR, width=6, alpha=0.9,
        )

        for qpu, (cx, cy) in centers.items():
            axis.add_patch(plt.Circle(
                (cx, cy), radius * 1.4, facecolor="none",
                edgecolor=TEXT_SECONDARY, linewidth=0.8, linestyle="--",
            ))
            axis.text(
                cx, cy + radius * 1.55,
                f"QPU {qpu} ({len(partition[qpu])}/{topology.qpu_capacities[qpu]})",
                ha="center", fontsize=8, color=TEXT_PRIMARY,
            )

        local_edges = [(u, v) for u, v in graph.edges() if qubit_map[u] == qubit_map[v]]
        remote_edges = [(u, v) for u, v in graph.edges() if qubit_map[u] != qubit_map[v]]
        peak = max((d["weight"] for _, _, d in graph.edges(data=True)), default=1)

        nx.draw_networkx_edges(
            graph, positions, edgelist=remote_edges, ax=axis, edge_color=SERIES[1],
            width=[0.4 + 2.0 * graph[u][v]["weight"] / peak for u, v in remote_edges],
            alpha=0.35,
        )
        nx.draw_networkx_edges(
            graph, positions, edgelist=local_edges, ax=axis, edge_color=SERIES[0],
            width=[0.6 + 2.4 * graph[u][v]["weight"] / peak for u, v in local_edges],
            alpha=0.8,
        )

        nodes = sorted(graph.nodes())

        nx.draw_networkx_nodes(
            graph, positions, nodelist=nodes, ax=axis, node_size=300,
            node_color=[_qpu_color(qubit_map[q], num_qpus) for q in nodes],
            edgecolors=[TEXT_PRIMARY if q in highlight else SURFACE for q in nodes],
            linewidths=[2.6 if q in highlight else 1.0 for q in nodes],
        )
        nx.draw_networkx_labels(graph, positions, ax=axis, font_size=7, font_color=TEXT_PRIMARY)

        if previous_partition is not None and highlight:
            old_map = _qubit_map(previous_partition)
            for qubit in sorted(highlight):
                ox, oy = centers[old_map[qubit]]
                nx_, ny_ = positions[qubit]
                axis.annotate(
                    "", xy=(nx_, ny_), xytext=(ox, oy),
                    arrowprops={"arrowstyle": "->", "color": TEXT_PRIMARY, "lw": 1.4},
                )

        handles = [
            Line2D([0], [0], color=SERIES[0], lw=2, label="local interaction (same QPU)"),
            Line2D([0], [0], color=SERIES[1], lw=2, alpha=0.6, label="remote interaction"),
            Line2D([0], [0], color=GRID_COLOR, lw=6, label="physical QPU link"),
        ]

        if highlight:
            handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=NEUTRAL_NODE,
                                  markeredgecolor=TEXT_PRIMARY, markeredgewidth=2.4,
                                  markersize=9, label="moved qubit"))

        axis.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
        axis.set_title(title, color=TEXT_PRIMARY, fontsize=11)
        axis.set_aspect("equal")
        axis.axis("off")

        if info_lines:
            figure.text(0.01, 0.01, "\n".join(info_lines), fontsize=7.5,
                        family="monospace", color=TEXT_SECONDARY, va="bottom")

        _save(figure, path)

    def topology(self, topology, path, title, congestion_counts=None):

        positions = self.topology_positions(topology)
        figure, axis = plt.subplots(figsize=(9, 7.5), facecolor=SURFACE)

        edges = [link.key for link in topology.all_links()]

        if congestion_counts is not None:
            peak = max(congestion_counts.values(), default=0) or 1
            colors = [SEQUENTIAL_BLUE(0.25 + 0.75 * congestion_counts.get(key, 0) / peak)
                      for key in edges]
        else:
            colors = [SERIES[0]] * len(edges)

        widths = [1.0 + 0.7 * topology.links[key].capacity for key in edges]

        nx.draw_networkx_edges(topology.graph, positions, edgelist=edges, ax=axis,
                               edge_color=colors, width=widths)
        nx.draw_networkx_nodes(
            topology.graph, positions, ax=axis, node_size=900, node_color=NEUTRAL_NODE,
            edgecolors=TEXT_PRIMARY, linewidths=1.2,
        )
        nx.draw_networkx_labels(
            topology.graph, positions, ax=axis, font_size=9, font_color=TEXT_PRIMARY,
            labels={q: f"QPU {q}\ncap {topology.qpu_capacities[q]}" for q in topology.qpu_ids},
        )

        labels = {}

        for key in edges:
            link = topology.links[key]
            text = f"lat {link.latency:.0f} | cap {link.capacity}\nF {link.fidelity:.3f} | BP {link.bell_pair_pool}"
            if congestion_counts is not None:
                text += f"\ncongested {congestion_counts.get(key, 0)} win"
            labels[key] = text

        nx.draw_networkx_edge_labels(
            topology.graph, positions, edge_labels=labels, ax=axis, font_size=6.5,
            font_color=TEXT_PRIMARY, rotate=False,
            bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": GRID_COLOR, "alpha": 0.92},
        )

        axis.set_title(title, color=TEXT_PRIMARY, fontsize=11)
        axis.axis("off")
        figure.text(0.01, 0.01, "edge width = link capacity"
                    + ("; edge shade = windows congested" if congestion_counts is not None else ""),
                    fontsize=8, color=TEXT_SECONDARY)
        _save(figure, path)

    def utilization_heatmap(self, link_state_history, threshold, path):

        links = sorted({row["link"] for row in link_state_history},
                       key=lambda label: tuple(int(p) for p in label.split("-")))
        steps = sorted({row["step"] for row in link_state_history})

        if not links or not steps:
            return

        link_index = {link: i for i, link in enumerate(links)}
        step_index = {step: i for i, step in enumerate(steps)}
        grid = np.zeros((len(links), len(steps)))

        for row in link_state_history:
            grid[link_index[row["link"]], step_index[row["step"]]] = row["utilization"]

        figure, axis = plt.subplots(
            figsize=(max(8, 0.16 * len(steps) + 3), max(3, 0.32 * len(links) + 1.6)),
            facecolor=SURFACE,
        )

        image = axis.imshow(grid, aspect="auto", cmap=SEQUENTIAL_BLUE, vmin=0.0, vmax=1.0,
                            interpolation="nearest",
                            extent=[steps[0] - 0.5, steps[-1] + 0.5, len(links) - 0.5, -0.5])

        congested = np.argwhere((grid >= threshold - 1e-9) & (grid < 1.0 - 1e-9))
        saturated = np.argwhere(grid >= 1.0 - 1e-9)

        axis.scatter([steps[c] for _, c in congested], [r for r, _ in congested], s=9,
                     marker="o", facecolor=STATUS_COLORS[CONGESTED], edgecolor="none")
        axis.scatter([steps[c] for _, c in saturated], [r for r, _ in saturated], s=11,
                     marker="s", facecolor=STATUS_COLORS[SATURATED], edgecolor="none")

        axis.set_yticks(range(len(links)))
        axis.set_yticklabels([f"link {link}" for link in links], fontsize=7, color=TEXT_SECONDARY)
        axis.set_xlabel("execution window (step)", color=TEXT_SECONDARY, fontsize=9)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=7)

        colorbar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.01)
        colorbar.set_ticks([0.0, threshold, 1.0])
        colorbar.set_ticklabels(["0%", f"{threshold:.0%} threshold", "100%"])
        colorbar.ax.tick_params(labelsize=7, colors=TEXT_SECONDARY)

        axis.legend(
            handles=[
                Line2D([0], [0], marker="o", color="none", markerfacecolor=STATUS_COLORS[CONGESTED],
                       markersize=6, label="congested (>= threshold)"),
                Line2D([0], [0], marker="s", color="none", markerfacecolor=STATUS_COLORS[SATURATED],
                       markersize=6, label="saturated (load = capacity)"),
            ],
            loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, frameon=False, fontsize=8,
        )
        axis.set_title("Per-link utilization (active load / capacity) by execution window",
                       color=TEXT_PRIMARY, fontsize=11, loc="left")
        _save(figure, path)

    def congestion_timeline(self, records, path, repartition_steps):

        steps = [r.step for r in records]

        figure, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                             facecolor=SURFACE)

        top.plot(steps, [r.congested_links_before_reroute for r in records], color=SERIES[0],
                 lw=2, label="congested links before rerouting")
        top.plot(steps, [r.congested_links_after_reroute for r in records], color=SERIES[1],
                 lw=2, label="congested links after rerouting")
        top.plot(steps, [r.saturated_links_after_reroute for r in records], color=SERIES[2],
                 lw=2, linestyle="--", label="saturated links after rerouting")
        top.set_ylabel("links", color=TEXT_SECONDARY, fontsize=9)
        top.set_title("Link-local congestion per execution window", color=TEXT_PRIMARY,
                      fontsize=11, loc="left")

        bottom.plot(steps, [r.deferred for r in records], color=SERIES[0], lw=2,
                    label="communications deferred this window")
        bottom.plot(steps, [r.queue_length_after for r in records], color=SERIES[1], lw=2,
                    label="queue length after window")
        bottom.plot(steps, [r.prevented_capacity_violations for r in records], color=SERIES[2],
                    lw=2, linestyle="--", label="prevented capacity violations")
        bottom.set_ylabel("count", color=TEXT_SECONDARY, fontsize=9)
        bottom.set_xlabel("execution window (step)", color=TEXT_SECONDARY, fontsize=9)

        for axis in (top, bottom):
            _style_axes(axis)
            for step in repartition_steps:
                axis.axvline(step, color=TEXT_SECONDARY, lw=0.8, linestyle=":")
            axis.legend(frameon=False, fontsize=8, loc="upper right")

        if repartition_steps:
            top.text(repartition_steps[0], top.get_ylim()[1], " accepted repartition",
                     fontsize=7, color=TEXT_SECONDARY, va="top")

        _save(figure, path)

    def rerouting_timeline(self, records, path):

        steps = np.array([r.step for r in records])
        rerouted = np.array([r.selective_reroutes for r in records])
        kept = np.array([r.affected_kept_route for r in records])
        unaffected = np.array([
            r.unaffected_communications if r.affected_communications else 0 for r in records
        ])

        figure, axis = plt.subplots(figsize=(11, 4.2), facecolor=SURFACE)

        axis.bar(steps, rerouted, color=SERIES[0], width=0.8, label="affected: rerouted",
                 edgecolor=SURFACE, linewidth=1)
        axis.bar(steps, kept, bottom=rerouted, color=SERIES[1], width=0.8,
                 label="affected: kept route (best feasible)", edgecolor=SURFACE, linewidth=1)
        axis.bar(steps, unaffected, bottom=rerouted + kept, color=GRID_COLOR, width=0.8,
                 label="unaffected (left untouched)", edgecolor=SURFACE, linewidth=1)

        _style_axes(axis)
        axis.set_xlabel("execution window (step)", color=TEXT_SECONDARY, fontsize=9)
        axis.set_ylabel("communications admitted this window", color=TEXT_SECONDARY, fontsize=9)
        axis.set_title("Selective rerouting on congestion windows", color=TEXT_PRIMARY,
                       fontsize=11, loc="left")
        axis.legend(frameon=False, fontsize=8, loc="upper right")
        _save(figure, path)

    def affected_link_utilization(self, entries, threshold, path):

        if not entries:
            return

        figure, axis = plt.subplots(figsize=(11, 4.5), facecolor=SURFACE)

        for entry in entries:
            axis.plot([entry["step"], entry["step"]],
                      [entry["utilization_before"], entry["utilization_after"]],
                      color=GRID_COLOR, lw=1.6, zorder=1)

        axis.scatter([e["step"] for e in entries], [e["utilization_before"] for e in entries],
                     s=36, facecolor=SURFACE, edgecolor=SERIES[1], linewidth=1.6, zorder=2,
                     label="before selective rerouting")
        axis.scatter([e["step"] for e in entries], [e["utilization_after"] for e in entries],
                     s=36, color=SERIES[0], zorder=3, edgecolor=SURFACE, linewidth=1,
                     label="after selective rerouting")

        axis.axhline(threshold, color=STATUS_COLORS[CONGESTED], lw=1.2, linestyle="--")
        axis.axhline(1.0, color=STATUS_COLORS[SATURATED], lw=1.2, linestyle=":")
        axis.text(axis.get_xlim()[1], threshold, " threshold", fontsize=7,
                  color=TEXT_SECONDARY, va="center")
        axis.text(axis.get_xlim()[1], 1.0, " capacity", fontsize=7,
                  color=TEXT_SECONDARY, va="center")

        _style_axes(axis)
        axis.set_ylim(0, 1.08)
        axis.set_xlabel("execution window (step)", color=TEXT_SECONDARY, fontsize=9)
        axis.set_ylabel("utilization of congested link", color=TEXT_SECONDARY, fontsize=9)
        axis.set_title("Congested-link utilization before and after selective rerouting",
                       color=TEXT_PRIMARY, fontsize=11, loc="left")
        axis.legend(frameon=False, fontsize=8, loc="lower right")
        _save(figure, path)

    def qaoa_vs_classical(self, blocks, path):

        compared = [b for b in blocks if b.qaoa_feasible]

        if not compared:
            return

        figure, axis = plt.subplots(figsize=(6.2, 5.6), facecolor=SURFACE)

        classical = [b.classical_energy for b in compared]
        quantum = [b.qaoa_energy for b in compared]

        low = min(classical + quantum)
        high = max(classical + quantum)

        axis.plot([low, high], [low, high], color=TEXT_SECONDARY, lw=1, linestyle="--")
        axis.scatter(classical, quantum, s=30, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
        axis.text(high, high, "  equal energy", fontsize=7, color=TEXT_SECONDARY, va="bottom",
                  ha="right")

        _style_axes(axis)
        axis.set_xlabel("classical block energy (same QUBO)", color=TEXT_SECONDARY, fontsize=9)
        axis.set_ylabel("best feasible QAOA sample energy", color=TEXT_SECONDARY, fontsize=9)
        axis.set_title("QAOA vs classical baseline per reroute block", color=TEXT_PRIMARY,
                       fontsize=11, loc="left")
        _save(figure, path)

"""Interactive 3D plots of lattices, to check the geometry by eye.

Each lattice gets two side-by-side views (drag to rotate, scroll to zoom):
  left:  the unit cell exactly as written in the file; node numbers match the file.
  right: the merged periodic graph that the GNN sees, tiled 2x2x2 so you can check
         the lattice looks right when cells are stacked. One cell's unique nodes and
         struts are highlighted.

Usage:
    python plot_lattice.py data/samples/sample_lattices.txt
    python plot_lattice.py data/raw/lattices.txt --names cub_Z08.0_E3 hex_Z08.0_E4
"""
import argparse
import itertools
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lattice_gnn.graph import build_periodic_graph, lattice_matrix
from lattice_gnn.parse import read_lattices

STRUT = "#2a78d6"
HIGHLIGHT = "#eb6834"
CELL = "#898781"
TILES = 2


def _segments(starts, ends):
    """x, y, z lists with None gaps, so one trace can draw many separate lines."""
    xyz = [[], [], []]
    for a, b in zip(starts, ends):
        for k in range(3):
            xyz[k] += [a[k], b[k], None]
    return xyz


def _box(L, size=1):
    """Edges of a parallelepiped spanning `size` cells along each lattice vector."""
    corners = np.array(list(itertools.product([0, size], repeat=3)))
    pairs = [(a, b) for a, b in itertools.combinations(range(8), 2)
             if np.abs(corners[a] - corners[b]).sum() == size]
    pts = corners @ L
    return _segments([pts[a] for a, _ in pairs], [pts[b] for _, b in pairs])


def _lines(xyz, color, width, name, dash=None):
    return go.Scatter3d(x=xyz[0], y=xyz[1], z=xyz[2], mode="lines", name=name,
                        line=dict(color=color, width=width, dash=dash), hoverinfo="skip")


def lattice_traces(lat):
    """Traces for one lattice, as (trace, column) pairs, plus a one-line summary."""
    g = build_periodic_graph(lat)
    L = g.lattice
    pos = lat.nodes @ L

    left = [
        _lines(_box(L), CELL, 2, "Unit cell", dash="dash"),
        _lines(_segments(pos[lat.bars[:, 0]], pos[lat.bars[:, 1]]), STRUT, 6, "Bars (file)"),
        go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers+text", name="Nodes (file)",
            marker=dict(size=5, color=STRUT), text=[str(i + 1) for i in range(len(pos))],
            textposition="top center",
            hovertext=[f"node {i + 1}<br>fractional ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
                       for i, p in enumerate(lat.nodes)],
            hoverinfo="text"),
    ]

    shifts = np.array(list(itertools.product(range(TILES), repeat=3)))
    starts = np.concatenate([(g.frac[g.edges[:, 0]] + s) @ L for s in shifts])
    ends = np.concatenate([(g.frac[g.edges[:, 1]] + g.offsets + s) @ L for s in shifts])
    tiled_nodes = np.concatenate([(g.frac + s) @ L for s in shifts])
    degree = np.bincount(g.edges.ravel(), minlength=len(g.frac))

    right = [
        _lines(_box(L, TILES), CELL, 2, f"{TILES}x{TILES}x{TILES} cells", dash="dash"),
        _lines(_segments(starts, ends), STRUT, 4, "Struts (tiled)"),
        go.Scatter3d(x=tiled_nodes[:, 0], y=tiled_nodes[:, 1], z=tiled_nodes[:, 2],
                     mode="markers", name="Nodes (tiled)", marker=dict(size=3, color=STRUT),
                     hoverinfo="skip"),
        _lines(_segments(g.frac[g.edges[:, 0]] @ L, (g.frac[g.edges[:, 1]] + g.offsets) @ L),
               HIGHLIGHT, 9, "Unique struts (one cell)"),
        go.Scatter3d(
            x=g.pos[:, 0], y=g.pos[:, 1], z=g.pos[:, 2], mode="markers+text",
            name="Unique nodes (one cell)", marker=dict(size=7, color=HIGHLIGHT),
            text=[chr(ord("A") + i) if i < 26 else str(i) for i in range(len(g.frac))],
            textposition="top center",
            hovertext=[f"unique node {i}<br>struts: {degree[i]}" for i in range(len(g.frac))],
            hoverinfo="text"),
    ]

    summary = (f"<b>{lat.name}</b>  ·  {len(g.frac)} unique nodes, {len(g.edges)} unique struts  ·  "
               f"Z_avg graph {g.z_avg:.2f} / file {lat.z_avg:.2f}"
               + ("  ·  file flags intersecting bars" if lat.intersecting else ""))
    return [(t, 1) for t in left] + [(t, 2) for t in right], summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="lattice data file")
    parser.add_argument("--names", nargs="+", help="only plot these lattices")
    parser.add_argument("--max", type=int, default=20, help="plot at most this many lattices")
    parser.add_argument("--out", default="plots/lattices.html", help="output HTML file")
    parser.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = parser.parse_args()

    lattices = read_lattices(args.path, strict=False)
    if args.names:
        lattices = [lat for lat in lattices if lat.name in set(args.names)]
    lattices = lattices[:args.max]
    if not lattices:
        raise SystemExit("No matching lattices found.")

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("As written in the file (one unit cell)",
                                        f"What the GNN sees: merged graph, tiled {TILES}x{TILES}x{TILES}"),
                        horizontal_spacing=0.02)
    counts, summaries = [], []
    for n, lat in enumerate(lattices):
        traces, summary = lattice_traces(lat)
        for trace, col in traces:
            trace.visible = n == 0
            trace.legendgroup = trace.name
            fig.add_trace(trace, row=1, col=col)
        counts.append(len(traces))
        summaries.append(summary)

    buttons = []
    for n, lat in enumerate(lattices):
        visible = [i == n for i, c in enumerate(counts) for _ in range(c)]
        buttons.append(dict(label=lat.name, method="update",
                            args=[{"visible": visible}, {"title.text": summaries[n]}]))

    axis = dict(showbackground=False, gridcolor="#e1e0d9", zerolinecolor="#e1e0d9")
    scene = dict(aspectmode="data", camera=dict(eye=dict(x=1.6, y=-1.9, z=1.1)),
                 xaxis=dict(title="x", **axis), yaxis=dict(title="y", **axis),
                 zaxis=dict(title="z", **axis))
    fig.update_layout(
        title=dict(text=summaries[0], x=0.5, y=0.985, font=dict(size=14)),
        scene=scene, scene2=scene,
        paper_bgcolor="#fcfcfb", font=dict(color="#52514e"),
        legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center"),
        updatemenus=[dict(buttons=buttons, x=0.0, xanchor="left", y=1.05, yanchor="bottom",
                          showactive=True)],
        margin=dict(l=10, r=10, t=110, b=40),
    )
    fig.update_annotations(yshift=10)   # subplot titles

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs=True, default_width="100%", default_height="95vh",
                   config={"responsive": True})
    print(f"Wrote {len(lattices)} lattices to {out}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()

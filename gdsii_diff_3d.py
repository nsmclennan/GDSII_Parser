#!/usr/bin/env python3
"""
Sample Usage:
    python gdsii_diff_3d.py file_a.gds file_b.gds [--output diff_3d.html] [--layers 1,2,3] [--extrude 0.5]

Requirements:
    pip install gdspy plotly numpy shapely
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import gdspy
import plotly.graph_objects as go
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union, triangulate
HAS_SHAPELY = True

# Colours
COLOUR_BOTH   = "rgba(59,  130, 246, 0.55)"   # blue  unchanged
COLOUR_A_ONLY = "rgba(239, 68,  68,  0.70)"   # red  only A
COLOUR_B_ONLY = "rgba(34,  197, 94,  0.70)"   # green only B
COLOUR_XOR    = "rgba(249, 115, 22,  0.65)"   # orange XOR region

EDGE_BOTH   = "rgba(59,  130, 246, 0.9)"
EDGE_A_ONLY = "rgba(239, 68,  68,  1.0)"
EDGE_B_ONLY = "rgba(34,  197, 94,  1.0)"
EDGE_XOR    = "rgba(249, 115, 22,  1.0)"


# Helpers

def load_gds(path: str) -> gdspy.GdsLibrary:
    lib = gdspy.GdsLibrary()
    lib.read_gds(path)
    return lib


def flatten_polygons(lib: gdspy.GdsLibrary) -> dict:
    result = defaultdict(list)
    for cell in lib.cells.values():
        flat = cell.get_polygons(by_spec=True)
        for (layer, dtype), polys in flat.items():
            result[(layer, dtype)].extend(polys)
    return dict(result)


def poly_signature(poly: np.ndarray, decimals: int = 4) -> frozenset:
    rounded = np.round(poly, decimals)
    pts = [tuple(p) for p in rounded]
    rotations = [pts[i:] + pts[:i] for i in range(len(pts))]
    return frozenset(map(tuple, min(rotations)))


def classify_polygons(polys_a, polys_b):
    sigs_a = {poly_signature(p): p for p in polys_a}
    sigs_b = {poly_signature(p): p for p in polys_b}
    ka, kb = set(sigs_a), set(sigs_b)
    return (
        [sigs_a[k] for k in ka - kb],   # only_a
        [sigs_b[k] for k in kb - ka],   # only_b
        [sigs_a[k] for k in ka & kb],   # shared
    )


# XOR

def shapely_xor_polys(polys_a, polys_b):
    if not HAS_SHAPELY:
        return []

    def to_union(polys):
        shapes = []
        for p in polys:
            try:
                s = ShapelyPolygon(p)
                if s.is_valid and not s.is_empty:
                    shapes.append(s)
            except Exception:
                pass
        return unary_union(shapes) if shapes else ShapelyPolygon()

    geom_a = to_union(polys_a)
    geom_b = to_union(polys_b)
    xor = geom_a.symmetric_difference(geom_b)

    result = []
    geoms = [xor] if xor.geom_type == "Polygon" else list(xor.geoms)
    for g in geoms:
        if not g.is_empty and g.geom_type == "Polygon":
            coords = np.array(g.exterior.coords)
            result.append(coords)
    return result


# 3D Extrusion

def triangulate_polygon(poly: np.ndarray):
    n = len(poly)
    if n < 3:
        return poly, []
    # Simple fan from vertex 0 — works for convex; for concave use shapely
    if HAS_SHAPELY:
        try:
            s = ShapelyPolygon(poly)
            if not s.is_valid:
                s = s.buffer(0)
            tris = triangulate(s)
            verts = list(poly)
            vert_map = {tuple(np.round(v, 6)): i for i, v in enumerate(verts)}
            triangles = []
            for tri in tris:
                if tri.is_empty:
                    continue
                c = np.array(tri.exterior.coords[:-1])
                idxs = []
                for pt in c:
                    key = tuple(np.round(pt, 6))
                    if key not in vert_map:
                        vert_map[key] = len(verts)
                        verts.append(pt)
                    idxs.append(vert_map[key])
                if len(idxs) == 3:
                    triangles.append(tuple(idxs))
            return np.array(verts), triangles
        except Exception:
            pass
    # Fallback: fan triangulation
    verts = poly
    triangles = [(0, i, i + 1) for i in range(1, n - 2)]
    return verts, triangles


def extrude_polygon_mesh(poly: np.ndarray, z_bot: float, z_top: float):
    poly2d = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
    if len(poly2d) < 3:
        return None

    verts2d, tris = triangulate_polygon(poly2d)
    n = len(verts2d)
    if not tris:
        return None

    # Bottom cap verts: indices 0..n-1  (z = z_bot)
    # Top    cap verts: indices n..2n-1 (z = z_top)
    x_b, y_b = verts2d[:, 0], verts2d[:, 1]
    x_t, y_t = x_b.copy(), y_b.copy()

    x = np.concatenate([x_b, x_t])
    y = np.concatenate([y_b, y_t])
    z = np.concatenate([np.full(n, z_bot), np.full(n, z_top)])

    ii, jj, kk = [], [], []

    # Bottom cap (reversed winding for outward normal)
    for (a, b, c) in tris:
        ii.append(a); jj.append(c); kk.append(b)

    # Top cap
    for (a, b, c) in tris:
        ii.append(a + n); jj.append(b + n); kk.append(c + n)

    # Side walls
    m = len(poly2d)
    for idx in range(m):
        nxt = (idx + 1) % m
        b0, b1 = idx, nxt          # bottom edge
        t0, t1 = idx + n, nxt + n  # top edge
        # Two triangles per quad
        ii += [b0, b0]; jj += [b1, t1]; kk += [t0, b1]

    return x.tolist(), y.tolist(), z.tolist(), ii, jj, kk


def make_mesh_trace(polys, z_bot, z_top, colour, edge_colour, name, legendgroup, showlegend=True):
    all_x, all_y, all_z = [], [], []
    all_i, all_j, all_k = [], [], []
    offset = 0

    for poly in polys:
        result = extrude_polygon_mesh(poly, z_bot, z_top)
        if result is None:
            continue
        x, y, z, ii, jj, kk = result
        all_x.extend(x); all_y.extend(y); all_z.extend(z)
        all_i.extend(v + offset for v in ii)
        all_j.extend(v + offset for v in jj)
        all_k.extend(v + offset for v in kk)
        offset += len(x)

    if not all_x:
        return None

    return go.Mesh3d(
        x=all_x, y=all_y, z=all_z,
        i=all_i, j=all_j, k=all_k,
        color=colour,
        flatshading=True,
        lighting=dict(diffuse=0.7, specular=0.3, ambient=0.4, roughness=0.6),
        lightposition=dict(x=1000, y=2000, z=3000),
        name=name,
        legendgroup=legendgroup,
        showlegend=showlegend,
        hovertemplate=f"<b>{name}</b><br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<br>z: %{{z:.3f}}<extra></extra>",
    )


def make_outline_trace(polys, z_top, edge_colour, name, legendgroup, showlegend=False):
    xs, ys, zs = [], [], []
    for poly in polys:
        p = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
        for pt in p:
            xs.append(pt[0]); ys.append(pt[1]); zs.append(z_top)
        # close loop
        xs.append(p[0][0]); ys.append(p[0][1]); zs.append(z_top)
        xs.append(None); ys.append(None); zs.append(None)

    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=edge_colour, width=2),
        name=name + " (outline)",
        legendgroup=legendgroup,
        showlegend=showlegend,
        hoverinfo="skip",
    )


# Figure building

def build_figure(path_a, path_b, layer_filter, extrude_height):
    print(f"Loading A: {path_a}")
    lib_a = load_gds(path_a)
    print(f"Loading B: {path_b}")
    lib_b = load_gds(path_b)

    polys_a = flatten_polygons(lib_a)
    polys_b = flatten_polygons(lib_b)

    all_keys = sorted(set(polys_a) | set(polys_b))
    if layer_filter:
        all_keys = [k for k in all_keys if k[0] in layer_filter]
    if not all_keys:
        sys.exit("No layers found (check --layers filter).")

    traces = []
    layer_z = {}   # (layer_key) -> z_bot
    Z_GAP = extrude_height * 2.5

    for idx, lk in enumerate(all_keys):
        z_bot = idx * (extrude_height + Z_GAP)
        z_top = z_bot + extrude_height
        layer_z[lk] = (z_bot, z_top)

        a_polys = polys_a.get(lk, [])
        b_polys = polys_b.get(lk, [])
        only_a, only_b, shared = classify_polygons(a_polys, b_polys)
        xor_polys = shapely_xor_polys(a_polys, b_polys)

        label = f"L{lk[0]}/D{lk[1]}"
        print(f"  {label}: {len(shared)} unchanged, {len(only_a)} only-A, "
              f"{len(only_b)} only-B, {len(xor_polys)} XOR polygons")

        first = True  # show legend entry only for first set in a layer

        if shared:
            t = make_mesh_trace(shared, z_bot, z_top, COLOUR_BOTH, EDGE_BOTH,
                                f"{label} — Unchanged", label, showlegend=first)
            if t: traces.append(t); first = False
            traces.append(make_outline_trace(shared, z_top, EDGE_BOTH, f"{label} — Unchanged", label))

        if only_a:
            t = make_mesh_trace(only_a, z_bot, z_top, COLOUR_A_ONLY, EDGE_A_ONLY,
                                f"{label} — Only A (del)", label, showlegend=first)
            if t: traces.append(t); first = False
            traces.append(make_outline_trace(only_a, z_top, EDGE_A_ONLY, f"{label} — Only A", label))

        if only_b:
            t = make_mesh_trace(only_b, z_bot, z_top, COLOUR_B_ONLY, EDGE_B_ONLY,
                                f"{label} — Only B (add)", label, showlegend=first)
            if t: traces.append(t); first = False
            traces.append(make_outline_trace(only_b, z_top, EDGE_B_ONLY, f"{label} — Only B", label))

        if xor_polys:
            t = make_mesh_trace(xor_polys, z_top, z_top + extrude_height * 0.15,
                                COLOUR_XOR, EDGE_XOR,
                                f"{label} — XOR diff", label, showlegend=first)
            if t: traces.append(t)
            traces.append(make_outline_trace(xor_polys, z_top + extrude_height * 0.15,
                                             EDGE_XOR, f"{label} — XOR", label))

    all_polys_flat = [p for lst in list(polys_a.values()) + list(polys_b.values()) for p in lst]
    if all_polys_flat:
        all_pts = np.vstack(all_polys_flat)
        xmin, xmax = all_pts[:, 0].min(), all_pts[:, 0].max()
        ymin, ymax = all_pts[:, 1].min(), all_pts[:, 1].max()
        pad = max(xmax - xmin, ymax - ymin) * 0.08 + 1
    else:
        xmin, xmax, ymin, ymax = -1, 1, -1, 1; pad = 1

    for lk in all_keys:
        z_bot, z_top = layer_z[lk]
        label = f"Layer {lk[0]} / Datatype {lk[1]}"
        # Grid plane
        traces.append(go.Mesh3d(
            x=[xmin - pad, xmax + pad, xmax + pad, xmin - pad],
            y=[ymin - pad, ymin - pad, ymax + pad, ymax + pad],
            z=[z_bot, z_bot, z_bot, z_bot],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color="rgba(255,255,255,0.03)",
            flatshading=True,
            showlegend=False,
            hoverinfo="skip",
            name=label + " plane",
        ))
        # Layer label at corner
        traces.append(go.Scatter3d(
            x=[xmin - pad * 0.5], y=[ymin - pad * 0.5], z=[(z_bot + z_top) / 2],
            mode="text",
            text=[label],
            textfont=dict(size=11, color="rgba(200,200,200,0.8)"),
            showlegend=False,
            hoverinfo="skip",
        ))

    name_a = Path(path_a).name
    name_b = Path(path_b).name

    layout = go.Layout(
        title=dict(
            text=f"<b>GDSII 3D Diff</b>  ·  <span style='color:#f87171'>{name_a}</span>"
                 f"  vs  <span style='color:#4ade80'>{name_b}</span>",
            font=dict(size=18, color="#f9fafb"),
            x=0.5,
        ),
        paper_bgcolor="#0f172a",
        scene=dict(
            bgcolor="#0f172a",
            xaxis=dict(title="X (µm)", color="#94a3b8", gridcolor="#1e293b",
                       showbackground=True, backgroundcolor="#0f172a"),
            yaxis=dict(title="Y (µm)", color="#94a3b8", gridcolor="#1e293b",
                       showbackground=True, backgroundcolor="#0f172a"),
            zaxis=dict(title="Layer (Z)", color="#94a3b8", gridcolor="#1e293b",
                       showbackground=True, backgroundcolor="#0f172a",
                       tickvals=[((z_bot + z_top) / 2)
                                 for z_bot, z_top in layer_z.values()],
                       ticktext=[f"L{k[0]}/D{k[1]}" for k in all_keys]),
            camera=dict(eye=dict(x=1.5, y=-1.8, z=1.4)),
            aspectmode="data",
        ),
        legend=dict(
            bgcolor="rgba(15,23,42,0.85)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color="#cbd5e1", size=11),
            itemsizing="constant",
            tracegroupgap=6,
        ),
        annotations=[
            dict(
                text=(
                    "<b>Legend</b><br>"
                    "<span style='color:#3b82f6'>■</span> Unchanged (A∩B)<br>"
                    "<span style='color:#ef4444'>■</span> Only in A (deleted)<br>"
                    "<span style='color:#22c55e'>■</span> Only in B (added)<br>"
                    "<span style='color:#f97316'>■</span> XOR difference<br><br>"
                    f"<span style='color:#f87171'>A: {name_a}</span><br>"
                    f"<span style='color:#4ade80'>B: {name_b}</span>"
                ),
                align="left",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.01, y=0.98,
                bgcolor="rgba(15,23,42,0.85)",
                bordercolor="#334155",
                borderwidth=1,
                font=dict(size=12, color="#cbd5e1"),
            )
        ],
        margin=dict(l=0, r=0, t=60, b=0),
        hoverlabel=dict(bgcolor="#1e293b", font_color="#f1f5f9", font_size=12),
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig

def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D GDSII diff viewer (outputs HTML).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gdsii_diff_3d.py design_v1.gds design_v2.gds --output my_diff.html
  
        """)
    parser.add_argument("file_a", help="First GDSII file")
    parser.add_argument("file_b", help="Second GDSII file")
    parser.add_argument("--output", default="gdsii_diff_3d.html",
                        help="Output HTML path (default: gdsii_diff_3d.html)")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer numbers, e.g. 1,2,10")
    parser.add_argument("--extrude", type=float, default=1.0,
                        help="Extrusion height per layer slab (default: 1.0 µm)")
    args = parser.parse_args()

    layer_filter = None
    if args.layers:
        try:
            layer_filter = [int(x.strip()) for x in args.layers.split(",")]
        except ValueError:
            sys.exit("--layers must be comma-separated integers, e.g.  --layers 1,2,10")

    fig = build_figure(args.file_a, args.file_b, layer_filter, args.extrude)
    fig.write_html(args.output, include_plotlyjs="cdn", full_html=True)
    print(f"\nSaved: {args.output}  (open in any browser)")


if __name__ == "__main__":
    main()

"""
Script to compare two GDSII files through a geometric XOR algorithm.

gdsii_compare.py input_a.gdsii input_b.gdsii --output 3d_model.html
"""

import argparse
import sys
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

# Packages for polygons and gds reading
import numpy as np
import gdspy
import plotly.graph_objects as go
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union, triangulate

COLOUR_BOTH = "rgba(59, 130, 246, 0.55)" # blue
COLOUR_A_ONLY = "rgba(239, 68, 68, 0.70)" # red
COLOUR_B_ONLY = "rgba(34, 197, 94, 0.70)" # green
COLOUR_XOR = "rgba(249, 115, 22, 0.65)" # orange

EDGE_BOTH = "rgba(59, 130, 246, 0.9)"
EDGE_A_ONLY = "rgba(239, 68, 68, 1.0)"
EDGE_B_ONLY = "rgba(34, 197, 94, 1.0)"
EDGE_XOR = "rgba(249, 115, 22, 1.0)"

LEGEND_HTML = (
    "<b>Legend</b><br>"
    "<span style='color:" + COLOUR_BOTH + "'></span> Unchanged (A B)<br>"
    "<span style='color:" + COLOUR_A_ONLY + "'></span> Only in A (deleted)<br>"
    "<span style='color:" + COLOUR_B_ONLY + "'></span> Only in B (added)<br>"
    "<span style='color:" + COLOUR_XOR + "'></span> XOR difference<br><br>"
)


def poly_bbox(poly):
    # Obtain bounding coordinates of the polygon box
    xs = poly[:, 0]
    ys = poly[:, 1]

    return {
        "xmin": float(xs.min()), "xmax": float(xs.max()),
        "ymin": float(ys.min()), "ymax": float(ys.max()),
    }


def poly_area(poly):
    # obtain area of polygon
    s = ShapelyPolygon(poly)
    if not s.is_valid:
        s = s.buffer(0)
    return float(s.area)

def poly_entries(polys):
    entries = []
    for p in polys:
        entries.append({
            "vertices": np.round(p, 6).tolist(),
            "bbox": poly_bbox(p),
            "area": round(poly_area(p), 6),
        })
    return entries

def poly_signature(poly, decimals = 4):
    # To obtain polygon signature, obtain rotations of the polygon
    rounded = np.round(poly, decimals)
    pts = [tuple(p) for p in rounded]
    rotations = [pts[i:] + pts[:i] for i in range(len(pts))]

    return frozenset(map(tuple, min(rotations)))

def classify_polygons(polys_a, polys_b):
    # Build polygon signature dicts and convert to sets for fast comparison
    sigs_a = {poly_signature(p): p for p in polys_a}
    sigs_b = {poly_signature(p): p for p in polys_b}
    ka, kb = set(sigs_a), set(sigs_b)

    return (
        [sigs_a[k] for k in ka - kb],
        [sigs_b[k] for k in kb - ka],
        [sigs_a[k] for k in ka & kb],
    )

def flatten_polygons(lib):
    # Remove cell references for polygon comparison
    result = defaultdict(list)
    for cell in lib.cells.values():
        flat = cell.get_polygons(by_spec=True)
        for (layer, dtype), polys in flat.items():
            result[(layer, dtype)].extend(polys)

    return dict(result)


def to_union(polys):
    # obtain union of the shapes
    shapes = []
    for p in polys:
        s = ShapelyPolygon(p)
        if s.is_valid and not s.is_empty:
            shapes.append(s)
    return unary_union(shapes) if shapes else ShapelyPolygon()

def shapely_xor_polys(polys_a, polys_b):
    # Perform xor on the union of the polygons
    geom_a = to_union(polys_a)
    geom_b = to_union(polys_b)
    xor = geom_a.symmetric_difference(geom_b)

    # Convert format to readable dict
    result = []
    geoms = [xor] if xor.geom_type == "Polygon" else list(xor.geoms)
    for g in geoms:
        if not g.is_empty and g.geom_type == "Polygon":
            coords = np.array(g.exterior.coords)
            result.append(coords)
    return result


def triangulate_polygon(poly):
    # convert polygons to triangles for extrusion
    n = len(poly)
    if n < 3:
        return poly, []
    s = ShapelyPolygon(poly)
    if not s.is_valid:
        s = s.buffer(0)
    verts = list(poly)
    vert_map = {tuple(np.round(v, 6)): i for i, v in enumerate(verts)}
    triangles = []
    for tri in triangulate(s):
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

# Extrusion

def extrude_polygon_mesh(poly, z_bot, z_top):
    # Process the input
    poly2d = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
    if len(poly2d) < 3:
        return None

    verts2d, tris = triangulate_polygon(poly2d)
    n = len(verts2d)
    if not tris:
        return None

    # Duplicate vertices for top and bottom edges
    x_b, y_b = verts2d[:, 0], verts2d[:, 1]
    x_t, y_t = x_b.copy(), y_b.copy()

    x = np.concatenate([x_b, x_t])
    y = np.concatenate([y_b, y_t])
    z = np.concatenate([np.full(n, z_bot), np.full(n, z_top)])

    ii, jj, kk = [], [], []

    # Build bottom surface
    for (a, b, c) in tris:
        ii.append(a); jj.append(c); kk.append(b)

    # Build top surface
    for (a, b, c) in tris:
        ii.append(a + n); jj.append(b + n); kk.append(c + n)

    # Side walls
    m = len(poly2d)
    for idx in range(m):
        nxt = (idx + 1) % m
        b0, b1 = idx, nxt
        t0, t1 = idx + n, nxt + n
        ii += [b0, b0]; jj += [b1, t1]; kk += [t0, b1]

    return x.tolist(), y.tolist(), z.tolist(), ii, jj, kk


def make_mesh_trace(polys, z_bot, z_top, colour, edge_colour, name, legendgroup, showlegend=True):
    all_x, all_y, all_z = [], [], []
    all_i, all_j, all_k = [], [], []
    offset = 0
    # Convert each 2D polygon into 3D through extrusion based on configured heights.
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

    # Build 3D mesh object
    return go.Mesh3d(
        x=all_x, y=all_y, z=all_z,
        i=all_i, j=all_j, k=all_k,
        color=colour,
        flatshading=True,
        name=name,
        legendgroup=legendgroup,
        showlegend=showlegend,
    )

def make_outline_trace(polys, z_top, edge_colour, name, legendgroup, showlegend=False):
    # Outline trace for 3D model output.
    xs, ys, zs = [], [], []
    for poly in polys:
        p = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
        for pt in p:
            xs.append(pt[0]); ys.append(pt[1]); zs.append(z_top)
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


# Reports

def write_text_report(report, out_path):
    # Summary details
    lines = []
    lines.append("=" * 70)
    lines.append("GDSII 3D DIFF REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated : {report['generated_at']}")
    lines.append(f"File A    : {report['file_a']}")
    lines.append(f"File B    : {report['file_b']}")
    lines.append("")
    s = report["summary"]
    lines.append("Overall Summary")
    lines.append("-" * 70)
    lines.append(f"  Layers compared : {s['total_layers_compared']}")
    lines.append(f"  Unchanged polys : {s['total_unchanged']}")
    lines.append(f"  Only in A (del) : {s['total_only_a']}")
    lines.append(f"  Only in B (add) : {s['total_only_b']}")
    lines.append(f"  XOR regions     : {s['total_xor_regions']}")
    lines.append("")

    # Details for each layer
    for layer in report["layers"]:
        c = layer["counts"]
        lines.append("-" * 70)
        lines.append(f"Layer {layer['layer']} / Datatype {layer['datatype']}")
        lines.append("-" * 70)
        lines.append(f"  Unchanged : {c['unchanged']}")
        lines.append(f"  Only A    : {c['only_a']}")
        lines.append(f"  Only B    : {c['only_b']}")
        lines.append(f"  XOR       : {c['xor_regions']}")

        if layer["only_a"]:
            lines.append("  Only-in-A polygons (deleted):")
            for i, e in enumerate(layer["only_a"], 1):
                bb = e["bbox"]
                lines.append(
                    f"    [{i}] area={e['area']:.4f} "
                    f"bbox=({bb['xmin']:.3f},{bb['ymin']:.3f}) -> "
                    f"({bb['xmax']:.3f},{bb['ymax']:.3f})"
                )

        if layer["only_b"]:
            lines.append("  Only-in-B polygons (added):")
            for i, e in enumerate(layer["only_b"], 1):
                bb = e["bbox"]
                lines.append(
                    f"    [{i}] area={e['area']:.4f} "
                    f"bbox=({bb['xmin']:.3f},{bb['ymin']:.3f}) -> "
                    f"({bb['xmax']:.3f},{bb['ymax']:.3f})"
                )

        if layer["xor"]:
            lines.append("  XOR difference regions:")
            for i, e in enumerate(layer["xor"], 1):
                bb = e["bbox"]
                lines.append(
                    f"    [{i}] area={e['area']:.4f} "
                    f"bbox=({bb['xmin']:.3f},{bb['ymin']:.3f}) -> "
                    f"({bb['xmax']:.3f},{bb['ymax']:.3f})"
                )

        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved text diff report: {out_path}")

def build_diff_report(path_a, path_b, all_keys, polys_a, polys_b, layer_results):
    # Build hash for report output formats
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_a": Path(path_a).name,
        "file_b": Path(path_b).name,
        "layers": [],
        "summary": {
            "total_layers_compared": len(all_keys),
            "total_unchanged": 0,
            "total_only_a": 0,
            "total_only_b": 0,
            "total_xor_regions": 0,
        },
    }

    for lk in all_keys:
        res = layer_results[lk]
        only_a, only_b, shared, xor_polys = (
            res["only_a"], res["only_b"], res["shared"], res["xor"]
        )



        layer_entry = {
            "layer": int(lk[0]),
            "datatype": int(lk[1]),
            "counts": {
                "unchanged": len(shared),
                "only_a": len(only_a),
                "only_b": len(only_b),
                "xor_regions": len(xor_polys),
            },
            "only_a": poly_entries(only_a),
            "only_b": poly_entries(only_b),
            "xor": poly_entries(xor_polys),
        }

        report["layers"].append(layer_entry)

        report["summary"]["total_unchanged"] += len(shared)
        report["summary"]["total_only_a"] += len(only_a)
        report["summary"]["total_only_b"] += len(only_b)
        report["summary"]["total_xor_regions"] += len(xor_polys)

    return report


def write_json_report(report, out_path):
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved JSON diff report: {out_path}")



def build_figure(path_a, path_b, layer_filter, extrude_height):
    print(f"Loading A: {path_a}")
    lib_a = gdspy.GdsLibrary()
    lib_a.read_gds(path_a)
    print(f"Loading B: {path_b}")
    lib_b = gdspy.GdsLibrary()
    lib_b.read_gds(path_b)

    polys_a = flatten_polygons(lib_a)
    polys_b = flatten_polygons(lib_b)

    # get working layers
    all_keys = sorted(set(polys_a) | set(polys_b))
    if layer_filter:
        all_keys = [k for k in all_keys if k[0] in layer_filter]
    if not all_keys:
        sys.exit("No layers found (check --layers filter).")

    traces = []
    layer_z = {}
    layer_results = {}
    Z_GAP = extrude_height * 2.5

    for idx, lk in enumerate(all_keys):
        # layer height limits
        z_bot = idx * (extrude_height + Z_GAP)
        z_top = z_bot + extrude_height
        layer_z[lk] = (z_bot, z_top)

        a_polys = polys_a.get(lk, [])
        b_polys = polys_b.get(lk, [])
        only_a, only_b, shared = classify_polygons(a_polys, b_polys)
        xor_polys = shapely_xor_polys(a_polys, b_polys)

        layer_results[lk] = {
            "only_a": only_a, "only_b": only_b,
            "shared": shared, "xor": xor_polys,
        }

        label = f"L{lk[0]}/D{lk[1]}"
        print(f"  {label}: {len(shared)} unchanged, {len(only_a)} only-A, "
              f"{len(only_b)} only-B, {len(xor_polys)} XOR polygons")

        first = True



        if shared:
            t = make_mesh_trace(shared, z_bot, z_top, COLOUR_BOTH, EDGE_BOTH,
                                 f"{label} - Unchanged", label, showlegend=first)
            if t: 
                traces.append(t)
                first = False
            traces.append(make_outline_trace(shared, z_top, EDGE_BOTH, f"{label} - Unchanged", label))

        if only_a:
            t = make_mesh_trace(only_a, z_bot, z_top, COLOUR_A_ONLY, EDGE_A_ONLY,
                                 f"{label} - Only A (del)", label, showlegend=first)
            if t: 
                traces.append(t)
                first = False
            traces.append(make_outline_trace(only_a, z_top, EDGE_A_ONLY, f"{label} - Only A", label))

        if only_b:
            t = make_mesh_trace(only_b, z_bot, z_top, COLOUR_B_ONLY, EDGE_B_ONLY,
                                 f"{label} - Only B (add)", label, showlegend=first)
            if t: 
                traces.append(t)
                first = False
            traces.append(make_outline_trace(only_b, z_top, EDGE_B_ONLY, f"{label} - Only B", label))

        if xor_polys:
            t = make_mesh_trace(xor_polys, z_top, z_top + extrude_height * 0.15,
                                 COLOUR_XOR, EDGE_XOR,
                                 f"{label} - XOR diff", label, showlegend=first)
            if t: 
                traces.append(t)
            traces.append(make_outline_trace(xor_polys, z_top + extrude_height * 0.15,
                                              EDGE_XOR, f"{label} - XOR", label))

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
        traces.append(go.Scatter3d(
            x=[xmin - pad * 0.5], y=[ymin - pad * 0.5], z=[(z_bot + z_top) / 2],
            mode="text",
            text=[label],
            textfont=dict(size=11),
            showlegend=False,
            hoverinfo="skip",
        ))

    name_a = Path(path_a).name
    name_b = Path(path_b).name

    # 3D model output structure based on plotly.
    layout = go.Layout(
        title=dict(
            text=f"GDSII 3D Diff  {name_a} vs {name_b}",
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="X (microm)"),
            yaxis=dict(title="Y (microm)"),
            zaxis=dict(
                title="Layer (Z)",
                tickvals=[((z_bot + z_top) / 2) for z_bot, z_top in layer_z.values()],
                ticktext=[f"L{k[0]}/D{k[1]}" for k in all_keys],
            ),
            camera=dict(eye=dict(x=1.5, y=-1.8, z=1.4)),
            aspectmode="data",
        ),
        legend=dict(
            itemsizing="constant",
            tracegroupgap=6,
        ),
        annotations=[
            dict(
                text=(
                    LEGEND_HTML + 
                    "<span style='color:" + COLOUR_A_ONLY + f"'>A: {name_a}</span><br>"
                    "<span style='color:" + COLOUR_B_ONLY + f"'>B: {name_b}</span>"
                ),
                align="left",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.01, y=0.98,
            ),
        ],
        margin=dict(l=0, r=0, t=60, b=0),
    )

    fig = go.Figure(data=traces, layout=layout)

    diff_report = build_diff_report(path_a, path_b, all_keys, polys_a, polys_b, layer_results)

    return fig, diff_report

# Start of main

parser = argparse.ArgumentParser(description="GDSII Comparison tool with HTML, JSON, and text reports.")
parser.add_argument("file_a")
parser.add_argument("file_b")
parser.add_argument("--output", default="gdsii_diff_3d.html")
parser.add_argument("--json", dest="json_output", default=None)
parser.add_argument("--text", dest="text_output", default=None)
parser.add_argument("--layers", default=None)
parser.add_argument("--extrude", type=float, default=1.0)
parser.add_argument("--no-json", action="store_true")
parser.add_argument("--no-text", action="store_true")
args = parser.parse_args()

# Filter layers
layer_filter = None
if args.layers:
    try:
        layer_filter = [int(x.strip()) for x in args.layers.split(",")]
    except ValueError:
        exit("--layers must be comma-separated integers, e.g. --layers 1,2,10")

# Run diff and get result
fig, diff_report = build_figure(args.file_a, args.file_b, layer_filter, args.extrude)
fig.write_html(args.output, include_plotlyjs="cdn", full_html=True)
print(f"\nSaved HTML: {args.output}")

output_stem = Path(args.output).with_suffix("")

# Output additional reports as selected
if not args.no_json:
    json_path = args.json_output or f"{output_stem}.json"
    write_json_report(diff_report, json_path)

if not args.no_text:
    text_path = args.text_output or f"{output_stem}.txt"
    write_text_report(diff_report, text_path)
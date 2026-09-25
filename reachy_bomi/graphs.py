#!/usr/bin/env python3
"""All matplotlib plotting for the grasp pipeline."""

import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import display

GRAPHS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "graphs")


def _save_fig(fig, name: str) -> None:
    """Save fig to GRAPHS_DIR as <timestamp>_<slug>.png, creating the dir if needed."""
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    slug = re.sub(r"[^\w.-]+", "_", name).strip("_")
    filename = f"{time.strftime('%Y%m%d_%H%M%S')}_{slug}.png"
    fig.savefig(os.path.join(GRAPHS_DIR, filename), dpi=150, bbox_inches="tight")


def show_point_cloud(point_cloud: np.ndarray, class_name: str) -> None:
    """Non-blocking 3D scatter view of a point cloud (Reachy coords,
    [m]), colored by height. Used for the intermediate pipeline stages
    (raw capture, distortion-corrected, isolated, final) -- see the
    calls in reachy_detection.build_object_point_cloud."""
    if point_cloud.shape[0] == 0:
        print("[WARN] Point cloud is empty, nothing to show")
        return

    fig = plt.figure(f"Point cloud - {class_name}")
    ax = fig.add_subplot(projection="3d")
    xs, ys, zs = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    scatter = ax.scatter(xs, ys, zs, c=zs, cmap="viridis", s=4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    fig.colorbar(scatter, ax=ax, shrink=0.6, label="z (m)")

    # Equal aspect ratio on all three axes
    ranges = point_cloud.max(axis=0) - point_cloud.min(axis=0)
    half_range = max(ranges.max() / 2.0, 1e-3)
    mid = point_cloud.mean(axis=0)
    ax.set_xlim(mid[0] - half_range, mid[0] + half_range)
    ax.set_ylim(mid[1] - half_range, mid[1] + half_range)
    ax.set_zlim(mid[2] - half_range, mid[2] + half_range)

    _save_fig(fig, f"point_cloud_{class_name}")

    display.place_figure(fig)
    plt.show(block=False)
    plt.pause(0.001)


def show_grasp_plan(geometry, plan) -> None:
    """Non-blocking 3D scatter of the object's point cloud with the planned 
    pre-grasp/grasp/lift EE positions marked on top."""
    point_cloud = geometry.point_cloud
    if point_cloud.shape[0] == 0:
        print("[WARN] Point cloud is empty, nothing to show")
        return

    fig = plt.figure(f"Grasp plan - {geometry.class_name} ({plan.arm_name})")
    ax = fig.add_subplot(projection="3d")

    xs, ys, zs = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    ax.scatter(xs, ys, zs, c=zs, cmap="viridis", s=4, label="object point cloud")

    pregrasp_pos = plan.pregrasp_matrix[:3, 3]
    grasp_pos = plan.grasp_matrix[:3, 3]
    lift_pos = plan.lift_matrix[:3, 3]

    ax.plot(*zip(pregrasp_pos, grasp_pos), c="blue", linestyle="--", linewidth=1.5, label="approach path")
    ax.scatter(*pregrasp_pos, c="orange", marker="^", s=80, label="pre-grasp EE")
    ax.scatter(*grasp_pos, c="red", marker="X", s=100, label="grasp EE")
    ax.scatter(*lift_pos, c="green", marker="^", s=80, label="lift EE")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(loc="upper left", fontsize=8)

    # Equal aspect ratio over the point cloud + waypoints
    all_points = np.vstack([point_cloud, [pregrasp_pos, grasp_pos, lift_pos]])
    ranges = all_points.max(axis=0) - all_points.min(axis=0)
    half_range = max(ranges.max() / 2.0, 1e-3)
    mid = all_points.mean(axis=0)
    ax.set_xlim(mid[0] - half_range, mid[0] + half_range)
    ax.set_ylim(mid[1] - half_range, mid[1] + half_range)
    ax.set_zlim(mid[2] - half_range, mid[2] + half_range)

    _save_fig(fig, f"grasp_plan_{geometry.class_name}_{plan.arm_name}")

    display.place_figure(fig)
    plt.show(block=False)
    # Several short pauses, not one: gives the window manager more chances
    # to map/raise the window before cv2.imshow + waitKey resume and starve
    #  matplotlib's event loop. 
    for _ in range(10):
        plt.pause(0.1)


def _table_plane_basis(table_normal: np.ndarray) -> tuple:
    """(normal, basis_u, basis_v): orthonormal in-plane basis of the table,
    basis_u ~ world X (same construction as reachy_selection's, kept local so
    this module stays dependency-free)."""
    normal = table_normal / np.linalg.norm(table_normal)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(reference, normal)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    basis_u = reference - normal * np.dot(reference, normal)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    return normal, basis_u, basis_v


def _set_equal_aspect(ax, points: np.ndarray, margin: float = 0.05) -> None:
    """Same half-range on the three axes, centred on the middle of the
    points' bounding box (not their mean: a dense cloud would drag the
    centre onto itself and push the far waypoints out of frame)."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    half_range = max((hi - lo).max() / 2.0, 1e-3) + margin
    mid = (lo + hi) / 2.0
    ax.set_xlim(mid[0] - half_range, mid[0] + half_range)
    ax.set_ylim(mid[1] - half_range, mid[1] + half_range)
    ax.set_zlim(mid[2] - half_range, mid[2] + half_range)


def _look_at_motion_plane(ax, start: np.ndarray, end: np.ndarray, fallback_direction: np.ndarray) -> None:
    """Orthographic view of the vertical plane through start and end, from
    the robot's point of view: the camera sits on the horizontal normal of
    that plane, on the side of the robot (world origin), so left and right on
    screen are the robot's own. fallback_direction is used when start and end
    share the same horizontal position."""
    direction = (end - start)[:2]
    if np.linalg.norm(direction) < 1e-3:
        direction = fallback_direction[:2]
    if np.linalg.norm(direction) < 1e-3:
        direction = np.array([1.0, 0.0])
    direction /= np.linalg.norm(direction)
    camera = np.array([direction[1], -direction[0]])
    # the robot stands at the origin: put the camera on its side of the plane
    if np.dot(camera, ((start + end) / 2.0)[:2]) > 0:
        camera = -camera
    ax.set_proj_type("ortho")
    ax.view_init(elev=20.0, azim=float(np.degrees(np.arctan2(camera[1], camera[0]))))
    ax.set_box_aspect((1.0, 1.0, 1.0))
    # the axis seen end-on collapses to a point: drop its ticks and label
    depth_axis = ax.xaxis if abs(camera[0]) >= abs(camera[1]) else ax.yaxis
    depth_axis.set_ticks([])
    depth_axis.set_label_text("")


def show_grasp_and_place_plan(geometry, plan, place_plan, target_point: np.ndarray,
                              cell_size_m: float = 0.12) -> None:
    """Non-blocking 3D view of the whole pick-and-place: the object's point
    cloud, the grasp waypoints (pre-grasp, grasp, lift), the place waypoints
    (transit at the lift height, place, retreat) and the table cell the user
    selected, drawn on the table plane around target_point so it is visible
    that the object is released right over it. place_plan is the GraspPlan
    returned by reachy_grasp.plan_place (lift/grasp/pregrasp matrices reused
    as transit/place/retreat); cell_size_m is reachy_selection's
    PLACE_GRID_CELL_SIZE_M (not imported to avoid a dependency cycle)."""
    point_cloud = geometry.point_cloud
    if point_cloud.shape[0] == 0:
        print("[WARN] Point cloud is empty, nothing to show")
        return

    fig = plt.figure(f"Grasp and place plan - {geometry.class_name} ({plan.arm_name})", figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")

    xs, ys, zs = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    ax.scatter(xs, ys, zs, c=zs, cmap="viridis", s=4, label="object point cloud")

    # Grasp: pre-grasp -> grasp -> lift
    pregrasp_pos = plan.pregrasp_matrix[:3, 3]
    grasp_pos = plan.grasp_matrix[:3, 3]
    lift_pos = plan.lift_matrix[:3, 3]
    # Place: lift -> transit (carry at the lift height) -> place -> retreat
    transit_pos = place_plan.lift_matrix[:3, 3]
    place_pos = place_plan.grasp_matrix[:3, 3]
    retreat_pos = place_plan.pregrasp_matrix[:3, 3]

    ax.plot(*zip(pregrasp_pos, grasp_pos), c="blue", linestyle="--", linewidth=1.5, label="approach")
    ax.plot(*zip(grasp_pos, lift_pos), c="green", linestyle="--", linewidth=1.5, label="lift")
    ax.plot(*zip(lift_pos, transit_pos), c="purple", linestyle=":", linewidth=1.8, label="carry (lift height)")
    ax.plot(*zip(transit_pos, place_pos), c="brown", linestyle="--", linewidth=1.5, label="lower")
    ax.plot(*zip(place_pos, retreat_pos), c="gray", linestyle="--", linewidth=1.5, label="retreat")

    ax.scatter(*pregrasp_pos, c="orange", marker="^", s=80, label="pre-grasp EE")
    ax.scatter(*grasp_pos, c="red", marker="X", s=100, label="grasp EE")
    ax.scatter(*lift_pos, c="green", marker="^", s=80, label="lift EE")
    ax.scatter(*transit_pos, c="purple", marker="s", s=70, label="transit EE")
    ax.scatter(*place_pos, c="brown", marker="X", s=100, label="place EE")
    ax.scatter(*retreat_pos, c="gray", marker="s", s=70, label="retreat EE")

    # Selected cell: a cell_size_m square of the placement grid around
    # target_point, drawn at the table level (the lowest point of the object
    # cloud along the table normal, i.e. where the object stands). A table
    # normal tilted more than 20 deg from vertical is a bad plane fit (same
    # check as reachy_grasp): fall back to world up, or the cell floats.
    normal = geometry.table_normal if geometry.table_normal is not None else np.array([0.0, 0.0, 1.0])
    if normal[2] / np.linalg.norm(normal) < np.cos(np.radians(20.0)):
        normal = np.array([0.0, 0.0, 1.0])
    normal, basis_u, basis_v = _table_plane_basis(normal)
    table_level = float(np.min(point_cloud @ normal))
    cell_center = target_point - normal * (float(np.dot(target_point, normal)) - table_level)
    half = cell_size_m / 2.0
    corners = np.array([
        cell_center + basis_u * du + basis_v * dv
        for du, dv in ((-half, -half), (half, -half), (half, half), (-half, half))
    ])
    cell = Poly3DCollection([corners], facecolor="gold", edgecolor="darkorange", alpha=0.35, linewidth=1.5)
    ax.add_collection3d(cell)
    ax.plot(*zip(*np.vstack([corners, corners[:1]])), c="darkorange", linewidth=1.5, label="selected cell")
    ax.scatter(*cell_center, c="darkorange", marker="+", s=120)
    # Vertical from the place pose down to the cell, to show it is released right over it
    ax.plot(*zip(place_pos, cell_center), c="darkorange", linestyle=":", linewidth=1.0)

    # labelpad: in the orthographic front view the tick labels sit right
    # where the default label goes, and the label lands on top of them
    ax.set_xlabel("x (m)", labelpad=12)
    ax.set_ylabel("y (m)", labelpad=12)
    ax.set_zlabel("z (m)", labelpad=12)
    ax.legend(loc="upper left", fontsize=7)

    _set_equal_aspect(ax, np.vstack([
        point_cloud, corners,
        [pregrasp_pos, grasp_pos, lift_pos, transit_pos, place_pos, retreat_pos],
    ]))

    # Front view of the motion: the carry runs in the vertical plane through
    # grasp and place, so look at that plane head-on (camera on its horizontal
    # normal, zero elevation, no perspective) and every waypoint height reads
    # directly off the z axis, with the object moving left to right.
    _look_at_motion_plane(ax, grasp_pos, place_pos, fallback_direction=grasp_pos - pregrasp_pos)

    _save_fig(fig, f"grasp_place_plan_{geometry.class_name}_{plan.arm_name}")

    display.place_figure(fig)
    plt.show(block=False)
    for _ in range(10):
        plt.pause(0.1)

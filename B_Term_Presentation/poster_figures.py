import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


# ============================================================
# Cleaner poster-style dynamical systems figures
#
# Design choices:
# - Figure 1 explicitly highlights one trajectory
# - Other figures rely mostly on the vector field geometry
# - Legends are boxed and overlaid inside the plots
# - Attractors drawn as black points
# - Unstable example uses a saddle instead of a source
# - Figure 1 draws the trajectory as a curve with arrows along its length
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "mathtext.default": "it",
})


# ----------------------------
# Colors
# ----------------------------
TRAJ_COLOR = "#1f4e79"
FIELD_COLOR = "#4f5b68"
AXIS_COLOR = "#9aa0a6"
BOUNDARY_COLOR = "#4b5563"
POINT_COLOR = "#111827"
POINT_TEXT_COLOR = "white"
BASIN_COLORS = ["#c7dcff", "#ffd98a"]


# ----------------------------
# Shared helpers
# ----------------------------
def savefig(fig, name):
    fig.savefig(f"{name}.png", bbox_inches="tight", transparent=False)
    fig.savefig(f"{name}.pdf", bbox_inches="tight", transparent=False)



def clean_axes(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.axhline(0, color=AXIS_COLOR, lw=0.8, alpha=0.85, zorder=0)
    ax.axvline(0, color=AXIS_COLOR, lw=0.8, alpha=0.85, zorder=0)

    for spine in ax.spines.values():
        spine.set_color("#d1d5db")
        spine.set_linewidth(0.8)

    ax.tick_params(
        colors="#6b7280",
        labelbottom=False,
        labelleft=False,
        length=3,
        width=0.7,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")



def integrate_field(field, x0, y0, dt=0.03, steps=500):
    xs = [x0]
    ys = [y0]
    x, y = x0, y0
    for _ in range(steps):
        k1x, k1y = field(x, y)
        k2x, k2y = field(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
        k3x, k3y = field(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y)
        k4x, k4y = field(x + dt * k3x, y + dt * k3y)
        x += (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        y += (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def add_labeled_circle(ax, x, y, label, size=260, fontsize=10):
    ax.scatter([x], [y], s=size, color=POINT_COLOR, zorder=8)
    if not label:
        return
    ax.text(
        x,
        y,
        label,
        color=POINT_TEXT_COLOR,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        zorder=9,
    )



def add_unstable_x(ax, x, y, size=70, lw=1.6):
    ax.scatter([x], [y], s=size, marker="x", color=POINT_COLOR, linewidths=lw, zorder=8)



def add_overlay_legend(ax, handles, loc="upper right", fontsize=17):
    legend = ax.legend(
        handles=handles,
        loc=loc,
        frameon=True,
        fontsize=fontsize,
        fancybox=True,
        framealpha=0.92,
        borderpad=0.55,
        labelspacing=0.6,
        handlelength=2.4,
        handletextpad=0.8,
    )
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#d1d5db")
    frame.set_linewidth(0.8)
    return legend



def build_field_grid(field, xlim, ylim, nx=240, ny=240):
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(x, y)
    U, V = field(X, Y)
    return X, Y, U, V



def draw_stream_field(
    ax,
    field,
    xlim,
    ylim,
    density=1.0,
    linewidth=1.55,
    arrowsize=1.45,
    maxlength=0.55,
    minlength=0.05,
    color=FIELD_COLOR,
    alpha=0.94,
    zorder=1,
    nx=240,
    ny=240,
    integration_direction="forward",
    **_,
):
    X, Y, U, V = build_field_grid(field, xlim, ylim, nx=nx, ny=ny)
    stream = ax.streamplot(
        X,
        Y,
        U,
        V,
        density=density,
        linewidth=linewidth,
        arrowsize=arrowsize,
        maxlength=maxlength,
        minlength=minlength,
        color=color,
        zorder=zorder,
        broken_streamlines=True,
        integration_direction=integration_direction,
    )
    stream.lines.set_alpha(alpha)
    stream.arrows.set_alpha(alpha)



def draw_quiver_field(
    ax,
    field,
    xlim,
    ylim,
    density=1.0,
    linewidth=1.55,
    arrowsize=1.45,
    maxlength=0.55,
    color=FIELD_COLOR,
    alpha=0.94,
    zorder=1,
    nx=None,
    ny=None,
    scale_by_magnitude=False,
    min_length_fraction=0.28,
    **_,
):
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]
    max_span = max(xspan, yspan)
    base_count = max(7, int(round(14 * density)))
    nx = nx or max(7, int(round(base_count * xspan / max_span)))
    ny = ny or max(7, int(round(base_count * yspan / max_span)))

    X, Y, U, V = build_field_grid(field, xlim, ylim, nx=nx, ny=ny)
    mag = np.hypot(U, V)
    mask = mag > 1e-8
    U_norm = np.zeros_like(U, dtype=float)
    V_norm = np.zeros_like(V, dtype=float)
    if scale_by_magnitude:
        ref_mag = np.percentile(mag[mask], 90)
        length_factor = np.zeros_like(mag, dtype=float)
        length_factor[mask] = np.clip(mag[mask] / max(ref_mag, 1e-8), 0.0, 1.0)
        length_factor[mask] = min_length_fraction + (1.0 - min_length_fraction) * length_factor[mask]
    else:
        length_factor = np.ones_like(mag, dtype=float)

    U_norm[mask] = maxlength * length_factor[mask] * U[mask] / mag[mask]
    V_norm[mask] = maxlength * length_factor[mask] * V[mask] / mag[mask]

    ax.quiver(
        X,
        Y,
        U_norm,
        V_norm,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=color,
        width=0.0028 * linewidth,
        headwidth=3.4 * arrowsize,
        headlength=4.6 * arrowsize,
        headaxislength=4.0 * arrowsize,
        pivot="mid",
        alpha=alpha,
        zorder=zorder,
    )



def draw_segmented_flow_field(
    ax,
    field,
    xlim,
    ylim,
    density=0.9,
    linewidth=1.48,
    arrowsize=1.35,
    maxlength=0.24,
    color=FIELD_COLOR,
    alpha=0.94,
    zorder=1,
    nx=None,
    ny=None,
    steps=8,
    seed_margin=0.055,
    **_,
):
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]
    max_span = max(xspan, yspan)
    base_count = max(7, int(round(16 * density)))
    nx = nx or max(7, int(round(base_count * xspan / max_span)))
    ny = ny or max(7, int(round(base_count * yspan / max_span)))

    x_pad = seed_margin * xspan
    y_pad = seed_margin * yspan
    seed_x = np.linspace(xlim[0] + x_pad, xlim[1] - x_pad, nx)
    seed_y = np.linspace(ylim[0] + y_pad, ylim[1] - y_pad, ny)

    segments = []
    tip_x = []
    tip_y = []
    tip_u = []
    tip_v = []
    ds = maxlength / max(steps, 1)

    for x0 in seed_x:
        for y0 in seed_y:
            x, y = x0, y0
            xs = [x]
            ys = [y]

            for _ in range(steps):
                dx, dy = field(x, y)
                mag = np.hypot(dx, dy)
                if mag < 1e-8:
                    break
                x_next = x + ds * dx / mag
                y_next = y + ds * dy / mag
                if not (xlim[0] <= x_next <= xlim[1] and ylim[0] <= y_next <= ylim[1]):
                    break
                xs.append(x_next)
                ys.append(y_next)
                x, y = x_next, y_next

            if len(xs) < 3:
                continue

            xs = np.array(xs)
            ys = np.array(ys)
            segments.append(np.column_stack([xs, ys]))

            dx = xs[-1] - xs[-2]
            dy = ys[-1] - ys[-2]
            mag = np.hypot(dx, dy)
            if mag < 1e-8:
                continue

            arrow_length = max(0.045, 0.28 * maxlength) * arrowsize
            tip_x.append(xs[-1])
            tip_y.append(ys[-1])
            tip_u.append(arrow_length * dx / mag)
            tip_v.append(arrow_length * dy / mag)

    lines = LineCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
        capstyle="round",
        zorder=zorder,
    )
    ax.add_collection(lines)

    ax.quiver(
        tip_x,
        tip_y,
        tip_u,
        tip_v,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=color,
        width=0.0028 * linewidth,
        headwidth=3.4 * arrowsize,
        headlength=4.6 * arrowsize,
        headaxislength=4.0 * arrowsize,
        pivot="tip",
        alpha=alpha,
        zorder=zorder + 0.1,
    )



def draw_field(ax, field, xlim, ylim, field_style="stream", **kwargs):
    if field_style == "stream":
        draw_stream_field(ax, field, xlim, ylim, **kwargs)
    elif field_style == "quiver":
        draw_quiver_field(ax, field, xlim, ylim, **kwargs)
    elif field_style == "segmented":
        draw_segmented_flow_field(ax, field, xlim, ylim, **kwargs)
    else:
        raise ValueError("field_style must be 'stream', 'quiver', or 'segmented'")



def trim_path_to_radius(xs, ys, stop_radius):
    radius = np.sqrt(xs**2 + ys**2)
    stop_candidates = np.flatnonzero(radius < stop_radius)
    if not len(stop_candidates):
        return xs, ys
    stop = max(stop_candidates[0], 2)
    return xs[:stop], ys[:stop]



def draw_arrowed_trajectory(
    ax,
    field,
    x0,
    y0,
    color,
    dt=0.03,
    steps=500,
    stop_radius=0.24,
    linewidth=2.2,
    arrow_count=12,
    arrow_length=0.26,
    arrow_scale=12,
):
    """
    Draw one integrated trajectory as a smooth curve with small arrows
    tangent to the path.
    """
    xs, ys = integrate_field(field, x0, y0, dt=dt, steps=steps)
    plot_xs, plot_ys = trim_path_to_radius(xs, ys, stop_radius)

    path = np.column_stack([plot_xs, plot_ys])
    ds = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
    arc = np.concatenate([[0.0], np.cumsum(ds)])
    if arc[-1] <= 0:
        return xs, ys

    ax.plot(
        plot_xs,
        plot_ys,
        color=color,
        lw=linewidth,
        alpha=1.0,
        solid_capstyle="round",
        zorder=6,
    )

    def point_at(distance):
        j = np.searchsorted(arc, distance)
        j = np.clip(j, 1, len(path) - 1)
        t = (distance - arc[j - 1]) / max(arc[j] - arc[j - 1], 1e-8)
        return path[j - 1] + t * (path[j] - path[j - 1])

    start = min(0.35, 0.12 * arc[-1])
    end = max(start, arc[-1] - 0.18)
    arrow_distances = np.linspace(start, end, arrow_count)

    for distance in arrow_distances:
        local_length = min(arrow_length, max(0.10, 0.42 * (arc[-1] - distance)))
        p0 = point_at(max(distance - 0.5 * local_length, 0.0))
        p1 = point_at(min(distance + 0.5 * local_length, arc[-1]))
        if np.linalg.norm(p1 - p0) < 0.04:
            continue
        arrow = FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=arrow_scale,
            lw=linewidth,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=7,
        )
        ax.add_patch(arrow)

    return xs, ys


# ============================================================
# Figure 1: One highlighted trajectory to an attractor
# ============================================================
def stable_field(x, y):
    r2 = x * x + y * y
    dx = -0.17 * x - 0.95 * y - 0.032 * r2 * x
    dy = 0.95 * x - 0.17 * y - 0.032 * r2 * y
    return dx, dy



def make_trajectory_figure(show_legend=True, field_style="stream", name_suffix=""):
    xlim = (-4.2, 4.2)
    ylim = (-4.2, 4.2)
    is_segmented = field_style == "segmented"

    fig, ax = plt.subplots(figsize=(6.1, 6.1))

    draw_field(
        ax,
        stable_field,
        xlim,
        ylim,
        field_style=field_style,
        density=0.78 if is_segmented else 1.15,
        linewidth=2.0 if is_segmented else 1.6,
        arrowsize=1.55 if is_segmented else 1.45,
        maxlength=0.36 if is_segmented else 0.34,
        alpha=0.94,
        zorder=1,
    )

    xs, ys = draw_arrowed_trajectory(
        ax,
        stable_field,
        3.4,
        2.5,
        color=TRAJ_COLOR,
        dt=0.03,
        steps=780,
        stop_radius=0.20,
        linewidth=2.05,
        arrow_count=13,
        arrow_length=0.28,
        arrow_scale=11,
    )
    ax.scatter(xs[0], ys[0], s=26, color=POINT_COLOR, zorder=7)

    add_labeled_circle(ax, 0, 0, r"$A^*$", size=300, fontsize=9)

    clean_axes(ax, xlim, ylim)

    if show_legend:
        handles = [
            Line2D([0], [0], color=TRAJ_COLOR, lw=2.3, label="Highlighted trajectory"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=POINT_COLOR, markeredgecolor=POINT_COLOR,
                   markersize=5.8, label="Initial state"),
            Line2D([0], [0], marker=r"$A^*$", color=POINT_COLOR,
                   lw=0, markersize=10.5, label="Attractor"),
        ]
        add_overlay_legend(ax, handles, loc="lower right")

    fig.tight_layout()
    savefig(fig, f"figure_1_trajectory{name_suffix}")
    return fig


# ============================================================
# Figure 2: Contraction
# Field contracts toward a shared trajectory
# ============================================================
def contraction_target(x):
    return 0.45 * np.sin(0.75 * x)



def contraction_field(x, y):
    target = contraction_target(x)
    target_slope = 0.45 * 0.75 * np.cos(0.75 * x)
    dx = np.ones_like(y, dtype=float)
    dy = target_slope * dx - 1.35 * (y - target)
    return dx, dy



def make_contraction_figure(show_legend=True, field_style="stream", name_suffix=""):
    xlim = (-3.6, 3.6)
    ylim = (-2.8, 2.8)

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    draw_field(
        ax,
        contraction_field,
        xlim,
        ylim,
        field_style=field_style,
        density=1.05,
        linewidth=1.6,
        arrowsize=1.45,
        maxlength=0.65,
        alpha=0.94,
        zorder=1,
    )

    curve_x = np.linspace(*xlim, 500)
    curve_y = contraction_target(curve_x)
    ax.plot(curve_x, curve_y, color=TRAJ_COLOR, lw=2.1, zorder=4)

    clean_axes(ax, xlim, ylim)

    if show_legend:
        handles = [
            Line2D([0], [0], color=TRAJ_COLOR, lw=2.1, label="Shared trajectory"),
        ]
        add_overlay_legend(ax, handles, loc="upper right")

    fig.tight_layout()
    savefig(fig, f"figure_2_contraction{name_suffix}")
    return fig


# ============================================================
# Figure 3: Unstable saddle
# ============================================================
def saddle_field(x, y):
    # Classic saddle: stable along y-axis, unstable along x-axis
    dx = 0.95 * x
    dy = -0.85 * y
    return dx, dy



def make_unstable_figure(show_legend=True, field_style="stream", name_suffix=""):
    xlim = (-2.8, 2.8)
    ylim = (-2.8, 2.8)

    fig, ax = plt.subplots(figsize=(6.1, 6.1))
    draw_field(
        ax,
        saddle_field,
        xlim,
        ylim,
        field_style=field_style,
        density=0.95,
        linewidth=1.55,
        arrowsize=1.45,
        maxlength=0.7,
        alpha=0.94,
        zorder=1,
    )

    add_unstable_x(ax, 0, 0, size=70, lw=1.6)

    clean_axes(ax, xlim, ylim)

    if show_legend:
        handles = [
            Line2D([0], [0], marker="x", color=POINT_COLOR,
                   lw=0, markersize=7.0, label="Unstable point"),
        ]
        add_overlay_legend(ax, handles, loc="upper right")

    fig.tight_layout()
    savefig(fig, f"figure_3_unstable_saddle{name_suffix}")
    return fig


# ============================================================
# Figure 4: Basins of attraction
# Mostly field + basin shading, no explicit trajectory curves
# ============================================================
def basin_field(x, y):
    dx = x - x**3
    dy = -y
    return dx, dy



def make_basin_figure(show_legend=True, field_style="stream", name_suffix=""):
    xx = np.linspace(-2.0, 2.0, 500)
    yy = np.linspace(-1.8, 1.8, 400)
    XX, YY = np.meshgrid(xx, yy)
    basin = np.where(XX < 0, 0, 1)

    fig, ax = plt.subplots(figsize=(6.8, 5.4))

    ax.contourf(
        XX,
        YY,
        basin,
        levels=[-0.5, 0.5, 1.5],
        colors=BASIN_COLORS,
        alpha=0.94,
        zorder=0,
    )

    xlim = (-2.0, 2.0)
    ylim = (-1.8, 1.8)
    is_segmented = field_style == "segmented"
    field_maxlength = 0.24 if field_style == "quiver" else 0.30

    draw_field(
        ax,
        basin_field,
        xlim,
        ylim,
        field_style=field_style,
        density=0.68 if is_segmented else 0.9,
        linewidth=1.9 if is_segmented else 1.48,
        arrowsize=1.5 if is_segmented else 1.35,
        maxlength=field_maxlength,
        minlength=0.04,
        alpha=0.94,
        zorder=1,
        scale_by_magnitude=field_style == "quiver",
        min_length_fraction=0.22,
    )

    ax.axvline(0, linestyle="--", linewidth=1.45, color=BOUNDARY_COLOR, alpha=0.9, zorder=3)

    add_labeled_circle(ax, -1, 0, r"$A_1^*$", size=360, fontsize=8.5)
    add_labeled_circle(ax, 1, 0, r"$A_2^*$", size=360, fontsize=8.5)
    add_unstable_x(ax, 0, 0, size=68, lw=1.6)

    clean_axes(ax, xlim, ylim)

    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=POINT_COLOR, markeredgecolor=POINT_COLOR,
                   markersize=7.5, label="Attractor"),
            Line2D([0], [0], marker="x", color=POINT_COLOR,
                   lw=0, markersize=7.0, label="Unstable point"),
            Line2D([0], [0], color=BOUNDARY_COLOR, lw=1.45, ls="--", label="Basin boundary"),
        ]
        add_overlay_legend(ax, handles, loc="upper center")

    fig.tight_layout()
    savefig(fig, f"figure_4_basins{name_suffix}")
    return fig


if __name__ == "__main__":
    for field_style, name_suffix in [("stream", ""), ("quiver", "_quiver")]:
        make_trajectory_figure(show_legend=True, field_style=field_style, name_suffix=name_suffix)
        make_contraction_figure(show_legend=True, field_style=field_style, name_suffix=name_suffix)
        make_unstable_figure(show_legend=True, field_style=field_style, name_suffix=name_suffix)
        make_basin_figure(show_legend=True, field_style=field_style, name_suffix=name_suffix)
    make_trajectory_figure(show_legend=True, field_style="segmented", name_suffix="_segmented")
    make_basin_figure(show_legend=True, field_style="segmented", name_suffix="_segmented")
    plt.show()

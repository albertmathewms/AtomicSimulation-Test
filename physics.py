from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Stores the most recent simulation metrics (KE, PE, forces, trajectories)
last_results = {}


def distpairs(p, rc):
    d = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = (np.sum((d**2), axis=2)) ** 0.5
    neighbours = (dist <= rc) & (dist != 0)

    return (d, dist, neighbours)


def compute_forces_and_pe(p, rc, k, r0):
    """
    Computes attraction and repulsion forces using the original force formula,
    along with the corresponding shifted potential energy.
    """
    d, dist, neighbours = distpairs(p, rc)

    # Force Formula
    magnitude = ((-k * ((dist) - r0)) * neighbours) / (dist + 1e-12)
    unit_vector = d / (dist[:, :, np.newaxis] + 1e-12)

    # Split into repulsive (0 < r < r0) and attractive (r0 <= r <= rc) regimes
    rep_mask = neighbours & (dist < r0)
    att_mask = neighbours & (dist >= r0)

    rep_mag = magnitude * rep_mask
    att_mag = magnitude * att_mask

    f_rep = np.sum((rep_mag[:, :, np.newaxis]) * unit_vector, axis=1)
    f_att = np.sum((att_mag[:, :, np.newaxis]) * unit_vector, axis=1)
    f_total = f_rep + f_att

    # Exact potential energy U(r) = k*(r - rc) - k*r0*ln(r / rc) for F(r) = -k*(r - r0)/r
    safe_dist = np.where(neighbours, dist, rc)
    pair_pe = np.where(neighbours, k * (safe_dist - rc) - k * r0 * np.log(safe_dist / rc), 0.0)

    # Count each unique pair (i < j) once using upper triangle mask
    upper = np.triu(np.ones((p.shape[0], p.shape[0]), dtype=bool), k=1)
    pe_rep = np.sum(pair_pe * (rep_mask & upper))
    pe_att = np.sum(pair_pe * (att_mask & upper))
    pe_total = pe_rep + pe_att

    # Cumulative scalar magnitudes of repulsive and attractive forces
    f_rep_scalar = np.sum(np.abs(rep_mag) * upper)
    f_att_scalar = np.sum(np.abs(att_mag) * upper)

    return f_total, f_rep, f_att, pe_total, pe_rep, pe_att, f_rep_scalar, f_att_scalar


def pair_forces_loop(p, rc, k, r0):
    d, dist, neighbours = distpairs(p, rc)
    f = np.zeros_like(p)
    magnitude = ((-k * ((dist) - r0)) * neighbours) / (dist + 1e-12)
    unit_vector = d / (dist[:, :, np.newaxis] + 1e-12)
    force = (magnitude[:, :, np.newaxis]) * unit_vector
    f = np.sum(force, axis=1)

    return (f)


def walls_reflect(p, v, l):
    for dim in range(3):
        hitp = p[:, dim] < 0
        v[hitp, dim] *= -1
        p[hitp, dim] = 0.0

        hitn = p[:, dim] > l
        v[hitn, dim] *= -1
        p[hitn, dim] = l

    return (p, v)


def step_once(p, v, m, dt, l, rc, k, r0):
    # Original time integration preserved unchanged
    f = pair_forces_loop(p, rc, k, r0)
    a = f / m
    v += a * dt
    p += v * dt
    p, v = walls_reflect(p, v, l)

    return (p, v)


def plot_energy_and_forces(results, rc, k, r0):
    """
    Plots Kinetic Energy, Potential Energy (Total, Repulsive, Attractive),
    Repulsive vs. Attractive Force magnitudes, and pairwise F(r) / U(r) curves.
    """
    time_axis = results["time"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.canvas.manager.set_window_title("Energy & Attraction/Repulsion Analysis")
    fig.suptitle("Particle Simulation: Attraction/Repulsion Forces & Energy", fontsize=13, fontweight="bold")

    # 1. Kinetic & Total Mechanical Energy
    ax1 = axes[0, 0]
    ax1.plot(time_axis, results["ke"], label="Kinetic Energy (KE)", color="#1f77b4", linewidth=1.8)
    ax1.plot(time_axis, results["total_e"], label="Total Energy (KE + PE)", color="#2ca02c", linestyle="--", linewidth=1.4)
    ax1.set_title("Kinetic & Total Mechanical Energy vs. Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Energy")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Potential Energy Breakdown
    ax2 = axes[0, 1]
    ax2.plot(time_axis, results["pe"], label="Total Potential Energy (PE)", color="#9467bd", linewidth=2.0)
    ax2.plot(time_axis, results["pe_rep"], label="Repulsive PE (r < r0)", color="#d62728", alpha=0.85, linewidth=1.4)
    ax2.plot(time_axis, results["pe_att"], label="Attractive PE (r0 ≤ r ≤ rc)", color="#ff7f0e", alpha=0.85, linewidth=1.4)
    ax2.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax2.set_title("Potential Energy Components vs. Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Potential Energy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Repulsive vs. Attractive Force Magnitudes
    ax3 = axes[1, 0]
    ax3.plot(time_axis, results["f_rep_mag"], label="Repulsive Force |F_rep| (r < r0)", color="#d62728", linewidth=1.6)
    ax3.plot(time_axis, results["f_att_mag"], label="Attractive Force |F_att| (r0 ≤ r ≤ rc)", color="#17becf", linewidth=1.6)
    ax3.set_title("Repulsive vs. Attractive Force Magnitudes Over Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Cumulative Force Magnitude")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Pairwise Force F(r) and Potential U(r) vs. Separation r
    ax4 = axes[1, 1]
    r_vals = np.linspace(0.1 * r0, rc * 1.15, 300)
    active = r_vals <= rc
    f_curve = np.where(active, (-k * (r_vals - r0)) / (r_vals + 1e-12), 0.0)
    u_curve = np.where(active, k * (r_vals - rc) - k * r0 * np.log(r_vals / rc), 0.0)

    ax4.plot(r_vals, f_curve, label="Pair Force F(r) [+ Repel / - Attract]", color="#d62728", linewidth=1.8)
    ax4.plot(r_vals, u_curve, label="Pair Potential U(r)", color="#9467bd", linestyle="--", linewidth=1.8)
    ax4.axvline(r0, color="green", linestyle=":", linewidth=1.4, label=f"Equilibrium r0 = {r0}")
    ax4.axvline(rc, color="gray", linestyle=":", linewidth=1.4, label=f"Cutoff rc = {rc}")
    ax4.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax4.set_title("Pairwise Force F(r) & Potential U(r) vs. Distance r")
    ax4.set_xlabel("Inter-particle Distance r")
    ax4.set_ylabel("Force / Potential")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8.5)

    plt.tight_layout()
    plt.savefig("energy_and_forces.png", dpi=150)
    plt.show(block=False)
    plt.pause(0.1)


def run_simulation(p, v, m, t, dt, l, rc, k, r0, show_plots=True):
    phistory = np.zeros((t, p.shape[0], 3))
    ke_hist = np.zeros(t)
    pe_hist = np.zeros(t)
    pe_rep_hist = np.zeros(t)
    pe_att_hist = np.zeros(t)
    f_rep_hist = np.zeros(t)
    f_att_hist = np.zeros(t)

    for step in range(t):
        p, v = step_once(p, v, m, dt, l, rc, k, r0)
        _, _, _, pe_tot, pe_rep, pe_att, fr_s, fa_s = compute_forces_and_pe(p, rc, k, r0)

        phistory[step] = p
        ke_hist[step] = 0.5 * np.sum(m * (v ** 2))
        pe_hist[step] = pe_tot
        pe_rep_hist[step] = pe_rep
        pe_att_hist[step] = pe_att
        f_rep_hist[step] = fr_s
        f_att_hist[step] = fa_s

    last_results.clear()
    last_results.update({
        "phistory": phistory,
        "time": np.arange(t) * dt,
        "ke": ke_hist,
        "pe": pe_hist,
        "pe_rep": pe_rep_hist,
        "pe_att": pe_att_hist,
        "total_e": ke_hist + pe_hist,
        "f_rep_mag": f_rep_hist,
        "f_att_mag": f_att_hist,
    })

    if show_plots:
        plot_energy_and_forces(last_results, rc, k, r0)

    return (phistory)


# ---------------------------------------------------------------------------
# Pinball flippers
# ---------------------------------------------------------------------------
# Lengths and positions are fractions of the box size l, so they scale with it.
FLIPPER_LEN_FRAC = 0.35             # flipper length
FLIPPER_PIVOT_X_FRAC = 0.05         # pivot distance from its side wall
FLIPPER_PIVOT_Z_FRAC = 0.25         # pivot height
FLIPPER_REST_DEG = -30.0            # resting angle (tip pointing down)
FLIPPER_UP_DEG = 35.0               # raised angle
FLIPPER_SWING_SPEED = np.radians(900.0)  # swing speed (rad/s)
FLIPPER_RESTITUTION = 0.95          # 1.0 = perfectly bouncy
BALL_RADIUS = 0.15                  # collision radius of a particle
SUBSTEPS = 8                        # sub-steps per frame (prevents tunnelling)
MAX_SPEED = 150.0                   # speed cap so flipper hits can't blow up the sim
FAST_3D_VIEW = True                 # True: wireframe box (fast). False: full matplotlib 3D axes


class Flipper:
    """A flat paddle spanning the full depth (y) of the box, swinging in the x-z plane."""

    def __init__(self, side, l):
        self.side = side  # +1 = left flipper, -1 = right flipper
        self.l = l
        self.length = FLIPPER_LEN_FRAC * l
        x = FLIPPER_PIVOT_X_FRAC * l
        self.pivot = np.array([x if side == 1 else l - x, FLIPPER_PIVOT_Z_FRAC * l])
        self.angle = np.radians(FLIPPER_REST_DEG)
        self.prev_angle = self.angle
        self.pressed = False

    def endpoints(self, angle=None):
        a = self.angle if angle is None else angle
        tip = self.pivot + self.length * np.array([self.side * np.cos(a), np.sin(a)])
        return self.pivot, tip

    def quad(self):
        """Vertices of the flipper plate for Poly3DCollection."""
        a, b = self.endpoints()
        return [[(a[0], 0, a[1]), (b[0], 0, b[1]), (b[0], self.l, b[1]), (a[0], self.l, a[1])]]

    def advance(self, h):
        self.prev_angle = self.angle
        target = np.radians(FLIPPER_UP_DEG if self.pressed else FLIPPER_REST_DEG)
        max_turn = FLIPPER_SWING_SPEED * h
        self.angle += np.clip(target - self.angle, -max_turn, max_turn)

    def collide(self, p, v, h):
        """Bounce all particles off the flipper at once (vectorised). Returns number of hits."""
        a0, b0 = self.endpoints(self.prev_angle)
        a1, b1 = self.endpoints()
        ab = b1 - a1

        q = p[:, [0, 2]]                                    # (n, 2) positions in the x-z plane
        s = np.clip((q - a1) @ ab / (ab @ ab), 0.0, 1.0)[:, None]
        c = a1 + s * ab                                     # closest point on the flipper
        diff = q - c
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        touching = dist < BALL_RADIUS                       # (n, 1)

        fallback = np.array([-ab[1], ab[0]]) / np.linalg.norm(ab)
        n = np.where(dist > 1e-9, diff / np.maximum(dist, 1e-12), fallback)

        # Velocity of the flipper surface at each contact point
        u = (c - (a0 + s * (b0 - a0))) / h

        vel = v[:, [0, 2]]
        vn = np.sum((vel - u) * n, axis=1, keepdims=True)   # relative speed along the normal
        bounce = touching & (vn < 0)                        # moving into the flipper

        v[:, [0, 2]] = np.where(bounce, vel - (1 + FLIPPER_RESTITUTION) * vn * n, vel)
        p[:, [0, 2]] = np.where(touching, c + n * BALL_RADIUS, q)  # push out of the flipper
        return int(bounce.sum())


def step_with_flippers(p, v, m, dt, l, rc, k, r0, flippers, substeps=SUBSTEPS, max_speed=MAX_SPEED):
    """
    Advances one frame of length dt using your original step_once(), split into
    smaller sub-steps with flipper motion and collisions applied in between.
    Returns (p, v, hits).
    """
    h = dt / substeps
    hits = 0
    for _ in range(substeps):
        for f in flippers:
            f.advance(h)
        p, v = step_once(p, v, m, h, l, rc, k, r0)
        for f in flippers:
            hits += f.collide(p, v, h)

        speed = np.linalg.norm(v, axis=1, keepdims=True)
        v *= np.minimum(1.0, max_speed / (speed + 1e-12))
    return p, v, hits


# ---------------------------------------------------------------------------
# Live energy / force dashboard
# ---------------------------------------------------------------------------
class LiveEnergyPlot:
    """
    Same four panels as plot_energy_and_forces(), but updated while the
    simulation runs. Shows a rolling window of the last `history` frames.

    Speed: the static parts of the figure (axes, grid, legends, the F(r)/U(r)
    panel) are drawn once and cached. Each refresh only redraws the moving
    lines (blitting). The y-axes are re-fitted only when the data leaves the
    visible range, which is what triggers an occasional full redraw.
    """

    KEYS = ("time", "ke", "total_e", "pe", "pe_rep", "pe_att", "f_rep", "f_att")

    def __init__(self, rc, k, r0, dt, history=1000, refresh_ms=100):
        self.data = {key: deque(maxlen=history) for key in self.KEYS}
        self.window = history * dt
        self._bg = None
        self._last_fit = {}

        self.fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
        self.fig.canvas.manager.set_window_title("Live Energy & Attraction/Repulsion Analysis")
        self.fig.suptitle("Particle Simulation (live): Attraction/Repulsion Forces & Energy",
                          fontsize=13, fontweight="bold")
        ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        # 1. Kinetic & total energy
        (self.l_ke,) = ax1.plot([], [], label="Kinetic Energy (KE)", color="#1f77b4", linewidth=1.8, animated=True)
        (self.l_tot,) = ax1.plot([], [], label="Total Energy (KE + PE)", color="#2ca02c",
                                 linestyle="--", linewidth=1.4, animated=True)
        ax1.set_title("Kinetic & Total Mechanical Energy vs. Time")
        ax1.set_ylabel("Energy")

        # 2. Potential energy breakdown
        (self.l_pe,) = ax2.plot([], [], label="Total Potential Energy (PE)", color="#9467bd",
                                linewidth=2.0, animated=True)
        (self.l_pe_rep,) = ax2.plot([], [], label="Repulsive PE (r < r0)", color="#d62728",
                                    alpha=0.85, linewidth=1.4, animated=True)
        (self.l_pe_att,) = ax2.plot([], [], label="Attractive PE (r0 ≤ r ≤ rc)", color="#ff7f0e",
                                    alpha=0.85, linewidth=1.4, animated=True)
        ax2.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax2.set_title("Potential Energy Components vs. Time")
        ax2.set_ylabel("Potential Energy")

        # 3. Repulsive vs attractive force magnitudes
        (self.l_frep,) = ax3.plot([], [], label="Repulsive Force |F_rep| (r < r0)", color="#d62728",
                                  linewidth=1.6, animated=True)
        (self.l_fatt,) = ax3.plot([], [], label="Attractive Force |F_att| (r0 ≤ r ≤ rc)", color="#17becf",
                                  linewidth=1.6, animated=True)
        ax3.set_title("Repulsive vs. Attractive Force Magnitudes Over Time")
        ax3.set_ylabel("Cumulative Force Magnitude")

        for ax in (ax1, ax2, ax3):
            ax.set_xlim(-self.window, 0)
            ax.set_xlabel("Time before now (s)")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", framealpha=0.9)
        self.groups = ((ax1, ("ke", "total_e")), (ax2, ("pe", "pe_rep", "pe_att")), (ax3, ("f_rep", "f_att")))

        # 4. Static pairwise F(r) and U(r) curves
        r_vals = np.linspace(0.1 * r0, rc * 1.15, 300)
        active = r_vals <= rc
        f_curve = np.where(active, (-k * (r_vals - r0)) / (r_vals + 1e-12), 0.0)
        u_curve = np.where(active, k * (r_vals - rc) - k * r0 * np.log(r_vals / rc), 0.0)
        ax4.plot(r_vals, f_curve, label="Pair Force F(r) [+ Repel / - Attract]", color="#d62728", linewidth=1.8)
        ax4.plot(r_vals, u_curve, label="Pair Potential U(r)", color="#9467bd", linestyle="--", linewidth=1.8)
        ax4.axvline(r0, color="green", linestyle=":", linewidth=1.4, label=f"Equilibrium r0 = {r0}")
        ax4.axvline(rc, color="gray", linestyle=":", linewidth=1.4, label=f"Cutoff rc = {rc}")
        ax4.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax4.set_title("Pairwise Force F(r) & Potential U(r) vs. Distance r")
        ax4.set_xlabel("Inter-particle Distance r")
        ax4.set_ylabel("Force / Potential")
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8.5)

        self.clock = self.fig.text(0.99, 0.985, "", ha="right", va="top", animated=True)
        self.pairs = (
            (self.l_ke, "ke"), (self.l_tot, "total_e"),
            (self.l_pe, "pe"), (self.l_pe_rep, "pe_rep"), (self.l_pe_att, "pe_att"),
            (self.l_frep, "f_rep"), (self.l_fatt, "f_att"),
        )
        self.artists = [line for line, _ in self.pairs] + [self.clock]

        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.09)  # room for wide tick labels (e.g. 30000)
        # Re-cache the static background after every full draw (incl. window resizes)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self.timer = self.fig.canvas.new_timer(interval=refresh_ms)
        self.timer.add_callback(self.refresh)
        self.timer.start()

    def _on_draw(self, event):
        cv = self.fig.canvas
        self._bg = cv.copy_from_bbox(self.fig.bbox)
        for a in self.artists:
            self.fig.draw_artist(a)

    def append(self, time, ke, pe, pe_rep, pe_att, f_rep, f_att):
        d = self.data
        d["time"].append(time)
        d["ke"].append(ke)
        d["pe"].append(pe)
        d["pe_rep"].append(pe_rep)
        d["pe_att"].append(pe_att)
        d["total_e"].append(ke + pe)
        d["f_rep"].append(f_rep)
        d["f_att"].append(f_att)

    def _fit_ylimits(self, arr, now):
        """Re-fit a y-axis only if the data left it, or has shrunk a lot (at most every 3 s)."""
        changed = False
        for ax, keys in self.groups:
            lo = min(arr[key].min() for key in keys)
            hi = max(arr[key].max() for key in keys)
            span = max(hi - lo, 1e-6)
            y0, y1 = ax.get_ylim()
            shrunk = span < 0.4 * (y1 - y0) and now - self._last_fit.get(ax, -1e9) > 3.0
            if lo < y0 or hi > y1 or shrunk:
                pad = 0.25 * span
                ax.set_ylim(lo - pad, hi + pad)
                self._last_fit[ax] = now
                changed = True
        return changed

    def refresh(self):
        if not self.data["time"]:
            return
        arr = {key: np.fromiter(self.data[key], dtype=float) for key in self.KEYS}
        now = arr["time"][-1]
        x = arr["time"] - now
        for line, key in self.pairs:
            line.set_data(x, arr[key])
        self.clock.set_text(f"t = {now:.2f} s")

        cv = self.fig.canvas
        if self._fit_ylimits(arr, now) or self._bg is None:
            self._bg = None          # skip blitting until the new background is cached
            cv.draw_idle()           # full redraw (rare)
            return
        cv.restore_region(self._bg)
        for a in self.artists:
            self.fig.draw_artist(a)
        cv.blit(self.fig.bbox)


# ---------------------------------------------------------------------------
# Live pinball simulation
# ---------------------------------------------------------------------------
def run_live_pinball(p, v, m, dt, l, rc, k, r0, show=True):
    """
    Runs the simulation live with controllable flippers and a live-updating
    energy dashboard.

    Controls (click the 3D window first):
        Left flipper  : A or Left arrow
        Right flipper : D or Right arrow
    """
    # Free up matplotlib's default shortcuts for the keys we use
    for name in ("keymap.back", "keymap.forward", "keymap.all_axes"):
        if name in plt.rcParams:
            plt.rcParams[name] = []

    left, right = Flipper(+1, l), Flipper(-1, l)
    flippers = (left, right)
    state = {"time": 0.0, "hits": 0}

    energy = LiveEnergyPlot(rc, k, r0, dt)

    fig = plt.figure()
    fig.canvas.manager.set_window_title("Pinball Simulation")
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(0, l)
    ax.set_ylim(0, l)
    ax.set_zlim(0, l)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("Pinball: A / Left = left flipper,  D / Right = right flipper")

    if FAST_3D_VIEW:
        # matplotlib's 3D ticks/panes/grid are slow to redraw every frame, so
        # replace them with a plain wireframe box.
        ax.set_axis_off()
        corners = np.array([[x, y, z] for x in (0, l) for y in (0, l) for z in (0, l)])
        for i in range(8):
            for j in range(i + 1, 8):
                if np.count_nonzero(corners[i] != corners[j]) == 1:  # corners joined by an edge
                    ax.plot(*zip(corners[i], corners[j]), color="gray", linewidth=0.8)

    scat = ax.scatter(p[:, 0], p[:, 1], p[:, 2], color="blue", s=50, depthshade=False)

    polys = []
    for f, color in ((left, "tab:red"), (right, "tab:green")):
        poly = Poly3DCollection(f.quad(), alpha=0.6, facecolor=color, edgecolor="black")
        ax.add_collection3d(poly)
        polys.append(poly)
    info = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def set_key(event, pressed):
        key = (event.key or "").lower()
        if key in ("a", "left"):
            left.pressed = pressed
        elif key in ("d", "right"):
            right.pressed = pressed

    fig.canvas.mpl_connect("key_press_event", lambda e: set_key(e, True))
    fig.canvas.mpl_connect("key_release_event", lambda e: set_key(e, False))

    def update(frame):
        _, _, hits = step_with_flippers(p, v, m, dt, l, rc, k, r0, flippers)
        state["time"] += dt
        state["hits"] += hits

        # Same energy/force metrics as run_simulation(), recorded live
        _, _, _, pe_tot, pe_rep, pe_att, fr_s, fa_s = compute_forces_and_pe(p, rc, k, r0)
        ke = 0.5 * np.sum(m * (v ** 2))
        energy.append(state["time"], ke, pe_tot, pe_rep, pe_att, fr_s, fa_s)

        scat._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        for poly, f in zip(polys, flippers):
            poly.set_verts(f.quad())
        info.set_text(f"Flipper hits: {state['hits']}")
        return scat

    ani = animation.FuncAnimation(fig, update, interval=20, cache_frame_data=False)

    handles = {"fig": fig, "energy": energy, "animation": ani, "update": update,
               "flippers": flippers, "state": state}
    if show:
        plt.show()  # blocks, so the animations stay alive
    return handles


if __name__ == "__main__":
    import sys
    from variables import p, v, m, t, dt, l, rc, k, r0

    if "--static" in sys.argv:
        # Original behaviour: run everything first, then show the static energy plots
        run_simulation(p, v, m, t, dt, l, rc, k, r0, show_plots=True)
        plt.show()
    else:
        # Live pinball simulation with live-updating energy graphs
        run_live_pinball(p.copy(), v.copy(), m, dt, l, rc, k, r0)
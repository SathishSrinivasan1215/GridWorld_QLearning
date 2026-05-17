
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

START     = (2, 1)
GOAL      = (5, 5)
OBSTACLES = {(3, 3), (3, 4), (3, 5), (4, 3)}
JUMP_FROM = (2, 4)
JUMP_TO   = (4, 4)

REWARD_GOAL = +10
REWARD_JUMP = +5
REWARD_STEP = -1

GRID_ROWS = GRID_COLS = 5


A_NORTH, A_SOUTH, A_EAST, A_WEST = 0, 1, 2, 3
ACTION_NAMES = ["North", "South", "East", "West"]


def step(state, action):

    r, c   = state
    jumped = False

    if state == JUMP_FROM and action == A_SOUTH:
        nr, nc = JUMP_TO
        jumped = True
    else:
        dr = [-1, +1,  0,  0][action]
        dc = [ 0,  0, +1, -1][action]
        nr, nc = r + dr, c + dc

    nr = max(1, min(GRID_ROWS, nr))
    nc = max(1, min(GRID_COLS, nc))

    if (nr, nc) in OBSTACLES:
        nr, nc = r, c

    next_state = (nr, nc)
    done       = (next_state == GOAL)

    if done:
        reward = REWARD_GOAL
    elif jumped:
        reward = REWARD_JUMP
    else:
        reward = REWARD_STEP

    return next_state, reward, done

def make_qtable():

    q = {}
    for r in range(1, GRID_ROWS + 1):
        for c in range(1, GRID_COLS + 1):
            s = (r, c)
            if s not in OBSTACLES:
                q[s] = [0.0, 0.0, 0.0, 0.0]   # North, South, East, West
    return q


def choose_action(q, state, epsilon):

    if random.random() < epsilon:
        return random.randint(0, 3)
    return int(np.argmax(q[state]))


def q_update(q, state, action, reward, next_state, alpha, gamma):

    best_next = max(q[next_state]) if next_state in q else 0.0
    td_target         = reward + gamma * best_next
    q[state][action] += alpha * (td_target - q[state][action])


def train(alpha, gamma=0.9, epsilon=0.2,
          n_episodes=100, early_stop_threshold=9.5, window=30):

    q           = make_qtable()
    rewards_log = []
    steps_log   = []

    print(f"\n-- Learning rate alpha = {alpha} --")

    early_stop_ep = None

    for ep in range(1, n_episodes + 1):
        state        = START
        total_reward = 0
        steps        = 0


        while True:
            action                    = choose_action(q, state, epsilon)
            next_state, reward, done  = step(state, action)
            q_update(q, state, action, reward, next_state, alpha, gamma)
            state        = next_state
            total_reward += reward
            steps        += 1
            if done:
                break

        rewards_log.append(total_reward)
        steps_log.append(steps)

        recent_mean = np.mean(rewards_log[-window:])

        if ep % 10 == 0:
            print(f"  Episode {ep:3d} | Steps: {steps:4d} | "
                  f"Reward: {total_reward:7.2f} | "
                  f"Mean(last {window}): {recent_mean:6.2f}")

        if (ep >= window and
                recent_mean > early_stop_threshold and
                early_stop_ep is None):
            early_stop_ep = ep
            print(f"  *** Early Stopping at Episode {ep} ***  "
                  f"(Mean reward over last {window} episodes = "
                  f"{recent_mean:.2f} > {early_stop_threshold})")
            print(f"  Episodes run: {ep} | "
                  f"Mean reward: {np.mean(rewards_log):.3f} | "
                  f"Final reward: {total_reward:.2f}")
            break

    # If no early stop, print final summary line
    if early_stop_ep is None:
        print(f"  Episodes run: {len(rewards_log)} | "
              f"Mean reward: {np.mean(rewards_log):.3f} | "
              f"Final reward: {rewards_log[-1]:.2f}")

    return q, rewards_log, steps_log, early_stop_ep


def print_qtable(q, alpha):
    print("\n" + "─" * 62)
    print(f"  Q-Table after training  α={alpha}")
    print("─" * 62)
    print(f"  {'State':>8}  {'North':>8}  {'South':>8}  {'East':>8}  {'West':>8}")
    print("─" * 62)
    for r in range(1, GRID_ROWS + 1):
        for c in range(1, GRID_COLS + 1):
            s = (r, c)
            if s in q:
                v = q[s]
                print(f"  {str(s):>8}  {v[0]:8.3f}  {v[1]:8.3f}  {v[2]:8.3f}  {v[3]:8.3f}")
    print("─" * 62)


def state_values(q):
    V = np.full((GRID_ROWS, GRID_COLS), np.nan)
    for r in range(1, GRID_ROWS + 1):
        for c in range(1, GRID_COLS + 1):
            s = (r, c)
            if s in q:
                V[r - 1][c - 1] = max(q[s])
    return V


def plot_state_values(V, alpha, ep_count, filename):

    fig, ax = plt.subplots(figsize=(6, 5.5))

    masked = np.ma.masked_invalid(V)
    vmin   = float(np.nanmin(V)) if not np.all(np.isnan(V)) else 0
    vmax   = float(np.nanmax(V)) if not np.all(np.isnan(V)) else 1
    im     = ax.imshow(masked, cmap="RdYlGn", aspect="equal",
                       vmin=vmin, vmax=vmax)
    cbar   = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("V(s) = max Q(s,a)", fontsize=9)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            s = (r + 1, c + 1)

            if s in OBSTACLES:

                ax.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, color="black", zorder=2))
                ax.text(c, r, "X", ha="center", va="center",
                        color="white", fontsize=11,
                        fontweight="bold", zorder=3)

            elif s == GOAL:

                ax.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    color="#27AE60", alpha=0.85, zorder=2))
                ax.text(c, r, f"+10\nGoal\nV={V[r, c]:.1f}",
                        ha="center", va="center",
                        color="white", fontsize=8,
                        fontweight="bold", zorder=3)

            elif s == START:

                ax.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    color="#E74C3C", alpha=0.75, zorder=2))
                ax.text(c, r, f"Start\nV={V[r, c]:.2f}",
                        ha="center", va="center",
                        color="white", fontsize=8,
                        fontweight="bold", zorder=3)

            elif not np.isnan(V[r, c]):
                ax.text(c, r, f"{V[r, c]:.2f}",
                        ha="center", va="center",
                        fontsize=9, zorder=3,
                        color="black")


    jr, jc = JUMP_FROM[0] - 1, JUMP_FROM[1] - 1
    ax.add_patch(plt.Rectangle(
        (jc - 0.5, jr - 0.5), 1, 1,
        fill=False, edgecolor="#2980B9", linewidth=3, zorder=4))
    ax.text(jc, jr - 0.35, "Jump→", ha="center", va="top",
            fontsize=7, color="#2980B9", fontweight="bold", zorder=5)

    tr, tc = JUMP_TO[0] - 1, JUMP_TO[1] - 1
    ax.add_patch(plt.Rectangle(
        (tc - 0.5, tr - 0.5), 1, 1,
        fill=False, edgecolor="#16A085", linewidth=3, zorder=4))
    ax.text(tc, tr - 0.35, "→Here", ha="center", va="top",
            fontsize=7, color="#16A085", fontweight="bold", zorder=5)


    for x in np.arange(-0.5, GRID_COLS, 1):
        ax.axvline(x, color="grey", linewidth=0.5, zorder=1)
    for y in np.arange(-0.5, GRID_ROWS, 1):
        ax.axhline(y, color="grey", linewidth=0.5, zorder=1)

    ax.set_xticks(range(GRID_COLS))
    ax.set_xticklabels([f"Col {c+1}" for c in range(GRID_COLS)], fontsize=8)
    ax.set_yticks(range(GRID_ROWS))
    ax.set_yticklabels([f"Row {r+1}" for r in range(GRID_ROWS)], fontsize=8)

    early = "  (early stop)" if ep_count < 100 else ""
    ax.set_title(f"State Values  V(s) = max Q(s,a)\n"
                 f"α = {alpha}   |   {ep_count} episodes{early}",
                 fontsize=11, fontweight="bold", pad=10)


    legend_items = [
        mpatches.Patch(facecolor="#E74C3C", label="Start [2,1]"),
        mpatches.Patch(facecolor="#27AE60", label="Goal [5,5] (+10)"),
        mpatches.Patch(facecolor="white",
                       edgecolor="#2980B9", linewidth=2,
                       label="Jump-from [2,4] (+5)"),
        mpatches.Patch(facecolor="white",
                       edgecolor="#16A085", linewidth=2,
                       label="Jump-to [4,4]"),
        mpatches.Patch(facecolor="black", label="Obstacle"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              bbox_to_anchor=(0.0, -0.28), ncol=3,
              fontsize=7, frameon=True)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"  Saved: {filename}")


def plot_training_curves(all_rewards, all_steps, alphas, filename):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    colors = ["#E74C3C", "#E67E22", "#27AE60",
              "#2980B9", "#8E44AD", "#2C3E50"]

    for i, (alpha, rewards, steps) in enumerate(
            zip(alphas, all_rewards, all_steps)):
        col = colors[i % len(colors)]
        eps = list(range(1, len(rewards) + 1))


        smooth_r = [np.mean(rewards[max(0, j - 9):j + 1])
                    for j in range(len(rewards))]
        smooth_s = [np.mean(steps[max(0, j - 9):j + 1])
                    for j in range(len(steps))]

        ax1.plot(eps, smooth_r, label=f"α={alpha}",
                 color=col, linewidth=1.8)
        ax2.plot(eps, smooth_s, label=f"α={alpha}",
                 color=col, linewidth=1.8)

    ax1.axhline(9.5, color="black", linestyle="--",
                linewidth=1, label="Early-stop threshold (9.5)")
    ax1.set_title("Mean Cumulative Reward per Episode\n(10-ep rolling average)",
                  fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Reward")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Mean Steps to Goal per Episode\n(10-ep rolling average)",
                  fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"  Saved: {filename}")

if __name__ == "__main__":
    print("=" * 62)
    print("  COM762 CW2 — Q-Learning Agent: 5×5 Grid World")
    print("=" * 62)


    ALPHAS    = [1.0, 0.75, 0.5, 0.25, 0.1, 0.01]
    GAMMA     = 0.9
    EPSILON   = 0.2
    N_EPS     = 100
    ES_THRESH = 9.5      # early-stop mean-reward threshold
    WINDOW    = 30       # rolling window

    output_dir = "."     # save PNGs next to the script
    all_rewards, all_steps = [], []
    summary = []

    for alpha in ALPHAS:
        q, rewards, steps, early_ep = train(
            alpha, GAMMA, EPSILON, N_EPS, ES_THRESH, WINDOW)

        print_qtable(q, alpha)

        all_rewards.append(rewards)
        all_steps.append(steps)

        ep_count = early_ep if early_ep else len(rewards)
        V        = state_values(q)


        tag   = str(alpha).replace(".", "_")
        fname = os.path.join(output_dir, f"state_values_alpha_{tag}.png")
        print(f"\n  Generating state-value grid for α={alpha} …")
        plot_state_values(V, alpha, ep_count, fname)

        summary.append({
            "alpha":        alpha,
            "episodes":     len(rewards),
            "mean_reward":  round(float(np.mean(rewards)), 4),
            "final_reward": rewards[-1],
            "early_stop":   early_ep,
        })

    print("\n-- Generating learning curves plot --")
    plot_training_curves(
        all_rewards, all_steps, ALPHAS,
        os.path.join(output_dir, "training_curves.png"))

    print("\n-- Results Summary --")
    print(f"   {'Alpha':>5} | {'Episodes':>8} | {'Mean Reward':>12} | {'Final Reward':>12}")
    print("-" * 50)
    for s in summary:
        print(f"  {s['alpha']:>6.2f} | {s['episodes']:>8} | "
              f"{s['mean_reward']:>12.4f} | {s['final_reward']:>12.2f}")

    print("\n  All done!  PNG files saved to:", os.path.abspath(output_dir))


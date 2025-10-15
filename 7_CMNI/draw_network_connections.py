#!/usr/bin/env python3
"""
Selective connection diagram with symmetric padding & edge anchors.

• Green  = mirror‑neuron candidates
• Red    = differentiating neurons
• Yellow = mix neurons
• Grey   = all other units

Changes 2025‑05‑25
──────────────────
1. **Symmetric gaps** – every neuron now has half the gap on its left and
   half on its right, so highlighted units appear visually centred.
2. **Edge anchors** – edges start at the *bottom* edge of a source box
   and finish at the *top* edge of the target box (height‑aware).
3. Geometry constants unchanged – tweak `NEURON_W/H`, `GAP_*` once and
   everything stays aligned.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from tensorflow.keras.models import load_model

# ────────────────────────────────────────────────────────────────────────
# STATIC PATHS
# ────────────────────────────────────────────────────────────────────────
MODEL_PATH = (
    '../5_Pretraining/checkpoints/'
    'checkpoints-20241010-075609-actrelu_bs25_dr0.12_ep500_nl2_nn17_lr4e-06-epoch303-valLoss0.0380/'
    'checkpoint-20241010-023625-actrelu_bs25_dr0.12_ep500_nl2_nn17_lr4e-06-epoch70-valLoss0.0440.h5'
)
PNG_NAME = 'network_connections.png'
OUT_LABELS = ['hop', 'jump', 'leap', 'help']

# ────────────────────────────────────────────────────────────────────────
# HIGHLIGHT LISTS (0‑based indices)
# ────────────────────────────────────────────────────────────────────────
# total mix
'''
MIRROR_MAP = {1: [3, 7, 12, 13], 2: [0, 3, 6, 13]}
DIFF_MAP   = {1: [9, 11], 2: [7]}
MIX_MAP    = {2: [1]}
'''

#mirror focus
'''
MIRROR_MAP = {1: [3, 7, 12, 13], 2: [0, 3, 6, 13]}
DIFF_MAP   = {1: [9, 11], 2: [7]}
MIX_MAP    = {2: [1]}
'''

#difference focus
'''
MIRROR_MAP = {1: [3, 7, 12, 13], 2: [0, 3, 6, 13]}
DIFF_MAP   = {1: [9, 11], 2: [7]}
MIX_MAP    = {2: [1]}
'''

#mix focus
'''
MIRROR_MAP = {1: [3, 12, 13], 2: []}
DIFF_MAP   = {1: [9, 11], 2: []}
MIX_MAP    = {2: [1]}
'''

#ad-hoc focus
#'''
#MIRROR_MAP = {1: [14, 15, 16], 2: [2, 4, 5, 8, 9, 10, 11, 12, 14, 15, 16]}
MIRROR_MAP = {1: [3, 7, 12, 13], 2: [0, 3, 6, 13]}
DIFF_MAP   = {1: [9, 11], 2: [7]}
MIX_MAP    = {2: [1]}
#'''

HIGHLIGHT_MAP = {
    1: MIRROR_MAP[1] + DIFF_MAP[1] + MIX_MAP.get(1, []),
    2: MIRROR_MAP[2] + DIFF_MAP[2] + MIX_MAP[2],
    3: list(range(len(OUT_LABELS)))
}

# ────────────────────────────────────────────────────────────────────────
# GEOMETRY / STYLE
# ────────────────────────────────────────────────────────────────────────
NEURON_W = 0.50
NEURON_H = 0.30
OTHER_W  = 0.06
OTHER_H  = NEURON_H
GAP_HIGHLIGHT = 1.00
GAP_OTHER     = 0.40
FONT_SIZE_IDX   = 14
FONT_SIZE_LABEL = 12
#COL_MIRROR, COL_DIFF, COL_MIX, COL_OTHER = '#2ecc71', '#e74c3c', '#dddd00', '#88a8bb'

COL_MIRROR = '#2ecc71'
COL_DIFF = '#e74c3c'
COL_MIX = '#dddd00'
COL_OTHER = '#88a8bb'

_BLUE = '#739dbe' # new colors for journal
_GREEN = '#2ecc71'
_RED = '#d11d20'
_YELLOW = '#dddd00'

COL_MIRROR = _GREEN
COL_DIFF = _RED
COL_MIX = _YELLOW
COL_OTHER = _BLUE

EDGE_MIRROR, EDGE_DIFF, EDGE_MIX = (0.05, 0.55, 0.05, 1.0), (0.80, 0.10, 0.10, 1.0), (0.90, 0.90, 0.10, 1.0)
ALPHA_MIN, LW_MIN, LW_MAX = 0.12, 0.1, 8.5

# ────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────

def dense_layers(m):
    return [l for l in m.layers if 'dense' in l.name.lower()]

def normalise(w):
    m = np.abs(w).max(); return w/m if m else w


def gen_x_positions(count, layer_idx):
    """Centre‑oriented positions: half‑gap on each side."""
    xs, x = [], 0.0
    for i in range(count):
        gap = GAP_HIGHLIGHT if i in HIGHLIGHT_MAP[layer_idx] else GAP_OTHER
        x += gap/2                # left half
        xs.append(x)
        x += gap/2                # right half
    return np.array(xs)


def draw_layer(ax, y, n, layer_idx, labels=None, x_offset=0.0):
    xs = gen_x_positions(n, layer_idx) + x_offset
    for i, x in enumerate(xs):
        if i in MIRROR_MAP.get(layer_idx, []):
            col = COL_MIRROR
        elif i in DIFF_MAP.get(layer_idx, []):
            col = COL_DIFF
        elif i in MIX_MAP.get(layer_idx, []):
            col = COL_MIX
        else:
            col = COL_OTHER
        width = NEURON_W if col != COL_OTHER else OTHER_W
        height = NEURON_H if col != COL_OTHER else OTHER_H
        #ax.add_patch(Rectangle((x-width/2, y-height/2), width, height, color=col, zorder=3))
        ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height, boxstyle="round,pad=0.03", color=col, zorder=3))
        if col != COL_OTHER:
            ax.text(x, y, str(i), ha='center', va='center', color='white', fontsize=FONT_SIZE_IDX, zorder=4)
        if labels and i < len(labels):
            ax.text(x, y-height/2-0.05, labels[i], ha='center', va='top', fontsize=FONT_SIZE_LABEL)
    return xs


def edge_colour(idx, layer_idx):
    if idx in MIRROR_MAP.get(layer_idx, []):
        return EDGE_MIRROR
    if idx in DIFF_MAP.get(layer_idx, []):
        return EDGE_DIFF
    if idx in MIX_MAP.get(layer_idx, []):
        return EDGE_MIX
    return None


def draw_edges(ax, xs0, y0, xs1, y1, W, src_idxs, tgt_idxs, layer_idx):
    Wn = normalise(W)
    for i in src_idxs:
        cbase = edge_colour(i, layer_idx)
        if cbase is None: continue
        y_start = y0 - NEURON_H/2   # bottom of source box
        for j in tgt_idxs:
            w = Wn[i, j]
            if w == 0: continue
            #if w <= 0: continue
            alpha = ALPHA_MIN + (1-ALPHA_MIN)*abs(w)
            lw = LW_MIN + (LW_MAX-LW_MIN)*abs(w)
            col = (*cbase[:3], alpha)
            if w < 0: col = '#dddddd'
            y_end = y1 + NEURON_H/2 # top of target box
            ax.plot([xs0[i], xs1[j]], [y_start, y_end], color=col, lw=lw, zorder=1)

# ────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    model = load_model(MODEL_PATH, compile=False)
    l1, l2, l3 = dense_layers(model)[:3]
    W12, W23 = l2.get_weights()[0], l3.get_weights()[0]

    y1, y2, y3 = 2, 1, 0
    fig, ax = plt.subplots(figsize=(14, 6)); ax.axis('off')

    xs1 = draw_layer(ax, y1, W12.shape[0], 1)
    # After xs1 = draw_layer(...):
    ax.text(
        xs1.max() + 0.5,    # a bit to the right of the rightmost neuron
        y1,                 # same vertical position
        '17-neuron hidden layer 1',
        va='center',
        ha='left',
        fontsize=FONT_SIZE_LABEL
    )

    xs2 = draw_layer(ax, y2, W23.shape[0], 2)
    # After xs2 = draw_layer(...):
    ax.text(
        xs2.max() + 0.5,    # a bit to the right of the rightmost neuron
        y2,
        '17-neuron hidden layer 2',
        va='center',
        ha='left',
        fontsize=FONT_SIZE_LABEL
    )

    # centre output under hidden‑2 span
    xs_out_tmp = gen_x_positions(len(OUT_LABELS), 3)
    x_offset = (xs2.max()+xs2.min() - xs_out_tmp.max())/2
    xs3 = draw_layer(ax, y3, len(OUT_LABELS), 3, labels=OUT_LABELS, x_offset=x_offset)
    # And for the output layer:
    ax.text(
        xs3.max() + 0.5,    # a bit to the right of the rightmost neuron
        y3,
        '4-neuron output layer',
        va='center',
        ha='left',
        fontsize=FONT_SIZE_LABEL
    )


    draw_edges(ax, xs1, y1, xs2, y2, W12, HIGHLIGHT_MAP[1], HIGHLIGHT_MAP[2], 1)
    draw_edges(ax, xs2, y2, xs3, y3, W23, HIGHLIGHT_MAP[2], HIGHLIGHT_MAP[3], 2)

    ax.set_title('Mirror (green) · Diff (red) · Mix (yellow)', fontsize=13, pad=28)
    plt.tight_layout(); plt.savefig(PNG_NAME, dpi=300)
    print('Saved →', PNG_NAME)

if __name__ == '__main__':
    main()

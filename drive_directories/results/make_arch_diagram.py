import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(7.0, 9.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

C_INPUT  = '#e8eef7'
C_EMBED  = '#cfe0f5'
C_LN     = '#fdebd0'
C_ATTN   = '#d4efdf'
C_MLP    = '#f5cba7'
C_HEAD   = '#d2b4de'
C_OUT    = '#aed6f1'
C_BLOCK  = '#fbfcfd'
C_BORDER = '#34495e'

def box(x, y, w, h, label, color, fontsize=10, weight='normal', edgecolor=None):
    ec = edgecolor if edgecolor else C_BORDER
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.12",
                       linewidth=1.1, facecolor=color, edgecolor=ec)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, label,
            ha='center', va='center', fontsize=fontsize, weight=weight)

def arrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='-|>', mutation_scale=12,
                        linewidth=1.2, color=C_BORDER)
    ax.add_patch(a)

def addnode(x, y, r=0.18):
    c = patches.Circle((x, y), r, facecolor='white', edgecolor=C_BORDER, linewidth=1.2)
    ax.add_patch(c)
    ax.text(x, y, '+', ha='center', va='center', fontsize=11, weight='bold')

cx = 5.0
W  = 5.6
xL = cx - W/2

box(xL, 0.4, W, 0.7, 'Input token IDs  (B, T)', C_INPUT, fontsize=10)
arrow(cx, 1.1, cx, 1.45)

box(xL,        1.45, W/2 - 0.05, 0.75, 'Token Embed\nwte (V$\\times$d)', C_EMBED, fontsize=9)
box(xL+W/2+0.05, 1.45, W/2 - 0.05, 0.75, 'Pos. Embed\nwpe (T$\\times$d)', C_EMBED, fontsize=9)
addnode(cx, 2.45)
arrow(xL + W/4,        2.20, cx - 0.18, 2.40)
arrow(xL + 3*W/4,      2.20, cx + 0.18, 2.40)
arrow(cx, 2.63, cx, 2.95)
box(xL, 2.95, W, 0.55, 'Dropout', C_INPUT, fontsize=9)
arrow(cx, 3.50, cx, 3.85)

bx = xL - 0.25
bw = W + 0.5
by = 3.85
bh = 5.6
block = FancyBboxPatch((bx, by), bw, bh,
                       boxstyle="round,pad=0.02,rounding_size=0.18",
                       linewidth=1.4, facecolor=C_BLOCK,
                       edgecolor=C_BORDER, linestyle='--')
ax.add_patch(block)
ax.text(bx + 0.18, by + bh - 0.32,
        'Transformer Block  $\\times\\,L$',
        ha='left', va='center', fontsize=10, weight='bold', style='italic')

y = 4.10
box(xL, y, W, 0.55, 'LayerNorm  (ln_1)', C_LN, fontsize=9)
arrow(cx, y + 0.55, cx, y + 0.90)

y2 = y + 0.90
box(xL, y2, W, 1.05,
    'Causal Self-Attention\n'
    r'q, k, v $\leftarrow$ c_attn(x);  scaled_dot_product_attention' '\n'
    'c_proj  $\\to$  resid_drop',
    C_ATTN, fontsize=8.5)
arrow(cx, y2 + 1.05, cx, y2 + 1.35)
addnode(cx, y2 + 1.55)
ax.annotate('residual', xy=(cx + 0.35, y2 + 1.55),
            xytext=(cx + 0.35, y2 + 1.55),
            ha='left', va='center', fontsize=8, style='italic', color='#555')

y3 = y2 + 1.85
box(xL, y3, W, 0.55, 'LayerNorm  (ln_2)', C_LN, fontsize=9)
arrow(cx, y3 + 0.55, cx, y3 + 0.90)

y4 = y3 + 0.90
box(xL, y4, W, 1.05,
    'MLP\nc_fc: d $\\to$ 4d  $\\to$  GELU  $\\to$  c_proj: 4d $\\to$ d',
    C_MLP, fontsize=8.5)
arrow(cx, y4 + 1.05, cx, y4 + 1.35)
addnode(cx, y4 + 1.55)

arrow(cx, y4 + 1.75, cx, y4 + 2.05)

y5 = y4 + 2.05 + 0.10
box(xL, y5, W, 0.55, 'Final LayerNorm  (ln_f)', C_LN, fontsize=9)
arrow(cx, y5 + 0.55, cx, y5 + 0.90)

y6 = y5 + 0.90
box(xL, y6, W, 0.65, 'LM Head  (lm_head, weight-tied to wte)', C_HEAD, fontsize=9)
arrow(cx, y6 + 0.65, cx, y6 + 1.00)

y7 = y6 + 1.00
box(xL, y7, W, 0.65, 'Logits  (B, T, V=4096)', C_OUT, fontsize=10, weight='bold')

ax.text(0.25, 13.55,
        'Decoder-only Transformer (nanoGPT-style)\n'
        'B = batch, T = 512, d = $d_{model}$, V = 4096, L = n_layer',
        ha='left', va='top', fontsize=10, weight='bold')

plt.tight_layout()
out = 'model_architecture.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved {out}')

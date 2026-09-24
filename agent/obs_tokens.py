"""Per-step observation tokens for the decoder-only policy.

Fixed layout per step (33 observation tokens, then the action tokens <ACT> ... <EOS>):
    9  PRODUCT tokens   market / stock / field state per product (Tracker features + product one-hot)
    1  GLOBAL token     time, money, shops, hands ...
    8  QUAD tokens      2 farms x 4 quadrants, each = 25 tiles x TILE_F features (+ side, quadrant one-hot)
    15 UNIT tokens      farmer + up to 14 hands: position one-hots, carried inventory, presence
Stored compactly (tiles as uint8 codes) and expanded to float features by `expand_*` (numpy or torch),
so the heavy expansion can run on the GPU during training.
"""
import numpy as np

from agent.action_tokens import V as ACTION_VOCAB
from agent.features import GF, PF, PRODUCTS

ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW",
         "SHEEP"]
IIDX = {k: i for i, k in enumerate(ITEMS)}
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
N_PROD, N_GLOB, N_QUAD, N_UNIT = 9, 1, 8, 15
N_OBS = N_PROD + N_GLOB + N_QUAD + N_UNIT          # 33
TILE_F = 22
PROD_W = PF + N_PROD                                # 41
GLOB_W = GF                                         # 16
QUAD_W = 25 * TILE_F + 1 + 4                        # 555
UNIT_W = 10 + 10 + len(ITEMS) + 3                   # 35
OBS_TYPES = (PROD_W, GLOB_W, QUAD_W, UNIT_W)
OBS_TOKEN_ID = [ACTION_VOCAB + i for i in range(4)]  # embedding ids of the 4 observation token types
TYPE_OF_SLOT = np.array([0] * N_PROD + [1] * N_GLOB + [2] * N_QUAD + [3] * N_UNIT, np.int8)


def tile_codes(tile, day):
    """uint8 [TILE_F] code for one tile."""
    f = np.zeros(TILE_F, np.uint8)
    if tile == "LOCKED":
        f[0] = 255
        return f
    if tile is None:
        f[1] = 255
        return f
    kind = tile.get("kind")
    if kind == "WEED":
        f[2] = 255
    elif kind == "PLANT":
        f[3] = 255
        f[7 + CROPS.index(tile["crop"])] = 255
        f[15] = min(255, max(0, day - tile.get("planted_day", day)) * 12)
        f[16] = min(255, tile.get("yield_units", 0) * 32)
        f[17] = 255 if tile.get("watered_today") else 0
        f[18] = min(255, tile.get("consecutive_unwatered", 0) * 127)
        f[19] = 255 if tile.get("fertilized_until_day", -1) >= day else 0
    else:
        f[4 if kind == "COOP" else 5] = 255
        a = tile.get("animal")
        if a:
            f[6] = 255
            f[12 + ANIMALS.index(a)] = 255
            f[15] = min(255, max(0, day - tile.get("placed_day", day)) * 12)
            f[16] = min(255, tile.get("yield_units", 0) * 40)
            f[17] = 255 if tile.get("fed_today") else 0
            f[18] = min(255, tile.get("consecutive_unfed", 0) * 127)
            f[19] = 255 if tile.get("cared_today") else 0
            f[20] = 255 if tile.get("fertilizer_available") else 0
            f[21] = min(255, tile.get("pending_care_bonus", 0) * 40)
    return f


def build_step(obs, P, G):
    """-> dict(prod f16 [9,PROD_W], glob f16 [GF], tiles u8 [2,100,TILE_F], units f16 [15,UNIT_W], n_units)."""
    me = int(obs["player"])
    day = int(obs["step"]) // 24
    prod = np.zeros((N_PROD, PROD_W), np.float16)
    prod[:, :PF] = P
    prod[np.arange(N_PROD), PF + np.arange(N_PROD)] = 1
    tiles = np.zeros((2, 100, TILE_F), np.uint8)
    for side, pid in enumerate((me, 1 - me)):
        rows = obs["farms"][pid]["tiles"]
        for y in range(10):
            for x in range(10):
                tiles[side, y * 10 + x] = tile_codes(rows[y][x], day)
    units = np.zeros((N_UNIT, UNIT_W), np.float16)
    farm = obs["farms"][me]
    pos = [farm["farmer"]] + list(farm["hands"])
    invs = obs["private"].get("inventories", [])
    for i, (x, y) in enumerate(pos[:N_UNIT]):
        units[i, x] = 1
        units[i, 10 + y] = 1
        inv = invs[i] if i < len(invs) else {}
        for k, n in inv.items():
            if k in IIDX:
                units[i, 20 + IIDX[k]] = np.log1p(n) / 3
        units[i, 32] = 1.0 if i == 0 else 0.0
        units[i, 33] = i / 15
        units[i, 34] = 1.0
    return {"prod": prod, "glob": G.astype(np.float16), "tiles": tiles, "units": units, "n_units": len(pos)}


def quad_features(tiles_f):
    """tiles_f float [..., 2, 100, TILE_F] (already /255) -> quad tokens [..., 8, QUAD_W].
    Works for numpy arrays and torch tensors."""
    is_torch = hasattr(tiles_f, "unflatten")
    lib_cat = (lambda xs, d: __import__("torch").cat(xs, d)) if is_torch else (lambda xs, d: np.concatenate(xs, d))
    shp = tiles_f.shape[:-3]
    t = tiles_f.reshape(*shp, 2, 10, 10, TILE_F)
    quads = []
    for side in range(2):
        for qy in range(2):
            for qx in range(2):
                q = t[..., side, qy * 5:(qy + 1) * 5, qx * 5:(qx + 1) * 5, :].reshape(*shp, 25 * TILE_F)
                quads.append(q)
    q = (__import__("torch").stack(quads, -2) if is_torch else np.stack(quads, -2))   # [...,8,25*TILE_F]
    extra = np.zeros((8, 5), np.float32)
    for i in range(8):
        extra[i, 0] = 1.0 if i >= 4 else 0.0
        extra[i, 1 + (i % 4)] = 1.0
    if is_torch:
        import torch
        e = torch.tensor(extra, device=q.device, dtype=q.dtype).expand(*shp, 8, 5)
    else:
        e = np.broadcast_to(extra.astype(q.dtype), (*shp, 8, 5))
    return lib_cat([q, e], -1)

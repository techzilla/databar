"""Teaching example: build a 2D layout from a few inputs, and cache the result.

Techniques shown in this file: computing a grid with numpy, dispatching on a
value with a dict instead of if/elif, list comprehensions, and @lru_cache to
avoid repeating expensive work. (Technical: it lays out one barcode frame.)
"""
from functools import lru_cache
import numpy as np

from . import spec

# Technique: an "enum by assignment" — give each category a small int name so the
# grid array can store a category per cell and comparisons read in plain words.
RING, CORNER_BLOCK, DIMS, ALIGN, FREE = 1, 2, 3, 4, 0

def marker_origin(marker_id, cols, rows):
    # Technique: dict dispatch — look the answer up by key instead of a 4-way if/elif.
    far_c, far_r = cols - 1 - spec.MARKER, rows - 1 - spec.MARKER
    return {0: (1, 1), 1: (1, far_c), 2: (far_r, far_c), 3: (far_r, 1)}[marker_id]

def dims_offsets(marker_id):
    # Technique: booleans as flags, then a nested list comprehension to build every
    # (row, col) pair in one expression instead of appending in a double loop.
    below = marker_id in (0, 1)
    left = marker_id in (0, 3)
    r0 = spec.MARKER + 1 if below else -1 - spec.DIMS_ROWS   # conditional expression
    c0 = 0 if left else -1
    return [(r0 + r, c0 + c) for r in range(spec.DIMS_ROWS) for c in range(spec.DIMS_COLS)]

def _crc8(value, nbits):
    # Technique: bit manipulation. Walk the bits high-to-low with >> and &, and keep
    # a running value masked to 8 bits with & 0xFF — a classic checksum loop.
    # (The leading _ marks this as a module-private helper by convention.)
    crc = 0
    for i in range(nbits - 1, -1, -1):
        bit = (value >> i) & 1
        crc = ((crc << 1) & 0xFF) ^ (0x07 if ((crc >> 7) ^ bit) else 0)
    return crc

def dims_bits(cols, rows):
    # Technique: pack several numbers into one integer with shift-and-OR, then read
    # bits back out. (cols << 10) | rows stacks two 10-bit fields side by side.
    v = (cols << 10) | rows
    word = (v << 8) | _crc8(v, 20)
    return [(word >> (spec.DIMS_BITS - 1 - i)) & 1 for i in range(spec.DIMS_BITS)]

def parse_dims_bits(bits):
    # Technique: the inverse — rebuild an integer from bits by shifting in one at a
    # time, then split the fields back out. Returning None signals "invalid input".
    word = 0
    for b in bits:
        word = (word << 1) | int(b)
    v, crc = word >> 8, word & 0xFF
    if _crc8(v, 20) != crc:          # guard: reject data that fails its checksum
        return None
    return v >> 10, v & 0x3FF         # & 0x3FF keeps the low 10 bits

@lru_cache(maxsize=None)              # Technique: memoise — compute once per id, cache forever
def marker_bits(marker_id):
    # Technique: build a numpy array, then use slicing (bits[1:-1, 1:-1]) to write a
    # whole inner rectangle at once instead of looping over cells.
    bits = np.zeros((spec.MARKER, spec.MARKER), dtype=np.uint8)
    bits[1:-1, 1:-1] = [[int(b) for b in row] for row in spec.ARUCO_BITS[marker_id]]
    return bits

# Technique: compute bounds from the spec constants so limits track the format.
MIN_COLS, MIN_ROWS = 2 * spec.CORNER + 12, 2 * (spec.CORNER + spec.DIMS_ROWS) + 6

class Layout:
    # Technique: a class bundles related data (grid arrays) with the code that
    # builds it. __init__ runs once and stores everything on self for later use.
    def __init__(self, cols, rows):
        # Technique: validate inputs first and raise early, so the rest of the
        # method can trust its data — "fail fast" instead of corrupting state.
        if not (MIN_COLS <= cols <= spec.MAX_GRID_DIM and MIN_ROWS <= rows <= spec.MAX_GRID_DIM):
            raise ValueError(f'grid {cols}x{rows} outside {MIN_COLS}x{MIN_ROWS} .. '
                             f'{spec.MAX_GRID_DIM}x{spec.MAX_GRID_DIM}')
        self.cols, self.rows = cols, rows
        # Technique: two parallel arrays. "kind" records what each cell is for;
        # "base" records its colour. Same shape, indexed together.
        kind = np.zeros((rows, cols), dtype=np.uint8)
        base = np.zeros((rows, cols), dtype=np.uint8)
        C = spec.CORNER        # short local alias for a value used many times below

        # Technique: fill rectangular regions with array slice assignment, and use
        # np.where as a vectorised if/else to map pattern bits to two colours.
        for mid in spec.CORNER_IDS:
            r0, c0 = marker_origin(mid, cols, rows)
            kind[r0 - 1:r0 - 1 + C, c0 - 1:c0 - 1 + C] = CORNER_BLOCK
            base[r0 - 1:r0 - 1 + C, c0 - 1:c0 - 1 + C] = spec.WHITE
            base[r0:r0 + spec.MARKER, c0:c0 + spec.MARKER] = np.where(
                marker_bits(mid), spec.WHITE, spec.BLACK)

        # Technique: build a list of coordinates by concatenating four comprehensions,
        # one per edge, to trace a loop clockwise.
        ring = ([(0, c) for c in range(C, cols - C)] +
                [(r, cols - 1) for r in range(C, rows - C)] +
                [(rows - 1, c) for c in range(cols - C - 1, C - 1, -1)] +
                [(r, 0) for r in range(rows - C - 1, C - 1, -1)])
        self.ring_cells = np.array(ring)
        # Technique: modulo (%) to cycle through a fixed set of values repeatedly.
        self.ring_colours = np.arange(len(ring)) % spec.NUM_COLOURS
        # Technique: fancy indexing — pass arrays of rows and cols to set many cells at once.
        kind[self.ring_cells[:, 0], self.ring_cells[:, 1]] = RING
        base[self.ring_cells[:, 0], self.ring_cells[:, 1]] = self.ring_colours

        # Technique: zip() walks two sequences in lockstep, pairing each position
        # with its bit, so one loop handles both.
        bits = dims_bits(cols, rows)
        for mid in spec.CORNER_IDS:
            r0, c0 = marker_origin(mid, cols, rows)
            for (dr, dc), b in zip(dims_offsets(mid), bits):
                kind[r0 + dr, c0 + dc] = DIMS
                base[r0 + dr, c0 + dc] = spec.WHITE if b else spec.BLACK

        # Technique: round() + max() clamp a computed count to a sensible minimum,
        # a common way to turn a rough formula into a safe integer.
        nx = max(2, round((cols - 5) / spec.ALIGN_SPACING) + 1)
        ny = max(2, round((rows - 5) / spec.ALIGN_SPACING) + 1)
        self.align_xs = np.round(np.linspace(2, cols - 3, nx)).astype(int)
        self.align_ys = np.round(np.linspace(2, rows - 3, ny)).astype(int)
        self.align_present = np.zeros((ny, nx), dtype=bool)
        # Technique: enumerate() gives index and value together; the nested loop
        # tests a 3x3 window with .all() before claiming it, a look-before-you-leap check.
        for i, y in enumerate(self.align_ys):
            for j, x in enumerate(self.align_xs):
                if (kind[y - 1:y + 2, x - 1:x + 2] == FREE).all():
                    self.align_present[i, j] = True
                    kind[y - 1:y + 2, x - 1:x + 2] = ALIGN
                    base[y - 1:y + 2, x - 1:x + 2] = spec.BLACK
                    base[y, x] = spec.WHITE

        self.kind, self.base = kind, base
        # Technique: a boolean mask selects elements — kind.ravel()[order] == FREE
        # keeps only the free cells. .reshape/.T/.ravel flatten the grid to 1D first.
        order = np.arange(rows * cols).reshape(rows, cols)
        if rows > cols:
            order = order.T
        order = order.ravel()
        free = order[kind.ravel()[order] == FREE]
        H = spec.HEADER_CELLS
        # Technique: slicing to split a sequence into head, tail and middle in one go.
        self.header_a, self.header_b = free[:H], free[-H:]
        self.data = free[H:-H]

    def marker_cell_corners(self, marker_id):
        # Technique: a method (function on self) that reuses the object's stored
        # cols/rows. Returns the four corners as a float32 array ready for later math.
        r0, c0 = marker_origin(marker_id, self.cols, self.rows)
        m = spec.MARKER
        return np.array([[c0, r0], [c0 + m, r0], [c0 + m, r0 + m], [c0, r0 + m]], dtype=np.float32)

@lru_cache(maxsize=16)   # Technique: cache the last 16 layouts so repeats are free
def get_layout(cols, rows):
    # Technique: a thin "factory" wrapper. Callers go through this so caching is
    # automatic and nobody constructs Layout directly.
    return Layout(cols, rows)

def grid_for_screen(screen_w, screen_h, profile=spec.DEFAULT_PROFILE):
    # Technique: a default argument (profile=...) lets callers omit the common case.
    # Technique: tuple unpacking pulls three settings out of one table row at once.
    cam_long, cam_short, px_per_cell = spec.PROFILES[profile]
    L, S = max(screen_w, screen_h), min(screen_w, screen_h)
    # Technique: min() of two ratios picks whichever dimension is the tighter fit.
    scale = spec.FRAMING * min(cam_long / L, cam_short / S)
    if L / S <= spec.SQUARISH_ASPECT + 1e-9:          # + 1e-9: tolerate float rounding
        scale = min(scale, spec.FRAMING * min(cam_short / L, cam_long / S))
    cells_long = round(L * scale / px_per_cell)
    cells_short = round(S * scale / px_per_cell)
    # Technique: a conditional expression swaps the pair depending on orientation.
    cols, rows = (cells_long, cells_short) if screen_w >= screen_h else (cells_short, cells_long)
    cell_px = min(screen_w // cols, screen_h // rows)
    if cell_px < spec.MIN_CELL_PX:          # guard: refuse an unreadable result, with a clear message
        raise ValueError(f'profile {profile} needs {cols}x{rows} cells; a {screen_w}x{screen_h} '
                         f'screen gives only {cell_px} px per cell (minimum {spec.MIN_CELL_PX})')
    return cols, rows, cell_px

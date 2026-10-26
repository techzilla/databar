"""Teaching example: a configuration module.

Technique shown here: collect every "magic number" in one place as named
constants, then import them everywhere instead of hard-coding values. One edit
changes the whole program, and each name documents what the number means.
(Technical: these constants define the barcode format both sides must agree on.)
"""
import numpy as np

FORMAT_VERSION = 1

# ── Palette ──────────────────────────────────────────────────────────────────
# Technique: a lookup table. Store the 8 colours in one array and refer to each
# by its index (0-7) everywhere else, so code passes around small ints, not RGB.
PALETTE_BGR = np.array([
    [  0,   0,   0],  # 0 black
    [255, 255, 255],  # 1 white
    [  0,   0, 255],  # 2 red
    [  0, 255,   0],  # 3 green
    [255,   0,   0],  # 4 blue
    [  0, 255, 255],  # 5 yellow
    [255,   0, 255],  # 6 magenta
    [255, 255,   0],  # 7 cyan
], dtype=np.uint8)
BLACK, WHITE = 0, 1           # Technique: name indices so code reads BLACK, not 0
BITS_PER_CELL = 3
NUM_COLOURS   = 1 << BITS_PER_CELL   # Technique: left-shift to compute 2**3 = 8

# ── Grid profiles ──────────────────────────────────────────────────────────────
# Technique: a dict as a small config table. The key names a preset; the value is
# a tuple of its settings. Beats a tangle of if/elif branches choosing numbers.
PROFILES = {
    '1080p-12': (1920, 1080, 12),
    '1080p-10': (1920, 1080, 10),
    '1080p-8':  (1920, 1080,  8),
    '1080p-7':  (1920, 1080,  7),
    '1080p-6':  (1920, 1080,  6),
    '1080p-5':  (1920, 1080,  5),
    '4k-12':    (3840, 2160, 12),
    '4k-10':    (3840, 2160, 10),
    '4k-8':     (3840, 2160,  8),
}
# Technique: pick one preset as the default so callers can omit the choice.
DEFAULT_PROFILE = '1080p-6'
FRAMING         = 0.5
SQUARISH_ASPECT = 4 / 3        # Technique: an expression (4/3) is clearer than 1.333
MIN_CELL_PX     = 4            # a named bound beats a bare literal in the check below
MAX_GRID_DIM    = 1023         # = 2**10 - 1, the largest value 10 bits can hold

# ── Timing ─────────────────────────────────────────────────────────────────────
# Technique: express durations in seconds as floats; the trailing comment gives units.
FRAME_TIME      = 0.100
FAST_FRAME_TIME = 0.050

# ── Frame layout (in cells) ─────────────────────────────────────────────────────
# Technique: more named constants. Later modules build the whole image from these
# sizes, so the layout is defined once here, never duplicated as literals.
ARUCO_DICT   = 'DICT_4X4_50'
MARKER       = 6
# Technique: a dict of tuples as table data. Each key maps to the rows of a small
# 2D pattern, stored as strings of '0'/'1' so it is easy to read and edit by hand.
ARUCO_BITS   = {
    0: ('1011', '0101', '0011', '0010'),
    1: ('0000', '1111', '1001', '1010'),
    2: ('0011', '0011', '0010', '1101'),
    3: ('1001', '1001', '0100', '0110'),
}
CORNER       = MARKER + 2      # Technique: derive one constant from another, don't restate 8
CORNER_IDS   = {0: 'tl', 1: 'tr', 2: 'br', 3: 'bl'}   # id -> human-readable label

# Technique: deriving a constant from others (rows * cols) keeps the three values
# consistent — change a dimension and the total updates itself.
DIMS_ROWS, DIMS_COLS = 4, 7
DIMS_BITS            = DIMS_ROWS * DIMS_COLS

ALIGN_SPACING = 14

# ── Header ─────────────────────────────────────────────────────────────────────
# Technique: a struct format string describes a binary record field-by-field;
# '>BIIIHH' means big-endian, then one byte, three 4-byte ints, two 2-byte ints.
HEADER_FMT    = '>BIIIHH'
HEADER_BYTES  = 19
HEADER_RS_K   = 17
HEADER_RS_NSYM = 14
HEADER_CELLS  = (HEADER_RS_K + HEADER_RS_NSYM) * 3   # arithmetic keeps values in step

# ── Inner code ──────────────────────────────────────────────────────────────────
# Technique: integer division (//) and shift (<<) compute related sizes from the
# base values, so the "how many cells per symbol" rule lives in one expression.
RS_BITS         = 9
RS_MAX_N        = (1 << RS_BITS) - 1          # 2**9 - 1 = 511, the max this field allows
CELLS_PER_SYM   = RS_BITS // BITS_PER_CELL
BLOCK_CRC_BYTES = 4
TARGET_BLOCK_N  = 255
DEFAULT_PARITY  = 0.20

# ── Outer code (LT) ─────────────────────────────────────────────────────────────
# Technique: group the knobs an algorithm is tuned by at the top, each named and
# commented, so tuning means editing clearly-labelled values, not hunting literals.
LT_C     = 0.3
LT_DELTA = 0.5
PRECODE_PERCENT = 10
PRECODE_MIN     = 17
MIN_K    = 128
MIN_T    = 16

# ── Stream metadata ──────────────────────────────────────────────────────────────
# Technique: a bytes literal (b'...') and hex (0xFFFFFFFF) spell out exact byte values.
STREAM_MAGIC  = b'DBR1'
MAX_NAME_BYTES = 255
MAX_FILE_BYTES = 0xFFFFFFFF - 1024

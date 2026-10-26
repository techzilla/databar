"""Teaching example: a pipeline that turns input into images on demand.

Techniques: do expensive setup once in __init__ then produce each frame lazily,
use string slicing to chop bytes into fixed-size pieces, and scale up an image
with numpy instead of pixel loops. Notice how it leans on the core library rather
than re-implementing the codes.
"""
import numpy as np

from databar.core import lt, spec, stream
from databar.core.layout import get_layout

class Sender:
    def __init__(self, file_bytes, name, cols, rows, parity=spec.DEFAULT_PARITY, stream_id=None):
        # Technique: __init__ does all the heavy preparation — pack the data, choose
        # parameters, split into fixed-size chunks — and stores results on self so
        # frame_cells() later is cheap. Each attribute is one computed fact.
        self.layout = get_layout(cols, rows)
        self.cols, self.rows = cols, rows
        self.data = stream.pack_stream(file_bytes, name)
        self.T, self.nsym = stream.choose_params(len(self.data), len(self.layout.data), parity)
        self.k, self.n, self.B = stream.block_layout(self.T, self.nsym, len(self.layout.data))
        self.K = -(-len(self.data) // self.T)                 # round-up division
        # Technique: pad to a whole multiple, then a comprehension slices the bytes into
        # equal T-byte pieces and reads each as one big integer.
        padded = self.data + bytes(self.K * self.T - len(self.data))
        source = [int.from_bytes(padded[i:i + self.T], 'big') for i in range(0, len(padded), self.T)]
        self.stream_id = stream_id or stream.new_stream_id()  # Technique: `or` for a default
        self.lt = lt.Encoder(self.stream_id, source)
        # Technique: precompute a reusable template once (here, filler colours) so it is
        # copied per frame instead of rebuilt.
        self._pad = (np.arange(len(self.layout.data)) % spec.NUM_COLOURS).astype(np.uint8)

    @property                       # Technique: expose a derived count as an attribute
    def first_pass_frames(self):
        return -(-self.lt.L // self.B)

    def frame_cells(self, f):
        # Technique: generate frame f on demand (not all frames up front), so the
        # stream can run forever. Build a flat array, overwrite named regions by index,
        # then reshape to 2D at the end.
        first_id = (f * self.B) & 0xFFFFFFFF          # & mask: wrap the counter into 32 bits
        payloads = [self.lt.symbol(first_id + j).to_bytes(self.T, 'big') for j in range(self.B)]
        block_cells = stream.encode_blocks(self.stream_id, first_id, payloads, self.nsym, self.k).ravel()
        hdr = stream.header_cells(self.stream_id, first_id, len(self.data), self.T, self.nsym)
        cells = self.layout.base.ravel().copy()       # copy so the template isn't mutated
        cells[self.layout.header_a] = hdr
        cells[self.layout.header_b] = hdr              # write the same header in two places
        data = self._pad.copy()
        data[:len(block_cells)] = block_cells
        cells[self.layout.data] = data
        return cells.reshape(self.rows, self.cols)

def render(cells, cell_px, screen_w=None, screen_h=None):
    # Technique: index a palette with a whole array (PALETTE_BGR[cells]) to colour every
    # cell at once, and np.repeat to enlarge without loops. A default of None makes the
    # centering step optional — the function does the simple thing unless asked for more.
    img = spec.PALETTE_BGR[cells]
    img = np.repeat(np.repeat(img, cell_px, axis=0), cell_px, axis=1)
    if screen_w is None:
        return img
    # Technique: place a small image onto a bigger canvas with slice assignment,
    # computing the top-left offset to centre it.
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    y0, x0 = (screen_h - img.shape[0]) // 2, (screen_w - img.shape[1]) // 2
    canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return canvas

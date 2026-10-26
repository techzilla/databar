"""Teaching example: turning structured data into bytes and back.

Techniques shown here: packing a binary record with struct, verifying integrity
with a hash, cleaning untrusted text with a regex, and converting between bytes,
bit-groups and array elements using numpy. Note the recurring pattern of a
pack/unpack (or encode/decode) pair that must be exact inverses of each other.
"""
import binascii, hashlib, os, re, struct, zlib
import numpy as np

from . import spec, rs

# ── Stream data: metadata + file ───────────────────────────────────────────────
# Technique: a format string describes a fixed binary layout; struct.calcsize then
# computes its byte length for us instead of us counting bytes by hand.
META_FMT  = '>4sQ32sH'
META_SIZE = struct.calcsize(META_FMT)

def cut_name(name):
    # Technique: encode-slice-decode to trim text to a byte budget without splitting a
    # multi-byte character — errors='ignore' drops a half character left at the cut.
    raw = name.encode('utf-8')[:spec.MAX_NAME_BYTES]
    return raw.decode('utf-8', errors='ignore').encode('utf-8')

def pack_stream(file_bytes, name):
    # Technique: validate, then assemble. struct.pack writes the fixed header; the
    # variable-length name and file are simply concatenated after it.
    if len(file_bytes) > spec.MAX_FILE_BYTES:
        raise ValueError(f'file is larger than {spec.MAX_FILE_BYTES} bytes')
    name_b = cut_name(name)
    meta = struct.pack(META_FMT, spec.STREAM_MAGIC, len(file_bytes),
                       hashlib.sha256(file_bytes).digest(), len(name_b))
    return meta + name_b + file_bytes

def unpack_stream(data):
    # Technique: the exact inverse of pack_stream, and defensive at every step —
    # check length, check the magic marker, check the hash. Never trust input blindly;
    # raise ValueError the moment something doesn't add up.
    if len(data) < META_SIZE:
        raise ValueError('stream too short for metadata')
    magic, file_len, digest, name_len = struct.unpack_from(META_FMT, data)
    if magic != spec.STREAM_MAGIC:
        raise ValueError('bad stream magic')
    body = data[META_SIZE:]
    if len(body) != name_len + file_len:
        raise ValueError('stream length does not match metadata')
    name = body[:name_len].decode('utf-8', errors='replace')
    file_bytes = body[name_len:]
    if hashlib.sha256(file_bytes).digest() != digest:   # integrity check before trusting data
        raise ValueError('whole-file SHA-256 mismatch')
    return name, file_bytes

# Technique: compile a regex once at module level (not per call) when it's reused.
_RESERVED = re.compile(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?$', re.IGNORECASE)

def sanitise_name(name, fallback='databar-output.bin'):
    # Technique: clean untrusted input step by step. Take the last path component,
    # replace dangerous characters in a comprehension, strip junk, and fall back to a
    # safe default if nothing usable is left — "never trust external data".
    name = name.replace('\\', '/').split('/')[-1]
    name = ''.join('_' if (ord(ch) < 32 or ch in '<>:"|?*' or ord(ch) == 127) else ch for ch in name)
    name = name.strip().rstrip('. ')
    if not name or name in ('.', '..'):
        return fallback
    if _RESERVED.match(name):
        name = '_' + name
    return name

# ── Bits, symbols and cells ────────────────────────────────────────────────────

def bytes_to_symbols(data, nsyms):
    # Technique: numpy for bit twiddling. unpackbits explodes bytes into 0/1, we pad to
    # a multiple, reshape into rows, and the matrix-multiply (@) with powers of two
    # collapses each row of bits back into one number — no per-bit Python loop.
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    bits = np.concatenate([bits, np.zeros(nsyms * spec.RS_BITS - len(bits), dtype=np.uint8)])
    return bits.reshape(-1, spec.RS_BITS) @ (1 << np.arange(spec.RS_BITS - 1, -1, -1))

def symbols_to_bytes(syms, nbytes):
    # Technique: the inverse. Broadcasting a shift (>>) with & 1 pulls every bit out at
    # once; packbits regroups them into bytes. The [..., :nbytes*8] slice trims padding.
    syms = np.asarray(syms, dtype=np.int64)
    bits = ((syms[..., None] >> np.arange(spec.RS_BITS - 1, -1, -1)) & 1).astype(np.uint8)
    bits = bits.reshape(*syms.shape[:-1], -1)[..., :nbytes * 8]
    return np.packbits(bits, axis=-1)

def symbols_to_cells(syms):
    # Technique: split each number into fixed-width digits via shift-and-mask, fully
    # vectorised. The same idea as writing a number in another base.
    syms = np.asarray(syms, dtype=np.int64)
    shifts = np.arange(spec.CELLS_PER_SYM - 1, -1, -1) * spec.BITS_PER_CELL
    cells = (syms[..., None] >> shifts) & (spec.NUM_COLOURS - 1)
    return cells.reshape(*syms.shape[:-1], -1).astype(np.uint8)

def cells_to_symbols(cells):
    # Technique: recombine digits by shifting each into place and summing — the inverse
    # of symbols_to_cells, again with no explicit loop.
    cells = np.asarray(cells, dtype=np.int64)
    cells = cells.reshape(*cells.shape[:-1], -1, spec.CELLS_PER_SYM)
    shifts = np.arange(spec.CELLS_PER_SYM - 1, -1, -1) * spec.BITS_PER_CELL
    return (cells << shifts).sum(axis=-1)

# ── Header ─────────────────────────────────────────────────────────────────────

def pack_header(stream_id, first_id, data_len, T, nsym):
    # Technique: append a checksum (CRC-16) over the body so the reader can detect
    # corruption. A tiny function composed from struct.pack keeps the format in one place.
    body = struct.pack(spec.HEADER_FMT, spec.FORMAT_VERSION, stream_id, first_id, data_len, T, nsym)
    return body + struct.pack('>H', binascii.crc_hqx(body, 0xFFFF))

def header_cells(stream_id, first_id, data_len, T, nsym):
    # Technique: composition — chain small converters (pack -> symbols -> encode ->
    # cells). Each step has one job, and the pipeline reads left to right.
    syms = bytes_to_symbols(pack_header(stream_id, first_id, data_len, T, nsym), spec.HEADER_RS_K)
    return symbols_to_cells(rs.encode_blocks(syms[None, :], spec.HEADER_RS_NSYM)[0])

class Header:
    # Technique: __slots__ fixes the allowed attributes, saving memory and catching
    # typo'd attribute names. Good for small objects created in large numbers.
    __slots__ = ('stream_id', 'first_id', 'data_len', 'T', 'nsym')

    def __init__(self, stream_id, first_id, data_len, T, nsym):
        self.stream_id, self.first_id, self.data_len, self.T, self.nsym = stream_id, first_id, data_len, T, nsym

    @property                       # Technique: a computed, read-only attribute
    def params(self):
        return (self.data_len, self.T, self.nsym)

    # Technique: define __eq__ and __hash__ together so instances compare by value and
    # can be used in sets/dicts. getattr over __slots__ avoids repeating field names.
    def __eq__(self, other):
        return isinstance(other, Header) and all(getattr(self, a) == getattr(other, a) for a in self.__slots__)

    def __hash__(self):
        return hash(tuple(getattr(self, a) for a in self.__slots__))

def parse_header_cells(cells):
    # Technique: a "parser" that returns None instead of raising, because failure is
    # expected and common here. Each check (error-correct, CRC, field sanity) is a
    # gate; the first failure returns None so the caller can simply try the next copy.
    syms = cells_to_symbols(np.asarray(cells)[None, :])
    fixed, ok = rs.decode_blocks(syms, spec.HEADER_RS_NSYM)
    if not ok[0]:
        return None
    raw = symbols_to_bytes(fixed[0, :spec.HEADER_RS_K], spec.HEADER_BYTES).tobytes()
    body, crc = raw[:-2], struct.unpack('>H', raw[-2:])[0]   # split off the trailing checksum
    if binascii.crc_hqx(body, 0xFFFF) != crc:
        return None
    version, stream_id, first_id, data_len, T, nsym = struct.unpack(spec.HEADER_FMT, body)
    if version != spec.FORMAT_VERSION or stream_id == 0 or data_len == 0 or T < 1 or nsym < 2:
        return None
    return Header(stream_id, first_id, data_len, T, nsym)

# ── Blocks ─────────────────────────────────────────────────────────────────────

def block_layout(T, nsym, data_cells):
    # Technique: return several related values as a tuple. -(-a // b) rounds up.
    k = -(-(T + spec.BLOCK_CRC_BYTES) * 8 // spec.RS_BITS)
    n = k + nsym
    return k, n, (data_cells // spec.CELLS_PER_SYM) // n

def choose_params(data_len, data_cells, parity=spec.DEFAULT_PARITY):
    # Technique: a search loop that remembers the best result so far and stops early
    # once "good enough" is reached (break). `continue` skips invalid candidates;
    # raising at the end handles the case where nothing worked.
    data_syms = data_cells // spec.CELLS_PER_SYM
    best = None
    for B in range(max(1, round(data_syms / spec.TARGET_BLOCK_N)), data_syms // 8 + 1):
        n = data_syms // B
        if n > spec.RS_MAX_N:
            continue
        nsym = max(2, round(n * parity))
        nsym += nsym % 2            # round up to an even number
        T = (n - nsym) * spec.RS_BITS // 8 - spec.BLOCK_CRC_BYTES
        if T < spec.MIN_T:
            break
        best = (T, nsym)
        if -(-data_len // T) >= spec.MIN_K:
            break
    if best is None:
        raise ValueError('grid too small to carry a single block')
    return best

def block_crc(stream_id, symbol_id, payload):
    # Technique: checksum the identity (ids) along with the payload, so a value can
    # only be accepted under the exact id it was created for.
    return zlib.crc32(struct.pack('>II', stream_id, symbol_id & 0xFFFFFFFF) + payload)

def encode_blocks(stream_id, first_id, payloads, nsym, k):
    # Technique: build a list of rows in a loop, then hand the whole batch to numpy at
    # once — gather first, compute in bulk after.
    rows = []
    for j, p in enumerate(payloads):
        raw = p + struct.pack('>I', block_crc(stream_id, first_id + j, p))
        rows.append(bytes_to_symbols(raw, k))
    return symbols_to_cells(rs.encode_blocks(np.array(rows), nsym))

def new_stream_id():
    # Technique: generate from a cryptographic random source and loop until the value
    # meets a condition (non-zero) — "retry until valid" instead of special-casing it.
    while True:
        sid = int.from_bytes(os.urandom(4), 'big')
        if sid:
            return sid

"""Teaching example: an error-correcting code (Reed-Solomon).

Techniques worth studying here: precompute lookup tables so a hard operation
(multiply) becomes a fast table lookup; do the bulk work in batches with numpy;
and spend the expensive algorithm only on the blocks that actually have errors.
(Technical: it detects and repairs misread symbols in received data.)
"""
from functools import lru_cache
import numpy as np

# Technique: hex and shift literals spell out exact values; the comment gives meaning.
PRIM  = 0x211                  # x^9 + x^4 + 1
FIELD = 1 << 9
ORDER = FIELD - 1              # 511

def _tables():
    # Technique: build lookup tables once at import time. Multiplying is slow, but
    # adding exponents and indexing a table is fast — so we precompute exp and log.
    # Technique: preallocate lists with [0] * n, then fill by index.
    exp = [0] * (2 * ORDER)
    log = [0] * FIELD
    x = 1
    for i in range(ORDER):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & FIELD:          # bit test: did the shift overflow the field?
            x ^= PRIM          # XOR folds it back in
    # Technique: assert a sanity invariant — catch a bad constant at startup, loudly.
    assert x == 1, 'PRIM is not primitive'
    for i in range(ORDER, 2 * ORDER):
        exp[i] = exp[i - ORDER]   # duplicate the table so index+index never overruns
    return exp, log

# Technique: compute shared data at module load, keep both a Python-list and a
# numpy copy of each table — the right form for scalar vs. batched code below.
EXP, LOG = _tables()
EXP_NP = np.array(EXP, dtype=np.int32)
LOG_NP = np.array(LOG, dtype=np.int32)

def gf_mul(a, b):
    # Technique: guard clause — handle the special (zero) case first, then the
    # general case stays simple: multiply becomes "add the logs, look up the exp".
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]]

def gf_div(a, b):
    # Technique: same guard pattern; divide is "subtract logs", with % to wrap around.
    if a == 0:
        return 0
    return EXP[(LOG[a] - LOG[b]) % ORDER]

def gf_pow(a, n):
    if a == 0:
        return 0
    return EXP[(LOG[a] * n) % ORDER]   # power is "multiply the log", same table trick

def _np_mul(a, b):
    # Technique: vectorise the scalar trick above over whole arrays. One expression
    # multiplies every element; np.where re-applies the zero rule without a loop.
    out = EXP_NP[(LOG_NP[a] + LOG_NP[b]) % ORDER]
    return np.where((a == 0) | (b == 0), 0, out)

def _poly_mul(p, q):
    # Technique: the schoolbook algorithm, with a twist — polynomials are lists of
    # coefficients, and in this number system "add" is XOR (^=).
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] ^= gf_mul(a, b)
    return out

@lru_cache(maxsize=None)        # Technique: cache — the generator is the same every call
def _generator(nsym):
    # Technique: build up a result by folding in one factor at a time in a loop.
    g = [1]
    for i in range(nsym):
        g = _poly_mul(g, [1, EXP[i + 1]])
    return g

@lru_cache(maxsize=None)
def _parity_matrix(k, nsym):
    # Technique: precompute a matrix so encoding later is just a batched multiply.
    g = _generator(nsym)
    rows = np.zeros((k, nsym), dtype=np.int32)
    # Technique: iterate backwards (range(k-1, -1, -1)) because each row is derived
    # from the next; a comprehension with zip does the per-element update in one line.
    rem = g[1:]
    for i in range(k - 1, -1, -1):
        rows[i] = rem
        lead = rem[0]
        rem = rem[1:] + [0]
        if lead:
            rem = [r ^ gf_mul(lead, c) for r, c in zip(rem, g[1:])]
    return rows

@lru_cache(maxsize=None)
def _syndrome_matrix(n, nsym):
    # Technique: build a 2D table without loops using broadcasting — a column vector
    # [:, None] times a row vector [None, :] expands into a full grid of values.
    i = np.arange(n)[:, None]
    j = np.arange(nsym)[None, :]
    return EXP_NP[((j + 1) * (n - 1 - i)) % ORDER]

def encode_blocks(data, nsym):
    # Technique: batch processing. Shapes (B, k, 1) and (1, k, nsym) broadcast so one
    # call encodes all B blocks at once; bitwise_xor.reduce sums (in XOR) along an axis.
    data = np.asarray(data, dtype=np.int32)          # asarray: accept lists or arrays
    k = data.shape[1]
    prod = _np_mul(data[:, :, None], _parity_matrix(k, nsym)[None, :, :])
    parity = np.bitwise_xor.reduce(prod, axis=1)
    return np.concatenate([data, parity], axis=1)    # append parity to the data

def syndromes(blocks, nsym):
    # Technique: the same batched multiply-and-reduce. A result of all zeros means
    # "no errors found" — a cheap check to run before any expensive repair.
    blocks = np.asarray(blocks, dtype=np.int32)
    n = blocks.shape[1]
    prod = _np_mul(blocks[:, :, None], _syndrome_matrix(n, nsym)[None, :, :])
    return np.bitwise_xor.reduce(prod, axis=1)

def _correct(block, synd, nsym):
    # Technique: mutate the caller's list in place and return a bool for success/failure
    # — a common pattern when the "answer" is the edit itself.
    n = len(block)
    # The next three stages are a known recipe; each is a tight loop over coefficients.
    C, B = [1], [1]
    L, m, b = 0, 1, 1
    for r in range(nsym):
        d = synd[r]
        for i in range(1, L + 1):
            if i < len(C):
                d ^= gf_mul(C[i], synd[r - i])
        if d == 0:
            m += 1
            continue
        coef = gf_div(d, b)
        shifted = [0] * m + [gf_mul(coef, x) for x in B]
        T = C[:]
        if len(shifted) > len(C):
            C = C + [0] * (len(shifted) - len(C))
        for i, x in enumerate(shifted):
            C[i] ^= x
        if 2 * L <= r:
            L, B, b, m = r + 1 - L, T, d, 1
        else:
            m += 1
    while len(C) > 1 and C[-1] == 0:   # Technique: trim trailing zeros off a list
        C.pop()
    # Technique: several cheap sanity checks combined with `or`; any failure bails
    # out early before the costly search below.
    if L == 0 or 2 * L > nsym or len(C) - 1 != L:
        return False
    # Stage 2 (Chien search): scan every position, collecting the ones that are wrong.
    positions = []
    for i in range(n):
        p = n - 1 - i
        xinv = EXP[(ORDER - p) % ORDER]
        acc, xp = 0, 1
        for c in C:
            acc ^= gf_mul(c, xp)
            xp = gf_mul(xp, xinv)
        if acc == 0:
            positions.append(i)
    if len(positions) != L:        # the count must match, or our answer is inconsistent
        return False
    # Stage 3 (Forney): for each wrong position, compute the value to XOR back in.
    omega = [0] * nsym
    for i, s in enumerate(synd):
        if s:
            for j, c in enumerate(C):
                if i + j < nsym:
                    omega[i + j] ^= gf_mul(s, c)
    for i in positions:
        p = n - 1 - i
        xinv = EXP[(ORDER - p) % ORDER]
        num, xp = 0, 1
        for o in omega:
            num ^= gf_mul(o, xp)
            xp = gf_mul(xp, xinv)
        den, xp2 = 0, 1
        x2 = gf_mul(xinv, xinv)
        for j in range(1, len(C), 2):
            den ^= gf_mul(C[j], xp2)
            xp2 = gf_mul(xp2, x2)
        if den == 0:
            return False
        block[i] ^= gf_div(num, den)   # apply the fix in place
    return True

def decode_blocks(blocks, nsym):
    # Technique: do the cheap test on everything first, then the expensive fix only
    # where needed. ~synd.any(axis=1) flags clean blocks; np.flatnonzero lists the
    # rest, and only those loop through _correct. "ok" reports which blocks are trusted.
    blocks = np.array(blocks, dtype=np.int32)
    synd = syndromes(blocks, nsym)
    ok = ~synd.any(axis=1)
    for bi in np.flatnonzero(~ok):
        block = blocks[bi].tolist()
        if _correct(block, synd[bi].tolist(), nsym):
            blocks[bi] = block
            ok[bi] = True
    return blocks, ok

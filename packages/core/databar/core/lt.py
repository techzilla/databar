"""Teaching example: a deterministic pseudo-random algorithm.

Techniques to notice: a reproducible random generator (same seed -> same output
on every machine), using Python's big integers and XOR as the core operation,
and an incremental solver that accepts input a piece at a time. The guiding idea
is determinism: the sender and receiver must agree without ever talking, so
nothing here is allowed to vary by platform (hence the hand-written log below).
(Technical: an LT fountain code that rebuilds data from any-enough packets.)

Symbols are plain Python ints — XOR of big ints is fast and allocates little.
"""
from bisect import bisect_right
from functools import lru_cache
import heapq, math

from . import spec

M64 = (1 << 64) - 1            # Technique: a bitmask to keep values in 64 bits (& M64)

class SplitMix64:
    # Technique: a tiny class holding one piece of state. Each .next() updates self
    # and returns a number, so the sequence is repeatable from the seed alone.
    def __init__(self, seed):
        self.state = seed & M64

    def next(self):
        # Technique: fixed integer constants + shifts/XOR/multiply, all masked to 64
        # bits, give the same stream everywhere — no reliance on the platform's RNG.
        self.state = (self.state + 0x9E3779B97F4A7C15) & M64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def below(self, n):
        # Technique: map a 64-bit random number into [0, n) with multiply-then-shift,
        # avoiding the bias and the division that plain % would introduce.
        return (self.next() * n) >> 64

_LN2 = 0.6931471805599453

def _ln(x):
    # Technique: implement a function from scratch for determinism. math.log can
    # differ in the last bit between systems, so we sum a series with only +, -, *, /
    # — operations guaranteed identical everywhere.
    m, e = math.frexp(x)        # split x into mantissa m and exponent e (exact)
    z = (m - 1.0) / (m + 1.0)
    z2, term, total = z * z, z, 0.0
    for k in range(1, 60, 2):   # add odd-numbered terms until the series converges
        total += term / k
        term *= z2
    return e * _LN2 + 2.0 * total

@lru_cache(maxsize=32)          # Technique: cache by arguments — same (K,c,delta) is reused
def degree_thresholds(K, c, delta):
    # Technique: precompute a cumulative distribution as a sorted list of integer
    # thresholds, so later a single random number + binary search picks an outcome.
    R = c * _ln(K / delta) * math.sqrt(K)
    spike = max(1, min(K, int(K / R))) if R > 0 else K   # clamp into range, guard R==0
    weights = [0.0] * (K + 1)
    weights[1] = 1.0 / K
    for d in range(2, K + 1):               # assign each outcome a weight
        weights[d] = 1.0 / (d * (d - 1))
    for d in range(1, spike):
        weights[d] += R / (d * K)
    weights[spike] += R * _ln(R / delta) / K if R > delta else 0.0
    total = sum(weights)
    # Technique: running sum (acc) turns weights into a cumulative scale, rescaled to
    # whole numbers out of 2**32 so the lookup needs no floats.
    thresholds, acc = [], 0.0
    for d in range(1, K + 1):
        acc += weights[d]
        thresholds.append(min(1 << 32, int(acc / total * (1 << 32))))
    thresholds[-1] = 1 << 32                 # force the last bucket to cover everything
    return thresholds

def _is_prime(n):
    # Technique: trial division only up to sqrt(n) (while f*f <= n) — a factor larger
    # than the square root would already have a matching smaller factor.
    if n < 2:
        return False
    f = 2
    while f * f <= n:
        if n % f == 0:
            return False
        f += 1
    return True

def precode_size(K):
    # Technique: -(-a // b) is the integer "round up" (ceiling division) idiom.
    # Then step forward to the next prime with a simple while loop.
    P = max(spec.PRECODE_MIN, -(-K * spec.PRECODE_PERCENT // 100))
    while not _is_prime(P):
        P += 1
    return P

@lru_cache(maxsize=8)
def precode_checks(K):
    # Technique: use a set to pick distinct items — keep drawing until the set holds
    # 3, which guarantees no duplicates. Seeding the RNG with K makes the "random"
    # layout identical on both sides.
    # Technique: a list of lists as a simple bucket structure (checks[b] grows).
    P = precode_size(K)
    checks = [[] for _ in range(P)]
    rng = SplitMix64(K)
    for i in range(K):
        picked = set()
        while len(picked) < 3:
            picked.add(rng.below(P))
        for b in picked:
            checks[b].append(i)
    return checks

def neighbours(stream_id, symbol_id, L):
    # Technique: early return for the simple case keeps the hard path uncluttered.
    if symbol_id < L:
        return [symbol_id]
    # Technique: seed from two values packed into one int, so each symbol id gives a
    # repeatable draw. bisect_right does the CDF lookup in log time.
    rng = SplitMix64((stream_id << 32) | (symbol_id & 0xFFFFFFFF))
    d = bisect_right(degree_thresholds(L, spec.LT_C, spec.LT_DELTA), rng.next() >> 32) + 1
    d = min(d, L)
    # Technique: choose the cheaper of two ways to pick d items — pick d directly, or
    # pick the (L-d) to leave out — whichever set is smaller.
    if d * 2 <= L:
        picked = set()
        while len(picked) < d:
            picked.add(rng.below(L))
        return list(picked)
    excluded = set()
    while len(excluded) < L - d:
        excluded.add(rng.below(L))
    return [i for i in range(L) if i not in excluded]   # comprehension with a filter

class Encoder:
    # Technique: precompute in __init__ (the parity symbols), then serve many cheap
    # requests from symbol(). Do expensive setup once, not per call.
    def __init__(self, stream_id, source_symbols):
        self.stream_id = stream_id
        self.K = len(source_symbols)
        # Technique: an accumulator loop — start acc at 0 and fold values in with XOR.
        parity = []
        for members in precode_checks(self.K):
            acc = 0
            for i in members:
                acc ^= source_symbols[i]
            parity.append(acc)
        self.intermediate = list(source_symbols) + parity
        self.L = len(self.intermediate)

    def symbol(self, symbol_id):
        # Technique: generate output on demand. Low ids return a stored symbol; higher
        # ids are XORed together on the fly, so the stream is effectively endless.
        if symbol_id < self.L:
            return self.intermediate[symbol_id]
        acc = 0
        for i in neighbours(self.stream_id, symbol_id, self.L):
            acc ^= self.intermediate[i]
        return acc

# Technique: named states for a small state machine — each symbol is in one of these.
UNRESOLVED, RESOLVED, INACTIVE = 0, 1, 2

class Decoder:
    # Technique: an incremental algorithm. Instead of waiting for all input, it
    # accepts one piece at a time with add() and makes as much progress as it can,
    # tracking everything in instance attributes. .done flips True when finished.
    # (Technical: peeling decoder with inactivation — solve the easy parts greedily,
    # set the stubborn unknowns aside, then finish them with a small linear solve.)

    def __init__(self, stream_id, K):
        # Technique: initialise every piece of bookkeeping up front, each with a
        # comment naming its job. Choosing the right container for each (below) is
        # half the design: bytearray/list for dense per-symbol data indexed by int,
        # dict for sparse maps, set for membership, list-as-heap for "smallest next".
        self.stream_id = stream_id
        self.K = K
        self.L = K + precode_size(K)
        self.state = bytearray(self.L)     # one state byte per symbol (compact)
        self.value = [0] * self.L
        self.mask = [0] * self.L
        self.unresolved = self.L
        self.recovered = 0          # how many source symbols are fully known
        self.inactive = []          # column -> symbol index
        self.column = {}            # symbol index -> column
        self.pending = {}           # equations not yet solvable
        self.index = {}             # symbol index -> which equations mention it
        self.heap = []              # priority queue: always pop the smallest equation
        self.pivots = {}            # the small side-system kept in solved form
        self.seen = set()           # ids already added, to ignore duplicates
        self.received = 0
        self.done = False
        self._next_key = 0
        for j, members in enumerate(precode_checks(K)):
            self._equation(list(members) + [K + j], 0, 0)

    @property                       # Technique: expose a derived value as an attribute
    def inactivated(self):
        return len(self.inactive)

    def add(self, symbol_id, value):
        # Technique: make the method idempotent — a duplicate id returns False and
        # changes nothing, so callers can feed the same data twice safely.
        if self.done or symbol_id in self.seen:
            return False
        self.seen.add(symbol_id)
        self.received += 1
        self._equation(neighbours(self.stream_id, symbol_id, self.L), value, 0)
        # Technique: `while cond and step(): pass` drives a process until it can make
        # no more progress — here, keep setting unknowns aside while peeling is stuck.
        if self.received >= self.K:
            while self.unresolved and self._inactivate():
                pass
        if not self.unresolved and len(self.pivots) == len(self.inactive):
            self._finish()
        return True

    def result(self):
        return self.value[:self.K]   # slice off just the original source symbols

    # ── Peeling ────────────────────────────────────────────────────────────────

    def _equation(self, members, value, mask):
        # Technique: classify inputs into buckets, then branch on how many remain.
        # Known terms are folded into value/mask immediately; only unknowns are kept.
        unknown = set()
        state = self.state          # local alias: a tiny speed-up for a hot loop
        for i in members:
            st = state[i]
            if st == RESOLVED:
                value ^= self.value[i]
                mask ^= self.mask[i]
            elif st == INACTIVE:
                mask ^= 1 << self.column[i]      # record dependence as a set bit
            else:
                unknown.add(i)
        # Three cases: nothing unknown (a constraint), exactly one (solve it now), or
        # many (park it until later). Dispatching on a count keeps each path small.
        if not unknown:
            self._constrain(mask, value)
        elif len(unknown) == 1:
            self._resolve(unknown.pop(), value, mask)
        else:
            key = self._next_key
            self._next_key += 1
            self.pending[key] = [unknown, value, mask]
            # Technique: setdefault builds a dict-of-sets reverse index lazily.
            for i in unknown:
                self.index.setdefault(i, set()).add(key)
            heapq.heappush(self.heap, (len(unknown), key))   # keep equations ordered by size

    def _resolve(self, i, value, mask):
        # Technique: an explicit stack instead of recursion. Solving one symbol can
        # unlock others; pushing them on a list avoids deep recursive calls.
        stack = [(i, value, mask)]
        while stack:
            i, value, mask = stack.pop()
            st = self.state[i]
            # Technique: guard + continue to handle the "already known" cases and skip
            # the rest of the loop body, keeping the main case un-indented.
            if st == RESOLVED:
                self._constrain(mask ^ self.mask[i], value ^ self.value[i])
                continue
            if st == INACTIVE:
                self._constrain(mask ^ (1 << self.column[i]), value)
                continue
            self.state[i] = RESOLVED
            self.value[i] = value
            self.mask[i] = mask
            self.unresolved -= 1
            if i < self.K and not mask:        # a source symbol with no leftover dependence
                self.recovered += 1
            # Technique: dict.pop(i, ()) reads-and-removes with a safe default, so each
            # equation that mentioned i is updated once. Shrinking an equation to one
            # unknown pushes more work onto the stack — the "peeling" cascade.
            for key in self.index.pop(i, ()):
                entry = self.pending.get(key)
                if entry is None:
                    continue
                entry[0].discard(i)
                entry[1] ^= value
                entry[2] ^= mask
                n = len(entry[0])
                if n == 1:
                    del self.pending[key]
                    j = next(iter(entry[0]))   # pull the lone remaining unknown out
                    self.index[j].discard(key)
                    stack.append((j, entry[1], entry[2]))
                elif n == 0:
                    del self.pending[key]
                    self._constrain(entry[2], entry[1])
                else:
                    heapq.heappush(self.heap, (n, key))

    def _inactivate(self):
        # Technique: lazy deletion from a heap. Entries are never removed when they
        # change; instead, on pop we check the size still matches and skip stale ones.
        # A for/while...else runs the else only if the loop never `break`s (= empty).
        while self.heap:
            n, key = heapq.heappop(self.heap)
            entry = self.pending.get(key)
            if entry is not None and len(entry[0]) == n:
                break
        else:
            return False
        # Technique: sorted() with a key function picks the symbol to keep (the one in
        # the fewest other equations), a greedy heuristic to keep later work small.
        members = sorted(entry[0], key=lambda i: len(self.index.get(i, ())))
        ready = []
        for i in members[1:]:
            col = len(self.inactive)
            self.inactive.append(i)
            self.column[i] = col
            self.state[i] = INACTIVE
            self.unresolved -= 1
            bit = 1 << col
            for k in self.index.pop(i, ()):
                e = self.pending.get(k)
                if e is None:
                    continue
                e[0].discard(i)
                e[2] ^= bit
                if len(e[0]) <= 1:
                    ready.append(k)
                else:
                    heapq.heappush(self.heap, (len(e[0]), k))
        for k in ready:
            e = self.pending.pop(k, None)   # may be gone already — tolerate it
            if e is None:
                continue
            if e[0]:
                j = e[0].pop()
                self.index[j].discard(k)
                self._resolve(j, e[1], e[2])
            else:
                self._constrain(e[2], e[1])
        return True

    # ── Inactive system ────────────────────────────────────────────────────────

    def _constrain(self, mask, value):
        # Technique: keep a linear system in "solved as we go" form. mask & -mask
        # isolates the lowest set bit, and .bit_length()-1 turns it into a column
        # number — a fast way to walk the bits of an integer used as a bit-set.
        pivots = self.pivots
        while mask:
            c = (mask & -mask).bit_length() - 1
            p = pivots.get(c)
            if p is None:
                pivots[c] = (mask, value)    # new pivot: store and stop
                return
            mask ^= p[0]                     # otherwise cancel against an existing pivot
            value ^= p[1]

    def _finish(self):
        # Technique: back-substitution — solve from the last column to the first, so
        # each row's other unknowns are already known by the time we reach it.
        n = len(self.inactive)
        sol = [0] * n
        for c in range(n - 1, -1, -1):
            row, value = self.pivots[c]
            rest, j = row >> (c + 1), c + 1
            while rest:
                shift = (rest & -rest).bit_length() - 1   # jump to the next set bit
                j += shift
                value ^= sol[j]
                rest >>= shift + 1
                j += 1
            sol[c] = value
        # Technique: precompute XOR lookup tables (256 entries per byte) so combining
        # many values becomes a few table lookups instead of a bit-by-bit loop —
        # trading a little memory for a lot of speed in the final substitution.
        tables = []
        for base in range(0, n, 8):
            t = [0] * 256
            for b in range(1, 256):
                low = b & -b
                col = base + low.bit_length() - 1
                t[b] = t[b ^ low] ^ (sol[col] if col < n else 0)   # build on a smaller entry
            tables.append(t)
        nbytes = (n + 7) // 8                # ceiling division, bits -> bytes
        for i in range(self.K):
            st = self.state[i]
            if st == INACTIVE:
                self.value[i] = sol[self.column[i]]
            elif self.mask[i]:
                v = self.value[i]
                # Technique: iterate the mask one byte at a time and apply its table.
                for t, b in zip(tables, self.mask[i].to_bytes(nbytes, 'little')):
                    if b:
                        v ^= t[b]
                self.value[i] = v
        self.recovered = self.K
        self.done = True
        # Technique: drop big working structures once finished to free memory.
        self.pending = self.index = self.heap = None

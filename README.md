# databar — a worked example for advanced programming

**databar is a small, complete Python codebase used here as teaching material for advanced coding concepts.** It is built around one realistic case study — encoding data for transmission across a noisy, one-way channel — but the application is beside the point. What matters is that a single coherent program exercises an unusually wide span of advanced topics at once: coding theory, probability, linear algebra over finite fields, numerical reproducibility, array programming, concurrency, and clean software architecture.

Every function is annotated with a comment naming the **technique** it demonstrates, so the code reads as a guided tour rather than a black box. The case study is just the scaffolding that lets these techniques appear together, interacting, the way they do in real software — instead of as isolated textbook snippets.

> This is a learning resource. The algorithms are genuine and correct; the repo exists so you can read, run, break, and extend them.

---

## Who this is for

Students and self-teachers who already know Python basics and want to see *advanced* ideas applied together in one honest, end-to-end program — the kind of thing a data-structures-and-algorithms course, a coding-theory unit, or a numerical-computing lab rarely shows in a single coherent place.

If you have ever asked *"I understand the concept in isolation, but what does it look like in real code?"* — that is the gap this repo fills.

---

## How to read it

Each source file opens with a docstring stating the techniques it teaches, and nearly every block carries a `# Technique: …` comment. A good path:

1. **[spec.py](packages/core/databar/core/spec.py)** — start here. Pure configuration: how to centralise constants and why.
2. **[layout.py](packages/core/databar/core/layout.py)** — numpy array construction, slicing, caching, validation.
3. **[rs.py](packages/core/databar/core/rs.py)** — Reed–Solomon coding over a finite field.
4. **[lt.py](packages/core/databar/core/lt.py)** — the LT fountain encoder and its degree distribution: the richest file, worth the most time.
5. **[stream.py](packages/core/databar/core/stream.py)** — binary formats and hashing.
6. **[sender.py](packages/encode/databar/encode/sender.py)** → **[display.py](packages/encode/databar/encode/display.py)** — generators and real-time concurrency.
7. **[cli.py](packages/encode/databar/encode/cli.py)** — how a library becomes a command-line tool.

---

## Concept map — where each technique lives

### Algorithms & data structures

| Concept | Where to look |
| --- | --- |
| Finite-field (Galois) arithmetic via log/exp lookup tables | [rs.py](packages/core/databar/core/rs.py) — `_tables`, `gf_mul` |
| Reed–Solomon encoding (systematic, batched) | [rs.py](packages/core/databar/core/rs.py) — `encode_blocks`, `_parity_matrix` |
| LT fountain code + LDPC precode (rateless coding) | [lt.py](packages/core/databar/core/lt.py) — `Encoder`, `precode_checks`, `neighbours` |
| CRC checksums (CRC-8/16/32) for integrity | [layout.py](packages/core/databar/core/layout.py) `_crc8`, [stream.py](packages/core/databar/core/stream.py) `block_crc` |

### Probability, numerics & reproducibility

| Concept | Where to look |
| --- | --- |
| Robust soliton degree distribution (why fountain codes work) | [lt.py](packages/core/databar/core/lt.py) — `degree_thresholds` |
| Sampling from a distribution via CDF + binary search | [lt.py](packages/core/databar/core/lt.py) — `neighbours` (`bisect`) |
| A deterministic PRNG (SplitMix64) for cross-platform agreement | [lt.py](packages/core/databar/core/lt.py) — `SplitMix64` |
| Bias-free range reduction (multiply-shift, not modulo) | [lt.py](packages/core/databar/core/lt.py) — `SplitMix64.below` |
| Re-implementing `ln` from IEEE ops to avoid platform drift | [lt.py](packages/core/databar/core/lt.py) — `_ln` |

### Array programming (numpy / SIMD thinking)

| Concept | Where to look |
| --- | --- |
| Vectorising a scalar operation over whole arrays | [rs.py](packages/core/databar/core/rs.py) — `_np_mul`, `encode_blocks` |
| Broadcasting (outer-product style) to split numbers into digits | [stream.py](packages/core/databar/core/stream.py) — `symbols_to_cells` |
| Bit-packing/unpacking and base conversion | [stream.py](packages/core/databar/core/stream.py) — `bytes_to_symbols` |
| Boolean masks and fancy indexing | [layout.py](packages/core/databar/core/layout.py) — `Layout` |

### Systems, concurrency & I/O

| Concept | Where to look |
| --- | --- |
| Producer/consumer threads with a `Condition` | [display.py](packages/encode/databar/encode/display.py) — `producer`, `texture` |
| Clock-driven timing (behaviour independent of hardware speed) | [display.py](packages/encode/databar/encode/display.py) — main loop |
| Lazy / on-demand generation of an endless stream | [sender.py](packages/encode/databar/encode/sender.py) — `frame_cells` |
| Optional dependencies, exit codes, `argparse` custom types | [cli.py](packages/encode/databar/encode/cli.py) |

### Software design & Python craft

| Concept | Where to look |
| --- | --- |
| Memoisation with `@lru_cache` | throughout `core/` |
| `__slots__`, custom `__eq__`/`__hash__`, `@property` | [stream.py](packages/core/databar/core/stream.py) — `Header` |
| Guard clauses and fail-fast validation | [layout.py](packages/core/databar/core/layout.py) `Layout.__init__`, [stream.py](packages/core/databar/core/stream.py) `pack_stream` |
| Package structure, `__all__`, `python -m` entry points | the `__init__.py` / `__main__.py` files |

---

## Academic angles worth pursuing

- **Information & coding theory.** The codebase combines an inner code (Reed–Solomon) with an outer rateless code (an LT fountain) — a compact lesson in why real systems concatenate codes. Companion reading: Shannon's channel capacity, the Singleton bound, and Luby's original LT paper.
- **Linear algebra over finite fields.** The codes are built on arithmetic over GF(2⁹) and GF(2), where XOR plays the role of addition. See the field tables in `rs.py` and the symbol combination in `lt.py`'s `Encoder`.
- **Probability & random graphs.** The robust soliton distribution in `degree_thresholds` is tuned so the code's underlying graph is *just barely* solvable. A hands-on entry to random-graph thresholds and the "coupon collector" intuition behind the precode.
- **Numerical reproducibility.** `_ln` and `SplitMix64` exist because two independent programs must agree bit-for-bit without ever communicating. A rare, concrete look at why `math.log` and `random` are *not* safe across machines — relevant to distributed systems and ML determinism.
- **Performance engineering.** Nearly every hot path trades a Python loop for a numpy expression, or a computation for a precomputed table. Benchmarking those swaps (see the exercises) teaches *where* speed actually comes from.
- **Designing for a one-way, lossy channel.** No acknowledgements, no retransmission, an arbitrary start time. A clean case study in robustness under constraints the sender cannot observe.

---

## Try it

Python 3.10+. Install the core library and the encoding side:

```sh
pip install ./packages/core ./packages/encode
```

Encode a file into the channel representation:

```sh
databar-encode report.pdf out.mkv
```

Or explore the core algorithms directly, with nothing but `databar-core` installed:

```python
from databar.core import lt
enc = lt.Encoder(stream_id=1, source_symbols=[10, 20, 30, 40])
print([enc.symbol(i) for i in range(8)])   # an endless stream of distinct coded symbols
```

---

## Suggested exercises

1. **Coding theory.** Vary the Reed–Solomon parity (`--parity`) and watch how `choose_params` trades symbol size `T` against overhead; relate parity to the RS(n, k) block shape.
2. **Probability.** Plot the degree distribution from `degree_thresholds` for several `K`. How does the "spike" move, and how does total overhead change?
3. **Determinism.** Replace `SplitMix64` with Python's `random` seeded identically, and explain why two independent programs could still disagree across machines.
4. **Performance.** Rewrite `_np_mul` as a plain Python double loop, benchmark both, and quantify the speed-up from vectorisation.
5. **Concurrency.** Remove the background producer thread in `display.py` and build frames inline; measure the throughput hit and reason about why.
6. **Array programming.** Call `grid_for_screen` for several screen sizes and aspect ratios, and explain the cell-count and clamping behaviour.

---

## Repository layout

| Path | Contents |
| --- | --- |
| [packages/core/](packages/core/) | `databar.core`: the shared core — constants, layout, RS and LT codes, binary packing |
| [packages/encode/](packages/encode/) | `databar.encode`: the encoding side and its command-line tool |

Start with the comments. Every file is written to be read.

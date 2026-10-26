"""Teaching example: a real-time loop with a background worker thread.

Techniques: the producer/consumer pattern (one thread builds frames ahead, the
main thread draws them), a threading.Condition to coordinate them safely, and
timing against a clock so behaviour doesn't depend on the hardware's speed.
"""
import threading, time
from collections import OrderedDict

import numpy as np

from databar.core import spec
from databar.core.layout import grid_for_screen
from .sender import Sender

LOOKAHEAD = 8                   # how many frames the worker may build ahead of the display

def run(file_bytes, name, profile, frame_time, parity, window_size=None, log=print):
    # Technique: import a heavy library inside the function, not at module top, so the
    # module loads fast and only pays for pyglet when this path actually runs.
    import pyglet
    from pyglet import gl

    # Technique: *window_size unpacks a tuple into positional arguments.
    if window_size:
        window = pyglet.window.Window(*window_size, vsync=True, caption='databar')
    else:
        window = pyglet.window.Window(fullscreen=True, vsync=True, caption='databar')
        window.set_mouse_visible(False)
    fb_w, fb_h = window.get_framebuffer_size()
    px_scale = fb_w / window.width                    # a ratio, named so later maths is clear
    cols, rows, cell_px = grid_for_screen(fb_w, fb_h, profile)
    sender = Sender(file_bytes, name, cols, rows, parity)
    log(f'Screen {fb_w}x{fb_h}: grid {cols}x{rows}, {cell_px} px per cell, profile {profile}')
    log(f'Stream 0x{sender.stream_id:08X}: {len(sender.data)} bytes, K={sender.K}, '
        f'{sender.B} symbols per frame, first pass {sender.first_pass_frames * frame_time:.0f} s')
    log('Press Esc to stop.')

    pyglet.image.Texture.default_min_filter = gl.GL_NEAREST   # keep sharp edges when scaling
    pyglet.image.Texture.default_mag_filter = gl.GL_NEAREST
    w_units, h_units = cols * cell_px / px_scale, rows * cell_px / px_scale
    x0, y0 = (window.width - w_units) / 2, (window.height - h_units) / 2

    # Technique: shared state between threads. An OrderedDict as a cache, a Condition as
    # the lock+signal, and a one-element list `wanted` as a mutable box threads can share.
    frames, lock, wanted = OrderedDict(), threading.Condition(), [0]
    stop = threading.Event()        # Technique: an Event is a thread-safe on/off flag

    def producer():
        # Technique: the worker. `with lock:` guards shared data; lock.wait() sleeps until
        # notified instead of busy-spinning. It also trims the cache so memory stays bounded.
        f = 0
        while not stop.is_set():
            with lock:
                while not stop.is_set() and (f - wanted[0] >= LOOKAHEAD):
                    lock.wait(0.05)
                f = max(f, wanted[0])          # skip ahead if the consumer jumped
            rgb = spec.PALETTE_BGR[sender.frame_cells(f)][:, :, ::-1]   # [::-1] flips BGR->RGB
            with lock:
                frames[f] = np.ascontiguousarray(rgb).tobytes()
                while len(frames) > 2 * LOOKAHEAD:
                    frames.popitem(last=False)   # drop the oldest cached frame
                lock.notify_all()
            f += 1

    # Technique: daemon=True lets the program exit even if this thread is still running.
    threading.Thread(target=producer, daemon=True).start()

    def texture(f):
        # Technique: a closure — this inner function reads `lock`, `frames`, `wanted`
        # from the enclosing scope. It requests frame f, then waits until it appears.
        with lock:
            wanted[0] = f
            lock.notify_all()
            while f not in frames:
                lock.wait()
            data = frames[f]
        img = pyglet.image.ImageData(cols, rows, 'RGB', data, pitch=-cols * 3)
        return img.get_texture()

    # Technique: drive animation off a monotonic clock (perf_counter), so which frame is
    # due is computed from elapsed time — independent of how fast the loop actually spins.
    t0 = time.perf_counter()
    shown, tex, last_log = -1, None, t0
    try:
        while not window.has_exit:
            window.dispatch_events()
            f = int((time.perf_counter() - t0) / frame_time)
            if f != shown:                  # only rebuild the texture when the frame changes
                tex, shown = texture(f), f
            window.clear()
            tex.blit(x0, y0, width=w_units, height=h_units)
            window.flip()
            now = time.perf_counter()
            if now - last_log > 2:          # Technique: throttle logging to every ~2 seconds
                last_log = now
                sent = (shown + 1) * sender.B
                print(f'\r  frame {shown + 1}, {sent} symbols sent ({sent / sender.K:.0%} of K)',
                      end='', flush=True)
    finally:
        # Technique: a finally block guarantees cleanup (stop the thread, close the window)
        # even if the loop errors out or is interrupted.
        stop.set()
        window.close()
        print()

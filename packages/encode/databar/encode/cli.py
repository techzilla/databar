"""Teaching example: a command-line program.

Techniques: define arguments with argparse (the module's docstring becomes the
help text), write a custom type that validates input, handle an optional
dependency gracefully, and report errors by exiting with a message and non-zero
status. main() is the single entry point.
"""
import argparse, os, sys

from databar.core import spec
from databar.core.layout import grid_for_screen
from .sender import Sender, render

def parse_size(text):
    # Technique: a custom argparse type — parse/validate once, and raise the library's
    # own error type so the user gets a clean message instead of a traceback.
    try:
        w, h = (int(v) for v in text.lower().split('x'))
        return w, h
    except ValueError:
        raise argparse.ArgumentTypeError(f'expected WIDTHxHEIGHT, got {text!r}')

def write_video(args, file_bytes, name, frame_time):
    # Technique: import-inside-try to treat a dependency as optional. If OpenCV isn't
    # installed, exit with a helpful message rather than crashing on ImportError.
    try:
        import cv2
    except ImportError:
        sys.exit('Writing video needs OpenCV: pip install "databar-encode[video]"')
    sw, sh = args.screen
    cols, rows, cell_px = grid_for_screen(sw, sh, args.profile)
    sender = Sender(file_bytes, name, cols, rows, args.parity)
    n_frames = -(-int(sender.first_pass_frames * args.loop_factor * 1000) // 1000)
    fps = 1 / frame_time
    print(f'File:      {name} ({len(file_bytes)} bytes)')
    print(f'Grid:      {cols}x{rows} cells, {cell_px} px per cell on {sw}x{sh}, profile {args.profile}')
    print(f'Stream ID: 0x{sender.stream_id:08X}; K={sender.K} symbols of {sender.T} B, '
          f'{sender.B} per frame, RS({sender.n},{sender.k}) blocks')
    print(f'Video:     {n_frames} frames at {fps:g} fps = {n_frames * frame_time:.1f} s per loop '
          f'(first pass {sender.first_pass_frames * frame_time:.1f} s)')

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'FFV1'), fps, (sw, sh))
    if not writer.isOpened():          # check a resource opened before using it
        sys.exit(f'Cannot open {args.output} for FFV1 video (use a .mkv or .avi name)')
    for f in range(n_frames):
        writer.write(render(sender.frame_cells(f), cell_px, sw, sh))
        if f % 25 == 0 or f == n_frames - 1:   # Technique: modulo to act every Nth iteration
            print(f'\r  frame {f + 1}/{n_frames}', end='', flush=True)   # \r redraws one line
    writer.release()                   # release the resource when done
    print(f'\nWritten: {args.output}')

def main():
    # Technique: build the parser, declaring each option once. Passing __doc__ reuses the
    # module docstring as help, so documentation and code can't drift apart.
    ap = argparse.ArgumentParser(prog='databar-encode', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('output', nargs='?', help='video file (.mkv or .avi); omit with --show')
    ap.add_argument('--show', action='store_true', help='display full screen instead of writing a video')
    ap.add_argument('--profile', choices=sorted(spec.PROFILES), default=spec.DEFAULT_PROFILE,
                    help=f'grid profile (default {spec.DEFAULT_PROFILE}); 4k-* only for 4K recordings')
    ap.add_argument('--fast', action='store_true',
                    help=f'{spec.FAST_FRAME_TIME * 1000:.0f} ms per data frame, for 60 fps recordings')
    ap.add_argument('--parity', type=float, default=spec.DEFAULT_PARITY,
                    help=f'share of each block spent on Reed-Solomon parity (default {spec.DEFAULT_PARITY})')
    ap.add_argument('--screen', type=parse_size, default=(1920, 1080),
                    help='video mode: screen size the video will be played on (default 1920x1080)')
    ap.add_argument('--loop-factor', type=float, default=1.5,
                    help='video mode: loop length as a multiple of one pass of the data (default 1.5)')
    ap.add_argument('--window', type=parse_size, help='show mode: use a window of this size, for testing')
    args = ap.parse_args()

    # Technique: validate combinations of arguments after parsing. `==` on two bools is a
    # neat "exactly one of these" test — error out if both or neither was given.
    if args.show == bool(args.output):
        ap.error('give either an output video file or --show')
    with open(args.input, 'rb') as f:   # Technique: `with` closes the file automatically
        file_bytes = f.read()
    name = os.path.basename(args.input)
    frame_time = spec.FAST_FRAME_TIME if args.fast else spec.FRAME_TIME   # pick based on a flag
    # Technique: wrap the real work in try/except so expected errors become a clean
    # message + exit code, not a scary traceback for the user.
    try:
        if args.show:
            from . import display
            display.run(file_bytes, name, args.profile, frame_time, args.parity, args.window)
        else:
            write_video(args, file_bytes, name, frame_time)
    except ValueError as e:
        sys.exit(f'Error: {e}')

# Technique: the __name__ == '__main__' guard runs main() only when this file is executed
# directly, not when it is imported — a standard Python idiom.
if __name__ == '__main__':
    main()

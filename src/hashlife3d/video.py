import subprocess

import colour
import numpy as np
from ranges import Range

from .camera import snapshot_from_grid
from .grid import LazyGrid
from .extent import CuboidExtent

def _ffmpeg_command(resolution, fps, output):
    return [
        'ffmpeg',
        '-y',  # Overwrite output file if exists
        '-f', 'rawvideo',
        '-pixel_format', 'rgb48le',
        '-s:v', f'{resolution.x}x{resolution.y}',
        '-r', str(fps),
        '-i', '-',
        # '-vf', 'format=yuv444p10le',
        '-c:v', 'libx265',
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-pix_fmt', 'yuv420p10le',
        '-x265-params', 'transfer=smpte2084:colorprim=bt2020:colormatrix=bt2020nc',
        '-crf', '16',
        '-preset', 'ultrafast',
        output
    ]

def _to_yuv(luminances, peak_brightness=1000):
    y_signal_float = colour.models.eotf_inverse_ST2084(luminances * peak_brightness)
    # Scale to 10-bit depth and cast to uint16
    y = (y_signal_float * 1023).round().astype(np.uint16)
    u = np.full_like(y, 512, dtype=np.uint16)
    v = np.full_like(y, 512, dtype=np.uint16)
    yuv_frame = np.dstack((y, u, v))
    return yuv_frame.tobytes()

def _to_rgb(luminances, peak_brightness=1000):
    y_signal_float = colour.models.eotf_inverse_ST2084(luminances * peak_brightness)
    # Scale to 16-bit depth and cast to uint16
    luma = np.clip((y_signal_float * (2**16-1)).round(), 0, 2**16-1).astype(np.uint16)
    r = g = b = luma
    rgb_frame = np.dstack((r, g, b))
    return rgb_frame.tobytes()

def create_video(grid: LazyGrid, speed_fn, view_fn, resolution, duration, output, initial_generation=0, fps=60, luminance_fn=lambda density, seconds: density, over_depth=lambda sec: 0):
    """
    Create a video of `grid`.

    :param grid: The grid to create a video of.
    :param speed_fn: A function mapping the time in seconds (starting at 0) into
        the number of generations to advance per second. Can be less than 1.
        Re-evaluated every frame.
    :param view_fn: A function mapping the time into the rectangle to be shown
        this frame. The height of the rectangle is ignored.
    :param resolution: The resolution of the video, in pixels.
    :param duration: The duration of the video, in frames.
    :param output: The output file to write the video to.
    :param initial_generation: The initial generation of the video.
    :param fps: The number of frames per second to generate the video at.
    :param luminance_fn: A function mapping the density (0 to 1) of a pixel to
        its luminance (0 to 1). It is passed the time in seconds as a second
        argument.
    """
    t = float(initial_generation)
    args = _ffmpeg_command(resolution, fps, output)
    process = subprocess.Popen(args, stdin=subprocess.PIPE)
    for frame in range(int(duration * fps)):
        seconds = frame / fps
        generations = speed_fn(seconds) / fps
        rectangle = view_fn(seconds)
        cuboid = CuboidExtent(
            x_range=rectangle.x_range,
            y_range=rectangle.y_range,
            t_range=Range(t, t + generations + over_depth(seconds))
        )
        print(f'Frame {frame + 1}/{duration * fps}, t={t:.1f}')
        print(f'  {cuboid}')
        densities = snapshot_from_grid(grid, cuboid, resolution)
        luminances = np.vectorize(lambda density: luminance_fn(density, seconds))(densities)
        assert np.all(luminances >= 0)
        assert np.all(luminances <= 1)
        assert luminances.shape == (resolution.y, resolution.x)
        frame_bytes = _to_rgb(luminances)
        process.stdin.write(frame_bytes)
        t += generations
    process.stdin.close()
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, args)

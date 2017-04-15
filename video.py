import datetime
import pathlib


def convert_timecode(tc):
    """Converts a string of the form "x:yy:zz.wwww" or the like to seconds"""
    components = reversed([float(x) for x in tc.split(':')])
    return sum(x * 60**n for n, x in enumerate(components))


def coroutine(func):
    def _coroutine(*args, **kwargs):
        coro = func(*args, **kwargs)
        next(coro)
        return coro
    return _coroutine


def harmonize_if(obj, instances, methods):
    for instance, method in zip(instances, methods):
        if isinstance(obj, instance):
            return method(obj)
    return obj


def harmonize_time(time):
    return None if time is None else harmonize_if(time,
        [int, str, datetime.timedelta],
        [float, convert_timecode, datetime.timedelta.total_seconds],
    )


def harmonize_path(path):
    return None if path is None else harmonize_if(path,
        [pathlib.Path],
        [str],
    )

import datetime
import io
import pathlib
import typing

import cv2
import numpy as np

rich_deps_missing = []
try:
    import matplotlib.cm
except ImportError:
    rich_deps_missing.append('matplotlib')

try:
    from PIL import Image as pil_image
    from PIL import ImageOps as pil_ops
except ImportError:
    rich_deps_missing.append('Pillow')

# from .util import convert_timecode, harmonize_time, harmonize_path

__all__ = ['Video']

ITA = typing.Iterator['Frame']
PATH = typing.Union[str, pathlib.Path]
TIME = typing.Union[float, int, datetime.timedelta]
OPT_TIME = typing.Union[None, TIME]

BLUE, GREEN, RED = range(3) # color layers


class Frame(object):

    cmap = 'Greys'
    thumb_size = 640, 480

    @classmethod
    def from_capture(cls, vc:cv2.VideoCapture):
        success, frame = vc.retrieve()
        assert success

        color_layer = RED # red color shows timecode better

        self = cls(
            frame_data=frame[...,color_layer].astype('float32') / 255,
            frame_number=int(vc.get(cv2.CAP_PROP_POS_FRAMES)),
            time=vc.get(cv2.CAP_PROP_POS_MSEC) / 1000,
        )

        return self

    def __init__(self, frame_data, frame_number, time):
        self.data = frame_data
        self.n = frame_number
        self.t = time

    def __repr__(self):
        return '<{}: {:s} px, frame no. {:d} (t={:0.2f})>'.format(
            self.__class__.__name__,
            'x'.join(str(x) for x in self.data.shape),
            self.n,
            self.t,
        )

    def __sub__(self, other):
        return DeltaFrame(self, other)

    def _raw_float(self):
        return self.data

    def _repr_png_(self):
        if rich_deps_missing:
            raise RuntimeError(
                'Missing dependencies for rich display: {}'
                .format(', '.join(rich_deps_missing))
            )

        cm = matplotlib.cm.get_cmap(self.cmap)

        buf = io.BytesIO()
        im = pil_image.fromarray(cm(self._raw_float(), bytes=True))
        im.thumbnail(self.thumb_size, pil_image.BICUBIC)
        im = pil_ops.expand(im, border=2, fill='black')
        im.save(buf, format='png')
        return buf.getvalue()


class DeltaFrame(Frame):

    cmap = 'bwr'

    def __init__(self, current, ref):
        super(DeltaFrame, self).__init__(
            current.data - ref.data,
            current.n,
            current.t,
        )
        self.dn = current.n - ref.n
        self.dt = current.t - ref.t

    def _raw_float(self):
        return (self.data + 1) / 2

    def abs(self):
        """Returns the "brighter" pixels (discards anything that got darker)"""
        return self.data.clip(min=0)


class FrameIndexer(object):
    """
    Setting/grabbing frames from the cv2.VideoCapture object
    around 0 is bizarre.
    set_to   after_set   after_grab    grab_2   actual (0-based)
    ------   ---------   ----------    ------   ================
        -1    43473.            0.0       1.0         0          BROKEN
         0        0.            1.0       2.0         1          BROKEN
         1        0.99          1.0       2.0         1
         2        1.99          2.0       3.0         2
         3        2.99          3.0       4.0
                        ⋮
     43472    43471.99      43472.0   43473.0      LAST (43472)
     43473    43472.99      43473.0       0.0      LAST          BROKEN ...AND AFTERWARDS >:(
     43474    43473.            0.0       1.0         0            n/a
     43475    43473.            0.0       1.0         0
                    (repeats)
    """
    def __init__(self, video:cv2.VideoCapture):
        self.video = video
        self.n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self._getframes(item)
        else:
            return self._getframe(item)

    def delta(self, frame_number):
        return self[frame_number] - self[frame_number - 1]

    def seek(self, frame_number):
        self._frame_pos = frame_number

    @property
    def _frame_pos(self):
        return self.video.get(cv2.CAP_PROP_POS_FRAMES)

    @_frame_pos.setter
    def _frame_pos(self, value):
        return self.video.set(cv2.CAP_PROP_POS_FRAMES, value)

    def _getframe(self, frame:int):
        if frame >= self.n_frames:
            raise IndexError('frame out of range')
        if int(frame) != frame:
            raise ValueError('cannot convert float to integer')
        frame = int(frame)

        # stupid transform to force logic onto the VideoCapture object (see docstring above)
        if frame == 0:
            self._frame_pos = -1
        else:
            self._frame_pos = frame

        success = self.video.grab()
        if frame != 0:
            assert success # grabbing the 0th frame "fails"

        # after grabbing, the reported frame is **actually** correct
        # (except when grabbing the last frame twice, but we guard against
        # that elsewhere)
        # assert self._frame_pos == frame

        # success, frame = self.video.retrieve()
        # assert success
        # return frame
        return Frame.from_capture(self.video)

    def _getframes(self, slice_):
        start = slice_.start
        if start is None:
            start = 0

        stop = slice_.stop
        if stop is None:
            stop = self.n_frames

        step = slice_.step
        if step is None:
            step = 1

        if not ((0 <= start <= self.n_frames) and
                (0 <= stop <= self.n_frames)):
            raise IndexError('frame out of range')

        for frame in range(start, stop, step):
            yield self._getframe(frame)


class Video(object):
    def __init__(self, filename:PATH):
        filename = harmonize_path(filename)
        self.cap = cv2.VideoCapture(filename)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.f = FrameIndexer(self.cap)

    def convert_timecode(self, timecode):
        """Coerce a float or string (base60) time-code into the frame number"""
        time = convert_timecode(timecode)
        approx_frame = time * self.fps
        frame = int(round(approx_frame))
        return frame

    @property
    def time(self):
        return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    @time.setter
    def time(self, value:float):
        s = self.cap.set(cv2.CAP_PROP_POS_MSEC, value * 1000)
        assert s, "failed to seek to specified time"

    def frames(self, start:int, end:int=None, length:int=None,
               normalize:bool=False, delta:bool=False) -> ITA:
        if length:
            end = start + length

        if delta:
            if start == 0:
                raise ValueError('cannot compute delta when starting at frame 0')
            frame_prev = self.f[start - 1]

        for frame in self.f[start:end]:
            if delta:
                yield frame - frame_prev
                frame_prev = frame
            else:
                yield frame

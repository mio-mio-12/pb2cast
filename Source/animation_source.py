"""Reader and sampler for Point Blank APF1/APF2/APF3 i3 animation packs.

The format stores clip descriptors followed by shared translation, rotation,
and scale dictionaries.  Individual animation tracks contain indices into
those dictionaries.  APF2 uses 16-bit indices and half-float dictionaries;
APF3 can contain both the older 32-bit/full-float and newer packed layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from bisect import bisect_right
import struct
from pathlib import Path


HEADER_SIZE = 0xA4
CLIP_RECORD_SIZE = 0x11C
TRACK_NAME_SIZE = 0x24
SEQUENCE_SIZE = 0x48


@dataclass(frozen=True)
class Transform:
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class Sequence:
    version: int
    flags: int
    key_count: int
    time_step: float
    start_time: float
    duration: float
    translation_interpolation: int
    rotation_interpolation: int
    scale_interpolation: int
    translation_indices: tuple[int, ...]
    rotation_indices: tuple[int, ...]
    scale_indices: tuple[int, ...]
    times: tuple[float, ...]


@dataclass(frozen=True)
class Clip:
    name: str
    version: int
    duration: float
    tracks: tuple[str, ...]
    sequences: tuple[Sequence, ...]


@dataclass(frozen=True)
class AnimationPack:
    source_name: str
    magic: str
    clips: tuple[Clip, ...]
    translations: tuple[tuple[float, float, float], ...]
    rotations: tuple[tuple[float, float, float, float], ...]
    scales: tuple[tuple[float, float, float], ...]

    def clip(self, suffix: str) -> Clip:
        wanted = suffix.replace("\\", "/").casefold()
        matches = [item for item in self.clips if item.name.replace("\\", "/").casefold().endswith(wanted)]
        if len(matches) != 1:
            raise KeyError(f"Expected one clip ending in {suffix!r}, found {len(matches)}")
        return matches[0]

    def sample(self, clip: Clip, time: float, *, loop: bool = True) -> dict[str, Transform]:
        return {
            name: _sample_sequence(sequence, time, self, loop=loop)
            for name, sequence in zip(clip.tracks, clip.sequences)
        }

    def sample_sequence(
        self, sequence: Sequence, time: float, *, loop: bool = True,
    ) -> Transform:
        """Sample one authored track without evaluating unrelated bones."""
        return _sample_sequence(sequence, time, self, loop=loop)


def _read_vector_array(data: bytes, offset: int, count: int, width: int, half: bool):
    component = "e" if half else "f"
    item_size = (2 if half else 4) * width
    end = offset + count * item_size
    if end > len(data):
        raise ValueError("Animation dictionary extends past end of file")
    fmt = "<" + component * width
    return tuple(struct.unpack_from(fmt, data, offset + index * item_size) for index in range(count)), end


def _index_count(flags: int, present: int, constant: int, key_count: int) -> int:
    if not flags & present:
        return 0
    return 1 if flags & constant else key_count


def parse_i3animpack(data: bytes, source_name: str = "<memory>") -> AnimationPack:
    if len(data) < HEADER_SIZE or data[:3] != b"APF" or data[3:4] not in {b"1", b"2", b"3"}:
        raise ValueError(f"{source_name}: unsupported i3AnimPack header")
    magic = data[:4].decode("ascii")
    (
        clip_count,
        float_translation_count,
        float_rotation_count,
        float_scale_count,
        data_offset,
        _reserved,
        data_size,
        _reserved2,
        half_translation_count,
        half_rotation_count,
        half_scale_count,
    ) = struct.unpack_from("<11I", data, 4)
    if data_offset < HEADER_SIZE + clip_count * CLIP_RECORD_SIZE:
        raise ValueError(f"{source_name}: invalid animation data offset")
    if data_offset + data_size > len(data):
        raise ValueError(f"{source_name}: truncated animation data")

    cursor = data_offset
    float_translations, cursor = _read_vector_array(data, cursor, float_translation_count, 3, False)
    float_rotations, cursor = _read_vector_array(data, cursor, float_rotation_count, 4, False)
    float_scales, cursor = _read_vector_array(data, cursor, float_scale_count, 3, False)
    half_translations, cursor = _read_vector_array(data, cursor, half_translation_count, 3, True)
    half_rotations, cursor = _read_vector_array(data, cursor, half_rotation_count, 4, True)
    half_scales, cursor = _read_vector_array(data, cursor, half_scale_count, 3, True)

    descriptors = []
    for index in range(clip_count):
        record = HEADER_SIZE + index * CLIP_RECORD_SIZE
        version, track_count, descriptor_duration, relative_offset = struct.unpack_from("<IIfI", data, record)
        raw_name = data[record + 20:record + CLIP_RECORD_SIZE].split(b"\0", 1)[0]
        name = raw_name.decode("utf-8", errors="replace")
        descriptors.append((version, track_count, descriptor_duration, relative_offset, name))

    clips = []
    for version, track_count, descriptor_duration, relative_offset, name in descriptors:
        clip_offset = data_offset + relative_offset
        sequence_offset = clip_offset + track_count * TRACK_NAME_SIZE
        indices_cursor = sequence_offset + track_count * SEQUENCE_SIZE
        if indices_cursor > len(data):
            raise ValueError(f"{source_name}: truncated clip {name}")
        tracks = []
        raw_sequences = []
        for track_index in range(track_count):
            track_offset = clip_offset + track_index * TRACK_NAME_SIZE
            raw_name = data[track_offset + 4:track_offset + TRACK_NAME_SIZE].split(b"\0", 1)[0]
            tracks.append(raw_name.decode("utf-8", errors="replace"))
            seq_offset = sequence_offset + track_index * SEQUENCE_SIZE
            flags, key_count, time_step = struct.unpack_from("<IIf", data, seq_offset)
            start_time, duration = struct.unpack_from("<ff", data, seq_offset + 0x1C)
            raw_sequences.append((
                flags, key_count, time_step, start_time, duration,
                data[seq_offset + 0x24], data[seq_offset + 0x25], data[seq_offset + 0x26],
            ))

        index_width = 2 if version == 2 else 4
        index_format = "<H" if index_width == 2 else "<I"

        def read_indices(count: int) -> tuple[int, ...]:
            nonlocal indices_cursor
            end = indices_cursor + count * index_width
            if end > len(data):
                raise ValueError(f"{source_name}: truncated indices in {name}")
            values = tuple(struct.unpack_from(index_format, data, indices_cursor + i * index_width)[0] for i in range(count))
            indices_cursor = end
            return values

        sequences = []
        for flags, key_count, time_step, start_time, duration, t_interp, r_interp, s_interp in raw_sequences:
            translation_indices = read_indices(_index_count(flags, 0x01, 0x10, key_count))
            rotation_indices = read_indices(_index_count(flags, 0x02, 0x20, key_count))
            scale_indices = read_indices(_index_count(flags, 0x04, 0x40, key_count))
            if flags & 0x08:
                scalar_size = 2 if version == 2 else 4
                scalar_format = "<e" if version == 2 else "<f"
                scalar_end = indices_cursor + key_count * scalar_size
                if scalar_end > len(data):
                    raise ValueError(f"{source_name}: truncated times in {name}")
                times = tuple(
                    struct.unpack_from(scalar_format, data, indices_cursor + i * scalar_size)[0]
                    for i in range(key_count)
                )
                indices_cursor = scalar_end
            else:
                times = ()
            sequences.append(Sequence(
                version=version,
                flags=flags,
                key_count=key_count,
                time_step=time_step,
                start_time=start_time,
                duration=duration,
                translation_interpolation=t_interp,
                rotation_interpolation=r_interp,
                scale_interpolation=s_interp,
                translation_indices=translation_indices,
                rotation_indices=rotation_indices,
                scale_indices=scale_indices,
                times=times,
            ))
        clip_duration = max((sequence.duration for sequence in sequences), default=descriptor_duration)
        clips.append(Clip(name, version, clip_duration, tuple(tracks), tuple(sequences)))

    # A clip version selects one complete dictionary representation.
    # Keep both concatenated only internally; sampling chooses by clip version.
    pack = AnimationPack(
        source_name=source_name,
        magic=magic,
        clips=tuple(clips),
        translations=tuple(float_translations) + tuple(half_translations),
        rotations=tuple(float_rotations) + tuple(half_rotations),
        scales=tuple(float_scales) + tuple(half_scales),
    )
    object.__setattr__(pack, "_float_translation_count", len(float_translations))
    object.__setattr__(pack, "_float_rotation_count", len(float_rotations))
    object.__setattr__(pack, "_float_scale_count", len(float_scales))
    return pack


def load_i3animpack(path: Path) -> AnimationPack:
    return parse_i3animpack(path.read_bytes(), str(path))


def _lerp(a, b, amount: float):
    return tuple(x + (y - x) * amount for x, y in zip(a, b))


def _slerp(a, b, amount: float):
    if a == b or amount <= 0.0:
        return tuple(a)
    if amount >= 1.0:
        return tuple(b)
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b = tuple(-value for value in b)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.999999:
        return _lerp(a, b, amount)
    angle = math.acos(dot)
    denominator = math.sin(angle)
    first = math.sin((1.0 - amount) * angle) / denominator
    second = math.sin(amount * angle) / denominator
    return tuple(first * x + second * y for x, y in zip(a, b))


def _dictionary(pack: AnimationPack, clip_version: int, kind: str):
    values = getattr(pack, kind)
    float_count = getattr(pack, f"_float_{kind[:-1]}_count")
    return values[float_count:] if clip_version >= 2 else values[:float_count]


def _sample_sequence(sequence: Sequence, time: float, pack: AnimationPack, *, loop: bool) -> Transform:
    if sequence.key_count <= 0:
        return Transform()
    local_time = max(0.0, time - sequence.start_time)
    if loop and sequence.duration > 0.0:
        local_time %= sequence.duration
    elif sequence.duration > 0.0:
        local_time = min(local_time, sequence.duration)

    if sequence.times:
        key = max(0, min(bisect_right(sequence.times, local_time) - 1, sequence.key_count - 1))
        next_key = min(key + 1, sequence.key_count - 1)
        interval = sequence.times[next_key] - sequence.times[key]
        amount = 0.0 if interval <= 0.0 else max(0.0, min(1.0, (local_time - sequence.times[key]) / interval))
    else:
        step = sequence.time_step if sequence.time_step > 1.0e-8 else 1.0
        position = local_time / step
        key = min(int(math.floor(position)), sequence.key_count - 1)
        next_key = min(key + 1, sequence.key_count - 1)
        amount = max(0.0, min(1.0, position - math.floor(position)))

    def values(kind: str, indices: tuple[int, ...], constant_flag: int, default):
        if not indices:
            return default, default
        # APF dictionaries can contain thousands of entries. Select their
        # versioned region by offset instead of copying the entire region for
        # every bone sample. Bounds remain relative to that region.
        dictionary = getattr(pack, kind)
        float_count = getattr(pack, f"_float_{kind[:-1]}_count")
        start = float_count if sequence.version >= 2 else 0
        count = len(dictionary) - start if sequence.version >= 2 else float_count
        first_index = indices[0 if sequence.flags & constant_flag else key]
        second_index = indices[0 if sequence.flags & constant_flag else next_key]
        if not (0 <= first_index < count and 0 <= second_index < count):
            raise IndexError("animation dictionary index outside versioned region")
        return dictionary[start + first_index], dictionary[start + second_index]

    translation_a, translation_b = values("translations", sequence.translation_indices, 0x10, (0.0, 0.0, 0.0))
    rotation_a, rotation_b = values("rotations", sequence.rotation_indices, 0x20, (0.0, 0.0, 0.0, 1.0))
    scale_a, scale_b = values("scales", sequence.scale_indices, 0x40, (1.0, 1.0, 1.0))
    translation = _lerp(translation_a, translation_b, amount) if sequence.translation_interpolation == 1 else translation_a
    rotation = _slerp(rotation_a, rotation_b, amount) if sequence.rotation_interpolation in {1, 4} else rotation_a
    scale = _lerp(scale_a, scale_b, amount) if sequence.scale_interpolation == 1 else scale_a
    return Transform(translation, rotation, scale)


# This decoder replaces i3Animation2's inconsistent float quotient/fmod lookup.
# Evaluate index and blend weight from ONE double-precision source position.
# Never repair duplicates after sampling: identical authored keys remain intact.
DECODER_REV = 2

def sample_count(duration, fps=30):
    # Match the existing native float metadata multiplication when deciding
    # whether the terminal sample exists. Evaluation time itself remains double.
    return math.ceil(struct.unpack('<f', struct.pack('<f', duration * fps))[0]) + 1

def decode_pack(data: bytes, source_name: str):
    pack = parse_i3animpack(data, source_name)
    clips = []
    for clip in pack.clips:
        # Installed character and weapon source rates were audited at <=30 Hz.
        # Evaluate source keys continuously; retain channel modes for inspection.
        times = [min(clip.duration, frame / 30.0)
                 for frame in range(sample_count(clip.duration))]
        tracks = []
        for name, sequence in zip(clip.tracks, clip.sequences):
            samples = []
            for time in times:
                value = pack.sample_sequence(sequence, time, loop=False)
                samples.append([*value.translation, *value.rotation, *value.scale])
            tracks.append(dict(name=name, flags=sequence.flags & 7, samples=samples,
                               channelModes=[sequence.translation_interpolation,
                                             sequence.rotation_interpolation,
                                             sequence.scale_interpolation]))
        clips.append(dict(name=clip.name, duration=clip.duration, tracks=tracks))
    return dict(fps=30, decoderRevision=DECODER_REV, clips=clips)

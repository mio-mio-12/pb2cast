"""Read-only I3R2/I3Pack inspection based on the supplied PB SDK format code."""

from __future__ import annotations

import json
import math
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence


def _rotate_decrypt(data: bytes, shift: int) -> bytes:
    """Match the SDK's Crypt.Decrypt byte rotation."""
    if not data:
        return data
    result = bytearray(data)
    last = result[-1]
    for index in range(len(result) - 1, 0, -1):
        result[index] = ((result[index - 1] << (8 - shift)) | (result[index] >> shift)) & 0xFF
    result[0] = ((last << (8 - shift)) | (result[0] >> shift)) & 0xFF
    return bytes(result)


def _dotnet_string(data: bytes, offset: int) -> tuple[str, int]:
    length = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("truncated .NET length-prefixed string")
        value = data[offset]
        offset += 1
        length |= (value & 0x7F) << shift
        if value & 0x80 == 0:
            break
        shift += 7
        if shift > 28:
            raise ValueError("invalid .NET length-prefixed string")
    end = offset + length
    if end > len(data):
        raise ValueError("truncated .NET length-prefixed string payload")
    return data[offset:end].decode("cp1252", errors="replace"), end


def _clean_name(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("cp1252", errors="replace")


@dataclass(frozen=True)
class I3Bone:
    index: int
    name: str
    bone_id: int
    parent: int
    matrix: tuple[float, ...]


@dataclass(frozen=True)
class I3AttachmentSocket:
    """Authored character-scene socket bound to a skeleton bone.

    Character I3S files keep weapon/equipment sockets outside the skinned
    palette as ``i3BoneRef -> i3Transform -> i3MatrixObject`` nodes.  The
    runtime preserves these nodes and attaches weapons to them; treating the
    referenced bone as the socket loses both the authored offset and the
    distinction between right, left, side, back, and thigh placements.
    """

    name: str
    bone_index: int
    bone_name: str
    matrix: tuple[float, ...]


@dataclass(frozen=True)
class I3Block:
    line: int
    block_id: int
    target: int
    type_name: str
    start: int
    size: int
    data: bytes


@dataclass(frozen=True)
class I3PackEntry:
    parent_name: str
    name: str
    start: int
    size: int
    data: bytes
    file_step: int


@dataclass
class I3File:
    source_name: str
    data: bytes
    text_lines: list[str]
    blocks: list[I3Block]
    crypted: bool = False

    def blocks_of_type(self, type_name: str) -> list[I3Block]:
        return [block for block in self.blocks if block.type_name == type_name]


def parse_i3_behavior_graph(root: I3File) -> dict[str, Any]:
    """Decode the authored state-to-resource links in an I3CHR graph.

    I3 serializes an ``i3Animation`` resource reference as a zero-sized block
    whose ``target`` indexes the file string table.  The owning ``i3AIState``
    embeds the exact pair ``(animation block id, 0x8000 | target)``.  Matching
    that pair is substantially safer than relying on block adjacency and, more
    importantly, preserves PB's distinct reload variants and POV/gender states.
    """
    animations = {
        block.block_id: block
        for block in root.blocks
        if block.type_name == "i3Animation"
        and 0 <= block.target < len(root.text_lines)
    }
    blocks_by_id = {block.block_id: block for block in root.blocks}
    # Scan each state payload once. The previous cross-product searched every
    # animation marker in every state, which is quadratic on character graphs.
    # Byte offsets remain unaligned and first-occurrence semantics unchanged.
    animation_markers = {
        struct.pack("<HH", animation_id, animation.target + 0x8000):
            (order, animation_id, animation)
        for order, (animation_id, animation) in enumerate(animations.items())
    }

    # i3AIContext serializes its parent context at name_end + 12 and its
    # owned i3AI at name_end + 40. i3AI serializes the number of owned states
    # at name_end + 36 followed by their 32-bit block IDs. Decode those
    # authored links instead of inferring context from animation filenames.
    ai_states: dict[int, tuple[int, ...]] = {}
    ai_names: dict[int, str] = {}
    for block in root.blocks:
        if block.type_name != "i3AI" or not block.data:
            continue
        try:
            ai_name, offset = _dotnet_string(block.data, 0)
        except ValueError:
            continue
        if offset + 40 > len(block.data):
            continue
        count = struct.unpack_from("<I", block.data, offset + 36)[0]
        if count > 4096 or offset + 40 + count * 4 > len(block.data):
            continue
        state_ids = struct.unpack_from(f"<{count}I", block.data, offset + 40)
        if not all(
            state_id in blocks_by_id
            and blocks_by_id[state_id].type_name == "i3AIState"
            for state_id in state_ids
        ):
            continue
        ai_names[block.block_id] = ai_name
        ai_states[block.block_id] = tuple(state_ids)

    contexts: dict[int, dict[str, Any]] = {}
    ai_context: dict[int, int] = {}
    for block in root.blocks:
        if block.type_name != "i3AIContext" or not block.data:
            continue
        try:
            context_name, offset = _dotnet_string(block.data, 0)
        except ValueError:
            continue
        if offset + 44 > len(block.data):
            continue
        parent_id = struct.unpack_from("<I", block.data, offset + 12)[0]
        ai_id = struct.unpack_from("<I", block.data, offset + 40)[0]
        if ai_id not in ai_states:
            continue
        contexts[block.block_id] = {
            "name": context_name,
            "block_id": block.block_id,
            "parent_block_id": (
                parent_id
                if parent_id in blocks_by_id
                and blocks_by_id[parent_id].type_name == "i3AIContext"
                else None
            ),
            "ai_block_id": ai_id,
            "ai_name": ai_names.get(ai_id, ""),
        }
        ai_context[ai_id] = block.block_id

    def context_path(ai_id: int) -> tuple[str, ...]:
        context_id = ai_context.get(ai_id)
        result: list[str] = []
        visited: set[int] = set()
        while context_id is not None and context_id not in visited:
            visited.add(context_id)
            context = contexts.get(context_id)
            if context is None:
                break
            result.append(str(context["name"]))
            context_id = context.get("parent_block_id")
        result.reverse()
        return tuple(result)

    state_owners: dict[int, list[dict[str, Any]]] = {}
    for ai_id, state_ids in ai_states.items():
        owner = {
            "ai": ai_names.get(ai_id, ""),
            "ai_block_id": ai_id,
            "contexts": context_path(ai_id),
        }
        for state_index, state_id in enumerate(state_ids):
            state_owners.setdefault(state_id, []).append({
                **owner,
                "state_index": state_index,
            })
    states: list[dict[str, Any]] = []
    linked_animation_ids: set[int] = set()
    for block in root.blocks:
        if block.type_name != "i3AIState" or not block.data:
            continue
        try:
            state_name, state_offset = _dotnet_string(block.data, 0)
        except ValueError:
            continue
        # i3AIState::pack::AI_STATE is aligned to eight bytes.  In current PB
        # archives its AIS1 payload stores timeScale and the destination
        # BlendTime at offsets 24 and 28.  BlendTime is part of animation
        # playback semantics, not cosmetic metadata: i3AIContext::Reset feeds
        # it directly to i3TransformSourceCombiner::AddAnimation.
        signature_offset = block.data.find(b"AIS1", state_offset)
        time_scale = 1.0
        blend_time = 0.0
        style = 0
        if signature_offset >= 0 and signature_offset + 32 <= len(block.data):
            style = struct.unpack_from("<I", block.data, signature_offset + 4)[0]
            parsed_scale, parsed_blend = struct.unpack_from(
                "<ff", block.data, signature_offset + 24
            )
            if math.isfinite(parsed_scale) and parsed_scale > 0.0:
                time_scale = parsed_scale
            if math.isfinite(parsed_blend) and 0.0 <= parsed_blend <= 10.0:
                blend_time = parsed_blend
        resources = []
        found_markers = {}
        for offset in range(max(0, len(block.data) - 3)):
            match = animation_markers.get(block.data[offset:offset + 4])
            if match is not None:
                order, animation_id, animation = match
                found_markers.setdefault(order, (animation_id, animation, offset))
        for order in sorted(found_markers):
            animation_id, animation, marker_offset = found_markers[order]
            linked_animation_ids.add(animation_id)
            resources.append({
                "kind": "animation",
                "path": root.text_lines[animation.target],
                "block_id": animation_id,
                "reference_offset": marker_offset,
            })
        states.append({
            "name": state_name,
            "block_id": block.block_id,
            "style": style,
            "time_scale": time_scale,
            "blend_time": blend_time,
            "resources": resources,
            "owners": state_owners.get(block.block_id, []),
        })

    scene_reference = next(
        (value for value in root.text_lines if value.casefold().endswith(".i3s")),
        "",
    )
    return {
        "scene_reference": scene_reference,
        "states": states,
        "animation_links": [
            {
                "state": state["name"],
                "path": resource["path"],
                "state_block_id": state["block_id"],
                "state_style": state["style"],
                "time_scale": state["time_scale"],
                "blend_time": state["blend_time"],
                "animation_block_id": resource["block_id"],
                "ai": owner.get("ai", ""),
                "ai_block_id": owner.get("ai_block_id"),
                "state_index": owner.get("state_index"),
                "contexts": list(owner.get("contexts", ())),
            }
            for state in states
            for resource in state["resources"]
            if resource["kind"] == "animation"
            for owner in (state.get("owners") or [{}])
        ],
        "ai_contexts": list(contexts.values()),
        "unowned_animation_paths": [
            root.text_lines[block.target]
            for block_id, block in animations.items()
            if block_id not in linked_animation_ids
        ],
    }


@dataclass
class I3Skin:
    """A named I3 skeleton and its model-space inverse-bind matrices."""

    bones: list[I3Bone]
    inverse_bind_matrices: list[tuple[float, ...]]
    bone_block_id: int = -1
    matrix_block_id: int = -1


@dataclass
class I3Mesh:
    """Geometry decoded from one i3Geometry/i3GeometryAttr pair."""

    name: str
    geometry_id: int
    vertex_array_id: int
    index_array_id: int
    vertex_flag: int
    vertex_stride: int
    positions: list[tuple[float, float, float]]
    normals: list[tuple[float, float, float]]
    uvs: list[tuple[float, float]]
    lightmap_uvs: list[tuple[float, float]]
    bone_indices: list[tuple[int, ...]]
    bone_weights: list[tuple[float, ...]]
    faces: list[tuple[int, int, int]]
    textures: dict[str, str]
    # Render attributes inherited through the owning i3AttrSet.  Keeping this
    # beside the texture bindings lets downstream exporters reproduce I3's
    # material tint, gloss, alpha test and culling instead of guessing from a
    # texture filename.
    material: dict[str, Any] = field(default_factory=dict)
    attachment_name: str = ""
    lod_index: int = 0
    source_name: str = ""  # Exact geometry node name; display labels may be synthetic.


def compact_i3_topology(
    mesh: I3Mesh,
) -> tuple[list[int], list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Compact one material subset and weld skin-compatible position seams.

    UVs and authored normals remain associated with each source face corner in
    the returned ``source_faces``. Duplicate positions with different skin
    influences deliberately remain separate.
    """
    key_to_compact: dict[tuple, int] = {}
    compact_sources: list[int] = []
    source_to_compact: dict[int, int] = {}
    compact_faces: list[tuple[int, int, int]] = []
    source_faces: list[tuple[int, int, int]] = []

    def skin_signature(index: int) -> tuple:
        if index >= len(mesh.bone_indices) or index >= len(mesh.bone_weights):
            return ()
        return (
            tuple(int(value) for value in mesh.bone_indices[index]),
            tuple(round(float(value), 7) for value in mesh.bone_weights[index]),
        )

    for source_face in mesh.faces:
        compact_face = []
        valid = True
        for source_index in source_face:
            if not 0 <= source_index < len(mesh.positions):
                valid = False
                break
            compact_index = source_to_compact.get(source_index)
            if compact_index is None:
                key = (
                    tuple(
                        round(float(value), 6)
                        for value in mesh.positions[source_index]
                    ),
                    skin_signature(source_index),
                )
                compact_index = key_to_compact.get(key)
                if compact_index is None:
                    compact_index = len(compact_sources)
                    key_to_compact[key] = compact_index
                    compact_sources.append(source_index)
                source_to_compact[source_index] = compact_index
            compact_face.append(compact_index)
        if valid and len(set(compact_face)) == 3:
            compact_faces.append(tuple(compact_face))
            source_faces.append(tuple(source_face))
    return compact_sources, compact_faces, source_faces


def parse_i3r2(data: bytes, source_name: str = "<memory>") -> I3File:
    crypted = False
    if data[:4] != b"I3R2":
        # A later PB pack generation masks only the four-byte I3R2 signature
        # with spaces while retaining the complete standard section table.
        # Do not rotate/decrypt its payload: doing so corrupts otherwise valid
        # meshes and textures (Wraith/Vampire/Jason Mermaid packs).
        if data[:4] in {b"    ", b"Blow", b"AVIX", b"UATH"} and len(data) >= 184:
            # ``Blow`` is another current-generation archive marker. Despite
            # the name, its section table and i3PackNode payload are ordinary
            # I3R2 data; only the four-byte signature differs. Treating the
            # entire file as rotate-encrypted is what previously made every
            # texture-only Blow weapon skin impossible to extract. AVIX and
            # UATH are additional installed header-only masks: their section
            # tables and i3PackNode entries validate as ordinary I3R2. The
            # strict structural checks below still apply to every block.
            data = b"I3R2" + data[4:]
            crypted = True
        else:
            decoded = _rotate_decrypt(data, 3)
            # Some PB collision resources rotate-decrypt to lowercase
            # ``i3R2``.  The SDK accepts the signature by bytes 1..3 and the
            # rest of the section table is identical, so normalize only the
            # type byte instead of rejecting otherwise valid character
            # collision geometry.
            if decoded[1:4] == b"3R2":
                decoded = b"I" + decoded[1:]
            if decoded[:4] != b"I3R2":
                raise ValueError(f"{source_name}: not an I3R2 file")
            data = decoded
            crypted = True

    if len(data) < 184:
        raise ValueError(f"{source_name}: I3R2 header is truncated")
    text_start = struct.unpack_from("<Q", data, 16)[0]
    text_size = struct.unpack_from("<Q", data, 24)[0]
    block_count = struct.unpack_from("<I", data, 32)[0]
    info_start = struct.unpack_from("<Q", data, 36)[0]
    info_size = struct.unpack_from("<Q", data, 44)[0]
    if text_start + text_size > len(data) or info_start + info_size > len(data):
        raise ValueError(f"{source_name}: I3R2 section exceeds file size")
    text = data[text_start:text_start + text_size].decode("cp1252", errors="replace")
    text_lines = text.rstrip("\x00").splitlines()

    expected_info_size = block_count * 28
    if info_size < expected_info_size:
        raise ValueError(f"{source_name}: block table is truncated")
    blocks: list[I3Block] = []
    for index in range(block_count):
        offset = info_start + index * 28
        line, block_id, target, _what, start, size, _unknown = struct.unpack_from("<IHHIqII", data, offset)
        end = start + size
        if end > len(data):
            raise ValueError(f"{source_name}: block {block_id} exceeds file size")
        type_name = text_lines[line] if line < len(text_lines) else f"Error ({line})"
        if target:
            target -= 32768
        blocks.append(I3Block(line, block_id, target, type_name, start, size, data[start:end]))
    return I3File(source_name, data, text_lines, blocks, crypted)


def parse_pack_entries(block_data: bytes, parent_data: bytes, parent_name: str = "<pack>") -> list[I3PackEntry]:
    """Read an i3PackNode block and its file table without modifying it."""
    name, offset = _dotnet_string(block_data, 0)
    if offset + 16 > len(block_data):
        raise ValueError("truncated i3PackNode header")
    offset += 12
    unknown_count = struct.unpack_from("<I", block_data, offset)[0]
    offset += 4 + unknown_count * 4
    offset += 56
    if offset + 12 > len(block_data):
        raise ValueError("truncated i3PackNode metadata")
    offset += 4
    pack_type = _rotate_decrypt(block_data[offset:offset + 8], 3)
    offset += 8
    if len(pack_type) < 6:
        raise ValueError("truncated i3PackNode type descriptor")
    file_type = pack_type[3]
    file_count = struct.unpack_from("<H", pack_type, 4)[0]
    file_step = 92 if file_type == 50 else 76
    entries: list[I3PackEntry] = []
    for _ in range(file_count):
        if offset + file_step > len(block_data):
            raise ValueError("truncated i3PackNode file table")
        row = _rotate_decrypt(block_data[offset:offset + file_step], 2)
        offset += file_step
        file_name = _clean_name(row[:52])
        if file_step == 76:
            start = struct.unpack_from("<H", row, 56)[0] * 65536 + struct.unpack_from("<H", row, 64)[0]
            size = struct.unpack_from("<H", row, 58)[0] * 65536 + struct.unpack_from("<H", row, 54)[0]
        else:
            start = struct.unpack_from("<H", row, 72)[0] * 65536 + struct.unpack_from("<H", row, 80)[0]
            size = struct.unpack_from("<H", row, 74)[0] * 65536 + struct.unpack_from("<H", row, 70)[0]
        end = start + size
        file_data = parent_data[start:end] if end <= len(parent_data) else b""
        entries.append(I3PackEntry(name or parent_name, file_name, start, size, file_data, file_step))
    return entries


def parse_bone_matrix_list(data: bytes, source_name: str = "<bone block>") -> list[I3Bone]:
    if len(data) < 48:
        raise ValueError(f"{source_name}: bone block is truncated")
    bone_count = struct.unpack_from("<I", data, 4)[0]
    offset = 48
    bones: list[I3Bone] = []
    record_size = 128
    if offset + bone_count * record_size > len(data):
        raise ValueError(f"{source_name}: bone records exceed block size")
    for index in range(bone_count):
        name = _clean_name(data[offset:offset + 32])
        bone_id = struct.unpack_from("<i", data, offset + 32)[0]
        matrix = struct.unpack_from("<16f", data, offset + 48)
        parent = struct.unpack_from("<i", data, offset + 112)[0]
        bones.append(I3Bone(index, name, bone_id, parent, tuple(matrix)))
        offset += record_size
    return bones


def parse_matrix_array(data: bytes, source_name: str = "<matrix array>") -> list[tuple[float, ...]]:
    """Decode i3MatrixArray, used by skinned I3S files for inverse binds."""
    if len(data) < 4:
        raise ValueError(f"{source_name}: matrix array is truncated")
    count = struct.unpack_from("<I", data, 0)[0]
    expected = 4 + count * 64
    if count > 4096 or expected > len(data):
        raise ValueError(f"{source_name}: matrix records exceed block size")
    return [struct.unpack_from("<16f", data, 4 + index * 64) for index in range(count)]


def extract_i3_skins(root: I3File) -> list[I3Skin]:
    """Return each bone palette paired with its nearest matching matrix array."""
    skeletons: list[tuple[I3Block, list[I3Bone]]] = []
    matrices: list[tuple[I3Block, list[tuple[float, ...]]]] = []
    for block in root.blocks_of_type("i3BoneMatrixListAttr"):
        try:
            skeletons.append((block, parse_bone_matrix_list(block.data, root.source_name)))
        except ValueError:
            continue
    for block in root.blocks_of_type("i3MatrixArray"):
        try:
            matrices.append((block, parse_matrix_array(block.data, root.source_name)))
        except ValueError:
            continue
    result: list[I3Skin] = []
    for bone_block, bones in skeletons:
        compatible = [item for item in matrices if len(item[1]) == len(bones)]
        if not compatible or not bones:
            continue
        preceding = [item for item in compatible if item[0].block_id < bone_block.block_id]
        matrix_block, inverse_binds = (
            max(preceding, key=lambda item: item[0].block_id)
            if preceding else min(
                compatible, key=lambda item: abs(item[0].block_id - bone_block.block_id)
            )
        )
        result.append(I3Skin(
            bones, inverse_binds, bone_block.block_id, matrix_block.block_id
        ))
    return result


def extract_i3_skin(root: I3File) -> I3Skin | None:
    """Return the first most-complete skeleton/inverse-bind pair in an I3S file."""
    return max(extract_i3_skins(root), key=lambda item: len(item.bones), default=None)


def _parse_geometry_reference(block: I3Block) -> tuple[str, int]:
    data = block.data
    if not data:
        return "", 0
    name_size = data[0]
    name_end = 1 + name_size
    if name_end > len(data):
        return "", 0
    name = data[1:name_end].decode("cp1252", errors="replace")
    remaining = len(data) - name_end
    link_offset = name_end + (44 if remaining == 48 else 20)
    if link_offset + 4 > len(data):
        return name, 0
    return name, struct.unpack_from("<I", data, link_offset)[0]


def _parse_named_block(block: I3Block) -> str:
    """Read the leading .NET string used by i3BoneRef/i3Geometry nodes."""
    try:
        name, _offset = _dotnet_string(block.data, 0)
    except (ValueError, struct.error):
        return ""
    return name


def _parse_bone_ref(block: I3Block) -> tuple[str, list[int]]:
    """Decode an i3BoneRef name and its child scene-node references."""
    name, offset = _dotnet_string(block.data, 0)
    offset += 8  # INF2 plus reserved int32
    if offset + 8 > len(block.data):
        raise ValueError("truncated i3BoneRef child count")
    count = struct.unpack_from("<q", block.data, offset)[0]
    offset += 8
    if count < 0 or count > 65536 or offset + count * 4 > len(block.data):
        raise ValueError("invalid i3BoneRef child count")
    refs = list(struct.unpack_from(f"<{count}i", block.data, offset)) if count else []
    return name, refs


def _parse_node_refs(block: I3Block) -> list[int]:
    """Decode the child list shared by unnamed i3Node graph roots."""
    _name, offset = _dotnet_string(block.data, 0)
    offset += 8  # INF2 plus reserved int32
    if offset + 8 > len(block.data):
        raise ValueError("truncated i3Node child count")
    count = struct.unpack_from("<i", block.data, offset)[0]
    offset += 8  # count plus reserved int32
    if count < 0 or count > 65536 or offset + count * 4 > len(block.data):
        raise ValueError("invalid i3Node child count")
    return list(struct.unpack_from(f"<{count}i", block.data, offset)) if count else []


def _parse_transform_refs(block: I3Block) -> tuple[list[int], int]:
    """Decode i3Transform children and its trailing i3MatrixObject link."""
    refs = _parse_node_refs(block)
    _name, offset = _dotnet_string(block.data, 0)
    offset += 16 + len(refs) * 4
    matrix_id = (
        struct.unpack_from("<i", block.data, offset)[0]
        if offset + 4 <= len(block.data) else 0
    )
    return refs, matrix_id


def _scene_graph_info_bindings(root: I3File) -> list[tuple[int, int]]:
    """Return legacy SGI1 ``(AttrSet, scene-root)`` relationships.

    Older rigid/map-object scenes keep render state and the transform tree as
    parallel roots joined only by i3SceneGraphInfo. Without this edge the
    geometry still decodes, but it loses materials and nested transforms.
    """
    result: list[tuple[int, int]] = []
    for block in root.blocks_of_type("i3SceneGraphInfo"):
        try:
            _name, offset = _dotnet_string(block.data, 0)
        except ValueError:
            continue
        if (
            offset + 20 > len(block.data)
            or block.data[offset:offset + 4] != b"SGI1"
        ):
            continue
        attr_set_id = struct.unpack_from("<i", block.data, offset + 8)[0]
        scene_root_id = struct.unpack_from("<i", block.data, offset + 16)[0]
        if scene_root_id > 0:
            result.append((attr_set_id, scene_root_id))
    return result


def parse_i3_attachment_sockets(root: I3File) -> list[I3AttachmentSocket]:
    """Decode authored ``*PointDummy`` sockets from an I3 scene graph.

    The trailing three int32 values in an ``i3BoneRef`` are the skeleton bone
    index plus two reserved fields.  Its first transform child supplies the
    local row-vector matrix used by the game.  Invalid/incomplete graph nodes
    are ignored instead of inventing a hand assignment.
    """

    by_id = {block.block_id: block for block in root.blocks}
    pending: list[tuple[str, int, tuple[float, ...]]] = []
    for block in root.blocks_of_type("i3BoneRef"):
        try:
            name, refs = _parse_bone_ref(block)
        except (ValueError, struct.error):
            continue
        if "pointdummy" not in name.casefold() or len(block.data) < 12:
            continue
        bone_index = struct.unpack_from("<i", block.data, len(block.data) - 12)[0]
        matrix = None
        for reference in refs:
            transform = by_id.get(reference)
            if transform is None or transform.type_name != "i3Transform":
                continue
            try:
                _children, matrix_id = _parse_transform_refs(transform)
            except (ValueError, struct.error):
                continue
            matrix_block = by_id.get(matrix_id)
            if (
                matrix_block is not None
                and matrix_block.type_name == "i3MatrixObject"
                and len(matrix_block.data) >= 64
            ):
                matrix = tuple(struct.unpack_from("<16f", matrix_block.data, 0))
                break
        if bone_index >= 0 and matrix is not None:
            pending.append((name, bone_index, matrix))

    palettes: list[list[I3Bone]] = []
    for block in root.blocks_of_type("i3BoneMatrixListAttr"):
        try:
            palette = parse_bone_matrix_list(block.data, root.source_name)
        except ValueError:
            continue
        if palette:
            palettes.append(palette)

    def expected_parent(socket_name: str) -> str:
        key = "".join(character for character in socket_name.casefold() if character.isalnum())
        if "weaponpointdummyright" in key:
            return "rhand"
        if "weaponpointdummyleft" in key:
            return "lhand"
        if "weaponpointdummyback" in key:
            return "spine3"
        if "weaponpointdummythigh" in key:
            return "rthigh"
        if "helmetpointdummy" in key:
            return "head"
        return ""

    sockets: list[I3AttachmentSocket] = []
    for name, bone_index, matrix in pending:
        candidates = [
            palette[bone_index].name
            for palette in palettes if bone_index < len(palette)
        ]
        expected = expected_parent(name)
        bone_name = next((
            candidate for candidate in candidates
            if "".join(
                character for character in candidate.casefold()
                if character.isalnum()
            ) == expected
        ), candidates[0] if len(set(candidates)) == 1 else "")
        sockets.append(I3AttachmentSocket(name, bone_index, bone_name, matrix))
    return sockets


def _row_matrix_multiply(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(
        sum(a[row * 4 + index] * b[index * 4 + column] for index in range(4))
        for row in range(4) for column in range(4)
    )


def geometry_model_matrices(root: I3File) -> dict[int, tuple[float, ...]]:
    """Return authored row-vector model matrices for scene geometry blocks."""
    identity = (
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    by_id = {block.block_id: block for block in root.blocks}
    transforms = [
        block for block in root.blocks
        if block.type_name in {"i3Transform", "i3Transform2"}
    ]
    referenced: set[int] = set()
    for transform in transforms:
        try:
            refs, _matrix_id = _parse_transform_refs(transform)
            referenced.update(refs)
        except (ValueError, struct.error):
            continue
    roots = [block.block_id for block in transforms if block.block_id not in referenced]
    result: dict[int, tuple[float, ...]] = {}

    def visit(node_id: int, parent: tuple[float, ...], active: set[int]) -> None:
        if node_id in active:
            return
        block = by_id.get(node_id)
        if block is None:
            return
        active = active | {node_id}
        try:
            if block.type_name in {"i3Transform", "i3Transform2"}:
                refs, matrix_id = _parse_transform_refs(block)
                matrix_block = by_id.get(matrix_id)
                local = (
                    tuple(struct.unpack_from("<16f", matrix_block.data, 0))
                    if matrix_block is not None
                    and matrix_block.type_name == "i3MatrixObject"
                    and len(matrix_block.data) >= 64 else identity
                )
                # I3/D3D uses row vectors: local is applied before its parent.
                model = _row_matrix_multiply(local, parent)
                for ref in refs:
                    visit(ref, model, active)
            elif block.type_name == "i3AttrSet":
                refs, _render_ids = _parse_attr_set(block.data)
                for ref in refs:
                    visit(ref, parent, active)
            elif block.type_name == "i3Node":
                for ref in _parse_node_refs(block):
                    visit(ref, parent, active)
            elif block.type_name == "i3BoneRef":
                _name, refs = _parse_bone_ref(block)
                for ref in refs:
                    visit(ref, parent, active)
            elif block.type_name == "i3Geometry":
                result.setdefault(block.block_id, parent)
        except (ValueError, struct.error):
            return

    for root_id in roots:
        visit(root_id, identity, set())
    return result


def _parse_geometry_attr(block: I3Block) -> tuple[int, int, int]:
    data = block.data
    base = 4 if data.startswith(b"GEO2") else 0
    if base + 17 > len(data):
        raise ValueError(f"{block.type_name}#{block.block_id}: truncated geometry attributes")
    return (
        struct.unpack_from("<I", data, base + 1)[0],
        struct.unpack_from("<I", data, base + 9)[0],
        struct.unpack_from("<I", data, base + 13)[0],
    )


def _parse_vertex_array(
    block: I3Block,
) -> tuple[
    int,
    int,
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[int, ...]],
    list[tuple[float, ...]],
]:
    data = block.data
    if len(data) < 40:
        raise ValueError(f"i3VertexArray#{block.block_id}: truncated header")
    flag, count = struct.unpack_from("<II", data, 4)
    if count == 0:
        return flag, 0, [], [], [], [], [], []
    payload_size = len(data) - 40
    if payload_size % count:
        raise ValueError(f"i3VertexArray#{block.block_id}: uneven vertex records")
    stride = payload_size // count
    if stride < 20:
        raise ValueError(f"i3VertexArray#{block.block_id}: unsupported {stride}-byte stride")

    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    lightmap_uvs: list[tuple[float, float]] = []
    all_indices: list[tuple[int, ...]] = []
    all_weights: list[tuple[float, ...]] = []
    blend_index_count = (flag >> 14) & 0xF
    explicit_weight_count = (flag >> 18) & 0xF
    # i3VertexFormat stores N blend indices and N-1 explicit weights. Both
    # two- and three-influence avatar meshes occur in shipped skin packs.
    # Testing only bit 0x80000 silently discarded the second influence of
    # one-explicit-weight (0x40000) meshes.
    blended_skin = (
        1 <= explicit_weight_count <= 3
        and blend_index_count == explicit_weight_count + 1
        and stride >= 36 + explicit_weight_count * 4
    )
    # 0x4000 is the common rigid-bone stream. Some articulated weapon meshes
    # use the equivalent wider-layout flag 0x40000 (often with 0x8000 for
    # additional packed vertex data); both store the rigid index at byte 32.
    rigid_skin = (
        not blended_skin and bool(flag & (0x4000 | 0x40000)) and stride >= 36
    )
    # I3's FVF-style bit 0x2 indicates an authored normal.  Current maps also
    # use a compact 0x881/28-byte stream: position, diffuse UV, lightmap UV.
    # Reading that stream as position, normal, UV consumes both UV sets as a
    # fake normal and leaves the lightmap at zero.
    has_normal = bool(flag & 0x2)
    # VF_DIFFUSE (0x8) inserts a packed RGBA value after the optional normal.
    # UVs and the skin palette therefore begin four bytes later.  Treating
    # these 72-byte head streams like the common 68-byte body stream turns UV
    # bytes into bone indices and catastrophically explodes the skinned mesh.
    vertex_color_size = 4 if flag & 0x8 else 0
    uv_offset = 12 + (12 if has_normal else 0) + vertex_color_size
    skin_offset = uv_offset + 8
    lightmap_offset = uv_offset + 8

    for index in range(count):
        offset = 40 + index * stride
        positions.append(struct.unpack_from("<3f", data, offset))
        if has_normal and stride >= 24:
            normals.append(struct.unpack_from("<3f", data, offset + 12))
        if stride >= uv_offset + 8:
            u, v = struct.unpack_from("<2f", data, offset + uv_offset)
            uvs.append((u, -v))
        else:
            u, v = struct.unpack_from("<2f", data, offset + 12)
            uvs.append((u, -v))

        if (
            bool(flag & 0x800) and not blended_skin and not rigid_skin
            and stride >= lightmap_offset + 8
        ):
            u, v = struct.unpack_from("<2f", data, offset + lightmap_offset)
            lightmap_uvs.append((u, -v))
        else:
            lightmap_uvs.append((0.0, 0.0))

        if blended_skin:
            packed = data[offset + skin_offset:offset + skin_offset + 4]
            raw_weights = struct.unpack_from(
                f"<{explicit_weight_count}f", data, offset + skin_offset + 4)
            weights = tuple(max(0.0, min(1.0, float(w))) for w in raw_weights)
            weights += (max(0.0, 1.0 - sum(weights)),)
            influences = [
                (packed[slot], weights[slot])
                for slot in range(blend_index_count)
                if weights[slot] > 1.0e-6
            ]
        elif rigid_skin:
            influences = [(data[offset + skin_offset], 1.0)]
        else:
            influences = []
        total = sum(weight for _, weight in influences)
        if total > 1.0e-8:
            all_indices.append(tuple(bone for bone, _ in influences))
            all_weights.append(tuple(weight / total for _, weight in influences))
        else:
            all_indices.append(())
            all_weights.append(())

    return flag, stride, positions, normals, uvs, lightmap_uvs, all_indices, all_weights


def _parse_index_array(block: I3Block) -> list[tuple[int, int, int]]:
    data = block.data
    if len(data) < 8:
        raise ValueError(f"i3IndexArray#{block.block_id}: truncated header")
    width = 2
    offset = 8
    if data.startswith(b"IIA2"):
        if len(data) < 32:
            raise ValueError(f"i3IndexArray#{block.block_id}: truncated IIA2 header")
        width = 4 if struct.unpack_from("<I", data, 12)[0] == 1 else 2
        offset = 32
    record_size = width * 3
    face_count = (len(data) - offset) // record_size
    code = "<3I" if width == 4 else "<3H"
    return [struct.unpack_from(code, data, offset + index * record_size) for index in range(face_count)]


def _parse_attr_set(data: bytes) -> tuple[list[int], list[int]]:
    """Decode the geometry and render-attribute references in i3AttrSet."""
    _name, offset = _dotnet_string(data, 0)
    if offset + 8 > len(data):
        raise ValueError("truncated i3AttrSet")
    offset += 4  # INF2
    _attr_type = struct.unpack_from("<i", data, offset)[0]
    offset += 4
    if offset + 8 > len(data):
        raise ValueError("truncated i3AttrSet geometry references")
    count_a = struct.unpack_from("<i", data, offset)[0]
    offset += 8  # count plus reserved int32
    if count_a < 0 or count_a > 65536 or offset + count_a * 4 > len(data):
        raise ValueError("invalid i3AttrSet geometry reference count")
    values_a = list(struct.unpack_from(f"<{count_a}i", data, offset)) if count_a else []
    offset += count_a * 4

    # Layout variants place material floats between A and ATS1. Locating the
    # explicit marker is safer and matches both old SDK and current-client data.
    marker = data.find(b"ATS1", offset)
    if marker < 0:
        # Current-client weapon scenes use a compact variant: UInt16 count
        # immediately followed by Int32 render-attribute block IDs.  K5, for
        # example, stores 7 followed by Material/Enable/Bind IDs here.
        # Older map/sky scenes put six bounding-box floats before the same
        # compact suffix.  Locate a UInt16 count whose Int32 ID array ends
        # exactly at the block boundary instead of assuming it starts here.
        # Requiring positive IDs also prevents arbitrary float bytes from
        # being mistaken for a render-reference table.
        for candidate in range(offset, len(data) - 1):
            remaining = len(data) - candidate - 2
            if remaining < 0 or remaining % 4:
                continue
            count_b = struct.unpack_from("<H", data, candidate)[0]
            if count_b != remaining // 4 or count_b > 65536:
                continue
            ids_offset = candidate + 2
            values_b = (
                list(struct.unpack_from(f"<{count_b}i", data, ids_offset))
                if count_b else []
            )
            if all(value > 0 for value in values_b):
                return values_a, values_b
        return values_a, []
    if marker + 16 > len(data):
        return values_a, []
    count_b = struct.unpack_from("<i", data, marker + 4)[0]
    ids_offset = marker + 16  # marker, count, reserved int64
    if count_b < 0 or count_b > 65536 or ids_offset + count_b * 4 > len(data):
        raise ValueError("invalid i3AttrSet render reference count")
    values_b = list(struct.unpack_from(f"<{count_b}i", data, ids_offset)) if count_b else []
    return values_a, values_b


def embedded_i3i_entries(
    data: bytes, source_name: str = "<Biah pack>"
) -> Iterator[I3PackEntry]:
    """Carve the self-sized I3IB streams used by current ``Biah`` packs.

    These common packs are not I3R2 scene containers.  Their texture payloads
    remain ordinary I3IB files, and each I3IB header supplies dimensions,
    mip-count, pixel format, and its embedded asset path, so no archive-table
    guessing is required.
    """
    cursor = 0
    ordinal = 0
    while True:
        start = data.find(b"I3IB", cursor)
        if start < 0:
            return
        cursor = start + 4
        if start + 60 > len(data):
            return
        width = struct.unpack_from("<H", data, start + 6)[0]
        height = struct.unpack_from("<H", data, start + 8)[0]
        mipmaps = max(1, struct.unpack_from("<H", data, start + 24)[0])
        comment_size = struct.unpack_from("<H", data, start + 26)[0]
        header_size = 60 + comment_size
        if not width or not height or start + header_size > len(data):
            continue

        format_a, format_b, format_c = (
            data[start + 10], data[start + 11], data[start + 13]
        )
        if format_a in {0x80, 0x81} and format_c in {0x80, 0xA0}:
            block_bytes, pixel_bytes = 8, 0
        elif format_a in {0x02, 0x04} and format_c == 0xA0:
            block_bytes, pixel_bytes = 16, 0
        elif format_a == 0x06 and format_c == 0x20:
            block_bytes, pixel_bytes = 0, 4
        elif format_a == 0x02 and format_b == 0x04 and format_c == 0x00:
            block_bytes, pixel_bytes = 0, 4
        elif format_a == 0x02 and format_b == 0x03 and format_c == 0x00:
            block_bytes, pixel_bytes = 0, 3
        else:
            continue

        body_size = 0
        mip_width, mip_height = width, height
        for _level in range(mipmaps):
            if block_bytes:
                body_size += (
                    max(1, (mip_width + 3) // 4)
                    * max(1, (mip_height + 3) // 4)
                    * block_bytes
                )
            else:
                body_size += mip_width * mip_height * pixel_bytes
            mip_width = max(1, mip_width // 2)
            mip_height = max(1, mip_height // 2)
        end = start + header_size + body_size
        if end > len(data):
            continue

        comment = data[start + 60:start + header_size]
        name = _clean_name(comment).replace("\\", "/")
        if not name:
            name = f"embedded_{ordinal:04d}.i3i"
        ordinal += 1
        yield I3PackEntry(
            parent_name=source_name,
            name=name,
            start=start,
            size=end - start,
            data=data[start:end],
            file_step=ordinal,
        )
        cursor = end


def biah_pack_entries(
    data: bytes, source_name: str = "<Biah pack>"
) -> Iterator[I3PackEntry]:
    """Decode the encrypted fixed-size file table in a current Biah pack."""
    if not data.startswith(b"Biah"):
        raise ValueError(f"{source_name}: not a Biah pack")

    descriptor_offset = -1
    file_count = 0
    file_step = 0
    search_end = min(len(data) - 8, 4096)
    for offset in range(4, search_end):
        descriptor = _rotate_decrypt(data[offset:offset + 8], 3)
        candidate_type = descriptor[3]
        candidate_count = struct.unpack_from("<H", descriptor, 4)[0]
        candidate_step = 92 if candidate_type == 50 else 76
        records_end = offset + 8 + candidate_count * candidate_step
        if not candidate_count or records_end > len(data):
            continue
        first = _rotate_decrypt(
            data[offset + 8:offset + 8 + candidate_step], 2
        )
        name = _clean_name(first[:52])
        if not name or not all(32 <= ord(char) < 127 for char in name):
            continue
        if candidate_step == 92:
            first_start = (
                struct.unpack_from("<H", first, 72)[0] * 65536
                + struct.unpack_from("<H", first, 80)[0]
            )
            first_size = (
                struct.unpack_from("<H", first, 74)[0] * 65536
                + struct.unpack_from("<H", first, 70)[0]
            )
        else:
            first_start = (
                struct.unpack_from("<H", first, 56)[0] * 65536
                + struct.unpack_from("<H", first, 64)[0]
            )
            first_size = (
                struct.unpack_from("<H", first, 58)[0] * 65536
                + struct.unpack_from("<H", first, 54)[0]
            )
        if first_start < records_end or first_start + first_size > len(data):
            continue
        descriptor_offset = offset
        file_count = candidate_count
        file_step = candidate_step
        break
    if descriptor_offset < 0:
        raise ValueError(f"{source_name}: Biah file table not found")

    records_start = descriptor_offset + 8
    for ordinal in range(file_count):
        record_offset = records_start + ordinal * file_step
        record = _rotate_decrypt(
            data[record_offset:record_offset + file_step], 2
        )
        name = _clean_name(record[:52]) or f"entry_{ordinal:04d}.bin"
        if file_step == 92:
            start = (
                struct.unpack_from("<H", record, 72)[0] * 65536
                + struct.unpack_from("<H", record, 80)[0]
            )
            size = (
                struct.unpack_from("<H", record, 74)[0] * 65536
                + struct.unpack_from("<H", record, 70)[0]
            )
        else:
            start = (
                struct.unpack_from("<H", record, 56)[0] * 65536
                + struct.unpack_from("<H", record, 64)[0]
            )
            size = (
                struct.unpack_from("<H", record, 58)[0] * 65536
                + struct.unpack_from("<H", record, 54)[0]
            )
        end = start + size
        payload = data[start:end] if 0 <= start <= end <= len(data) else b""
        yield I3PackEntry(
            parent_name=source_name,
            name=name,
            start=start,
            size=size,
            data=payload,
            file_step=file_step,
        )


def _texture_block_name(root: I3File, texture: I3Block) -> str:
    if 0 < texture.target < len(root.text_lines):
        name = root.text_lines[texture.target]
        if name:
            return name
    if 0 < texture.block_id < len(root.text_lines):
        # Current weapon scenes reuse sparse numeric IDs. Their texture block
        # ID is also the text-table line containing the asset path.
        name = root.text_lines[texture.block_id]
        if name:
            return name
    if len(texture.data) > 60:
        name_size = texture.data[26]
        return texture.data[60:60 + name_size].decode(
            "cp1252", errors="replace"
        )
    return ""


def _external_texture_block_map(
    root: I3File, references: Sequence[str] | None,
) -> dict[int, str]:
    """Map RSC texture dependencies onto unresolved I3 texture placeholders."""
    if not references:
        return {}
    unique_references: list[str] = []
    seen: set[str] = set()
    for reference in references:
        key = str(reference).casefold()
        if reference and key not in seen:
            seen.add(key)
            unique_references.append(str(reference))
    unresolved = [
        block for block in root.blocks
        if block.type_name in {"i3Texture", "i3TextureObject"}
        and not _texture_block_name(root, block)
        and not block.data.startswith((b"I3IB", b"DDS "))
    ]
    if not unresolved or not unique_references:
        return {}
    if len(unique_references) == 1:
        return {
            block.block_id: unique_references[0] for block in unresolved
        }
    return {
        block.block_id: reference
        for block, reference in zip(unresolved, unique_references)
    }


def _texture_refs_from_render_ids(
    root: I3File, render_ids: list[int],
    external_texture_blocks: dict[int, str] | None = None,
) -> dict[str, str]:
    by_id: dict[int, list[I3Block]] = {}
    for block in root.blocks:
        by_id.setdefault(block.block_id, []).append(block)
    result: dict[str, str] = {}
    for render_id in render_ids:
        binding = next(
            (item for item in by_id.get(render_id, []) if "BindAttr" in item.type_name),
            None,
        )
        if binding is None or len(binding.data) < 2:
            continue
        texture_id = struct.unpack_from("<H", binding.data, 0)[0]
        texture = next(
            (
                item for item in by_id.get(texture_id, [])
                if item.type_name in {"i3Texture", "i3TextureObject"}
            ),
            None,
        )
        if texture is None:
            continue
        name = _texture_block_name(root, texture)
        if not name:
            name = (external_texture_blocks or {}).get(texture.block_id, "")
        if not name:
            continue
        kind = "diffuse"
        lowered = binding.type_name.casefold()
        if "luxmap" in lowered:
            kind = "lightmap"
        elif "normal" in lowered:
            kind = "normal"
        elif "specular" in lowered:
            kind = "specular"
        elif "emissive" in lowered:
            kind = "emissive"
        elif "reflectmask" in lowered:
            kind = "reflect_mask"
        elif "reflect" in lowered:
            kind = "reflection"
        result.setdefault(kind, name)
    return result


def embedded_i3_textures(root: I3File) -> list[I3PackEntry]:
    """Expose textures embedded directly in an I3 scene as file entries.

    PB sky scenes keep each cubemap face in an ``i3Texture`` block and retain
    the original authoring path (usually a PSD) only as metadata.  Returning
    an I3I filename lets the normal texture decoder consume those blocks
    without inventing a second image parser.
    """
    entries: list[I3PackEntry] = []
    used_names: set[str] = set()
    for ordinal, texture in enumerate(
        block for block in root.blocks
        if block.type_name in {"i3Texture", "i3TextureObject"}
    ):
        if not texture.data.startswith((b"I3IB", b"DDS ")):
            continue
        name = ""
        if 0 < texture.target < len(root.text_lines):
            name = root.text_lines[texture.target]
        elif 0 < texture.block_id < len(root.text_lines):
            name = root.text_lines[texture.block_id]
        elif len(texture.data) > 60:
            comment_size = struct.unpack_from("<H", texture.data, 26)[0]
            name = _clean_name(texture.data[60:60 + comment_size])
        leaf = Path(name.replace("\\", "/")).name if name else ""
        stem = Path(leaf).stem or f"embedded_texture_{ordinal:04d}"
        suffix = ".dds" if texture.data.startswith(b"DDS ") else ".i3i"
        candidate = f"{stem}{suffix}"
        if candidate.casefold() in used_names:
            candidate = f"{stem}__{ordinal:04d}{suffix}"
        used_names.add(candidate.casefold())
        entries.append(I3PackEntry(
            parent_name=root.source_name,
            name=candidate,
            start=texture.start,
            size=len(texture.data),
            data=texture.data,
            file_step=ordinal,
        ))
    return entries


def _material_from_render_ids(root: I3File, render_ids: list[int]) -> dict[str, Any]:
    """Decode the stable I3 render attributes used by scene materials.

    I3 material payloads are four RGBA colours followed by a Direct3D-style
    specular power.  Boolean render switches are one byte.  AlphaFunc stores
    an integer reference value followed by the comparison enum.
    """
    by_id = {block.block_id: block for block in root.blocks}
    state: dict[str, Any] = {}
    enable_names = {
        "i3TextureEnableAttr": "texture_enabled",
        "i3NormalMapEnableAttr": "normal_enabled",
        "i3SpecularMapEnableAttr": "specular_enabled",
        "i3EmissiveMapEnableAttr": "emissive_enabled",
        "i3ReflectMapEnableAttr": "reflection_enabled",
        "i3ReflectMaskMapEnableAttr": "reflect_mask_enabled",
        "i3LuxMapEnableAttr": "lightmap_enabled",
        "i3LightingEnableAttr": "lighting_enabled",
        "i3AlphaTestEnableAttr": "alpha_test",
        "i3BlendEnableAttr": "blend_enabled",
        "i3ZWriteEnableAttr": "z_write",
    }
    for render_id in render_ids:
        block = by_id.get(render_id)
        if block is None:
            continue
        if block.type_name == "i3MaterialAttr" and len(block.data) >= 68:
            values = struct.unpack_from("<17f", block.data)
            state.update({
                "ambient": tuple(values[0:4]),
                "diffuse": tuple(values[4:8]),
                "specular": tuple(values[8:12]),
                "emissive": tuple(values[12:16]),
                "specular_power": float(values[16]),
            })
        elif block.type_name in enable_names and block.data:
            state[enable_names[block.type_name]] = bool(block.data[0])
        elif block.type_name == "i3AlphaFuncAttr" and len(block.data) >= 5:
            reference = struct.unpack_from("<I", block.data, 0)[0]
            state["alpha_reference"] = max(0.0, min(1.0, reference / 255.0))
            state["alpha_function"] = int(block.data[4])
        elif block.type_name == "i3FaceCullModeAttr" and block.data:
            state["face_cull_mode"] = int.from_bytes(
                block.data[:min(4, len(block.data))], "little"
            )
        elif block.type_name in {"i3SrcBlendAttr", "i3DestBlendAttr"} and block.data:
            key = "source_blend" if "Src" in block.type_name else "destination_blend"
            state[key] = int.from_bytes(block.data[:min(4, len(block.data))], "little")
        elif block.type_name == "i3BlendModeAttr" and len(block.data) >= 2:
            state["source_blend"] = int(block.data[0])
            state["destination_blend"] = int(block.data[1])
            if len(block.data) >= 3:
                state["blend_operation"] = int(block.data[2])
    return state


def _mesh_texture_refs(root: I3File, geometry_id: int) -> dict[str, str]:
    """Legacy direct-AttrSet lookup used when graph traversal is unavailable."""
    render_ids = _mesh_render_ids(root, geometry_id)
    return _texture_refs_from_render_ids(root, render_ids) if render_ids else {}


def _mesh_render_ids(root: I3File, geometry_id: int) -> list[int]:
    """Return render IDs from the AttrSet directly owning one geometry."""
    for attr_set in root.blocks_of_type("i3AttrSet"):
        try:
            geometry_ids, render_ids = _parse_attr_set(attr_set.data)
        except (ValueError, struct.error):
            continue
        if geometry_id in geometry_ids:
            return render_ids
    return []


def _geometry_graph_contexts(root: I3File) -> dict[int, tuple[str, list[int], int]]:
    """Resolve geometry ownership and inherited render state from the I3 graph."""
    by_id = {block.block_id: block for block in root.blocks}
    contexts: dict[int, tuple[str, list[int], int]] = {}

    def visit(
        node_id: int, owner: str, inherited: list[int], lod_index: int,
        active: set[int],
    ) -> None:
        if node_id in active:
            return
        block = by_id.get(node_id)
        if block is None:
            return
        active = active | {node_id}
        try:
            if block.type_name == "i3BoneRef":
                bone_name, refs = _parse_bone_ref(block)
                for ref in refs:
                    visit(ref, bone_name or owner, inherited, lod_index, active)
                return
            if block.type_name == "i3Node":
                for ref in _parse_node_refs(block):
                    visit(ref, owner, inherited, lod_index, active)
                return
            if block.type_name in {"i3Transform", "i3Transform2"}:
                refs, _matrix_id = _parse_transform_refs(block)
                for ref in refs:
                    visit(ref, owner, inherited, lod_index, active)
                return
            if block.type_name == "i3AttrSet":
                refs, render_ids = _parse_attr_set(block.data)
                # Nearest state comes first so its bindings win by kind.
                combined = render_ids + [item for item in inherited if item not in render_ids]
                for ref in refs:
                    visit(ref, owner, combined, lod_index, active)
                return
            if block.type_name == "i3Geometry":
                contexts.setdefault(block.block_id, (owner, inherited, lod_index))
        except (ValueError, struct.error):
            return

    roots: list[int] = []
    for lod in root.blocks_of_type("i3LOD"):
        if len(lod.data) >= 20:
            roots.append(struct.unpack_from("<i", lod.data, 16)[0])
    for attr_set_id, scene_root_id in _scene_graph_info_bindings(root):
        render_ids: list[int] = []
        attr_set = by_id.get(attr_set_id)
        if attr_set is not None and attr_set.type_name == "i3AttrSet":
            try:
                _refs, render_ids = _parse_attr_set(attr_set.data)
            except (ValueError, struct.error):
                pass
        visit(scene_root_id, "", render_ids, 0, set())
        roots.append(scene_root_id)
    if not roots:
        # Map worlds commonly omit i3LOD/i3Node entirely.  Their authored
        # hierarchy starts at one or more top-level i3AttrSet blocks, each of
        # which contributes a group lightmap before referencing the material
        # AttrSets below it.  Starting from every child AttrSet loses that
        # inherited state, so identify the actual unreferenced graph roots.
        graph_types = {"i3BoneRef", "i3Node", "i3AttrSet"}
        candidates = [block for block in root.blocks if block.type_name in graph_types]
        referenced: set[int] = set()
        for block in candidates:
            try:
                if block.type_name == "i3BoneRef":
                    _name, refs = _parse_bone_ref(block)
                elif block.type_name == "i3Node":
                    refs = _parse_node_refs(block)
                else:
                    refs, _render_ids = _parse_attr_set(block.data)
                referenced.update(refs)
            except (ValueError, struct.error):
                continue
        roots.extend(
            block.block_id for block in candidates
            if block.block_id not in referenced
        )
        if not roots:
            roots.extend(block.block_id for block in candidates)
    for lod_index, root_id in enumerate(roots):
        visit(root_id, "", [], lod_index, set())
    return contexts


def parse_i3_meshes(
    root: I3File,
    external_texture_references: Sequence[str] | None = None,
) -> list[I3Mesh]:
    """Decode renderable meshes, including the verified I3 skin payload."""
    by_id = {block.block_id: block for block in root.blocks}
    vertex_cache: dict[int, tuple[Any, ...]] = {}
    index_cache: dict[int, list[tuple[int, int, int]]] = {}
    graph_contexts = _geometry_graph_contexts(root)
    external_texture_blocks = _external_texture_block_map(
        root, external_texture_references
    )
    meshes: list[I3Mesh] = []
    for geometry in root.blocks_of_type("i3Geometry"):
        name, attr_id = _parse_geometry_reference(geometry)
        attributes = by_id.get(attr_id)
        if attributes is None or attributes.type_name != "i3GeometryAttr":
            continue
        try:
            triangle_count, vertex_id, index_id = _parse_geometry_attr(attributes)
            vertex_block = by_id.get(vertex_id)
            if vertex_block is None or vertex_block.type_name != "i3VertexArray":
                continue
            if vertex_id not in vertex_cache:
                vertex_cache[vertex_id] = _parse_vertex_array(vertex_block)
            (
                flag, stride, positions, normals, uvs, lightmap_uvs,
                bone_indices, bone_weights,
            ) = vertex_cache[vertex_id]
            index_block = by_id.get(index_id)
            if index_block is not None and index_block.type_name == "i3IndexArray":
                if index_id not in index_cache:
                    index_cache[index_id] = _parse_index_array(index_block)
                faces = index_cache[index_id]
            else:
                faces = [
                    (face * 3, face * 3 + 1, face * 3 + 2)
                    for face in range(triangle_count)
                ]
            faces = [
                face for face in faces[:triangle_count]
                if max(face, default=-1) < len(positions)
            ]
        except (ValueError, struct.error):
            continue
        attachment_name, inherited_render_ids, lod_index = graph_contexts.get(
            geometry.block_id, ("", [], 0)
        )
        render_ids = inherited_render_ids or _mesh_render_ids(
            root, geometry.block_id
        )
        textures = _texture_refs_from_render_ids(
            root, render_ids, external_texture_blocks
        )
        material = _material_from_render_ids(root, render_ids)
        meshes.append(I3Mesh(
            name=name or f"geometry_{geometry.block_id}",
            source_name=name,
            geometry_id=geometry.block_id,
            vertex_array_id=vertex_id,
            index_array_id=index_id,
            vertex_flag=flag,
            vertex_stride=stride,
            positions=positions,
            normals=normals,
            uvs=uvs,
            lightmap_uvs=lightmap_uvs,
            bone_indices=bone_indices,
            bone_weights=bone_weights,
            faces=faces,
            textures=textures,
            material=material,
            # The owner comes from the actual i3BoneRef/i3AttrSet scene graph,
            # not block-table proximity.
            attachment_name=attachment_name,
            lod_index=lod_index,
        ))
    return meshes


def i3i_to_dds(data: bytes, source_name: str = "<texture>") -> bytes:
    """Convert the SDK's I3IB texture container to a standard DDS stream."""
    if data.startswith(b"DDS "):
        return data
    if len(data) <= 60 or not data.startswith(b"I3IB"):
        raise ValueError(f"{source_name}: not an I3IB/DDS texture")
    width = struct.unpack_from("<H", data, 6)[0]
    height = struct.unpack_from("<H", data, 8)[0]
    mipmaps = max(1, struct.unpack_from("<H", data, 24)[0])
    comment_size = struct.unpack_from("<H", data, 26)[0]
    body_offset = 60 + comment_size
    if not width or not height or body_offset > len(data):
        raise ValueError(f"{source_name}: invalid I3IB dimensions or payload")
    body = data[body_offset:]

    fourcc = None
    rgb_bits = 0
    masks = (0, 0, 0, 0)
    pixel_flags = 0x4  # DDPF_FOURCC
    if data[10] in {0x80, 0x81} and data[13] in {0x80, 0xA0}:
        fourcc = b"DXT1"
    elif data[10] == 0x02 and data[13] == 0xA0:
        fourcc = b"DXT3"
    elif data[10] == 0x04 and data[13] == 0xA0:
        fourcc = b"DXT5"
    elif data[10] == 0x06 and data[13] == 0x20:
        pixel_flags, rgb_bits = 0x41, 32
        masks = (0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    elif data[10] == 0x02 and data[11] == 0x04 and data[13] == 0x00:
        pixel_flags, rgb_bits = 0x40, 32
        masks = (0x00FF0000, 0x0000FF00, 0x000000FF, 0)
    elif data[10] == 0x02 and data[11] == 0x03 and data[13] == 0x00:
        pixel_flags, rgb_bits = 0x40, 24
        masks = (0x00FF0000, 0x0000FF00, 0x000000FF, 0)
    else:
        raise ValueError(f"{source_name}: unsupported I3IB pixel format")

    header = bytearray(128)
    header[:4] = b"DDS "
    struct.pack_into("<I", header, 4, 124)
    flags = 0x00001007 | (0x00080000 if fourcc else 0x00000008)
    if mipmaps > 1:
        flags |= 0x00020000
    struct.pack_into("<I", header, 8, flags)
    struct.pack_into("<II", header, 12, height, width)
    linear_size = max(1, ((width + 3) // 4) * ((height + 3) // 4))
    linear_size *= 8 if fourcc == b"DXT1" else (16 if fourcc else max(1, rgb_bits // 8) * width)
    struct.pack_into("<I", header, 20, linear_size)
    struct.pack_into("<I", header, 28, mipmaps)
    struct.pack_into("<I", header, 76, 32)
    struct.pack_into("<I", header, 80, pixel_flags)
    if fourcc:
        header[84:88] = fourcc
    else:
        struct.pack_into("<I", header, 88, rgb_bits)
        struct.pack_into("<4I", header, 92, *masks)
    caps = 0x1000 | (0x400008 if mipmaps > 1 else 0)
    struct.pack_into("<I", header, 108, caps)
    return bytes(header) + body


def _bone_report(bones: list[I3Bone]) -> dict[str, Any]:
    return {
        "bone_count": len(bones),
        "bones": [
            {
                "index": bone.index,
                "name": bone.name,
                "id": bone.bone_id,
                "parent": bone.parent,
                "matrix_column_major": list(bone.matrix),
            }
            for bone in bones
        ],
    }


def _matrix_to_transform(matrix: list[float]) -> dict[str, Any]:
    """Convert the SDK's column-major bind matrix into AGR TRS fields."""
    sx = math.sqrt(matrix[0] ** 2 + matrix[1] ** 2 + matrix[2] ** 2) or 1.0
    sy = math.sqrt(matrix[4] ** 2 + matrix[5] ** 2 + matrix[6] ** 2) or 1.0
    sz = math.sqrt(matrix[8] ** 2 + matrix[9] ** 2 + matrix[10] ** 2) or 1.0
    r00, r01, r02 = matrix[0] / sx, matrix[4] / sy, matrix[8] / sz
    r10, r11, r12 = matrix[1] / sx, matrix[5] / sy, matrix[9] / sz
    r20, r21, r22 = matrix[2] / sx, matrix[6] / sy, matrix[10] / sz
    trace = r00 + r11 + r22
    if trace > 0.0:
        root = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * root
        qx = (r21 - r12) / root
        qy = (r02 - r20) / root
        qz = (r10 - r01) / root
    elif r00 > r11 and r00 > r22:
        root = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / root
        qx = 0.25 * root
        qy = (r01 + r10) / root
        qz = (r02 + r20) / root
    elif r11 > r22:
        root = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / root
        qx = (r01 + r10) / root
        qy = 0.25 * root
        qz = (r12 + r21) / root
    else:
        root = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / root
        qx = (r02 + r20) / root
        qy = (r12 + r21) / root
        qz = 0.25 * root
    return {
        "translation": [matrix[12], matrix[13], matrix[14]],
        "rotation_xyzw": [qx, qy, qz, qw],
        "scale": [sx, sy, sz],
    }


def skeleton_manifest(report: dict[str, Any]) -> dict[str, Any]:
    """Convert an inspect-i3 report into AGR skeleton definitions."""
    skeletons: list[dict[str, Any]] = []
    entries = [{"source": report["source"], "block_id": item.get("block_id"), "skeleton": item} for item in report["skeletons"]]
    for entry in report["pack_entries"]:
        for item in entry.get("skeletons", []):
            entries.append({"source": entry["name"], "block_id": item.get("block_id"), "skeleton": item})
    for ordinal, entry in enumerate(entries):
        item = entry["skeleton"]
        if "bones" not in item:
            continue
        source_name = entry["source"]
        block_id = entry["block_id"]
        skeleton_id = f"{source_name}#skeleton-{block_id if block_id is not None else ordinal}"
        bones = []
        for bone in item["bones"]:
            parent_id = int(bone["id"])
            bones.append({
                "name": bone["name"],
                "parent_index": None if parent_id < 0 else parent_id,
                "bind_transform": _matrix_to_transform(bone["matrix_column_major"]),
                "bind_matrix_column_major": bone["matrix_column_major"],
            })
        skeletons.append({"id": skeleton_id, "source": source_name, "block_id": block_id, "bones": bones})
    return {
        "format": "pointblank-agr.skeleton-manifest",
        "version": 1,
        "source_asset": report["source"],
        "skeletons": skeletons,
    }


def inspect_i3(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    root = parse_i3r2(source.read_bytes(), source.as_posix())
    report: dict[str, Any] = {
        "format": "pointblank-agr.i3-inspection",
        "version": 1,
        "source": str(source.resolve()),
        "crypted_input": root.crypted,
        "block_count": len(root.blocks),
        "block_types": dict(Counter(block.type_name for block in root.blocks)),
        "skeletons": [],
        "pack_entries": [],
    }
    for block in root.blocks_of_type("i3BoneMatrixListAttr"):
        try:
            report["skeletons"].append({"source": root.source_name, "block_id": block.block_id, **_bone_report(parse_bone_matrix_list(block.data, root.source_name))})
        except ValueError as exc:
            report["skeletons"].append({"source": root.source_name, "block_id": block.block_id, "error": str(exc)})

    for block in root.blocks_of_type("i3PackNode"):
        entries = parse_pack_entries(block.data, root.data, root.source_name)
        for entry in entries:
            entry_report: dict[str, Any] = {
                "name": entry.name,
                "parent_name": entry.parent_name,
                "start": entry.start,
                "size": entry.size,
                "file_step": entry.file_step,
                "parseable_i3r2": False,
                "block_types": {},
                "skeletons": [],
            }
            try:
                child = parse_i3r2(entry.data, entry.name)
                entry_report["parseable_i3r2"] = True
                entry_report["block_types"] = dict(Counter(block.type_name for block in child.blocks))
                for child_block in child.blocks_of_type("i3BoneMatrixListAttr"):
                    try:
                        entry_report["skeletons"].append({"block_id": child_block.block_id, **_bone_report(parse_bone_matrix_list(child_block.data, entry.name))})
                    except ValueError as exc:
                        entry_report["skeletons"].append({"block_id": child_block.block_id, "error": str(exc)})
            except ValueError as exc:
                entry_report["parse_error"] = str(exc)
            report["pack_entries"].append(entry_report)
    return report


def write_i3_report(report: dict[str, Any], output: str | Path) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def write_skeleton_manifest(manifest: dict[str, Any], output: str | Path) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

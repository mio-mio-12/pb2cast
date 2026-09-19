"""Current-client Script.i3Pack metadata used by preview composition.

The supplied PB SDK describes an older registry generation.  This module
decodes the shipped current-client Weapon.Pef directly, caches the read-only
index by the Script pack fingerprint, and resolves one selected/derived weapon
pack to its authored runtime class and character-animation family.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Any

from studio_i3 import (
    I3Block,
    I3File,
    _dotnet_string,
    _rotate_decrypt,
    parse_i3r2,
)

from i3 import archive_entries


WEAPON_FIELDS = (
    "UiPath",
    "_UIShapeIndex",
    "ClassMeta",
    "LinkedToCharaAI",
    "FastReloadAnimation",
    "LoadBulletLeftBarrel",
    "LoadMagazineLeftBarrel",
    "LoadMagazineReady",
    "LoadMagToLoadBullet",
    "AttachedSubWeapon",
    "UsingMagazine",
    "ReloadLoopAnimation",
    "ReloadBulletCount",
    "ReloadTime",
    "_ResName",
    "_ResName_I3S",
    "ITEMID",
)

EXTENSION_FIELDS = (
    "Type",
    "ITEMID",
    "UseExtShape",
)


def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _resource_identity(value: Any) -> str:
    stem = Path(str(value).replace("\\", "/")).stem
    stem = re.sub(r"^(?:weapon|wpn)[_-]*", "", stem, flags=re.IGNORECASE)
    return _normalized(stem)


def _decode_pef(data: bytes, source_name: str) -> I3File:
    if data[:4] == b'I3R2':
        return parse_i3r2(data, source_name)
    # i3PackFile::Open/Read uses READ_UNIT=2048 and BitRotateDecript(...,3)
    # independently on EACH secure-entry block, including the short tail.
    # Whole-entry rotation corrupts every block boundary, including section
    # offsets, string lengths and item IDs. Never mask that corruption by
    # silently dropping "stale" blocks or overwriting the signature byte.
    read_unit = 2048
    decoded = b''.join(_rotate_decrypt(data[i:i+read_unit], 3)
                       for i in range(0, len(data), read_unit))
    if decoded[:4] != b'I3R2':
        raise ValueError(f'{source_name}: secure PEF did not decode to I3R2')
    root = parse_i3r2(decoded, source_name)
    return I3File(root.source_name, root.data, root.text_lines, root.blocks, True)


def _global_block_id(block: I3Block) -> int:
    raw_target = 0 if block.target == 0 else block.target + 32768
    return raw_target * 65536 + block.block_id


def _parse_array(data: bytes) -> tuple[str, list[Any]]:
    name, offset = _dotnet_string(data, 0)
    if offset + 12 > len(data):
        return name, []
    offset += 4
    value_type, count = struct.unpack_from("<II", data, offset)
    offset += 8
    if count > 1_000_000:
        return name, []
    values: list[Any] = []
    if value_type == 0 and offset + count * 4 <= len(data):
        values = list(struct.unpack_from(f"<{count}i", data, offset))
    elif value_type == 1 and offset + count * 4 <= len(data):
        values = list(struct.unpack_from(f"<{count}f", data, offset))
    elif value_type == 2:
        for _index in range(count):
            if offset + 8 > len(data):
                break
            marker = data[offset:offset + 4]
            size = struct.unpack_from("<I", data, offset + 4)[0]
            offset += 8
            byte_count = size * 2 if marker == b"RGS3" else size
            if offset + byte_count > len(data):
                break
            raw = data[offset:offset + byte_count]
            offset += byte_count
            values.append(raw.decode(
                "utf-16le" if marker == b"RGS3" else "cp1252",
                errors="replace",
            ))
    elif value_type in (3, 4, 5):
        width = value_type - 1
        end = offset + count * width * 4
        if end <= len(data):
            flat = struct.unpack_from(f"<{count * width}f", data, offset)
            values = [
                list(flat[index:index + width])
                for index in range(0, len(flat), width)
            ]
    return name, values


def _parse_key(data: bytes) -> tuple[str, list[int], list[int]]:
    name, offset = _dotnet_string(data, 0)
    if offset + 76 > len(data):
        return name, [], []
    offset += 4 + 8
    child_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4 + 60
    if child_count > 100_000 or offset + child_count * 4 + 16 > len(data):
        return name, [], []
    child_refs = list(struct.unpack_from(
        f"<{child_count}I", data, offset,
    )) if child_count else []
    offset += child_count * 4
    if data[offset:offset + 4] != b"RGK1":
        return name, child_refs, []
    value_count = struct.unpack_from("<I", data, offset + 4)[0]
    offset += 16
    if value_count > 1_000_000 or offset + value_count * 8 > len(data):
        return name, child_refs, []
    return name, child_refs, [
        struct.unpack_from("<Q", data, offset + index * 8)[0]
        for index in range(value_count)
    ]


def _representative(values: list[Any]) -> Any:
    cleaned = []
    for value in values:
        if isinstance(value, str):
            value = "".join(
                character for character in value
                if character >= " " or character in "\t\n"
            ).strip()
        if value not in (None, ""):
            cleaned.append(value)
    if not cleaned:
        return None
    encoded = Counter(
        json.dumps(value, sort_keys=True) for value in cleaned
    ).most_common(1)[0][0]
    return json.loads(encoded)


def _weapon_records(
    root: I3File, *, include_field_values: bool = False,
) -> list[dict[str, Any]]:
    nodes: dict[int, dict[str, Any]] = {}
    for block in root.blocks:
        try:
            if block.type_name == "i3RegArray":
                name, values = _parse_array(block.data)
                nodes[_global_block_id(block)] = {
                    "kind": "array", "name": name, "values": values,
                }
            elif block.type_name == "i3RegKey":
                name, child_refs, value_refs = _parse_key(block.data)
                nodes[_global_block_id(block)] = {
                    "kind": "key", "name": name,
                    "child_refs": child_refs,
                    "value_refs": value_refs,
                }
        except (ValueError, struct.error, UnicodeError):
            continue

    records = []
    for node in nodes.values():
        if node["kind"] != "key":
            continue
        fields = {}
        field_values = {}
        for reference in node["value_refs"]:
            value = nodes.get(reference)
            if value and value["kind"] == "array" and include_field_values:
                field_values.setdefault(value["name"], []).append(value["values"])
            if value and value["kind"] == "array" and value["name"] in WEAPON_FIELDS:
                fields[value["name"]] = _representative(value["values"])
        # Extension keys also carry ITEMID, but they are child descriptors,
        # not weapon database records.  The client reaches them through the
        # parent CWeaponInfo::_ReadExtensionKey loop.  Keep that graph shape
        # instead of flattening an ExtensionN into a false weapon record.
        has_resource_identity = any(fields.get(name) not in (None, "") for name in (
            "_ResName", "_ResName_I3S", "ClassMeta", "LinkedToCharaAI",
        ))
        # Full audits must retain item records with deliberately empty
        # resources too. Keep the compact render projection unchanged.
        full_item_record = (include_field_values and fields.get('ITEMID')
                            and not re.fullmatch(r'Extension[1-4]', node['name'], re.IGNORECASE))
        if not has_resource_identity and not full_item_record:
            continue
        extensions = []
        for reference in node.get("child_refs", []):
            child = nodes.get(reference)
            if (
                not child
                or child.get("kind") != "key"
                or not re.fullmatch(r"Extension[1-4]", child.get("name", ""))
            ):
                continue
            extension_fields = {}
            extension_values = {}
            for value_reference in child.get("value_refs", []):
                value = nodes.get(value_reference)
                if value and value["kind"] == "array" and include_field_values:
                    extension_values.setdefault(value["name"], []).append(value["values"])
                if (
                    value
                    and value.get("kind") == "array"
                    and value.get("name") in EXTENSION_FIELDS
                ):
                    extension_fields[value["name"]] = _representative(
                        value.get("values", [])
                    )
            if extension_fields:
                extensions.append({
                    "key": child["name"],
                    **extension_fields,
                    **({"_FieldValues": extension_values} if include_field_values else {}),
                })
        extensions.sort(key=lambda value: value["key"])
        records.append({
            "key": node["name"],
            **fields,
            "Extensions": extensions,
            **({"_FieldValues": field_values} if include_field_values else {}),
        })
    records.sort(key=lambda value: str(value.get("key", "")).casefold())
    return records


def _script_fingerprint(script_pack: Path) -> tuple[str, Path]:
    stat = script_pack.stat()
    token = hashlib.sha1(
        f"{script_pack.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|v4-secure-blocks".encode(
            "utf-8", errors="replace",
        )
    ).hexdigest()[:20]
    return token, script_pack


def load_weapon_metadata_index(
    pb_root: str | Path, cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(pb_root).resolve()
    script_pack = root / "Pack" / "Script.i3Pack"
    token, script_pack = _script_fingerprint(script_pack)
    cache_path = (
        Path(cache_dir) / f"weapon_script_metadata_{token}.json"
        if cache_dir is not None else None
    )
    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == token:
                return cached
        except (OSError, ValueError):
            pass

    entry = next(
        (
            value for value in archive_entries(script_pack)
            if value.name.casefold() == "weapon.pef"
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"Weapon.Pef was not found in {script_pack}")
    records = _weapon_records(_decode_pef(entry.data, "Weapon.Pef"))
    result = {
        "fingerprint": token,
        "script_pack": str(script_pack),
        "records": records,
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
        temporary.replace(cache_path)
        # Keep older fingerprinted projections for reproducible baselines.
        # Cache version changes must not erase the approved audit inputs.
    return result


def resolve_weapon_metadata(
    pb_root: str | Path,
    asset_path: str | Path,
    resolved_geometry: str | Path,
    asset_metadata: dict | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve one selected pack to an exact or deterministic PEF record."""
    index = load_weapon_metadata_index(pb_root, cache_dir)
    metadata = asset_metadata or {}
    selected = _resource_identity(Path(asset_path).stem)
    geometry = _resource_identity(Path(resolved_geometry).stem)
    variant = metadata.get("weapon_variant") or {}
    variant_base = _resource_identity(variant.get("base", ""))
    identities = {
        "selected": selected,
        "geometry": geometry,
        "variant_base": variant_base,
    }

    ranked = []
    for record in index["records"]:
        aliases = {
            "key": _resource_identity(record.get("key", "")),
            "variant": _resource_identity(record.get("_ResName_I3S", "")),
            "geometry": _resource_identity(record.get("_ResName", "")),
        }
        score = 0
        reason = ""
        if selected and aliases["key"] == selected:
            score, reason = 160, "exact Weapon.Pef key"
        elif selected and aliases["variant"] == selected:
            score, reason = 155, "exact _ResName_I3S variant"
        elif selected and aliases["geometry"] == selected:
            score, reason = 145, "exact selected _ResName"
        elif geometry and aliases["key"] == geometry:
            score, reason = 140, "exact geometry-provider key"
        elif geometry and aliases["variant"] == geometry:
            score, reason = 135, "exact geometry-provider _ResName_I3S"
        elif geometry and aliases["geometry"] == geometry:
            score, reason = 130, "exact geometry-provider _ResName"
        elif variant_base and aliases["geometry"] == variant_base:
            score, reason = 120, "exact texture-variant base geometry"
        if score:
            # Prefer the non-durability/base item when semantically tied.
            key = str(record.get("key", ""))
            if "durability" not in key.casefold():
                score += 2
            ranked.append((score, key.casefold(), record, reason))

    if not ranked:
        return {
            "status": "unmatched",
            "identities": identities,
            "script_fingerprint": index["fingerprint"],
        }
    ranked.sort(key=lambda item: (-item[0], item[1]))
    top_score = ranked[0][0]
    top = [value for value in ranked if value[0] == top_score]
    record = top[0][2]
    signatures = {
        (
            value[2].get("ClassMeta"),
            value[2].get("LinkedToCharaAI"),
            value[2].get("_ResName"),
            value[2].get("_ResName_I3S"),
        )
        for value in top
    }
    return {
        "status": "exact" if top_score >= 145 else "provider_fallback",
        "reason": top[0][3],
        "score": top_score,
        "ambiguous": len(signatures) > 1,
        "identities": identities,
        "script_fingerprint": index["fingerprint"],
        "record": record,
    }

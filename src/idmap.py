import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union


def _read_id_file(path: Path) -> List[str]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n").strip()
            if "\t" in line:
                line = line.split("\t")[-1].strip()
            lines.append(line)
    return lines


@dataclass(frozen=True)
class IDMap:
    entities: List[str]
    relations: List[str]
    name_map: Dict[str, str]

    @classmethod
    def from_dir(
        cls,
        data_dir: Union[str, Path],
        name_map_path: Union[str, Path, None] = None,
    ) -> "IDMap":
        data_dir = Path(data_dir)
        ent_path = data_dir / "entities.txt"
        rel_path = data_dir / "relations.txt"
        if not ent_path.exists():
            raise FileNotFoundError(f"Missing entities file: {ent_path}")
        if not rel_path.exists():
            raise FileNotFoundError(f"Missing relations file: {rel_path}")
        name_map: Dict[str, str] = {}
        if name_map_path is not None:
            name_map_file = Path(name_map_path)
            if not name_map_file.is_absolute():
                name_map_file = data_dir / name_map_file
            if not name_map_file.exists():
                raise FileNotFoundError(f"Missing entity name map: {name_map_file}")
            name_map = json.loads(name_map_file.read_text(encoding="utf-8"))
            if not isinstance(name_map, dict):
                raise ValueError("entity name map must be a JSON object {mid: name}")
        return cls(
            entities=_read_id_file(ent_path),
            relations=_read_id_file(rel_path),
            name_map=name_map,
        )

    def entity_text(self, eid: Union[int, str]) -> str:
        if isinstance(eid, str):
            return self.name_map.get(eid, eid)
        if 0 <= eid < len(self.entities):
            mid = self.entities[eid]
            return self.name_map.get(mid, mid)
        return f"<UNK_ENTITY:{eid}>"

    def relation_text(self, rid: Union[int, str]) -> str:
        if isinstance(rid, str):
            return rid
        if 0 <= rid < len(self.relations):
            return self.relations[rid]
        return f"<UNK_REL:{rid}>"

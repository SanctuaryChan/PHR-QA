from dataclasses import dataclass
from pathlib import Path
from typing import List, Union


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

    @classmethod
    def from_dir(cls, data_dir: Union[str, Path]) -> "IDMap":
        data_dir = Path(data_dir)
        ent_path = data_dir / "entities.txt"
        rel_path = data_dir / "relations.txt"
        if not ent_path.exists():
            raise FileNotFoundError(f"Missing entities file: {ent_path}")
        if not rel_path.exists():
            raise FileNotFoundError(f"Missing relations file: {rel_path}")
        return cls(entities=_read_id_file(ent_path), relations=_read_id_file(rel_path))

    def entity_text(self, eid: Union[int, str]) -> str:
        if isinstance(eid, str):
            return eid
        if 0 <= eid < len(self.entities):
            return self.entities[eid]
        return f"<UNK_ENTITY:{eid}>"

    def relation_text(self, rid: Union[int, str]) -> str:
        if isinstance(rid, str):
            return rid
        if 0 <= rid < len(self.relations):
            return self.relations[rid]
        return f"<UNK_REL:{rid}>"


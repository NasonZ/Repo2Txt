"""The directory tree of a repository, with token totals."""

from dataclasses import dataclass, field

from .models import Repository


@dataclass
class Node:
    path: str
    tokens: int = 0
    files: int = 0
    children: dict[str, "Node"] = field(default_factory=dict)

    @property
    def is_dir(self) -> bool:
        return bool(self.children)

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


@dataclass
class Entry:
    """One line of a map: a file, or a directory whose contents may be collapsed."""

    path: str
    tokens: int
    depth: int
    files: int | None = None  # set for directories


def build_tree(token_map: dict[str, int]) -> Node:
    """Build the tree for `{path: tokens}`; every directory carries its totals."""
    root = Node("")
    for path, tokens in token_map.items():
        node, walked = root, ""
        for part in path.split("/"):
            node.tokens += tokens
            node.files += 1
            walked = f"{walked}/{part}" if walked else part
            node = node.children.setdefault(part, Node(walked))
        node.tokens, node.files = tokens, 1
    return root


def of(repo: Repository) -> Node:
    """The tree of a repository, built once."""
    if "tree" not in repo.cache:
        repo.count()
        repo.cache["tree"] = build_tree({p: f.tokens for p, f in repo.files.items()})
    return repo.cache["tree"]


def find(root: Node, prefix: str) -> Node | None:
    """The node at a directory path such as `src/repo2txt`, or None."""
    node = root
    for part in filter(None, prefix.strip("/").split("/")):
        node = node.children.get(part)
        if node is None:
            return None
    return node


def max_depth(node: Node) -> int:
    return 1 + max((max_depth(c) for c in node.children.values()), default=-1)


def entries(node: Node, depth: int) -> list[Entry]:
    """List what is under `node` down to `depth` levels, directories before files."""
    out: list[Entry] = []

    def walk(current: Node, level: int) -> None:
        for child in sorted(current.children.values(), key=lambda n: (not n.is_dir, n.name)):
            if child.is_dir:
                out.append(Entry(child.path + "/", child.tokens, level, child.files))
                if level < depth:
                    walk(child, level + 1)
            else:
                out.append(Entry(child.path, child.tokens, level))

    walk(node, 1)
    return out


def render(items: list[Entry]) -> str:
    """Indent by depth; show names, not whole paths, as the indentation carries the rest."""
    lines = []
    for e in items:
        name = e.path.rstrip("/").rsplit("/", 1)[-1] + ("/" if e.files is not None else "")
        count = f" ({e.files} files)" if e.files is not None else ""
        lines.append(f"{'  ' * (e.depth - 1)}{name} {e.tokens:,}{count}")
    return "\n".join(lines)

"""Directory tree built from a flat path-to-token mapping."""

from dataclasses import dataclass, field


@dataclass
class TreeNode:
    name: str
    is_dir: bool
    tokens: int = 0
    children: dict[str, "TreeNode"] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        if not self.is_dir:
            return self.tokens
        return sum(c.total_tokens for c in self.children.values())


def build_tree(name: str, token_map: dict[str, int]) -> TreeNode:
    root = TreeNode(name, is_dir=True)
    for path, tokens in token_map.items():
        *dirs, filename = path.split("/")
        node = root
        for part in dirs:
            node = node.children.setdefault(part, TreeNode(part, is_dir=True))
        node.children[filename] = TreeNode(filename, is_dir=False, tokens=tokens)
    return root


def render_tree(root: TreeNode, show_tokens: bool = False) -> str:
    """Render the tree with box-drawing connectors, directories before files."""

    def label(node: TreeNode) -> str:
        text = node.name + ("/" if node.is_dir else "")
        return f"{text} (~{node.total_tokens:,} tokens)" if show_tokens else text

    lines = [label(root)]

    def walk(node: TreeNode, prefix: str) -> None:
        children = sorted(node.children.values(), key=lambda n: (not n.is_dir, n.name))
        for i, child in enumerate(children):
            last = i == len(children) - 1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{label(child)}")
            walk(child, prefix + ("    " if last else "│   "))

    walk(root, "")
    return "\n".join(lines)

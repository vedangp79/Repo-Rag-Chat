from pathlib import Path

DEFAULT_EXTS = {
    ".md", ".txt",
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".sql",
    ".json", ".yaml", ".yml",
    ".html", ".css",
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp",
    ".go", ".rs",
}

def is_probably_binary(text: str) -> bool:
    # quick heuristic: lots of NULs or very low printable ratio
    if "\x00" in text:
        return True
    printable = sum(ch.isprintable() or ch in "\n\r\t" for ch in text)
    return (printable / max(len(text), 1)) < 0.85

def list_files(root_dir: str, exts=DEFAULT_EXTS, max_bytes: int = 200_000):
    root = Path(root_dir)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name

        # Skip dotfiles, AppleDouble, backup/temp
        if name.startswith(".") or name.startswith("._"):
            continue
        if name.endswith("~") or name.endswith(".swp") or name.endswith(".tmp"):
            continue

        # Extension allow-list
        if p.suffix.lower() not in exts:
            continue

        try:
            if p.stat().st_size > max_bytes:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if is_probably_binary(text):
            continue

        rel = str(p.relative_to(root))
        yield rel, text

def repo_tree(root_dir: str, max_depth: int = 4):
    """
    Returns a simple tree string (limited depth to avoid spam).
    """
    root = Path(root_dir)
    lines = [f"{root.name}/"]
    def walk(d: Path, prefix: str, depth: int):
        if depth > max_depth:
            lines.append(prefix + "…")
            return
        entries = sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        for i, e in enumerate(entries):
            last = (i == len(entries) - 1)
            joint = "└─ " if last else "├─ "
            if e.is_dir():
                lines.append(prefix + joint + e.name + "/")
                walk(e, prefix + ("   " if last else "│  "), depth + 1)
            else:
                lines.append(prefix + joint + e.name)
    walk(root, "", 1)
    return "\n".join(lines)

def safe_read(root_dir: str, rel_path: str, max_bytes: int = 200_000) -> str:
    """
    Safely read a file by relative path (prevents path traversal).
    """
    base = Path(root_dir).resolve()
    target = (base / rel_path).resolve()
    if not str(target).startswith(str(base)):
        return "[Security] Invalid path."
    if not target.exists() or not target.is_file():
        return "[Error] File not found."
    if target.stat().st_size > max_bytes:
        return "[Error] File too large to preview."
    try:
        return target.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return f"[Error] Cannot read file: {e}"

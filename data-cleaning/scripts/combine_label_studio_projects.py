#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Script: Combine Label Studio project exports into one raw dataset
#
# What it does:
# - Takes a folder of Label Studio YOLO exports, one subfolder per project
#   (e.g. label_studio_exports/tf_rbb/project_14, project_38, ...), each
#   with images/, labels/ and a class list in classes.txt, notes.json or both.
#   If --src itself contains images/ and labels/, it is treated as one project.
# - Reads each project's class list:
#     * only classes.txt or only notes.json: that file is used;
#     * both: they must list the same classes in the same order, otherwise it
#       stops and names the project (remove or fix the wrong file).
# - Combines the class lists. A Label Studio export only lists the classes used
#   in that project, so projects can have different lists. The combined list
#   holds every class found in any project (e.g. project A: corrosion, spall
#   and project B: corrosion, crack -> corrosion, crack, spall). The order
#   follows the projects' own lists where they set one, and is alphabetical
#   otherwise (Label Studio exports list classes alphabetically, so the result
#   is normally plain alphabetical order). Names that differ only in case,
#   spacing or "_" vs space count as the same class; the first spelling seen
#   is kept, and the report lists the variants.
# - Copies every image and label into <--dest>/images and <--dest>/labels
#   (normally a new folder in data/data_raw/), renumbering each label row's class id to the combined list, so
#   one id means one class across the whole dataset. Labels of projects whose
#   list already matches the combined one are copied unchanged.
# - Writes classes.txt and notes.json for the combined list, so the output
#   looks like a single Label Studio export, and combine_report.txt (also
#   printed): projects found, images and boxes per project, classes per
#   project, the combined class list and any problems.
# - Keeps every class and every image: nothing is cleaned here. Choose the
#   classes to train on afterwards with clean_to_x_classes.py.
# - Checks before writing anything. It stops with an error if:
#     1. no project folder (with images/ and labels/) is found;
#     2. a project has neither classes.txt nor notes.json, or the two disagree;
#     3. the same file name is used in more than one project (the same photo
#        exported from two projects, or two photos with one name), listing
#        them; rename or remove the duplicates in the export first.
#   Label rows that are not plain YOLO box rows (exactly 5 fields: class cx cy
#   w h, class id within the project's list, values 0-1) are skipped and
#   counted in the report rather than stopping the run. Images with no label
#   file are copied without one; label files with no image are skipped.
#
# How to use (paths are relative to the current directory; examples assume you
# are in data-cleaning/, with exports kept in label_studio_exports/ at the repo
# root):
# - Combine the projects of one export into data/data_raw:
#   python scripts/combine_label_studio_projects.py \
#     --src  ../label_studio_exports/tf_rbb \
#     --dest ../data/data_raw/tf_rbb_all
# - Check the export and see the report without writing anything:
#   python scripts/combine_label_studio_projects.py --src ... --dest ... --dry-run
# - Replace an existing output folder:
#   python scripts/combine_label_studio_projects.py --src ... --dest ... --overwrite
# - Next step (choose the classes to keep):
#   python scripts/clean_to_x_classes.py --src ../data/data_raw/tf_rbb_all \
#     --dest ../data/data-5defect/tf_rbb-5defect \
#     --keep corrosion crack spall plant_growth crack_glass
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # Fallback if tqdm isn't installed.
    tqdm = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_EXAMPLES = 10


@dataclass
class Project:
    root: Path
    classes: list[str]
    class_source: str
    images: dict[str, Path] = field(default_factory=dict)  # file stem -> image path
    labels: dict[str, Path] = field(default_factory=dict)  # file stem -> label path
    id_map: dict[int, int] = field(default_factory=dict)   # project class id -> combined id

    @property
    def name(self) -> str:
        return self.root.name


def class_key(name: str) -> str:
    """Matching key: case, spacing and '_' vs space don't matter."""
    return " ".join(name.replace("_", " ").split()).lower()


def natural_key(path: Path):
    """Sort project_9 before project_10."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def find_projects(src: Path) -> list[Path]:
    def is_project(folder: Path) -> bool:
        return (folder / "images").is_dir() and (folder / "labels").is_dir()

    if is_project(src):
        return [src]
    return sorted((p for p in src.iterdir() if p.is_dir() and is_project(p)), key=natural_key)


def read_class_list(project: Path) -> tuple[list[str] | None, str, str | None]:
    """Return (class names in id order, which file(s) they came from, problem or None)."""
    classes_txt = project / "classes.txt"
    notes_json = project / "notes.json"
    from_txt = from_json = None
    if classes_txt.is_file():
        from_txt = [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    if notes_json.is_file():
        try:
            categories = json.loads(notes_json.read_text(encoding="utf-8")).get("categories") or []
            by_id = {int(c["id"]): str(c["name"]).strip() for c in categories}
        except (ValueError, KeyError, TypeError) as error:
            return None, "", f"notes.json could not be read ({error})"
        if sorted(by_id) != list(range(len(by_id))):
            return None, "", "notes.json category ids are not 0, 1, 2, ... without gaps"
        from_json = [by_id[i] for i in range(len(by_id))]

    if from_txt is None and from_json is None:
        return None, "", "has neither classes.txt nor notes.json"
    if from_txt is not None and from_json is not None:
        if from_txt != from_json:
            return None, "", (
                "classes.txt and notes.json disagree:\n"
                f"        classes.txt: {from_txt}\n"
                f"        notes.json : {from_json}"
            )
        return from_txt, "classes.txt + notes.json", None
    if from_txt is not None:
        return from_txt, "classes.txt", None
    return from_json, "notes.json", None


def combine_class_lists(lists: list[list[str]]) -> tuple[list[str], bool]:
    """Union of all class lists, in an order consistent with every list, ties broken
    alphabetically. Returns (combined names, whether the lists' orders conflicted)."""
    spelling: dict[str, str] = {}
    after: dict[str, set[str]] = {}  # key -> keys that must come after it
    for names in lists:
        keys = [class_key(name) for name in names]
        for name, key in zip(names, keys):
            spelling.setdefault(key, name)
            after.setdefault(key, set())
        for earlier, later in zip(keys, keys[1:]):
            after[earlier].add(later)

    # Topological order, ties broken alphabetically.
    incoming = Counter(later for laters in after.values() for later in laters)
    ready = sorted(k for k in after if incoming[k] == 0)
    ordered: list[str] = []
    while ready:
        key = ready.pop(0)
        ordered.append(key)
        for later in after[key]:
            incoming[later] -= 1
            if incoming[later] == 0:
                ready.append(later)
        ready.sort()
    conflict = len(ordered) < len(after)
    if conflict:  # projects disagree on order: fall back to alphabetical
        ordered = sorted(after)
    return [spelling[key] for key in ordered], conflict


def check_label_row(parts: list[str], num_classes: int) -> str | None:
    if len(parts) != 5:
        return f"expected 5 fields (class cx cy w h), found {len(parts)}"
    try:
        class_id = int(parts[0])
    except ValueError:
        return f"class id '{parts[0]}' is not a whole number"
    if not 0 <= class_id < num_classes:
        return f"class id {class_id} is outside the project's class list (0-{num_classes - 1})"
    try:
        values = [float(value) for value in parts[1:]]
    except ValueError:
        return "box values are not all numbers"
    if not all(0.0 <= value <= 1.0 for value in values):
        return "box values must be between 0 and 1"
    return None


def format_table(headers: list[str], rows: list[list]) -> list[str]:
    cells = [headers] + [[f"{v:,}" if isinstance(v, int) else str(v) for v in row] for row in rows]
    widths = [max(len(row[col]) for row in cells) for col in range(len(headers))]
    return [
        "  " + row[0].ljust(widths[0]) + "  " + "  ".join(cell.rjust(widths[i + 1]) for i, cell in enumerate(row[1:]))
        for row in cells
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine Label Studio project exports (one subfolder per project) into one raw dataset "
        "with one class list covering every project."
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Export folder holding one subfolder per project (each with images/, labels/ and "
        "classes.txt and/or notes.json), e.g. ../label_studio_exports/tf_rbb.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Output folder for the combined dataset, normally in data/data_raw, e.g. ../data/data_raw/tf_rbb_all.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace --dest if it already exists and is not empty.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run every check and print the report without writing anything."
    )
    args = parser.parse_args()

    src = args.src.resolve()
    if not src.is_dir():
        raise SystemExit(f"--src folder not found: {src}")
    dest = args.dest.resolve()
    if dest == src or dest in src.parents or src in dest.parents:
        raise SystemExit(f"--dest ({dest}) overlaps with --src ({src}); choose another folder.")
    if dest.exists() and not dest.is_dir():
        raise SystemExit(f"--dest exists and is not a folder: {dest}")
    if dest.exists() and any(dest.iterdir()) and not args.overwrite and not args.dry_run:
        raise SystemExit(f"--dest already exists and is not empty: {dest}\nPass --overwrite to replace it.")

    # ---- Find projects and read their class lists ----
    project_dirs = find_projects(src)
    if not project_dirs:
        raise SystemExit(f"No project folders found in {src} (a project folder contains images/ and labels/).")

    projects: list[Project] = []
    problems: list[str] = []
    for root in project_dirs:
        classes, source, problem = read_class_list(root)
        if problem:
            problems.append(f"  {root.name}: {problem}")
            continue
        keys = [class_key(name) for name in classes]
        duplicated = sorted({name for name, key in zip(classes, keys) if keys.count(key) > 1})
        if duplicated:
            problems.append(f"  {root.name}: its class list names the same class twice: {duplicated}")
            continue
        project = Project(root=root, classes=classes, class_source=source)
        project.images = {p.stem: p for p in sorted((root / "images").iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS}
        project.labels = {p.stem: p for p in sorted((root / "labels").glob("*.txt"))}
        projects.append(project)
    if problems:
        raise SystemExit("ERROR: problems with the projects' class lists:\n" + "\n".join(problems))

    # ---- Same file name in more than one project ----
    owners: dict[str, list[str]] = {}
    for project in projects:
        for stem in sorted(set(project.images) | set(project.labels)):
            owners.setdefault(stem, []).append(project.name)
    clashes = {stem: names for stem, names in owners.items() if len(names) > 1}
    if clashes:
        lines = [f"ERROR: {len(clashes)} file name{'s are' if len(clashes) != 1 else ' is'} used in more than one project:"]
        lines += [f"  {stem}: {', '.join(names)}" for stem, names in list(clashes.items())[:MAX_EXAMPLES]]
        if len(clashes) > MAX_EXAMPLES:
            lines.append(f"  ... and {len(clashes) - MAX_EXAMPLES} more")
        lines.append("Rename or remove the duplicates in the export, then run again.")
        raise SystemExit("\n".join(lines))

    # ---- Combined class list and id mapping ----
    combined, order_conflict = combine_class_lists([project.classes for project in projects])
    combined_id = {class_key(name): index for index, name in enumerate(combined)}
    for project in projects:
        project.id_map = {old: combined_id[class_key(name)] for old, name in enumerate(project.classes)}

    # ---- Read, check and (unless dry run) write ----
    if not args.dry_run:
        if dest.exists():
            shutil.rmtree(dest)
        (dest / "images").mkdir(parents=True)
        (dest / "labels").mkdir(parents=True)

    stats = {p.name: Counter() for p in projects}
    boxes = {p.name: Counter() for p in projects}
    bad_rows: list[str] = []
    jobs = [(project, stem) for project in projects for stem in sorted(set(project.images) | set(project.labels))]
    iterable = tqdm(jobs, desc="Combining", unit="file") if tqdm is not None else jobs
    for project, stem in iterable:
        image, label = project.images.get(stem), project.labels.get(stem)
        if image is None:
            stats[project.name]["labels without image (skipped)"] += 1
            continue
        stats[project.name]["images"] += 1
        new_rows = []
        if label is None:
            stats[project.name]["images without label"] += 1
        else:
            for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
                parts = line.split()
                if not parts:
                    continue
                problem = check_label_row(parts, len(project.classes))
                if problem:
                    bad_rows.append(f"  {project.name}/labels/{label.name}:{line_no}: {problem}")
                    stats[project.name]["invalid rows (skipped)"] += 1
                    continue
                new_id = project.id_map[int(parts[0])]
                boxes[project.name][new_id] += 1
                new_rows.append(" ".join([str(new_id), *parts[1:]]))
        if not args.dry_run:
            shutil.copy2(image, dest / "images" / image.name)
            if label is not None:
                (dest / "labels" / f"{stem}.txt").write_text(
                    "\n".join(new_rows) + ("\n" if new_rows else ""), encoding="utf-8"
                )

    # ---- Report ----
    report = [
        f"Report from : combine_label_studio_projects.py, {datetime.now():%Y-%m-%d %H:%M}",
        f"Source      : {src}",
        f"Output      : {dest}",
        f"Projects    : {len(projects)}",
        "",
        "Combined class list (id: name):",
        *[f"  {i}: {name}" for i, name in enumerate(combined)],
    ]
    if order_conflict:
        report.append(
            "  Note: projects list their classes in conflicting orders, so the combined list is alphabetical."
        )
    differing = [p for p in projects if [class_key(n) for n in p.classes] != [class_key(n) for n in combined]]
    report += ["", f"Projects whose class list differs from the combined list: {len(differing)}"]
    for project in differing:
        missing = [name for name in combined if class_key(name) not in {class_key(n) for n in project.classes}]
        detail = f"ids renumbered; lacks {', '.join(missing)}" if missing else "same classes in a different order; ids renumbered"
        report.append(f"  {project.name} ({len(project.classes)} classes, from {project.class_source}): {detail}")
    spellings = {}
    for project in projects:
        for name in project.classes:
            spellings.setdefault(class_key(name), set()).add(name)
    variants = {key: names for key, names in spellings.items() if len(names) > 1}
    if variants:
        report.append("")
        report.append("Class names spelled differently across projects (treated as one class):")
        report += [f"  {sorted(names)}" for names in variants.values()]

    report += ["", "Per project (boxes per combined class id):"]
    rows = []
    for project in projects:
        s = stats[project.name]
        rows.append([
            project.name, s["images"], sum(boxes[project.name].values()),
            ", ".join(f"{i}:{n}" for i, n in sorted(boxes[project.name].items())) or "-",
        ])
    total_boxes = Counter()
    for counter in boxes.values():
        total_boxes.update(counter)
    rows.append(["total", sum(stats[p.name]["images"] for p in projects), sum(total_boxes.values()), ""])
    report += format_table(["project", "images", "boxes", "boxes by class id"], rows)

    report += ["", "Boxes per class (combined):"]
    report += [f"  {i}: {name:<28} {total_boxes[i]:>8,}" for i, name in enumerate(combined)]

    notes = []
    for project in projects:
        for what, count in sorted(stats[project.name].items()):
            if what != "images" and count:
                notes.append(f"  {project.name}: {what}: {count}")
    report += ["", "Problems found:" if notes or bad_rows else "Problems found: none"]
    report += notes
    if bad_rows:
        report.append("  Invalid label rows (first examples):")
        report += ["  " + row for row in bad_rows[:MAX_EXAMPLES]]
        if len(bad_rows) > MAX_EXAMPLES:
            report.append(f"    ... and {len(bad_rows) - MAX_EXAMPLES} more")

    print()
    print("\n".join(report))
    print()
    if args.dry_run:
        print("Dry run: all checks passed; nothing was written.")
        return

    (dest / "classes.txt").write_text("\n".join(combined) + "\n", encoding="utf-8")
    notes_json = {
        "categories": [{"id": i, "name": name} for i, name in enumerate(combined)],
        "info": {
            "description": f"Combined from {len(projects)} Label Studio project exports by combine_label_studio_projects.py",
            "source": str(src),
            "projects": [p.name for p in projects],
            "date": f"{datetime.now():%Y-%m-%d}",
        },
    }
    (dest / "notes.json").write_text(json.dumps(notes_json, indent=2) + "\n", encoding="utf-8")
    (dest / "combine_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Combined dataset written to: {dest}")


if __name__ == "__main__":
    main()

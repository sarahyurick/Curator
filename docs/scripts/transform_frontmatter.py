#!/usr/bin/env python3
"""
Transform frontmatter from legacy schema to v2 schema.

Legacy schema:
- categories → topics
- personas → content.audience
- difficulty → content.difficulty
- content_type → content.type
- modality → facets.modality

Usage:
    python transform_frontmatter.py [--dry-run] [path...]
    
Examples:
    python transform_frontmatter.py --dry-run ../*.md
    python transform_frontmatter.py ../about/index.md
    python transform_frontmatter.py ../**/*.md
"""

import argparse
import re
import sys
from pathlib import Path

import yaml


# Persona normalization map
PERSONA_MAP = {
    "data-scientist-focused": "Data Scientist",
    "mle-focused": "Machine Learning Engineer",
    "admin-focused": "Cluster Administrator",
    "devops-focused": "DevOps Professional",
    # Pass through already-normalized values
    "Data Scientists": "Data Scientist",
    "Machine Learning Engineers": "Machine Learning Engineer",
    "Cluster Administrators": "Cluster Administrator",
    "DevOps Professionals": "DevOps Professional",
}

# Content type normalization map
CONTENT_TYPE_MAP = {
    "tutorial": "Tutorial",
    "concept": "Concept",
    "reference": "Reference",
    "workflow": "Workflow",
    "index": "Index",
    "example": "Example",
    "troubleshooting": "Troubleshooting",
    "get-started": "Get Started",
    "getting-started": "Get Started",
}

# Difficulty normalization map
DIFFICULTY_MAP = {
    "beginner": "Beginner",
    "intermediate": "Intermediate",
    "advanced": "Advanced",
    "reference": "Intermediate",  # Reference isn't a difficulty level
}


def extract_frontmatter(content: str) -> tuple[dict | None, str, str]:
    """
    Extract YAML frontmatter from markdown content.
    
    Handles both proper format and corrupted format (missing newline after ---).
    
    Returns:
        Tuple of (frontmatter_dict, frontmatter_raw, body)
        frontmatter_dict is None if no valid frontmatter found
    """
    if not content.startswith("---"):
        return None, "", content
    
    # Find the closing --- (with or without trailing newline)
    # IMPORTANT: Check corrupted formats FIRST because \n---\n might match
    # a horizontal rule inside the content body
    
    # Try corrupted format: \n---( (no newline after ---)
    end_match = re.search(r'\n---(?=\()', content[3:])
    if end_match:
        end_pos = end_match.start() + 3
        frontmatter_raw = content[4:end_pos]
        body = content[end_pos + 4:]  # Skip \n--- but keep the (
    else:
        # Try corrupted format with immediate content: \n---#
        end_match = re.search(r'\n---(?=[#\w])', content[3:])
        if end_match:
            end_pos = end_match.start() + 3
            frontmatter_raw = content[4:end_pos]
            body = content[end_pos + 4:]
        else:
            # Try proper format: \n---\n (but only the FIRST occurrence)
            end_match = re.search(r'\n---\n', content[3:])
            if end_match:
                end_pos = end_match.start() + 3  # Account for initial ---
                frontmatter_raw = content[4:end_pos]  # Skip opening ---\n
                body = content[end_pos + 5:]  # Skip \n---\n
            else:
                return None, "", content
    
    try:
        frontmatter = yaml.safe_load(frontmatter_raw)
        return frontmatter, frontmatter_raw, body
    except yaml.YAMLError as e:
        print(f"  Warning: YAML parse error: {e}")
        return None, frontmatter_raw, body


def transform_frontmatter(fm: dict) -> dict:
    """
    Transform frontmatter from legacy to v2 schema.
    
    Args:
        fm: Legacy frontmatter dictionary
        
    Returns:
        Transformed v2 frontmatter dictionary
    """
    new_fm = {}
    
    # === Description (unchanged) ===
    if "description" in fm:
        new_fm["description"] = fm["description"]
    
    # === Topics (from categories) ===
    if "topics" in fm:
        new_fm["topics"] = fm["topics"]
    elif "categories" in fm:
        categories = fm["categories"]
        if isinstance(categories, list):
            new_fm["topics"] = categories
        else:
            new_fm["topics"] = [categories]
    
    # === Tags (unchanged) ===
    if "tags" in fm:
        new_fm["tags"] = fm["tags"]
    
    # === Content classification ===
    content = {}
    
    # content.type
    if "content" in fm and isinstance(fm["content"], dict) and "type" in fm["content"]:
        content["type"] = fm["content"]["type"]
    elif "content_type" in fm:
        ct = fm["content_type"]
        content["type"] = CONTENT_TYPE_MAP.get(ct.lower(), ct.title())
    
    # content.difficulty
    if "content" in fm and isinstance(fm["content"], dict) and "difficulty" in fm["content"]:
        content["difficulty"] = fm["content"]["difficulty"]
    elif "difficulty" in fm:
        diff = fm["difficulty"]
        content["difficulty"] = DIFFICULTY_MAP.get(diff.lower(), diff.title())
    
    # content.audience
    if "content" in fm and isinstance(fm["content"], dict) and "audience" in fm["content"]:
        content["audience"] = fm["content"]["audience"]
    elif "personas" in fm:
        personas = fm["personas"]
        if isinstance(personas, list):
            content["audience"] = [PERSONA_MAP.get(p, p) for p in personas]
        else:
            content["audience"] = [PERSONA_MAP.get(personas, personas)]
    
    if content:
        new_fm["content"] = content
    
    # === Facets ===
    facets = {}
    
    if "facets" in fm and isinstance(fm["facets"], dict):
        facets = fm["facets"].copy()
    elif "modality" in fm:
        facets["modality"] = fm["modality"]
    
    if facets:
        new_fm["facets"] = facets
    
    # === Optional fields (pass through if present) ===
    if "status" in fm:
        new_fm["status"] = fm["status"]
    
    if "dates" in fm:
        new_fm["dates"] = fm["dates"]
    
    if "social" in fm:
        new_fm["social"] = fm["social"]
    
    if "only" in fm:
        new_fm["only"] = fm["only"]
    
    # === Title (only if explicitly set, not derived from H1) ===
    if "title" in fm and isinstance(fm["title"], dict):
        # Only keep title.page and title.social if they add value
        title = {}
        if "page" in fm["title"]:
            title["page"] = fm["title"]["page"]
        if "social" in fm["title"]:
            title["social"] = fm["title"]["social"]
        if title:
            new_fm["title"] = title
    
    return new_fm


def format_yaml_value(value, indent=0):
    """Format a YAML value with proper indentation."""
    prefix = "  " * indent
    
    if isinstance(value, list):
        if all(isinstance(v, str) and len(v) < 40 for v in value):
            # Short list - inline format
            return value
        # Long list - multiline format
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value


def format_list_inline(items: list) -> str:
    """Format a list as inline YAML array: ["item1", "item2"]."""
    formatted = []
    for item in items:
        if isinstance(item, str):
            # Quote strings that need it
            if any(c in item for c in [",", "[", "]", ":", "#"]):
                formatted.append(f'"{item}"')
            else:
                formatted.append(item)
        else:
            formatted.append(str(item))
    return "[" + ", ".join(formatted) + "]"


def frontmatter_to_yaml(fm: dict) -> str:
    """
    Convert frontmatter dict to YAML string with consistent formatting.
    
    Uses inline arrays for compactness: ["item1", "item2"]
    Outputs fields in a consistent order for readability.
    """
    # Define field order
    field_order = [
        "title",
        "description", 
        "social",
        "topics",
        "tags",
        "content",
        "facets",
        "status",
        "dates",
        "only",
    ]
    
    # Build ordered dict
    ordered = {}
    for field in field_order:
        if field in fm:
            ordered[field] = fm[field]
    
    # Add any remaining fields not in order
    for field in fm:
        if field not in ordered:
            ordered[field] = fm[field]
    
    # Custom YAML formatting with inline arrays
    lines = []
    for key, value in ordered.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, list):
                    # Inline array format
                    lines.append(f"  {subkey}: {format_list_inline(subvalue)}")
                else:
                    lines.append(f"  {subkey}: {subvalue}")
        elif isinstance(value, list):
            # Inline array format
            lines.append(f"{key}: {format_list_inline(value)}")
        else:
            # Handle multiline strings
            if isinstance(value, str) and ("\n" in value or len(value) > 80):
                lines.append(f'{key}: "{value}"')
            else:
                lines.append(f"{key}: {json_safe_str(value)}")
    
    return "\n".join(lines)


def json_safe_str(value):
    """Format a value safely for YAML output."""
    if isinstance(value, str):
        # Quote if contains special characters
        if any(c in value for c in [":", "#", "[", "]", "{", "}", ",", "&", "*", "!", "|", ">", "'", '"', "%", "@", "`"]):
            return f'"{value}"'
        # Quote if starts with special characters
        if value and value[0] in ["-", "?", ":", " "]:
            return f'"{value}"'
        return value
    return value


def process_file(filepath: Path, dry_run: bool = False, force: bool = False) -> bool:
    """
    Process a single markdown file.
    
    Returns:
        True if file was modified (or would be in dry-run), False otherwise
    """
    print(f"\nProcessing: {filepath}")
    
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  Error reading file: {e}")
        return False
    
    frontmatter, raw, body = extract_frontmatter(content)
    
    if frontmatter is None:
        print("  No frontmatter found, skipping")
        return False
    
    # Check if already v2 schema
    is_v2 = (
        "content" in frontmatter and isinstance(frontmatter.get("content"), dict) and
        ("type" in frontmatter.get("content", {}) or "difficulty" in frontmatter.get("content", {}))
    )
    
    if is_v2 and not force and "categories" not in frontmatter and "personas" not in frontmatter and "content_type" not in frontmatter:
        print("  Already v2 schema, skipping (use --force to reformat)")
        return False
    
    # Transform
    new_frontmatter = transform_frontmatter(frontmatter)
    
    # Skip if transformation results in empty frontmatter
    if not new_frontmatter:
        print("  Empty frontmatter after transformation, skipping")
        return False
    
    # Generate new YAML
    new_yaml = frontmatter_to_yaml(new_frontmatter)
    new_content = f"---\n{new_yaml}\n---\n{body}"
    
    if dry_run:
        print("  [DRY RUN] Would transform to:")
        print("  ---")
        for line in new_yaml.split("\n"):
            print(f"  {line}")
        print("  ---")
        return True
    
    # Write back
    try:
        filepath.write_text(new_content, encoding="utf-8")
        print("  ✅ Transformed successfully")
        return True
    except Exception as e:
        print(f"  Error writing file: {e}")
        return False


def find_markdown_files(paths: list[str]) -> list[Path]:
    """Find all markdown files from given paths (supports globs)."""
    files = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == ".md":
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob("*.md"))
        elif "*" in path_str:
            # Handle glob patterns
            base = Path(path_str).parent
            pattern = Path(path_str).name
            if base.exists():
                files.extend(base.glob(pattern))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(
        description="Transform frontmatter from legacy to v2 schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Files or directories to process (default: current directory)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be changed without modifying files"
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["_build", "_extensions", "apidocs"],
        help="Directories to exclude (default: _build, _extensions, apidocs)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-process files that are already v2 schema (to update formatting)"
    )
    
    args = parser.parse_args()
    
    # Find files
    files = find_markdown_files(args.paths)
    
    # Filter excluded directories
    files = [
        f for f in files 
        if not any(excl in str(f) for excl in args.exclude)
    ]
    
    if not files:
        print("No markdown files found")
        return 1
    
    print(f"Found {len(files)} markdown files")
    if args.dry_run:
        print("[DRY RUN MODE - no files will be modified]")
    
    # Process files
    modified = 0
    for filepath in files:
        if process_file(filepath, args.dry_run, args.force):
            modified += 1
    
    print(f"\n{'Would modify' if args.dry_run else 'Modified'}: {modified}/{len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())


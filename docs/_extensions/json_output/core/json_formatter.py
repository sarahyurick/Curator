"""JSON data formatting and structure building."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util import logging

from docs._extensions.json_output.utils import get_document_url, get_setting

from .document_discovery import DocumentDiscovery

if TYPE_CHECKING:
    from .builder import JSONOutputBuilder

logger = logging.getLogger(__name__)


def get_nested_value(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely get a nested value from a dictionary.

    Args:
        data: Dictionary to traverse
        *keys: Sequence of keys to traverse
        default: Default value if not found

    Returns:
        Value at the nested path or default
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def normalize_metadata_for_json(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize metadata from legacy schema to v2 schema for JSON output.

    Supports both legacy flat fields and new nested structure.

    Args:
        metadata: Raw frontmatter metadata

    Returns:
        Normalized metadata dict
    """
    normalized = {}

    # === Title ===
    # v2: title.page, title.nav, title.social
    title_data = metadata.get("title")
    if isinstance(title_data, dict):
        normalized["title"] = title_data
    elif title_data:
        normalized["title"] = {"page": title_data}

    # === Description ===
    if metadata.get("description"):
        normalized["description"] = metadata["description"]

    # === Social metadata ===
    social_data = metadata.get("social")
    if isinstance(social_data, dict):
        normalized["social"] = social_data

    # === Tags ===
    if metadata.get("tags"):
        tags = metadata["tags"]
        normalized["tags"] = tags if isinstance(tags, list) else [tags]

    # === Topics (v2) / Categories (legacy) ===
    topics = metadata.get("topics") or metadata.get("categories")
    if topics:
        normalized["topics"] = topics if isinstance(topics, list) else [topics]

    # === Content classification ===
    content = {}

    # content.type or content_type
    content_type = get_nested_value(metadata, "content", "type") or metadata.get("content_type")
    if content_type:
        content["type"] = content_type

    # content.difficulty or difficulty
    difficulty = get_nested_value(metadata, "content", "difficulty") or metadata.get("difficulty")
    if difficulty:
        content["difficulty"] = difficulty

    # content.audience or personas
    audience = get_nested_value(metadata, "content", "audience")
    if not audience and metadata.get("personas"):
        # Normalize legacy personas to human-readable audience
        personas = metadata["personas"]
        if isinstance(personas, list):
            audience_map = {
                "data-scientist-focused": "Data Scientist",
                "mle-focused": "Machine Learning Engineer",
                "admin-focused": "Cluster Administrator",
                "devops-focused": "DevOps Professional",
            }
            audience = [audience_map.get(p, p) for p in personas]
        else:
            audience = [personas]

    if audience:
        content["audience"] = audience if isinstance(audience, list) else [audience]

    if content:
        normalized["content"] = content

    # === Facets (v2) / modality (legacy) ===
    facets = {}
    modality = get_nested_value(metadata, "facets", "modality") or metadata.get("modality")
    if modality:
        facets["modality"] = modality

    if facets:
        normalized["facets"] = facets

    # === Dates ===
    dates = metadata.get("dates")
    if isinstance(dates, dict):
        normalized["dates"] = dates

    # === Status ===
    if metadata.get("status"):
        normalized["status"] = metadata["status"]

    # === Only (content gating) ===
    if metadata.get("only"):
        normalized["only"] = metadata["only"]

    # === Cascade/product info (pass through) ===
    if metadata.get("cascade"):
        normalized["cascade"] = metadata["cascade"]

    return normalized


class JSONFormatter:
    """Handles JSON data structure building and formatting."""

    def __init__(self, app: Sphinx, json_builder: "JSONOutputBuilder"):
        self.app = app
        self.env = app.env
        self.config = app.config
        self.json_builder = json_builder
        
        # Cache product/book metadata from conf.py
        self._product_metadata = self._extract_product_metadata()
    
    def _extract_product_metadata(self) -> dict[str, Any]:
        """Extract product and book metadata from Sphinx config."""
        metadata = {}
        
        # Book info from standard Sphinx config
        if hasattr(self.config, "project"):
            metadata["book"] = {
                "title": self.config.project,
                "version": getattr(self.config, "release", ""),
            }
        
        # Product info - try multiple sources
        product_name = None
        product_family = None
        
        # Check html_context (common pattern)
        html_context = getattr(self.config, "html_context", {})
        if isinstance(html_context, dict):
            product_name = html_context.get("product_name")
            product_family = html_context.get("product_family")
        
        # Check myst_substitutions (NeMo Curator pattern)
        myst_subs = getattr(self.config, "myst_substitutions", {})
        if isinstance(myst_subs, dict):
            if not product_name:
                product_name = myst_subs.get("product_name_short") or myst_subs.get("product_name")
            if not product_family:
                # Derive from product_name if contains family indicator
                full_name = myst_subs.get("product_name", "")
                if "NeMo" in full_name:
                    product_family = ["NeMo"]
        
        # Fallback: extract from project name
        if not product_name and hasattr(self.config, "project"):
            project = self.config.project
            # Extract product name from patterns like "NeMo-Curator" or "NVIDIA NeMo Curator"
            if "-" in project:
                product_name = project.split("-")[-1]
            elif " " in project:
                parts = project.replace("NVIDIA", "").strip().split()
                product_name = parts[-1] if parts else project
            else:
                product_name = project
        
        if product_name:
            metadata["product"] = {
                "name": product_name,
            }
            if product_family:
                metadata["product"]["family"] = product_family if isinstance(product_family, list) else [product_family]
        
        # Site info
        site_name = html_context.get("site_name") if isinstance(html_context, dict) else None
        if site_name:
            metadata["site"] = {"name": site_name}
        
        return metadata
    
    def _get_parent_path(self, docname: str) -> list[str]:
        """Get parent document path(s) as an array."""
        parents = []
        
        if "/" in docname:
            parts = docname.split("/")
            # Build parent paths from root to immediate parent
            for i in range(len(parts) - 1):
                if i == 0:
                    parent_path = parts[0] + "/index"
                else:
                    parent_path = "/".join(parts[:i+1]) + "/index"
                
                # Check if parent exists
                if parent_path in self.env.all_docs or parts[i] == "index":
                    parents.append(parent_path if parts[i] != "index" else "/".join(parts[:i]) + "/index")
            
            # Also add root index as top-level parent
            if docname != "index" and "index" in self.env.all_docs:
                if "index" not in parents:
                    parents.insert(0, "index")
        elif docname != "index":
            # Top-level doc, parent is main index
            if "index" in self.env.all_docs:
                parents.append("index")
        
        return parents

    def add_metadata_fields(self, data: dict[str, Any], metadata: dict[str, Any]) -> None:
        """Add all metadata fields to JSON data structure using v2 schema."""
        # Normalize metadata to v2 schema
        normalized = normalize_metadata_for_json(metadata)

        # === Core fields ===
        if normalized.get("description"):
            data["description"] = normalized["description"]

        if normalized.get("tags"):
            data["tags"] = normalized["tags"]

        if normalized.get("topics"):
            data["topics"] = normalized["topics"]

        # === Content classification (v2 nested structure) ===
        content = normalized.get("content", {})
        if content:
            data["content"] = content

        # === Facets (v2 nested structure) ===
        facets = normalized.get("facets", {})
        if facets:
            data["facets"] = facets

        # === Social metadata ===
        social = normalized.get("social", {})
        if social:
            data["social"] = social

        # === Dates ===
        dates = normalized.get("dates", {})
        if dates:
            data["dates"] = dates

        # === Status ===
        if normalized.get("status"):
            data["status"] = normalized["status"]

        # === Content gating ===
        if normalized.get("only"):
            data["only"] = normalized["only"]

        # === Legacy field passthrough for backward compatibility ===
        # Keep these for any consumers that expect flat fields
        if metadata.get("author"):
            data["author"] = metadata["author"]

    def build_child_json_data(self, docname: str, include_content: bool | None = None) -> dict[str, Any]:
        """Build optimized JSON data for child documents (LLM/search focused)."""
        if include_content is None:
            include_content = get_setting(self.config, "include_child_content", True)

        # Get document title
        title = self.env.titles.get(docname, nodes.title()).astext() if docname in self.env.titles else ""

        # Extract metadata for tags/categories
        metadata = self.json_builder.extract_document_metadata(docname)
        content_data = self.json_builder.extract_document_content(docname) if include_content else {}

        # Build optimized data structure for search engines
        data = {
            "id": docname,  # Use 'id' for search engines
            "title": title,
            "url": get_document_url(self.app, docname),
        }
        
        # Add parent references as array
        parents = self._get_parent_path(docname)
        if parents:
            data["parent"] = parents

        # Add metadata fields
        self.add_metadata_fields(data, metadata)
        
        # Add product/book metadata from conf.py
        if self._product_metadata:
            data.update(self._product_metadata)

        # Add search-specific fields
        if include_content:
            self._add_content_fields(data, content_data, docname, title)

        return data

    def build_json_data(self, docname: str) -> dict[str, Any]:
        """Build optimized JSON data structure for LLM/search use cases."""
        # Get document title
        title = self.env.titles.get(docname, nodes.title()).astext() if docname in self.env.titles else ""

        # Extract metadata and content
        metadata = self.json_builder.extract_document_metadata(docname)
        content_data = self.json_builder.extract_document_content(docname)

        # Build data structure
        data = {
            "id": docname,
            "title": title,
            "url": get_document_url(self.app, docname),
            "last_modified": datetime.now(timezone.utc).isoformat(),
        }
        
        # Add parent references as array
        parents = self._get_parent_path(docname)
        if parents:
            data["parent"] = parents

        # Add metadata fields
        self.add_metadata_fields(data, metadata)
        
        # Add product/book metadata from conf.py
        if self._product_metadata:
            data.update(self._product_metadata)

        # Add markdown content (preserve content classification object if it exists)
        if content_data.get("content"):
            # If content is already an object (classification metadata), preserve it and add markdown separately
            if isinstance(data.get("content"), dict):
                # Content classification already set, add markdown content as separate field
                data["content_text"] = content_data["content"]
                data["format"] = content_data.get("format", "text")
            else:
                # No content classification, use content for markdown
                data["content"] = content_data["content"]
                data["format"] = content_data.get("format", "text")

        if content_data.get("summary"):
            data["summary"] = content_data["summary"]

        if content_data.get("headings"):
            data["headings"] = [{"text": h["text"], "level": h["level"]} for h in content_data["headings"]]

        return data

    def _add_content_fields(
        self, data: dict[str, Any], content_data: dict[str, Any], docname: str, title: str
    ) -> None:
        """Add content-related fields to JSON data."""
        self._add_primary_content(data, content_data)
        self._add_summary_content(data, content_data)
        self._add_headings_content(data, content_data)
        self._add_optional_features(data, content_data)
        self._add_document_metadata(data, content_data, docname, title)

    def _add_primary_content(self, data: dict[str, Any], content_data: dict[str, Any]) -> None:
        """Add primary content with length limits."""
        if not content_data.get("content"):
            return

        content_max_length = get_setting(self.config, "content_max_length", 50000)
        content = content_data["content"]

        if content_max_length > 0 and len(content) > content_max_length:
            content = content[:content_max_length] + "..."

        data["content"] = content
        data["format"] = content_data.get("format", "text")
        data["content_length"] = len(content_data["content"])  # Original length
        data["word_count"] = len(content_data["content"].split()) if content_data["content"] else 0

    def _add_summary_content(self, data: dict[str, Any], content_data: dict[str, Any]) -> None:
        """Add summary with length limits."""
        if not content_data.get("summary"):
            return

        summary_max_length = get_setting(self.config, "summary_max_length", 500)
        summary = content_data["summary"]

        if summary_max_length > 0 and len(summary) > summary_max_length:
            summary = summary[:summary_max_length] + "..."

        data["summary"] = summary

    def _add_headings_content(self, data: dict[str, Any], content_data: dict[str, Any]) -> None:
        """Add headings for structure/navigation."""
        if not content_data.get("headings"):
            return

        # Simplify headings for LLM use
        data["headings"] = [
            {"text": h["text"], "level": h["level"], "id": h.get("id", "")} for h in content_data["headings"]
        ]
        # Add searchable heading text
        data["headings_text"] = " ".join([h["text"] for h in content_data["headings"]])

    def _add_optional_features(self, data: dict[str, Any], content_data: dict[str, Any]) -> None:
        """Add optional search enhancement features."""
        if get_setting(self.config, "extract_keywords", True) and "keywords" in content_data:
            keywords_max_count = get_setting(self.config, "keywords_max_count", 50)
            keywords = (
                content_data["keywords"][:keywords_max_count] if keywords_max_count > 0 else content_data["keywords"]
            )
            data["keywords"] = keywords

        if get_setting(self.config, "extract_code_blocks", True) and "code_blocks" in content_data:
            data["code_blocks"] = content_data["code_blocks"]

        if get_setting(self.config, "extract_links", True) and "links" in content_data:
            data["links"] = content_data["links"]

        if get_setting(self.config, "extract_images", True) and "images" in content_data:
            data["images"] = content_data["images"]

    def _add_document_metadata(
        self, data: dict[str, Any], content_data: dict[str, Any], docname: str, title: str
    ) -> None:
        """Add document type and section metadata."""
        if get_setting(self.config, "include_doc_type", True):
            discovery = DocumentDiscovery(self.app, self.json_builder)
            data["doc_type"] = discovery.detect_document_type(docname, title, content_data.get("content", ""))

        if get_setting(self.config, "include_section_path", True):
            discovery = DocumentDiscovery(self.app, self.json_builder)
            data["section_path"] = discovery.get_section_path(docname)

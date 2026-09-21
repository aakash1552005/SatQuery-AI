"""
SatQuery AI -- Query Intent Parser
Section 14: Query Contract & Natural Language Intent Parser
"""

from __future__ import annotations

import re
from typing import Optional
from src.contracts.query_contracts import QueryIntent, TaskType


class QueryParser:
    """
    Analyzes natural language queries to extract structured intent,
    task categorization, required inputs, and target entities.
    """

    TEMPORAL_KEYWORDS = [
        "changed", "change", "changes", "difference", "differences",
        "between", "expansion", "growth", "shrinkage", "deforestation",
        "construction", "before and after", "over time", "timeline",
        "temporal", "evolution", "trend"
    ]

    GROUNDING_KEYWORDS = [
        "where is", "where are", "locate", "highlight", "bounding box",
        "bbox", "find the", "pinpoint", "show where", "box the",
        "coordinates of", "exact location"
    ]

    FUSION_KEYWORDS = [
        "fuse", "fusion", "both sensors", "both observations",
        "optical and sar", "sar and optical", "radar and optical",
        "combine", "cross-modal", "multimodal"
    ]

    SAR_KEYWORDS = [
        "radar", "sar", "backscatter", "dielectric", "surface roughness",
        "polarization", "vv", "vh", "hh", "hv", "speckle",
        "microwave", "penetration", "double bounce"
    ]

    MEASUREMENT_KEYWORDS = [
        "area", "square meters", "hectares", "sq km", "percentage",
        "percent", "how much", "quantify", "measure", "extent"
    ]

    TARGET_ENTITY_PATTERNS = [
        r"\b(water\s*body|lake|river|reservoir|ocean|pond)\b",
        r"\b(forest|trees|vegetation|greenery|canopy)\b",
        r"\b(urban|building|buildings|city|runway|airport|road|infrastructure)\b",
        r"\b(agriculture|cropland|farmland|crop|field)\b",
        r"\b(bare\s*soil|sand|desert|rock)\b",
        r"\b(flood|flooding|inundation)\b",
        r"\b(burn\s*scar|fire|deforestation)\b",
    ]

    def parse(self, query: str) -> QueryIntent:
        """
        Parse raw natural language query into a structured QueryIntent.
        """
        q_lower = query.lower().strip()
        entities = self._extract_entities(q_lower)
        measurement_required = any(m in q_lower for m in self.MEASUREMENT_KEYWORDS)

        # 1. Bi-temporal change query check
        if any(w in q_lower for w in self.TEMPORAL_KEYWORDS):
            return QueryIntent(
                raw_query=query,
                task_type=TaskType.TEMPORAL_CHANGE,
                requires_temporal_pair=True,
                requires_optical_sar=False,
                requires_spatial_evidence=True,
                requested_outputs=["change_mask", "measurement", "explanation"],
                measurement_required=measurement_required,
                target_entities=entities,
                confidence=0.95,
            )

        # 2. Optical-SAR fusion query check
        if any(w in q_lower for w in self.FUSION_KEYWORDS):
            return QueryIntent(
                raw_query=query,
                task_type=TaskType.OPTICAL_SAR_ANALYSIS,
                requires_temporal_pair=False,
                requires_optical_sar=True,
                requires_spatial_evidence=True,
                requested_outputs=["optical_evidence", "sar_evidence", "fused_mask", "agreement_tier"],
                measurement_required=measurement_required,
                target_entities=entities,
                confidence=0.92,
            )

        # 3. Grounding query check
        if any(w in q_lower for w in self.GROUNDING_KEYWORDS):
            return QueryIntent(
                raw_query=query,
                task_type=TaskType.SINGLE_IMAGE_GROUNDING,
                requires_temporal_pair=False,
                requires_optical_sar=False,
                requires_spatial_evidence=True,
                requested_outputs=["bbox", "highlight_overlay", "confidence"],
                measurement_required=measurement_required,
                target_entities=entities,
                confidence=0.90,
            )

        # 4. Explicit SAR query check
        if any(w in q_lower for w in self.SAR_KEYWORDS):
            return QueryIntent(
                raw_query=query,
                task_type=TaskType.SINGLE_IMAGE_VQA_SAR,
                requires_temporal_pair=False,
                requires_optical_sar=False,
                requires_spatial_evidence=True,
                requested_outputs=["backscatter_stats", "feature_analysis", "structured_answer"],
                measurement_required=measurement_required,
                target_entities=entities,
                confidence=0.92,
            )

        # 5. Caption / Describe check
        if any(w in q_lower for w in ["caption", "describe the scene", "summarize image", "overview"]):
            return QueryIntent(
                raw_query=query,
                task_type=TaskType.SINGLE_IMAGE_CAPTION,
                requires_temporal_pair=False,
                requires_optical_sar=False,
                requires_spatial_evidence=False,
                requested_outputs=["caption_text", "key_attributes"],
                measurement_required=False,
                target_entities=entities,
                confidence=0.88,
            )

        # 6. Default to Optical VQA (will be re-evaluated by router depending on input sensor modality)
        return QueryIntent(
            raw_query=query,
            task_type=TaskType.SINGLE_IMAGE_VQA_OPTICAL,
            requires_temporal_pair=False,
            requires_optical_sar=False,
            requires_spatial_evidence=False,
            requested_outputs=["answer", "confidence"],
            measurement_required=measurement_required,
            target_entities=entities,
            confidence=0.85,
        )

    def _extract_entities(self, query_lower: str) -> list[str]:
        """Extract geospatial target entities mentioned in the query."""
        found = []
        for pattern in self.TARGET_ENTITY_PATTERNS:
            matches = re.findall(pattern, query_lower)
            for m in matches:
                clean = m.strip()
                if clean not in found:
                    found.append(clean)
        return found

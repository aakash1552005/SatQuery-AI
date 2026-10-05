"""
SatQuery AI -- Evidence Store
Section 19: Structured Evidence Store for Multimodal Remote Sensing Claims.

Stores every spatial, spectral, and numerical claim as a verifiable contract:
{
  "claim": "A water body is present",
  "source": "optical_sar_fusion",
  "evidence_type": "region",
  "bbox": [ymin, xmin, ymax, xmax],
  "confidence": {"evidence": "high", "fusion_agreement": "both_agree"},
  "inputs": ["sentinel2_01", "sentinel1_01"],
  "analysis_source": "deterministic"
}
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field


class EvidenceType(str, enum.Enum):
    """Supported evidence types per Section 19."""
    REGION = "region"
    BBOX = "bbox"
    MASK = "mask"
    MEASUREMENT = "measurement"
    METADATA = "metadata"
    MODEL_OUTPUT = "model_output"
    TEMPORAL_DIFFERENCE = "temporal_difference"


class EvidenceItem(BaseModel):
    """
    A single factual, verifiable evidence record supporting a system claim.
    """
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    claim: str
    source: str
    evidence_type: EvidenceType
    bbox: Optional[list[float]] = Field(
        None,
        description="Geographic [min_lon, min_lat, max_lon, max_lat] or pixel [ymin, xmin, ymax, xmax]"
    )
    confidence: dict = Field(default_factory=dict)
    inputs: list[str] = Field(default_factory=list)
    analysis_source: str = "deterministic"
    properties: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_geojson_feature(self) -> dict:
        """Convert evidence item to an RFC 7946 GeoJSON Feature."""
        geometry = None
        if self.bbox and len(self.bbox) == 4:
            min_x, min_y, max_x, max_y = self.bbox
            geometry = {
                "type": "Polygon",
                "coordinates": [[
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y],
                    [min_x, min_y],
                ]]
            }

        return {
            "type": "Feature",
            "id": self.evidence_id,
            "geometry": geometry,
            "properties": {
                "claim": self.claim,
                "source": self.source,
                "evidence_type": self.evidence_type.value,
                "confidence": self.confidence,
                "inputs": self.inputs,
                "analysis_source": self.analysis_source,
                "timestamp": self.timestamp,
                **self.properties,
            }
        }


class EvidenceStore:
    """
    Central repository tracking all evidence supporting claims in SatQuery AI.
    """

    def __init__(self):
        self._items: dict[str, EvidenceItem] = {}

    def add(
        self,
        claim: str,
        source: str,
        evidence_type: EvidenceType | str,
        confidence: dict,
        inputs: list[str],
        bbox: Optional[list[float]] = None,
        analysis_source: str = "deterministic",
        properties: Optional[dict[str, Any]] = None,
    ) -> EvidenceItem:
        """Create, store, and return a new EvidenceItem."""
        if isinstance(evidence_type, str):
            evidence_type = EvidenceType(evidence_type)

        item = EvidenceItem(
            claim=claim,
            source=source,
            evidence_type=evidence_type,
            bbox=bbox,
            confidence=confidence,
            inputs=inputs,
            analysis_source=analysis_source,
            properties=properties or {},
        )
        self._items[item.evidence_id] = item
        return item

    def get(self, evidence_id: str) -> Optional[EvidenceItem]:
        return self._items.get(evidence_id)

    def get_by_input(self, file_id: str) -> list[EvidenceItem]:
        """Retrieve all evidence associated with a specific input file."""
        return [item for item in self._items.values() if file_id in item.inputs]

    def get_all(self) -> list[EvidenceItem]:
        return list(self._items.values())

    def to_geojson(self) -> dict:
        """Export all spatial evidence as an RFC 7946 GeoJSON FeatureCollection."""
        features = [
            item.to_geojson_feature()
            for item in self._items.values()
            if item.bbox is not None
        ]
        return {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "features": features,
        }

    def clear(self):
        self._items.clear()

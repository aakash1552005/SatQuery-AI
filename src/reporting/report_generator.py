"""
SatQuery AI -- Field Pack & Intelligence Report Generator
Section 26: Air-Gapped Field Pack Export, RFC 7946 GeoJSON, and PDF/HTML Briefs.

Generates:
1. evidence.geojson (RFC 7946 compliant GeoJSON vector boundaries)
2. result.json (Complete structured analysis contract)
3. execution_trace.json (Factual tool execution sequence)
4. mission_brief.html (Self-contained, printable mission intelligence report)
5. offline_field_viewer.html (Zero-network standalone interactive viewer for field rescue teams)
6. All packaged into a single air-gapped 1-Click Field Pack (.zip)
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.contracts.query_contracts import AnalysisResult
from src.contracts.raster_contracts import RasterMetadata
from src.verification.evidence_store import EvidenceStore


class FieldPackGenerator:
    """
    Assembles air-gapped intelligence field packs for emergency responders
    operating in zero-connectivity disaster zones.
    """

    @staticmethod
    def generate_geojson(
        result: AnalysisResult,
        input_metadata: list[RasterMetadata],
        evidence_store: Optional[EvidenceStore] = None,
    ) -> dict:
        """Construct RFC 7946 GeoJSON FeatureCollection from analysis evidence and raster bounds."""
        features = []

        # Add raster footprints
        for idx, meta in enumerate(input_metadata):
            if meta.bounds:
                b = meta.bounds
                if isinstance(b, dict):
                    b_left, b_bottom, b_right, b_top = b.get("left", 0.0), b.get("bottom", 0.0), b.get("right", 0.0), b.get("top", 0.0)
                else:
                    b_left = getattr(b, "left", 0.0)
                    b_bottom = getattr(b, "bottom", 0.0)
                    b_right = getattr(b, "right", 0.0)
                    b_top = getattr(b, "top", 0.0)
                features.append({
                    "type": "Feature",
                    "id": f"raster_footprint_{idx + 1}",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [b_left, b_bottom],
                            [b_right, b_bottom],
                            [b_right, b_top],
                            [b_left, b_top],
                            [b_left, b_bottom],
                        ]]
                    },
                    "properties": {
                        "name": meta.filename,
                        "modality": meta.modality.value,
                        "crs": meta.crs,
                        "dimensions": f"{meta.width}x{meta.height}",
                        "feature_class": "raster_bounding_box",
                    }
                })

        # Add spatial items from evidence store if present
        if evidence_store:
            for item in evidence_store.get_all():
                features.append(item.to_geojson_feature())

        return {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "properties": {
                "title": "SatQuery AI Intelligence Vectors",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "task": result.task.value,
                "mechanism": result.mechanism,
                "status": result.status,
            },
            "features": features,
        }

    @staticmethod
    def generate_html_brief(
        query: str,
        result: AnalysisResult,
        trace_data: dict,
        input_metadata: list[RasterMetadata],
    ) -> str:
        """Generate a self-contained, publication-ready HTML Mission Intelligence Brief."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        inputs_html = "".join(
            f"<li><b>{m.filename}</b>: {m.modality.value.upper()} ({m.width}x{m.height}, CRS={m.crs})</li>"
            for m in input_metadata
        )
        evidence_html = "".join(f"<li>{e}</li>" for e in result.evidence)
        limitations_html = "".join(f"<li>{l}</li>" for l in result.limitations)

        trace_steps_html = "".join(
            f"<tr><td>{s.get('name')}</td><td>{s.get('status')}</td><td>{s.get('details', '')}</td></tr>"
            for s in trace_data.get("steps", [])
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SatQuery AI &mdash; Mission Intelligence Brief</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.5; color: #1e293b; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
    h1 {{ color: #0f172a; border-bottom: 2px solid #0284c7; padding-bottom: 8px; margin-bottom: 4px; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; background: #e0f2fe; color: #0369a1; }}
    .badge-ready {{ background: #dcfce7; color: #15803d; }}
    .meta-box {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 14px; margin: 16px 0; font-size: 13px; }}
    .answer-box {{ background: #f0fdf4; border-left: 4px solid #16a34a; padding: 14px; margin: 16px 0; font-size: 15px; font-weight: 500; color: #14532d; }}
    table {{ width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; }}
    th {{ background: #f1f5f9; }}
    .warning {{ color: #b45309; font-size: 13px; }}
</style>
</head>
<body>
    <h1>SATQUERY AI &mdash; MISSION INTELLIGENCE BRIEF</h1>
    <p style="color: #64748b; font-size: 12px;">Generated at: {timestamp} | Classification: OPERATIONAL_UNCLASSIFIED</p>

    <div class="meta-box">
        <div><b>User Natural Language Query:</b> "{query}"</div>
        <div style="margin-top: 4px;"><b>Task Classification:</b> <span class="badge">{result.task.value.upper()}</span> | <b>Execution Mechanism:</b> <code>{result.mechanism}</code></div>
        <div style="margin-top: 4px;"><b>Verification Status:</b> <span class="badge badge-ready">{result.status}</span></div>
    </div>

    <h3>Verified Analytical Answer</h3>
    <div class="answer-box">
        {result.answer}
    </div>

    <h3>Input Satellite Observations</h3>
    <ul>
        {inputs_html}
    </ul>

    <h3>Deterministic Physical Evidence</h3>
    <ul>
        {evidence_html}
    </ul>

    <h3>Physical Constraints & Scientific Limitations</h3>
    <ul class="warning">
        {limitations_html}
    </ul>

    <h3>Factual Execution Trace Telemetry</h3>
    <table>
        <thead>
            <tr><th>Step Name</th><th>Status</th><th>Details</th></tr>
        </thead>
        <tbody>
            {trace_steps_html}
        </tbody>
    </table>

    <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 30px 0;">
    <p style="font-size: 11px; color: #94a3b8; text-align: center;">SatQuery AI (SatSense) &mdash; Sovereign Remote Sensing Intelligence &mdash; ISRO / SAC SIH PS 26167</p>
</body>
</html>"""

    @staticmethod
    def generate_offline_viewer(geojson_dict: dict, answer_text: str) -> str:
        """
        Generate a 100% air-gapped, zero-dependency HTML/SVG map viewer
        that renders GeoJSON geometries locally without requiring CDN or internet access.
        """
        features_json = json.dumps(geojson_dict)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SatQuery AI &mdash; Offline Field Viewer</title>
<style>
    body {{ margin: 0; padding: 20px; font-family: -apple-system, sans-serif; background: #0a0e17; color: #f1f5f9; }}
    #header {{ margin-bottom: 16px; border-bottom: 1px solid #1e293b; padding-bottom: 12px; }}
    #mapCanvas {{ width: 100%; height: 500px; background: #111827; border: 1px solid #00d4ff; border-radius: 8px; }}
    .badge {{ padding: 4px 8px; border-radius: 4px; background: #00d4ff; color: #0a0e17; font-weight: bold; font-size: 12px; }}
    .legend {{ margin-top: 12px; display: flex; gap: 16px; font-size: 13px; }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; }}
    .box {{ width: 12px; height: 12px; border-radius: 2px; }}
</style>
</head>
<body>
    <div id="header">
        <h2>SatQuery AI &mdash; Air-Gapped Offline Field Viewer <span class="badge">100% OFFLINE</span></h2>
        <p style="color: #94a3b8; font-size: 13px;">{answer_text}</p>
        <div class="legend">
            <div class="legend-item"><span class="box" style="background: rgba(0, 212, 255, 0.4); border: 1px solid #00d4ff;"></span> Satellite Observation Footprint</div>
            <div class="legend-item"><span class="box" style="background: rgba(245, 158, 11, 0.6); border: 1px solid #f59e0b;"></span> Delineated Area of Interest</div>
        </div>
    </div>

    <svg id="mapCanvas" viewBox="0 0 800 500"></svg>

    <script>
    const data = {features_json};
    const svg = document.getElementById('mapCanvas');

    // Simple robust local SVG projection for offline field laptop viewing
    if (data.features && data.features.length) {{
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        data.features.forEach(f => {{
            if (f.geometry && f.geometry.coordinates) {{
                f.geometry.coordinates[0].forEach(pt => {{
                    minX = Math.min(minX, pt[0]);
                    maxX = Math.max(maxX, pt[0]);
                    minY = Math.min(minY, pt[1]);
                    maxY = Math.max(maxY, pt[1]);
                }});
            }}
        }});

        const padX = (maxX - minX) * 0.1 || 0.01;
        const padY = (maxY - minY) * 0.1 || 0.01;
        minX -= padX; maxX += padX;
        minY -= padY; maxY += padY;

        function toSvgX(lon) {{ return ((lon - minX) / (maxX - minX)) * 760 + 20; }}
        function toSvgY(lat) {{ return 480 - ((lat - minY) / (maxY - minY)) * 440; }}

        data.features.forEach((f, idx) => {{
            if (f.geometry && f.geometry.coordinates) {{
                const pts = f.geometry.coordinates[0].map(pt => `${{toSvgX(pt[0])}},${{toSvgY(pt[1])}}`).join(' ');
                const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                poly.setAttribute('points', pts);
                poly.setAttribute('fill', idx === 0 ? 'rgba(0, 212, 255, 0.15)' : 'rgba(245, 158, 11, 0.25)');
                poly.setAttribute('stroke', idx === 0 ? '#00d4ff' : '#f59e0b');
                poly.setAttribute('stroke-width', '2');
                svg.appendChild(poly);
            }}
        }});
    }}
    </script>
</body>
</html>"""

    @classmethod
    def create_field_pack_bytes(
        cls,
        query: str,
        result: AnalysisResult,
        trace_data: dict,
        input_metadata: list[RasterMetadata],
        evidence_store: Optional[EvidenceStore] = None,
    ) -> bytes:
        """Package all reports, GeoJSON, trace, and offline viewer into an in-memory ZIP archive."""
        geojson_dict = cls.generate_geojson(result, input_metadata, evidence_store)
        html_brief = cls.generate_html_brief(query, result, trace_data, input_metadata)
        offline_viewer = cls.generate_offline_viewer(geojson_dict, result.answer or "")

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # 1. Evidence GeoJSON
            zip_file.writestr("evidence.geojson", json.dumps(geojson_dict, indent=2))
            # 2. Result Contract JSON
            zip_file.writestr("result.json", result.model_dump_json(indent=2))
            # 3. Execution Trace JSON
            zip_file.writestr("execution_trace.json", json.dumps(trace_data, indent=2))
            # 4. Printable Intelligence Brief HTML
            zip_file.writestr("mission_intelligence_brief.html", html_brief)
            # 5. Offline Field Viewer HTML
            zip_file.writestr("offline_field_viewer.html", offline_viewer)
            # 6. Readme
            zip_file.writestr(
                "README_FIELD_PACK.txt",
                "SATQUERY AI (SATSENSE) -- AIR-GAPPED DISASTER RESPONSE FIELD PACK\n"
                "=================================================================\n"
                "Contains deterministic geospatial intelligence vectors, mission brief,\n"
                "and offline HTML viewer for field teams operating with zero connectivity.\n"
            )

        buffer.seek(0)
        return buffer.getvalue()

/**
 * SatQuery AI — Cloudflare Edge Worker & Static Assets Gateway
 * Provides edge telemetry (/api/status, /api/health, /api/capabilities)
 * and serves static assets from app/frontend.
 */

const CAPABILITIES = {
  upload: "READY",
  metadata_inspection: "READY",
  compatibility_check: "READY",
  single_image_vqa_optical: "READY",
  single_image_vqa_sar: "READY",
  single_image_grounding: "READY",
  temporal_change: "READY",
  optical_sar_fusion: "READY",
  verification_engine: "READY",
  field_pack_generator: "READY",
  geochat: "UNAVAILABLE",
  changechat: "BLOCKED_LICENSE",
  croma: "DISABLED",
  rs_adaptation: "READY"
};

const CAPABILITY_DETAILS = {
  single_image_vqa_optical: {
    name: "Single-Image Optical VQA",
    status: "READY",
    primary_engine: "Qwen2-VL-7B-Instruct (4-bit NF4)",
    fallback_engine: "Rule-based Spectral Classifier",
    description: "Visual question answering on optical/multispectral satellite imagery."
  },
  single_image_vqa_sar: {
    name: "Single-Image SAR VQA",
    status: "READY",
    primary_engine: "Deterministic SAR Physics Engine",
    description: "Radar backscatter analysis, Refined Lee speckle filtering, Otsu water extraction."
  },
  single_image_grounding: {
    name: "Visual Grounding / Localization",
    status: "READY",
    primary_engine: "Normalized Difference Spectral Reticle",
    description: "Spatial coordinate extraction and bounding box detection for target features."
  },
  temporal_change: {
    name: "Bi-Temporal Change Detection",
    status: "READY",
    primary_engine: "Deterministic Change Engine (L1 + Index Difference)",
    description: "Change magnitude calculation, physical gating, and changed-area mapping."
  },
  optical_sar_fusion: {
    name: "Optical-SAR Sensor Fusion",
    status: "READY",
    primary_engine: "Cross-Modal Consensus Engine",
    description: "Optical land cover cross-verified with SAR radar backscatter agreement matrix."
  },
  verification_engine: {
    name: "Evidence Verifier & Numerical Guard",
    status: "READY",
    primary_engine: "EvidenceVerifier + NumericalGuard",
    description: "Zero-tolerance token verification locking text claims strictly to GIS calculations."
  },
  field_pack_generator: {
    name: "Air-Gapped Field Pack Exporter",
    status: "READY",
    primary_engine: "FieldPackGenerator",
    description: "Generates offline HTML field dossier, RFC 7946 GeoJSON, and cryptographic checksums."
  }
};

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "*",
  "Content-Type": "application/json"
};

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Handle CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    // Telemetry & Status endpoint
    if (url.pathname === "/api/status" || url.pathname === "/api/health") {
      return new Response(
        JSON.stringify({
          status: "running",
          service: "satquery-ai",
          version: "1.0.0",
          runtime_mode: "EDGE_TERMINAL",
          capabilities: CAPABILITIES
        }),
        { headers: CORS_HEADERS }
      );
    }

    // Capability Registry endpoint
    if (url.pathname === "/api/capabilities") {
      return new Response(
        JSON.stringify({
          runtime_mode: "EDGE_TERMINAL",
          capabilities: CAPABILITY_DETAILS
        }),
        { headers: CORS_HEADERS }
      );
    }

    // Pass through to Cloudflare Static Assets
    if (env.ASSETS) {
      return env.ASSETS.fetch(request);
    }

    return new Response("SatQuery AI Edge Terminal", {
      status: 200,
      headers: { "Content-Type": "text/plain" }
    });
  }
};

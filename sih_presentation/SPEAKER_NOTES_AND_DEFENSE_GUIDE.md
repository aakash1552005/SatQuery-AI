# SatQuery AI — SIH 2026 Speaker Script & Jury Defense Guide
**Problem Statement ID**: 26167 | **Theme**: Space Technology | **Organization**: ISRO / Department of Space (SAC)  
**Team**: SatSense (ID: 148995) | **Institution**: Vel Tech R&D Institute  
**Framework**: Dr. Arvind A. R’s *Hackathon & Project Preparation Guide*

---

## Part 1: The Winning 3-Minute Live Pitch Script

*Rule from Dr. Arvind's Guide*: **Prepare like an engineer. Present like a storyteller. Prove like a researcher. Think like a jury.**

```
[00:00 - 00:30] SLIDE 1 & SLIDE 2: THE EMOTIONAL HOOK & PROBLEM
[00:30 - 01:15] SLIDE 3: SYSTEM ARCHITECTURE & 6-STEP LIFECYCLE
[01:15 - 01:45] SLIDE 4: FEASIBILITY, TCO & AIR-GAPPED DEFENSE
[01:45 - 02:20] SLIDE 5: QUANTIFIED IMPACT & ASSAM BENCHMARK
[02:20 - 03:00] SLIDE 6: STRATEGIC ROADMAP, STANDARDS & JURY ASK
```

---

### [00:00 – 00:30] Slide 1 & 2: Opening Hook & Problem Reality

> *"Good morning, esteemed jury members from ISRO and SAC. When catastrophic monsoon floods submerge Assam, emergency commanders on rescue boats need a fast, certified answer to a single question: **'How much land is underwater right now, and which evacuation routes are cut off?'**
>
> But today, that assessment takes **4 hours** of manual GIS work. Worse, optical satellites like Cartosat and Sentinel-2 are **78% blind** because monsoon storm clouds cover the entire valley. And regular AI chatbots cannot read spatial coordinates, hallucinate areas, and cannot be trusted with human lives.
>
> We built **SatQuery AI**: an interactive vision-language assistant that gives disaster officers instant, certified ground measurements through plain English queries with **zero AI guesswork**."*

---

### [00:30 – 01:15] Slide 3: Technical Approach & The 6-Step Lifecycle

> *"Here is how we ensure 100% scientific truth. SatQuery AI is a **Careful Coordinator**, not an unchecked black box. We enforce an uncompromising 6-step lifecycle:
>
> 1. **Understand**: Pydantic v2 parses the natural query into a typed contract with strict intent, sensor, and date constraints.
> 2. **Check**: Our Data Readiness Gate validates coordinate reference systems, affine transforms, GSD, and spatial overlap. If data doesn't align, we trigger an **Honest Refusal** rather than hallucinating.
> 3. **Choose**: Our deterministic router directs tasks strictly to certified algorithms—no free-form LLM tool hallucinations.
> 4. **Analyze**: Optical and SAR data are quarantined. For radar, we execute Lee speckle filtering and Otsu thresholding in linear power space. Our **Elevation-Gated Radar** incorporates DEM slope checks to eliminate mountain shadow false alarms.
> 5. **Verify**: We enforce a **Deterministic Math Guard**: flooded area is computed directly from GeoTIFF affine transform matrices.
> 6. **Explain**: We output interactive agreement maps, QGIS-compliant RFC 7946 GeoJSON, and an audit trace.
>
> Notice the feedback loop: disaster commanders can interactively refine queries or zoom into specific critical bridges in real time."*

---

### [01:15 – 01:45] Slide 4: Feasibility, TCO & Air-Gapped Security

> *"Is this feasible in the real world? Yes, along three dimensions:
>
> - **Economic Feasibility**: SatQuery AI is built entirely on open-source FOSS tools—FastAPI, Rasterio, GDAL, and PyTorch. We eliminate proprietary GIS licenses like ArcGIS Server, **saving State Disaster Management Authorities ₹15 to 20 Lakhs every year**.
> - **Compute Feasibility**: We engineered an emergency **CPU fallback mode**. If high-end GPU clusters lose power during a cyclone, core flood mapping still runs on a standard laptop in minutes.
> - **Sovereign Security**: Our containerized architecture is **100% air-gapped**. It runs locally on-premise, guaranteeing that sensitive ISRO and defence satellite feeds never touch external cloud servers."*

---

### [01:45 – 02:20] Slide 5: Quantified Impact & The Assam Benchmark

> *"Let’s look at the numbers. In our Assam flood validation benchmark:
>
> - **Analysis Time**: Dropped from **4 hours to under 3 minutes**—a **95% reduction in response latency**.
> - **Cloud Resilience**: Optical sensors were blind under 78% cloud cover. SatQuery AI's C-band SAR fusion achieved **100% cloud penetration**, revealing **11.2 square kilometers of hidden floodwaters** that optical satellites completely missed.
> - **Accuracy**: We measured **18.40 square kilometers (1,840 hectares)** within **1% of certified GIS ground truth**.
> - **Tactical Accessibility**: We package results into **Offline Tactical Field Packs under 10 MB**, allowing NDRF rescue teams to view maps and vectors on field tablets with zero internet connectivity.
>
> This directly advances **UN SDG 9 (Infrastructure)**, **SDG 11 (Safe Cities)**, and **SDG 13 (Climate Action)**."*

---

### [02:20 – 03:00] Slide 6: Strategic Roadmap & Jury Closing Ask

> *"Where are we today, and where are we going?
>
> - **NOW**: Our working prototype has **99 passing automated test cases**, proven on real BigEarthNet Sentinel-1 and Sentinel-2 pairs with zero synthetic data pollution.
> - **NEXT (3–6 Months)**: Native integration with **ISRO Bhuvan STAC API** and pilot deployment with the Assam and Odisha State Disaster Management Authorities.
> - **THEN (1 Year)**: Dual-frequency L+S band support for the upcoming **NASA-ISRO SAR (NISAR)** satellite and edge deployment on NDRF command vehicles.
> - **FUTURE**: The core engine for a **Pan-India National Automated Disaster Intelligence Engine (NADIE)**.
>
> Esteemed judges: SatQuery AI bridges the critical gap between raw ISRO space pixels and instant, lifesaving decisions. We are scientifically defensible, reproducible, and ready for deployment.
>
> Thank you, and we welcome your questions."*

---

## Part 2: 10 Lethal Jury Questions & Bulletproof Answers

### Q1: "LLMs are notorious for hallucinating numbers. How can ISRO trust your flood area calculations?"
> **Answer**: *"We never let the LLM do arithmetic or generate coordinates. The LLM only parses the user's intent into a typed Pydantic schema. Once parsed, the flood mask is extracted deterministically using Otsu thresholding on Lee-filtered SAR backscatter, and the flooded area is calculated using pure linear algebra: `Area = N_pixels * GSD_x * GSD_y / 1,000,000` via GeoTIFF affine transform matrices. The LLM merely reads and formats the certified number."*

### Q2: "SAR penetrates clouds, but in hilly terrain like Assam and Northeast India, radar shadow mimics water. How do you prevent massive false alarms?"
> **Answer**: *"That is exactly why we created **Elevation-Gated Radar**. Specular water reflection and radar mountain shadows both yield low backscatter (<-18 dB). SatQuery AI pulls the corresponding SRTM or CartoDEM digital elevation model and calculates terrain slope and aspect. Any low-backscatter pixel on a slope greater than 5 degrees or on a radar shadow backslope is strictly masked out from the flood mask."*

### Q3: "How do you handle coordinate and projection differences between Cartosat and RISAT/Sentinel-1?"
> **Answer**: *"Step 2 of our pipeline is the **Data Readiness Gate**. We read GeoTIFF CRS metadata using GDAL/Rasterio. If the coordinate systems differ, we automatically reproject both rasters to a shared UTM projection. If the spatial overlap between the two footprints is less than 80%, SatQuery AI halts and triggers an **Honest Refusal**, explaining that the images do not sufficiently intersect to perform valid cross-modal analysis."*

### Q4: "What happens if a field officer asks a question outside your system's capability, like predicting next week's flood?"
> **Answer**: *"Our router enforces a **Hard Refusal Gate**. If a query asks for predictive hydrologic forecasting or requires images not present in the workspace, the system halts with a structured message explaining the scientific limitation: 'Only historical or current acquisitions are available; hydrological flood prediction requires rainfall gauge integration.' We never guess."*

### Q5: "How does this run in a remote flood shelter or rescue boat with no internet?"
> **Answer**: *"We export **Offline Tactical Field Packs**. This is a compressed ZIP package (<10 MB) containing an interactive, offline Leaflet HTML map, RFC 7946 GeoJSON vectors of the flood perimeter, and a lightweight PDF intelligence dossier. Responders can load it onto any offline tablet or smartphone without a cellular signal."*

### Q6: "Why do you quarantine optical and SAR data instead of feeding both to a single multimodal VLM?"
> **Answer**: *"Optical vision-language models like LLaVA or GeoChat were pre-trained on RGB photographs. SAR backscatter represents geometric surface roughness and dielectric permittivity, not visible light. Feeding raw SAR backscatter directly into an optical vision encoder causes severe hallucination. We treat SAR as a deterministic radar signal, extract calibrated binary masks, and only fuse optical and SAR at the decision and evidence layer."*

### Q7: "Why did you process SAR backscatter in linear power instead of decibels?"
> **Answer**: *"Decibels are logarithmic units ($10 \log_{10} P$). Performing linear arithmetic like spatial averaging, Lee speckle filtering, or polarization ratios on decibel values is a known remote-sensing mathematical violation. In SatQuery AI, all SAR filters and ratios convert dB to linear power ($\sigma^0_{\text{linear}} = 10^{\sigma^0_{\text{dB}}/10}$), perform the math, and only convert back to dB for visualization."*

### Q8: "What is your testing and validation standard? Did you test this on fake data?"
> **Answer**: *"We have 99 automated test cases passing with zero errors. All physics contracts (NDVI zero-denominator guard, Lee filter border handling, dB-to-linear roundtrip, affine coordinate conversion) are unit tested. Our production pipeline runs strictly on real calibrated Sentinel-1 SAR and Sentinel-2 optical GeoTIFF pairs from the official BigEarthNet benchmark. Zero synthetic data is used in production."*

### Q9: "How much would it cost a state government like Assam or Odisha to run this?"
> **Answer**: *"Software licensing cost is **zero**. SatQuery AI is 100% built on open-source FOSS tools, saving ₹15 to 20 Lakhs per year compared to proprietary enterprise GIS licenses like ArcGIS Server. For compute, it can run on existing state data center servers or edge laptops with CPU fallback during emergency power blackouts."*

### Q10: "How will this integrate with ISRO's existing platforms like Bhuvan?"
> **Answer**: *"SatQuery AI outputs all raster and vector products compliant with OGC STAC API v1.0 and RFC 7946 GeoJSON. Any output layer generated by SatQuery AI can be dropped directly into ISRO Bhuvan GeoAI, QGIS, or ArcGIS without format conversion."*

---

## Part 3: Second-by-Second 60-Second Video Storyboard

| Time | Visual on Screen | Spoken Voiceover / Audio |
| :---: | :--- | :--- |
| **00:00 - 00:05** | Satellite zoom-in on flood-submerged Assam under heavy white clouds $\rightarrow$ NDRF emblem $\rightarrow$ 4-hour countdown clock. | *"When floods strike, optical satellites are 78% blind under storm clouds, and manual GIS mapping takes 4 hours disaster teams don't have."* |
| **00:05 - 00:15** | SatQuery AI UI on screen. User types: *"How many hectares are flooded in Nagaon district?"* System instant-parses typed contract. | *"Meet SatQuery AI: an interactive assistant fusing ISRO optical and radar imagery to see through clouds in seconds."* |
| **00:15 - 00:35** | **HERO DEMO**: Leaflet side-by-side swipe viewer. Left = cloudy optical image. Right = radar water mask penetrating the cloud. Affine area pop-up: `18.40 km² (1,840 ha) certified`. | *"Our elevation-gated radar penetrates dense clouds, while our deterministic math guard calculates exact ground area directly from GeoTIFF affine metadata—zero AI hallucination."* |
| **00:35 - 00:45** | User uploads mismatched images. SatQuery displays an amber alert: *"Refusal: CRS mismatch detected. Halting safely."* | *"When data is invalid, SatQuery refuses honestly rather than guessing."* |
| **00:45 - 00:55** | Single-click: *"Export Offline Tactical Field Pack"* $\rightarrow$ PDF dossier and vector GeoJSON downloaded (<10 MB). | *"Analysis time drops from 4 hours to 3 minutes, delivering actionable maps to rescue boats even without internet."* |
| **00:55 - 01:00** | Team SatSense logo + ISRO / SIH 2026 splash screen. Subtitle: *Turning space pixels into lifesaving decisions.* | *"SatQuery AI: Turning raw space pixels into instant field decisions."* |

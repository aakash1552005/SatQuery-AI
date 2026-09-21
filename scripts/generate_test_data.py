"""
Generate synthetic test GeoTIFFs for Day 1 engineering validation.
These are NOT real satellite imagery -- they exist purely to test
the upload -> inspect -> validate pipeline.
"""

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
except ImportError:
    print("ERROR: rasterio is required. Install with: pip install rasterio")
    exit(1)

from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "samples"


def create_optical_sample():
    """3-band RGB optical GeoTIFF."""
    out_dir = OUTPUT_DIR / "optical"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "synthetic_optical_rgb.tif"

    height, width = 256, 256
    # Simulate land cover: green area + water (dark blue) + urban (gray)
    data = np.random.randint(30, 200, (3, height, width), dtype=np.uint8)
    # Water region (low reflectance)
    data[:, 100:150, 50:150] = np.array([10, 20, 60])[:, None, None]
    # Vegetation (high green)
    data[:, 0:80, 0:120] = np.array([30, 140, 40])[:, None, None]

    transform = from_bounds(72.0, 20.0, 72.1, 20.1, width, height)

    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=3,
        dtype="uint8", crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(data)
        dst.update_tags(
            SENSOR="SYNTHETIC_OPTICAL",
            ACQUISITION_DATE="2025-01-15",
        )

    print(f"  Created: {path} (3-band optical, EPSG:4326)")
    return path


def create_multispectral_sample():
    """4-band multispectral GeoTIFF (R, G, B, NIR)."""
    out_dir = OUTPUT_DIR / "optical"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "synthetic_multispectral_4band.tif"

    height, width = 256, 256
    data = np.random.randint(100, 3000, (4, height, width), dtype=np.uint16)
    # High NIR for vegetation
    data[3, 0:100, 0:150] = np.random.randint(3000, 5000, (100, 150))
    # Low NIR for water
    data[3, 150:200, 50:200] = np.random.randint(50, 200, (50, 150))

    transform = from_bounds(77.5, 12.9, 77.6, 13.0, width, height)

    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=4,
        dtype="uint16", crs=CRS.from_epsg(32643),
        transform=transform,
    ) as dst:
        dst.write(data)
        dst.set_band_description(1, "Red")
        dst.set_band_description(2, "Green")
        dst.set_band_description(3, "Blue")
        dst.set_band_description(4, "NIR")
        dst.update_tags(
            SENSOR="SYNTHETIC_MULTISPECTRAL",
            ACQUISITION_DATE="2025-03-20",
        )

    print(f"  Created: {path} (4-band multispectral, EPSG:32643)")
    return path


def create_sar_sample():
    """2-band SAR GeoTIFF (VV + VH polarization)."""
    out_dir = OUTPUT_DIR / "sar"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "synthetic_sar_vv_vh.tif"

    height, width = 256, 256
    # Simulate SAR backscatter in dB (float32, negative values typical)
    vv = np.random.uniform(-25.0, -5.0, (height, width)).astype(np.float32)
    vh = np.random.uniform(-30.0, -10.0, (height, width)).astype(np.float32)
    # Water region (very low backscatter)
    vv[100:180, 60:200] = np.random.uniform(-28.0, -22.0, (80, 140)).astype(np.float32)
    vh[100:180, 60:200] = np.random.uniform(-32.0, -26.0, (80, 140)).astype(np.float32)

    data = np.stack([vv, vh])
    transform = from_bounds(72.0, 20.0, 72.1, 20.1, width, height)

    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=2,
        dtype="float32", crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(data)
        dst.set_band_description(1, "VV")
        dst.set_band_description(2, "VH")
        dst.update_tags(
            SENSOR="SYNTHETIC_SAR",
            PLATFORM="Sentinel-1",
            ACQUISITION_DATE="2025-01-15",
            POLARIZATION="VV+VH",
        )

    print(f"  Created: {path} (2-band SAR VV/VH, EPSG:4326)")
    return path


def create_temporal_pair():
    """Bi-temporal optical pair (same area, different dates)."""
    out_dir = OUTPUT_DIR / "temporal"
    out_dir.mkdir(parents=True, exist_ok=True)

    height, width = 256, 256
    transform = from_bounds(77.5, 12.9, 77.6, 13.0, width, height)

    # T1 -- mostly vegetation
    t1_data = np.random.randint(30, 180, (3, height, width), dtype=np.uint8)
    t1_data[:, 0:200, 0:200] = np.array([35, 130, 40])[:, None, None]

    path_t1 = out_dir / "temporal_t1_2024_jan.tif"
    with rasterio.open(
        path_t1, "w", driver="GTiff",
        height=height, width=width, count=3,
        dtype="uint8", crs=CRS.from_epsg(32643),
        transform=transform,
    ) as dst:
        dst.write(t1_data)
        dst.update_tags(ACQUISITION_DATE="2024-01-10", SENSOR="SYNTHETIC")

    # T2 -- urban development in center
    t2_data = t1_data.copy()
    t2_data[:, 80:160, 80:180] = np.array([160, 155, 150])[:, None, None]

    path_t2 = out_dir / "temporal_t2_2025_jan.tif"
    with rasterio.open(
        path_t2, "w", driver="GTiff",
        height=height, width=width, count=3,
        dtype="uint8", crs=CRS.from_epsg(32643),
        transform=transform,
    ) as dst:
        dst.write(t2_data)
        dst.update_tags(ACQUISITION_DATE="2025-01-10", SENSOR="SYNTHETIC")

    print(f"  Created: {path_t1}")
    print(f"  Created: {path_t2}")
    return path_t1, path_t2


def create_optical_sar_pair():
    """Co-registered optical + SAR pair (same area)."""
    out_dir = OUTPUT_DIR / "optical_sar"
    out_dir.mkdir(parents=True, exist_ok=True)

    height, width = 256, 256
    transform = from_bounds(72.0, 20.0, 72.1, 20.1, width, height)

    # Optical
    opt_data = np.random.randint(40, 180, (3, height, width), dtype=np.uint8)
    opt_data[:, 100:180, 60:200] = np.array([10, 30, 80])[:, None, None]  # water

    path_opt = out_dir / "pair_optical.tif"
    with rasterio.open(
        path_opt, "w", driver="GTiff",
        height=height, width=width, count=3,
        dtype="uint8", crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(opt_data)
        dst.update_tags(SENSOR="SYNTHETIC_OPTICAL", ACQUISITION_DATE="2025-01-15")

    # SAR
    vv = np.random.uniform(-20.0, -5.0, (height, width)).astype(np.float32)
    vh = np.random.uniform(-25.0, -10.0, (height, width)).astype(np.float32)
    vv[100:180, 60:200] = np.random.uniform(-28.0, -22.0, (80, 140)).astype(np.float32)
    vh[100:180, 60:200] = np.random.uniform(-32.0, -26.0, (80, 140)).astype(np.float32)

    path_sar = out_dir / "pair_sar.tif"
    with rasterio.open(
        path_sar, "w", driver="GTiff",
        height=height, width=width, count=2,
        dtype="float32", crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(np.stack([vv, vh]))
        dst.set_band_description(1, "VV")
        dst.set_band_description(2, "VH")
        dst.update_tags(
            SENSOR="SYNTHETIC_SAR", PLATFORM="Sentinel-1",
            ACQUISITION_DATE="2025-01-15", POLARIZATION="VV+VH",
        )

    print(f"  Created: {path_opt}")
    print(f"  Created: {path_sar}")
    return path_opt, path_sar


def main():
    print("SatQuery AI -- Generating synthetic test GeoTIFFs")
    print("=" * 50)
    print("NOTE: These are NOT real satellite imagery.")
    print("      For engineering validation only.")
    print()

    create_optical_sample()
    create_multispectral_sample()
    create_sar_sample()
    create_temporal_pair()
    create_optical_sar_pair()

    print()
    print("Done. All synthetic samples created in data/samples/")


if __name__ == "__main__":
    main()

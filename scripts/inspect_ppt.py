import pptx
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ppt_path = Path("PPT - SatSense .pptx")
prs = pptx.Presentation(str(ppt_path))
print(f"Presentation: {ppt_path.name}")
print(f"Total Slides: {len(prs.slides)}")
print(f"Dimensions: {prs.slide_width.inches:.2f} x {prs.slide_height.inches:.2f} inches (16:9 widescreen)")

for i, slide in enumerate(prs.slides):
    print(f"\n{'='*60}")
    print(f"SLIDE {i+1}")
    print(f"{'='*60}")
    for s_idx, shape in enumerate(slide.shapes):
        if shape.has_text_frame:
            paras = [p.text.strip() for p in shape.text_frame.paragraphs if p.text.strip()]
            text = " \n    ".join(paras)
            if text:
                print(f"  Shape {s_idx} ({shape.name}):\n    {text}")
        elif shape.has_table:
            print(f"  Shape {s_idx} [TABLE] ({shape.table.rows} rows, {shape.table.columns} cols)")
            for r in shape.table.rows:
                row_txt = [c.text.strip() for c in r.cells]
                print(f"    {' | '.join(row_txt)}")
        else:
            print(f"  Shape {s_idx} ({shape.name}) [{shape.shape_type}]")

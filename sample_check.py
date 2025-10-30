import json
from pathlib import Path
import contract_ocr_service as svc
rows = []
for pdf in sorted(Path('sample docs').glob('*.pdf')):
    images = svc.pdf_to_images(pdf)
    pages = [svc.preprocess_pil(img) for img in images]
    texts = []
    header_hint = None
    for idx, img in enumerate(pages):
        t, c, tsv, size = svc.ocr_image(img)
        texts.append(t)
        if idx == 0:
            header_hint = svc.header_text_from_tsv(tsv, size, top_ratio=0.18)
    full_text = '\n\n'.join(texts)
    fields = svc.extract_fields(full_text, header_hint=header_hint)
    rows.append({'file': pdf.name, **fields.model_dump()})
print(json.dumps(rows, indent=2, default=str))

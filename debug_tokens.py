import re
from pathlib import Path
import contract_ocr_service as svc

text = Path('debug/full_raw.txt').read_text(encoding='utf-8')
block = ['SURE BIMICHAEL J. AND NANCY A.', 'Wed §=6BICKETT FAMILY TRUST']
candidate = svc._extend_with_previous_uppercase(text, ' '.join(block))
print('candidate:', candidate)
cleaned = candidate.replace('\u00a0', ' ')
cleaned = re.sub(r"[^A-Za-z0-9&.,' ]+", ' ', cleaned)
print('cleaned tokens:', cleaned.split())

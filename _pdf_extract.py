import fitz
from pathlib import Path
base = Path('C:/Users/lahma/OneDrive/Documents/CVM1')
names = ['IJNTI2505002.pdf',
         'Gutierrez-Perez_No_Bells_Just_Whistles_Sports_Field_Registration_by_Leveraging_Geometric_CVPRW_2024_paper.pdf',
         '2410.07401v1.pdf']
out = Path('papers_text.txt')
with open(out, 'w', encoding='utf-8') as f:
    for name in names:
        doc = fitz.open(str(base / name))
        f.write(f'\n\n======== {name} ({doc.page_count} pages) ========\n\n')
        for i in range(doc.page_count):
            f.write(f'\n--- page {i+1} ---\n')
            f.write(doc[i].get_text())
        doc.close()
print('wrote', out, 'bytes=', out.stat().st_size)

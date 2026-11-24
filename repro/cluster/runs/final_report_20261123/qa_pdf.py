import json
from pathlib import Path
import pymupdf as fitz
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent;PDF=ROOT/'final_draft.pdf';OUT=ROOT/'qa';OUT.mkdir(exist_ok=True)
def main():
 doc=fitz.open(PDF);report=[];images=[]
 for i,page in enumerate(doc):
  pix=page.get_pixmap(matrix=fitz.Matrix(1.4,1.4),alpha=False);p=OUT/f'page_{i+1:02d}.png';pix.save(p);images.append(p)
  text=page.get_text();boxes=[]
  for block in page.get_text('dict')['blocks']:
   if 'lines' not in block:continue
   for line in block['lines']:
    for span in line['spans']:
     x0,y0,x1,y1=span['bbox']
     if x0<0 or y0<0 or x1>page.rect.width+1 or y1>page.rect.height+1:boxes.append({'text':span['text'],'bbox':span['bbox']})
  report.append({'page':i+1,'words':len(text.split()),'first_lines':text.splitlines()[:3],'text_outside_page':boxes})
  assert not boxes,(i+1,boxes)
 for start in range(0,len(images),4):
  sheet=Image.new('RGB',(1000,1350),'#e4e8ed');draw=ImageDraw.Draw(sheet)
  for k,p in enumerate(images[start:start+4]):
   im=Image.open(p);im.thumbnail((480,630));x=(k%2)*500+10;y=(k//2)*675+30;sheet.paste(im,(x,y));draw.text((x,y-20),f'Page {start+k+1}',fill='#172b43')
  sheet.save(OUT/f'contact_{start//4+1:02d}.jpg',quality=92)
 (OUT/'layout_report.json').write_text(json.dumps({'pages':len(doc),'page_checks':report},indent=2));print(json.dumps({'pages':len(doc),'words_per_page':[x['words'] for x in report],'out_of_page_spans':sum(len(x['text_outside_page']) for x in report)}))
if __name__=='__main__':main()

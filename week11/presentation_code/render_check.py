from pathlib import Path
import json
import pymupdf as fitz
from PIL import Image,ImageDraw
D=Path("/home/a/ppt_workspace/ppt_info/RL_week11");O=Path('/home/a/ppt_workspace/output/RL_week11');reports={}
for stem in ['RL_week11','RL_week11_design_report']:
 doc=fitz.open(O/(stem+'.pdf'));out=D/'rendered'/stem;out.mkdir(parents=True,exist_ok=True);thumbs=[];report=[]
 for stale in out.glob('*.png'):stale.unlink()
 for i,page in enumerate(doc):
  page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(out/f'{i+1:02}.png')
  im=Image.open(out/f'{i+1:02}.png');im.thumbnail((480,500 if 'report' in stem else 270));thumbs.append(im.copy())
  outside=[b[:4] for b in page.get_text('blocks') if b[0]<-1 or b[1]<-1 or b[2]>page.rect.width+1 or b[3]>page.rect.height+1]
  report.append(dict(page=i+1,outside=outside,words=len(page.get_text().split())))
 for start in range(0,len(thumbs),6):
  group=thumbs[start:start+6];cellh=(520 if 'report' in stem else 290);sheet=Image.new('RGB',(1440,cellh*((len(group)+2)//3)),'#dddddd');dr=ImageDraw.Draw(sheet)
  for j,im in enumerate(group):sheet.paste(im,(j%3*480,j//3*cellh));dr.text((j%3*480+5,(j//3+1)*cellh-18),str(start+j+1),fill='black')
  sheet.save(out/f'contact_{start//6:02}.png')
 reports[stem]=report
(D/'render_report.json').write_text(json.dumps(reports,indent=2));print({k:dict(pages=len(v),outside=sum(len(r['outside']) for r in v)) for k,v in reports.items()})

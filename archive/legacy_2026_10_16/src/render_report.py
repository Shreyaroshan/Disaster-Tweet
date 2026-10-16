"""Render the Markdown report to PDF; requires reportlab (bundled document runtime)."""
from pathlib import Path
import re, html
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
ROOT=Path(__file__).resolve().parents[1]
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=9.5,leading=14,spaceAfter=7))
styles.add(ParagraphStyle(name='TableCustom',fontName='Helvetica',fontSize=7.2,leading=10))
styles['Title'].fontName='Helvetica-Bold'; styles['Title'].fontSize=20; styles['Title'].leading=25; styles['Title'].alignment=TA_LEFT; styles['Title'].spaceAfter=16
styles['Heading2'].fontSize=13; styles['Heading2'].leading=17; styles['Heading2'].spaceBefore=12; styles['Heading2'].spaceAfter=7
styles['Heading3'].fontSize=11; styles['Heading3'].leading=14

def fmt(s):
    s=s.replace('–','-').replace('—',' - ').replace('→',' to ').replace('≤','<=').replace('’',"'")
    s=html.escape(s.replace('`',''))
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    return s

def para(s,style='BodyCustom'): return Paragraph(fmt(s),styles[style])
width=A4[0]-88
story=[]; lines=(ROOT/'reports/interim_report.md').read_text().splitlines(); i=0
while i<len(lines):
    line=lines[i].strip()
    if not line: i+=1; continue
    if line.startswith('|'):
        data=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[-: ]+',c) for c in cells): data.append([para(c,'TableCustom') for c in cells])
            i+=1
        cols=len(data[0]); widths=([105,85,88,108,width-386] if cols==5 else [width/cols]*cols)
        if cols==8: widths=[40,49,48,43,42,67,62,width-351]
        table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7edf2')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#c8d1da')),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        story.extend([KeepTogether([table]) if len(data)<=7 else table,Spacer(1,10)]); continue
    if line.startswith('!['):
        match=re.match(r'!\[(.*?)\]\((.*?)\)',line)
        img=Image(str(ROOT/'reports'/match[2])); scale=min(width/img.imageWidth,230/img.imageHeight)
        img.drawWidth=img.imageWidth*scale; img.drawHeight=img.imageHeight*scale
        story.append(KeepTogether([img,Spacer(1,10)])); i+=1; continue
    if line.startswith('# '): story.append(para(line[2:],'Title'))
    elif line.startswith('## '):
        if line == '## Individual contribution statement': story.append(PageBreak())
        story.append(para(line[3:],'Heading2'))
    elif line.startswith('### '): story.append(para(line[4:],'Heading3'))
    elif re.match(r'^\d+\. ',line): story.append(para(line))
    else:
        block=line
        while i+1<len(lines) and lines[i+1].strip() and not lines[i+1].startswith(('#','|','![')):
            i+=1; block+=' '+lines[i].strip()
        story.append(para(block))
    i+=1

def footer(canvas,doc):
    canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#596673'))
    canvas.drawString(44,25,'ICT 4442 | Disaster tweet classification | Interim')
    canvas.drawRightString(A4[0]-44,25,str(doc.page))
path=ROOT/'reports/interim_report.pdf'
SimpleDocTemplate(str(path),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=42,bottomMargin=44,title='Disaster tweet classification interim report',author='Project team; prepared with Codex assistance').build(story,onFirstPage=footer,onLaterPages=footer)
print(path)

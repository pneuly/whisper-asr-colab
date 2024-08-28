import logging
from pathlib import Path
from docx import Document
from typing import Optional
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_LINE_SPACING, WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Mm, RGBColor

DEFAULT_FONT_SIZE = 12
SERIF_FONT = "ＭＳ 明朝"
SANS_FONT = "ＭＳ ゴシック"

def set_rFonts(style, key, value):
    style._element.rPr.rFonts.set(qn(f'w:{key}'), value)

def create_attribute(element, name, value):
    element.set(qn(name), value)


class DocxGenerator():
    def __init__(self, doc: Optional[Document] = None):
        # Create a new Word document
        self.doc = doc if doc else Document()
        self.txtfilename = ""
        self.docfilename = ""
        self.styles = {
            "normal" : self.doc.styles['Normal'],
            "speaker" : self.doc.styles["Heading 1"],
            "ts1" : self.doc.styles.add_style('FirstParagraph', WD_STYLE_TYPE.PARAGRAPH),
            "ts2" : self.doc.styles.add_style('SecondParagraph', WD_STYLE_TYPE.PARAGRAPH),
        }

        self.init_document()
        self.add_page_number()


    def init_document(self):
        sec = self.doc.sections[-1]
        sec.page_height = Mm(297)
        sec.page_width = Mm(210)
        sec.top_margin = Mm(20)
        sec.bottom_margin = Mm(15)
        sec.left_margin = Mm(25)
        sec.right_margin = Mm(25)
        sec.footer_distance = Mm(8)

        style_normal = self.styles['normal']
        style_normal.font.name = SERIF_FONT
        style_normal.font.name_eastasia = SERIF_FONT
        style_normal.font.size = Pt(DEFAULT_FONT_SIZE)

        style_speaker = self.styles["speaker"]
        style_speaker.font.color.rgb = RGBColor(0, 0, 0)
        style_speaker.font.bold = False
        style_speaker.font.name = SANS_FONT
        set_rFonts(style_speaker, "asciiTheme", SANS_FONT)
        style_speaker.font.size = Pt(DEFAULT_FONT_SIZE)
        style_speaker.paragraph_format.space_before = Pt(DEFAULT_FONT_SIZE)
        style_speaker.paragraph_format.space_after = Pt(0)
        style_speaker.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

        style_ts1 = self.styles["ts1"]
        style_ts1.font.name = SERIF_FONT
        style_ts1.font.name_eastasia = SERIF_FONT
        style_ts1.font.size = Pt(DEFAULT_FONT_SIZE)
        style_ts1.paragraph_format.space_before = Pt(0)
        style_ts1.paragraph_format.space_after = Pt(0)
        style_ts1.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        style_ts1.paragraph_format.first_line_indent = Pt(- DEFAULT_FONT_SIZE)

        style_ts2 = self.styles["ts2"]
        style_ts2.base_style = style_ts1
        style_ts2.paragraph_format.first_line_indent = Pt(DEFAULT_FONT_SIZE)

        sec.footer.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    def add_page_number(self):
        fldChar1 = OxmlElement('w:fldChar')
        create_attribute(fldChar1, 'w:fldCharType', 'begin')
        instrText = OxmlElement('w:instrText')
        create_attribute(instrText, 'xml:space', 'preserve')
        instrText.text = "PAGE"
        fldChar2 = OxmlElement('w:fldChar')
        create_attribute(fldChar2, 'w:fldCharType', 'end')
        run = self.doc.sections[-1].footer.paragraphs[0].add_run()
        run._r.append(fldChar1)
        run._r.append(instrText)
        run._r.append(fldChar2)

    def txt_to_word(self, txtfilename: str):
        self.txtfilename = txtfilename
        # Open the text file in read mode with utf8 encoding
        with open(txtfilename, 'r', encoding='utf8') as f:
            lines = f.readlines()

        # Process each line
        pline_count = 0
        for line in lines:
            # Blank line
            if line.strip() == "":
                continue
            # Time and speaker info
            if line.startswith('['):
                elements = line.split(' ')
                speaker = elements[3].strip()
                time = " ".join(elements[:3])
                self.doc.add_paragraph(speaker + ' ' + time, style=self.styles["speaker"])
                pline_count = 1
                continue
            # Transcript
            if pline_count == 1:
                self.doc.add_paragraph("○　", style=self.styles["ts1"])
            else:
                self.doc.add_paragraph("", style=self.styles["ts2"])
            self.doc.paragraphs[-1].add_run(line.strip())
            pline_count += 1

        self.docfilename = f"{Path(txtfilename).stem}.docx"
        self.doc.save(self.docfilename)

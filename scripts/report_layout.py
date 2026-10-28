"""Small deterministic A4 PDF/Markdown renderer for the assignment reports."""

from pathlib import Path

import pymupdf


PAGE_WIDTH, PAGE_HEIGHT = 595, 842
MARGIN = 36
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN
BOTTOM = 802
INK = (0.12, 0.17, 0.22)
ACCENT = (0.08, 0.29, 0.44)
FONT_DIR = Path("C:/Windows/Fonts")
REGULAR = FONT_DIR / "arial.ttf"
BOLD = FONT_DIR / "arialbd.ttf"


def wrap(text, font, size, width):
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            trial = (line + " " + word).strip()
            if line and font.text_length(trial, fontsize=size) > width:
                lines.append(line)
                line = ""
            while font.text_length(word, fontsize=size) > width:
                cut = 1
                while cut < len(word) and font.text_length(word[:cut + 1], fontsize=size) <= width:
                    cut += 1
                lines.append(word[:cut])
                word = word[cut:]
            line = (line + " " + word).strip()
        lines.append(line)
    return lines


class Report:
    def __init__(self, directory, part, name):
        self.directory, self.part, self.name = Path(directory), part, name
        self.document = pymupdf.open()
        self.font = pymupdf.Font(fontfile=str(REGULAR))
        self.bold = pymupdf.Font(fontfile=str(BOLD))
        self.markdown = [f"# SC4001 — Part {part}", "", f"Author: {name} | October 11, 2026", ""]
        self.layout = []

    def page(self, title, *, first=False):
        self.current = self.document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        self.current.insert_font(fontname="ReportArial", fontfile=str(REGULAR))
        self.current.insert_font(fontname="ReportArialBold", fontfile=str(BOLD))
        if first:
            self.current.insert_text((MARGIN, 48), f"SC4001 · Part {self.part}", fontsize=17,
                fontname="ReportArialBold", color=ACCENT)
            self.current.insert_text((MARGIN, 65), f"{self.name}   |   Neural Networks and Deep Learning   |   October 11, 2026",
                fontsize=9, fontname="ReportArial", color=INK)
            self.y = 81
        else:
            self.current.insert_text((MARGIN, 44), title, fontsize=14,
                fontname="ReportArialBold", color=ACCENT)
            self.y = 58
        self.current.draw_line((MARGIN, PAGE_HEIGHT - 27), (PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 27),
                               color=(.75, .79, .82), width=.4)
        self.current.insert_text((MARGIN, PAGE_HEIGHT - 14), f"SC4001 | Part {self.part} | {self.name}",
            fontsize=8, fontname="ReportArial", color=INK)
        self.current.insert_text((PAGE_WIDTH - MARGIN - 32, PAGE_HEIGHT - 14), f"{len(self.document)} / 5",
            fontsize=8, fontname="ReportArial", color=INK)
        self.markdown.extend(["", f"<!-- PDF page {len(self.document)} -->", f"## {title}", ""])

    def reserve(self, height, description):
        if self.y + height > BOTTOM:
            raise ValueError(f"Part {self.part} page {len(self.document)} overflows at {description}: "
                             f"y={self.y:.1f}, height={height:.1f}, bottom={BOTTOM}")
        self.layout.append({"page": len(self.document), "kind": description,
                            "top": round(self.y, 2), "bottom": round(self.y + height, 2)})

    def text(self, text, *, size=10.5, bold=False, color=INK, gap=6, inset=0):
        font = self.bold if bold else self.font
        missing = {character for character in text if not character.isspace() and not font.has_glyph(ord(character))}
        if missing:
            raise ValueError(f"Report font lacks these glyphs: {sorted(missing)}")
        lines = wrap(text, font, size, CONTENT_WIDTH - 2 * inset)
        leading = size * 1.22
        self.reserve(len(lines) * leading + gap, "text")
        for line in lines:
            self.current.insert_text((MARGIN + inset, self.y + size), line,
                fontsize=size, fontname="ReportArialBold" if bold else "ReportArial", color=color)
            self.y += leading
        self.y += gap
        self.markdown.extend([text, ""])

    def section(self, text):
        self.text(text, size=11.5, bold=True, color=ACCENT, gap=5)
        self.markdown[-2] = f"### {text}"

    def table(self, headers, rows, *, widths=None, highlight=None, size=9.3):
        weights = widths or [1] * len(headers)
        widths = [CONTENT_WIDTH * value / sum(weights) for value in weights]
        data = [headers, *rows]
        leading = size * 1.16
        for index, row in enumerate(data):
            is_header = index == 0
            selected = index - 1 == highlight
            font = self.bold if is_header or selected else self.font
            wrapped = [wrap(str(value), font, size, width - 10) for value, width in zip(row, widths)]
            height = max(map(len, wrapped)) * leading + 9
            self.reserve(height, "table header" if is_header else "table row")
            fill = (.90, .94, .97) if is_header else ((.94, .97, .99) if selected else (1, 1, 1))
            self.current.draw_rect(pymupdf.Rect(MARGIN, self.y, PAGE_WIDTH - MARGIN, self.y + height),
                color=None, fill=fill)
            x = MARGIN
            for column, lines in enumerate(wrapped):
                for j, line in enumerate(lines):
                    self.current.insert_text((x + 5, self.y + 4 + size + j * leading), line,
                        fontsize=size, fontname="ReportArialBold" if is_header or selected else "ReportArial", color=INK)
                x += widths[column]
            self.current.draw_line((MARGIN, self.y + height), (PAGE_WIDTH - MARGIN, self.y + height),
                color=(.8, .84, .87), width=.35)
            self.y += height
        self.y += 7
        self.markdown.extend(["| " + " | ".join(str(value).replace("\n", " ") for value in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |"])
        for row in rows:
            self.markdown.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
        self.markdown.append("")

    def figures(self, assets, *, caption, height=None):
        gap = 12
        width = (CONTENT_WIDTH - gap * (len(assets) - 1)) / len(assets)
        sources = [pymupdf.open(self.directory / "assets" / f"{name}.pdf") for name in assets]
        heights = [width * doc[0].rect.height / doc[0].rect.width for doc in sources]
        actual_height = max(heights) if height is None else height
        self.reserve(actual_height + 3, "figure: " + ", ".join(assets))
        for index, source in enumerate(sources):
            x = MARGIN + index * (width + gap)
            self.current.show_pdf_page(pymupdf.Rect(x, self.y, x + width, self.y + actual_height),
                source, 0, keep_proportion=True)
            source.close()
        self.y += actual_height + 3
        for name in assets:
            self.markdown.extend([f"![{caption}](assets/{name}.png)", ""])
        self.text(caption, size=8.6, gap=7)

    def reference(self, text, url=None):
        top = self.y
        self.text(text, size=8.3, gap=3)
        if url:
            self.current.insert_link({"kind": pymupdf.LINK_URI,
                "from": pymupdf.Rect(MARGIN, top, PAGE_WIDTH - MARGIN, self.y), "uri": url})
            self.markdown[-2] = f"[{text}]({url})"

    def save(self, slug):
        if len(self.document) != 5:
            raise ValueError("Reports must contain exactly five pages")
        self.document.set_metadata({"title": f"SC4001 Part {self.part} report", "author": self.name,
            "subject": "Assignment answers from completed saved experiments", "creator": "Project venv / PyMuPDF"})
        self.document.subset_fonts()
        # Arial aliases several Unicode characters to identical glyphs (e.g.
        # '-' / soft hyphen, ';' / Greek question mark, space / NBSP). MuPDF's
        # default reverse mapping can select the alias when extracting text.
        # Preserve the report's original Unicode spelling in its ToUnicode maps.
        characters = set("\n".join(self.markdown) + f"SC4001 | Part {self.part} | {self.name} 1 / 5 ·")
        report_fonts = {item[4]: item[0] for index in range(len(self.document))
                        for item in self.document[index].get_fonts() if item[4].startswith("ReportArial")}
        for resource, xref in report_fonts.items():
            font = self.bold if resource == "ReportArialBold" else self.font
            mapping = {}
            for character in sorted(characters):
                if character.isspace() and character != " ":
                    continue
                glyph = font.has_glyph(ord(character))
                if glyph:
                    if glyph in mapping and mapping[glyph] != character:
                        raise ValueError(f"Ambiguous Unicode glyph in report: {mapping[glyph]!r}, {character!r}")
                    mapping[glyph] = character
            commands = ["/CIDInit /ProcSet findresource begin", "12 dict begin", "begincmap",
                "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
                "/CMapName /ReportUnicode def", "/CMapType 2 def", "1 begincodespacerange",
                "<0000> <FFFF>", "endcodespacerange"]
            items = list(mapping.items())
            for start in range(0, len(items), 100):
                chunk = items[start:start + 100]
                commands.append(f"{len(chunk)} beginbfchar")
                commands.extend(f"<{glyph:04X}> <{character.encode('utf-16-be').hex().upper()}>"
                                for glyph, character in chunk)
                commands.append("endbfchar")
            commands.extend(["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"])
            cmap = self.document.get_new_xref()
            self.document.update_object(cmap, "<<>>")
            self.document.update_stream(cmap, "\n".join(commands).encode("ascii"))
            self.document.xref_set_key(xref, "ToUnicode", f"{cmap} 0 R")
        path = self.directory / f"Part{self.part}_report_{slug}.pdf"
        self.document.save(path, garbage=4, deflate=True)
        self.document.close()
        path.with_suffix(".md").write_text("\n".join(self.markdown) + "\n", encoding="utf-8")
        return path, self.layout

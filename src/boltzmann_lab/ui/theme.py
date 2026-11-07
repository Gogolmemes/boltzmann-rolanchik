"""Readable presentation theme shared by widgets, graphs and documents."""

from PyQt6.QtGui import QColor, QFont, QPalette

from boltzmann_lab.utils.resources import load_stylesheet

PLOT_BACKGROUND = "#0b1220"
DEMON_TEXT_COLORS = ("#991b1b", "#166534", "#1e40af")


def relative_luminance(color):
    color = QColor(color)
    channels = (color.redF(), color.greenF(), color.blueF())
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return sum(c * weight for c, weight in zip(linear, (0.2126, 0.7152, 0.0722)))


def contrast_text_color(background):
    """Choose black or white by the greater luminance contrast ratio."""
    luminance = relative_luminance(background)
    white_contrast = 1.05 / (luminance + 0.05)
    black_contrast = (luminance + 0.05) / 0.05
    return QColor("#ffffff" if white_contrast >= black_contrast else "#000000")


def apply_presentation_theme(app):
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 18, QFont.Weight.DemiBold))
    palette = QPalette()
    colors = {
        QPalette.ColorRole.Window: "#f8fafc",
        QPalette.ColorRole.WindowText: "#0f172a",
        QPalette.ColorRole.Base: "#ffffff",
        QPalette.ColorRole.AlternateBase: "#e2e8f0",
        QPalette.ColorRole.Text: "#0f172a",
        QPalette.ColorRole.Button: "#e2e8f0",
        QPalette.ColorRole.ButtonText: "#0f172a",
        QPalette.ColorRole.Highlight: "#1e40af",
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.ToolTipBase: "#ffffff",
        QPalette.ColorRole.ToolTipText: "#0f172a",
    }
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#475569"))
    app.setPalette(palette)
    app.setStyleSheet(load_stylesheet())


def presentation_document(html):
    """Apply the same readable type and opaque backgrounds to embedded help."""
    style = """<style>
        html, body, table, td, th, div { background-color: #ffffff !important; }
        body, p, li, td, th, span, font {
            color: #0f172a !important;
            font-family: 'Segoe UI', Arial, sans-serif !important;
            font-size: 24px !important; font-weight: 600 !important; line-height: 1.6;
        }
        body { padding: 24px; }
        h1, h2, h3 { color: #0f172a !important; font-weight: 800 !important; }
        h1, h1 font { font-size: 38px !important; }
        h2, h2 font { font-size: 32px !important; border-bottom: 3px solid #1e40af; }
        h3, h3 font { font-size: 28px !important; }
        a, a span, a font { color: #075985 !important; text-decoration: underline; }
        img { max-width: 100%; height: auto; }
    </style>"""
    if "</head>" in html:
        return html.replace("</head>", style + "</head>", 1)
    return style + html

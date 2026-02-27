"""
INKFORGE — Rendering Service

Converts stroke sequences to visual output (PNG, PDF, SVG)
using CairoSVG and Pillow at 300 DPI print resolution.
"""

import io
import math
from pathlib import Path

import cairosvg
from PIL import Image
import svgwrite


# Paper sizes in mm
PAPER_SIZES = {
    "a4": (210, 297),
    "a5": (148, 210),
    "us_letter": (215.9, 279.4),
    "us_legal": (215.9, 355.6),
}

# Ink color presets
INK_COLORS = {
    "black": "#1a1a1a",
    "blue": "#1e3a8a",
    "dark_blue": "#1e40af",
    "sepia": "#704214",
    "red": "#991b1b",
    "green": "#166534",
}

# Paper texture colors
PAPER_COLORS = {
    "white": "#ffffff",
    "cream": "#fffef0",
    "aged": "#f5f0e1",
    "lined": "#ffffff",
    "graph": "#ffffff",
}


class Renderer:
    """
    Renders stroke sequences to exportable formats.

    Supports:
        - SVG generation from raw strokes
        - SVG → PNG conversion at 300 DPI
        - SVG → PDF conversion for print
        - Paper texture overlay
        - Ink color application
    """

    def __init__(self, export_dir: str = "./exports", dpi: int = 300) -> None:
        """
        Initialize the renderer.

        Args:
            export_dir: Directory to save exported files.
            dpi: Output resolution in dots per inch.
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def strokes_to_svg(
        self,
        strokes: list[tuple[float, float, int, int, int]],
        ink_color: str = "black",
        stroke_width_base: float = 1.5,
        paper_size: str = "a4",
        paper_color: str = "white",
        add_lines: bool = False,
        line_spacing: float = 8.0,
        margin_left: float = 20.0,
        margin_top: float = 25.0,
        scale: float = 1.0,
    ) -> str:
        """
        Convert stroke tuples to SVG string.

        Args:
            strokes: List of (Δx, Δy, p1, p2, p3) tuples.
            ink_color: Ink color name or CSS color string.
            stroke_width_base: Base stroke width in SVG units.
            paper_size: Paper size name ("a4", "us_letter", etc.).
            paper_color: Paper background color.
            add_lines: Whether to add ruled lines.
            line_spacing: Spacing between ruled lines (mm).
            margin_left: Left margin offset (mm).
            margin_top: Top margin offset (mm).
            scale: Scale factor for strokes.

        Returns:
            SVG document as string.
        """
        # Get paper dimensions
        width_mm, height_mm = PAPER_SIZES.get(paper_size, PAPER_SIZES["a4"])

        # Resolve colors
        ink = INK_COLORS.get(ink_color, ink_color)
        paper = PAPER_COLORS.get(paper_color, paper_color)

        # Create SVG document
        dwg = svgwrite.Drawing(
            size=(f"{width_mm}mm", f"{height_mm}mm"),
            viewBox=f"0 0 {width_mm} {height_mm}",
        )

        # Add background
        dwg.add(dwg.rect(insert=(0, 0), size=(width_mm, height_mm), fill=paper))

        # Add ruled lines if requested
        if add_lines:
            self._add_ruled_lines(dwg, width_mm, height_mm, line_spacing, margin_left)

        # Convert strokes to absolute coordinates
        x, y = margin_left, margin_top
        current_path_points = []
        paths = []

        for i, (dx, dy, p1, p2, p3) in enumerate(strokes):
            x += dx * scale
            y += dy * scale

            if p1 == 1:  # Pen down - drawing
                current_path_points.append((x, y))
            else:  # Pen up or end
                if len(current_path_points) >= 2:
                    paths.append(current_path_points.copy())
                current_path_points = []

                if p3 == 1:  # End of sequence
                    break

        # Add final path if exists
        if len(current_path_points) >= 2:
            paths.append(current_path_points)

        # Draw all paths
        for path_points in paths:
            if len(path_points) < 2:
                continue

            # Create smooth path using quadratic bezier curves
            path_d = self._points_to_smooth_path(path_points)

            dwg.add(dwg.path(
                d=path_d,
                stroke=ink,
                stroke_width=stroke_width_base,
                fill="none",
                stroke_linecap="round",
                stroke_linejoin="round",
            ))

        return dwg.tostring()

    def strokes_to_svg_with_positions(
        self,
        word_data: list[dict],
        ink_color: str = "black",
        stroke_width_base: float = 1.5,
        paper_size: str = "a4",
        paper_color: str = "white",
        add_lines: bool = False,
        scale: float = 0.15,
    ) -> str:
        """
        Convert document word data (with positions) to SVG.

        Args:
            word_data: List of dicts with "strokes", "position", "word" keys.
            ink_color: Ink color.
            stroke_width_base: Base stroke width.
            paper_size: Paper size.
            paper_color: Background color.
            add_lines: Add ruled lines.
            scale: Scale factor for strokes.

        Returns:
            SVG document string.
        """
        width_mm, height_mm = PAPER_SIZES.get(paper_size, PAPER_SIZES["a4"])
        ink = INK_COLORS.get(ink_color, ink_color)
        paper = PAPER_COLORS.get(paper_color, paper_color)

        dwg = svgwrite.Drawing(
            size=(f"{width_mm}mm", f"{height_mm}mm"),
            viewBox=f"0 0 {width_mm} {height_mm}",
        )

        dwg.add(dwg.rect(insert=(0, 0), size=(width_mm, height_mm), fill=paper))

        if add_lines:
            self._add_ruled_lines(dwg, width_mm, height_mm, 8.0, 20.0)

        # Render each word at its position
        for word_info in word_data:
            strokes = word_info["strokes"]
            pos_x, pos_y = word_info["position"]

            x, y = pos_x, pos_y
            current_path_points = []
            paths = []

            for dx, dy, p1, p2, p3 in strokes:
                x += dx * scale
                y += dy * scale

                if p1 == 1:
                    current_path_points.append((x, y))
                else:
                    if len(current_path_points) >= 2:
                        paths.append(current_path_points.copy())
                    current_path_points = []
                    if p3 == 1:
                        break

            if len(current_path_points) >= 2:
                paths.append(current_path_points)

            for path_points in paths:
                if len(path_points) < 2:
                    continue
                path_d = self._points_to_smooth_path(path_points)
                dwg.add(dwg.path(
                    d=path_d,
                    stroke=ink,
                    stroke_width=stroke_width_base,
                    fill="none",
                    stroke_linecap="round",
                    stroke_linejoin="round",
                ))

        return dwg.tostring()

    def _points_to_smooth_path(self, points: list[tuple[float, float]]) -> str:
        """
        Convert points to smooth SVG path using quadratic bezier curves.

        Args:
            points: List of (x, y) coordinates.

        Returns:
            SVG path d attribute string.
        """
        if len(points) < 2:
            return ""

        # Start at first point
        d = f"M {points[0][0]:.2f},{points[0][1]:.2f}"

        if len(points) == 2:
            # Simple line for 2 points
            d += f" L {points[1][0]:.2f},{points[1][1]:.2f}"
            return d

        # Use quadratic bezier for smoothing
        for i in range(1, len(points) - 1):
            # Control point is current point
            cx, cy = points[i]
            # End point is midpoint between current and next
            nx, ny = points[i + 1]
            ex, ey = (cx + nx) / 2, (cy + ny) / 2
            d += f" Q {cx:.2f},{cy:.2f} {ex:.2f},{ey:.2f}"

        # Final line to last point
        d += f" L {points[-1][0]:.2f},{points[-1][1]:.2f}"

        return d

    def _add_ruled_lines(
        self,
        dwg: svgwrite.Drawing,
        width: float,
        height: float,
        spacing: float,
        margin: float,
    ) -> None:
        """Add horizontal ruled lines to SVG."""
        line_color = "#d4d4d4"
        y = margin + spacing

        while y < height - margin:
            dwg.add(dwg.line(
                start=(margin, y),
                end=(width - margin, y),
                stroke=line_color,
                stroke_width=0.3,
            ))
            y += spacing

    def export_png(
        self,
        svg_content: str,
        output_path: str | None = None,
        transparent: bool = False,
        background_color: str = "#ffffff",
    ) -> Path | bytes:
        """
        Render SVG to PNG at target DPI.

        Args:
            svg_content: SVG document string.
            output_path: Output file path. If None, returns bytes.
            transparent: Whether to use transparent background.
            background_color: Background color if not transparent.

        Returns:
            Path to saved file, or PNG bytes if no output_path.
        """
        # Calculate scale for DPI
        # SVG default is 96 DPI, we want 300 DPI
        scale = self.dpi / 96.0

        # Convert SVG to PNG bytes
        png_bytes = cairosvg.svg2png(
            bytestring=svg_content.encode("utf-8"),
            scale=scale,
            background_color=None if transparent else background_color,
        )

        if output_path is None:
            return png_bytes

        # Save to file
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(png_bytes)

        return output

    def export_pdf(
        self,
        svg_content: str,
        output_path: str | None = None,
        paper_size: str = "a4",
    ) -> Path | bytes:
        """
        Render SVG to PDF at print resolution.

        Args:
            svg_content: SVG document string.
            output_path: Output file path. If None, returns bytes.
            paper_size: "a4" or "us_letter".

        Returns:
            Path to saved file, or PDF bytes if no output_path.
        """
        # Determine target page dimensions from paper_size
        size_mm = PAPER_SIZES.get(paper_size, PAPER_SIZES["a4"])
        # CairoSVG uses CSS pixels (96 dpi). 1 inch = 25.4 mm.
        px_per_mm = 96.0 / 25.4
        output_width = size_mm[0] * px_per_mm
        output_height = size_mm[1] * px_per_mm

        # Convert SVG to PDF bytes with explicit page size
        pdf_bytes = cairosvg.svg2pdf(
            bytestring=svg_content.encode("utf-8"),
            output_width=output_width,
            output_height=output_height,
        )

        if output_path is None:
            return pdf_bytes

        # Save to file
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(pdf_bytes)

        return output

    def export_svg(
        self,
        svg_content: str,
        output_path: str | None = None,
    ) -> Path | str:
        """
        Save raw SVG to file.

        Args:
            svg_content: SVG document string.
            output_path: Output file path. If None, returns content.

        Returns:
            Path to saved file, or SVG string if no output_path.
        """
        if output_path is None:
            return svg_content

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(svg_content, encoding="utf-8")

        return output

    def render_to_image(
        self,
        strokes: list[tuple[float, float, int, int, int]],
        width: int = 800,
        height: int = 200,
        ink_color: str = "black",
        background_color: str = "white",
        stroke_width: float = 2.0,
        scale: float = 1.0,
        padding: float = 20.0,
    ) -> Image.Image:
        """
        Render strokes directly to a PIL Image (for previews).

        Args:
            strokes: List of stroke tuples.
            width: Output width in pixels.
            height: Output height in pixels.
            ink_color: Ink color.
            background_color: Background color.
            stroke_width: Stroke width.
            scale: Scale factor.
            padding: Padding around content.

        Returns:
            PIL Image object.
        """
        # Create minimal SVG
        ink = INK_COLORS.get(ink_color, ink_color)
        bg = PAPER_COLORS.get(background_color, background_color)

        dwg = svgwrite.Drawing(size=(width, height))
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill=bg))

        # Convert strokes
        x, y = padding, padding
        current_path_points = []
        paths = []

        for dx, dy, p1, p2, p3 in strokes:
            x += dx * scale
            y += dy * scale

            if p1 == 1:
                current_path_points.append((x, y))
            else:
                if len(current_path_points) >= 2:
                    paths.append(current_path_points.copy())
                current_path_points = []
                if p3 == 1:
                    break

        if len(current_path_points) >= 2:
            paths.append(current_path_points)

        for path_points in paths:
            if len(path_points) < 2:
                continue
            path_d = self._points_to_smooth_path(path_points)
            dwg.add(dwg.path(
                d=path_d,
                stroke=ink,
                stroke_width=stroke_width,
                fill="none",
                stroke_linecap="round",
                stroke_linejoin="round",
            ))

        # Convert to PIL Image
        png_bytes = cairosvg.svg2png(bytestring=dwg.tostring().encode("utf-8"))
        return Image.open(io.BytesIO(png_bytes))

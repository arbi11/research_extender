"""Vector drawings read from PDF pages converted to SVG: strokes, scales, chains and simplification."""
import math
import unittest

from femm_brain import vector_figure


def _svg(paths, height=792):
    body = "".join(paths)
    glyph = '<defs><g id="glyph-0-0"><path d="M 0 0 L 5 5 Z"/></g></defs>'
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 612 {height}">{glyph}{body}</svg>'


class VectorFigureTests(unittest.TestCase):
    def test_strokes_are_read_in_pdf_coordinates_with_their_colour(self):
        text = _svg(['<path fill="none" stroke="rgb(0%, 0%, 100%)" d="M 10 20 L 30 20 L 30.06 19.94 M 5 5 L 5 9" '
                     'transform="matrix(1, 0, 0, -1, 0, 792)"/>',
                     '<path fill="none" stroke="rgb(0%, 0%, 0%)" d="M 0 0 C 0 10 10 10 10 0"/>',
                     '<path fill="rgb(0%, 0%, 0%)" d="M 1 1 L 2 2"/>'])
        pieces = vector_figure.segments(text, min_length=0.1)
        blue = [(p, q) for colour, p, q in pieces if colour == "rgb(0%,0%,100%)"]
        # The flip matrix maps PDF coordinates to SVG ones; reading back gives the PDF coordinates, tails dropped.
        self.assertEqual(blue, [((10, 20), (30, 20)), ((5, 5), (5, 9))])
        black = [(p, q) for colour, p, q in pieces if colour == "rgb(0%,0%,0%)"]
        self.assertEqual(len(black), 8)  # a cubic in eight straight pieces; glyphs in <defs> and unstroked paths skipped
        self.assertAlmostEqual(black[0][0][1], 792)
        self.assertAlmostEqual(black[-1][1][0], 10)
        self.assertAlmostEqual(max(792 - q[1] for _, _, q in [(c, p, q) for c, p, q in pieces if c == "rgb(0%,0%,0%)"]), 7.5)

    def test_unequal_scales_show_as_an_ellipse_and_are_recovered(self):
        points = [(100 + 59.2169 * math.cos(t), 200 + 58.5493 * math.sin(t)) for t in (k * 2 * math.pi / 37 for k in range(37))]
        x0, y0, a, b, residual = vector_figure.fit_axes_ellipse(points)
        self.assertAlmostEqual(x0, 100, places=6)
        self.assertAlmostEqual(y0, 200, places=6)
        self.assertAlmostEqual(a / b, 59.2169 / 58.5493, places=9)
        self.assertLess(residual, 1e-9)
        with self.assertRaises(ValueError):
            vector_figure.fit_axes_ellipse(points[:3])

    def test_pieces_chain_into_polylines_and_simplify(self):
        square = [((0, 0), (1, 0)), ((1, 0.02), (1, 1)), ((0, 1), (1, 1)), ((0, 1), (0, 0.01))]
        loops = vector_figure.chains(square, tolerance=0.05)
        self.assertEqual(len(loops), 1)
        self.assertEqual(len(loops[0]), 5)
        self.assertLess(math.dist(loops[0][0], loops[0][-1]), 0.05)
        arc = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in range(0, 91)]
        coarse = vector_figure.simplify(arc, tolerance=0.01)
        self.assertEqual(coarse[0], arc[0])
        self.assertEqual(coarse[-1], arc[-1])
        self.assertLess(len(coarse), 12)
        for x, y in arc:  # every dropped point stays within the tolerance of the kept polyline
            self.assertLess(min(abs(math.hypot(x, y) - 1), 1), 0.011)
        self.assertEqual(vector_figure.simplify([(0, 0), (1, 1e-9), (2, 0)], 1e-6), [(0, 0), (2, 0)])
        with self.assertRaises(ValueError):
            vector_figure.chains(square, tolerance=0)


if __name__ == "__main__":
    unittest.main()

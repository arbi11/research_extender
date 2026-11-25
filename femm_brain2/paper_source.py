"""Read cached papers as data; reproduce an explicitly located vector-plot extraction.

This is mechanical extraction, not automatic interpretation of arbitrary papers.
The coding agent supplies the panel, colour, axes and semantic interpretation.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

PAPER_CACHE = Path.home() / '.cache/femm-brain/papers'


def cached_paper(source, cache=PAPER_CACHE):
    path = Path(cache) / source['sha256'] / 'source.pdf'
    if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
        raise ValueError('Paper PDF fingerprint differs from the reviewed source')
    return path


def paper_text(source, cache=PAPER_CACHE):
    path = cached_paper(source, cache)
    return subprocess.run(['pdftotext', '-layout', str(path), '-'], capture_output=True,
                          text=True, check=True, timeout=30).stdout.split('\f')


def _linear_fit(points):
    xbar = sum(p[0] for p in points)/len(points)
    ybar = sum(p[1] for p in points)/len(points)
    slope = sum((x-xbar)*(y-ybar) for x,y in points)/sum((x-xbar)**2 for x,y in points)
    intercept = ybar-slope*xbar
    residual = max(abs(slope*x+intercept-y) for x,y in points)
    return slope, intercept, residual


def extract_vector_markers(source, recipe, cache=PAPER_CACHE):
    """Extract one colour's filled marker centres in an agent-selected PDF panel.

    SVG path coordinates must already be in page space. Transformed markers are
    refused rather than silently interpreted incorrectly. No source code is run.
    """
    pdf = cached_paper(source, cache)
    with tempfile.TemporaryDirectory(prefix='femm-brain2-paper-') as directory:
        svg = Path(directory) / 'panel.svg'
        subprocess.run(['pdftocairo', '-svg', '-f', str(recipe['pdf_page']), '-l',
                        str(recipe['pdf_page']), str(pdf), str(svg)], check=True,
                       capture_output=True, timeout=30)
        content = svg.read_bytes()
    root = ET.fromstring(content)
    selected = []
    box = recipe['marker_box_pdf_points']
    for element in root.iter():
        if element.get('fill', '').replace(' ', '') != recipe['fill'].replace(' ', ''):
            continue
        if element.get('transform'):
            raise ValueError('Transformed plot markers need an explicit transform parser')
        path = element.get('d', '')
        if not path:
            continue
        numbers = [float(n) for n in re.findall(r'-?\d+(?:\.\d+)?', path)]
        if len(numbers) % 2:
            raise ValueError('Marker path does not contain coordinate pairs')
        xs, ys = numbers[::2], numbers[1::2]
        x, y = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            selected.append([x, y])
    selected.sort()
    if len(selected) != recipe['expected_markers']:
        raise ValueError(f"Expected {recipe['expected_markers']} markers, extracted {len(selected)}")
    slope, intercept, residual = _linear_fit(recipe['y_calibration'])
    left, right = recipe['x_axis_pdf_points']
    span = recipe['angle_span_degrees']
    samples = []
    for index, (x, y) in enumerate(selected):
        uniform_angle = index * span/(len(selected)-1)
        image_angle = (x-left)/(right-left)*span
        if abs(uniform_angle-image_angle) > recipe['angle_uncertainty_degrees']:
            raise ValueError('FEM markers do not follow the declared equally spaced angles')
        samples.append({'angle_deg': uniform_angle, 'value': slope*y+intercept,
                        'page_x': x, 'page_y': y, 'image_angle_deg': image_angle})
    # Last marker duplicates the first at one electrical period; do not count it twice.
    unique = samples[:-1]
    count = len(unique)
    cos_part = 2/count*sum(s['value']*math.cos(2*math.pi*k/count) for k,s in enumerate(unique))
    sin_part = 2/count*sum(s['value']*math.sin(2*math.pi*k/count) for k,s in enumerate(unique))
    return {'source_sha256': source['sha256'], 'pdf_page': recipe['pdf_page'],
            'method': 'Vector FEM marker centres, interior gridline calibration, endpoint excluded DFT',
            'y_calibration_max_residual': residual, 'samples': samples,
            'fundamental_peak_V': math.hypot(cos_part, sin_part),
            'absolute_uncertainty_V': recipe['absolute_uncertainty_V']}


def verify_source(knowledge, cache=PAPER_CACHE):
    """Check page locators/short quotes and reproduce the saved marker readings."""
    for paper in knowledge['papers']:
        pages = paper_text(paper['source'], cache)
        normalized = lambda s: re.sub(r'\s+', '', s).casefold()
        for name, parameter in paper['parameters'].items():
            if parameter['provenance'] == 'quoted':
                page = pages[parameter['pdf_page']-1]
                if normalized(parameter['quote']) not in normalized(page):
                    raise ValueError('Quoted parameter is absent from its PDF page: '+name)
        extracted = extract_vector_markers(paper['source'], paper['extraction_recipe'], cache)
        saved = paper['extraction']
        if len(extracted['samples']) != len(saved['samples']) or any(
            abs(actual['value']-expected['value']) > 1e-9 or abs(actual['angle_deg']-expected['angle_deg']) > 1e-9
            for actual,expected in zip(extracted['samples'], saved['samples'])):
            raise ValueError('Stored figure samples differ from the cached paper')
        if abs(extracted['fundamental_peak_V']-paper['acceptance']['emf_fundamental_peak_V']) > 1e-9:
            raise ValueError('No-load target differs from the frozen figure extraction')
    return {'status': 'passed', 'papers': len(knowledge['papers']),
            'checks': 'PDF hashes, quoted values/page locators and vector FEM-marker extraction'}

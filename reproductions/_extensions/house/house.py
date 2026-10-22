"""The house style, for the Python a Quarto document runs.

This module ships inside the extension, so it travels with every project that
installs the kit. A project does not import it directly. Each keeps a thin
local module (``bookstyle.py``, ``paperstyle.py``, or ``style.py`` in a note)
which puts this directory on ``sys.path``, derived from its own ``__file__``,
imports ``house``, and adds only what is genuinely local.

``aggregate`` is imported lazily, inside the functions that need it, so a
project that does not use the package does not need it installed.

**Targets.** A document and a slide differ in the width a table may take. Each
also names the face figure text is set in, serif for both since review, so a
slide can take a different face with a one-line change.
``configure(target=...)`` picks one, once, in the project's setup; a document
is the default. ``figures()`` then applies the figure style for it.

**Library figures.** A figure ``aggregate`` draws, such as ``a.plot()``, is
sized by the library, not by ``figure()``. ``figures()`` and ``configure()``
tell aggregate the target's text width and height cap through
``aggregate.plots.fit_page``, so a library figure wider than the page is drawn
at the page width, with the house type, rather than drawn wider and shrunk.

**Tables.** ``GT`` is ``greater_tables.GT`` with the house style. It publishes
all three types, html, latex and plain text, and Quarto takes the right one for
each format: LaTeX in a PDF, HTML on the web. No document pins or detects its
target, so the stale ``QUARTO_FIG_FORMAT`` that a jupyter daemon carries from
one format's pass to the next no longer matters. The mechanical house rules
from ``gt-house-style.md`` are implemented here once:

1. publish html, latex and plain text, and let Quarto choose;
2. the width budget is the text width of the target, with ``tikz_scale`` 0.9
   as a ceiling, and widths are measured in STIX Two, the house text face;
3. integer-valued floats print as integers;
4. a column whose name contains ``err`` or ``resid`` prints in scientific
   notation.

Frames that ``aggregate.qd`` prints as text are not touched: they are printed
output, and the preamble sets them in the code face at the code size.
"""
import logging
import re
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

__all__ = ['FACES', 'GT', 'MIMES', 'TARGETS', 'WIDTHS', 'configure', 'figsize',
           'figure', 'figures', 'house_style', 'receipt', 'show']

# Every step down greater_tables' scale ladder is logged at WARNING, and Quarto
# captures a kernel's stderr into the rendered cell, so a wide table would
# carry a "tikz_scale drops to 0.75" banner. Stepping down is the design
# working, not a fault. Raise the level to see them when tuning a budget.
logging.getLogger('greater_tables').setLevel(logging.ERROR)

#: What every table publishes. Quarto keeps the one its format wants.
MIMES = ('text/html', 'text/latex', 'text/plain')

#: Per target settings. ``table_width_em`` is the text width in em of the
#: body size: a document's 16.0cm block is 455pt, which at 11pt is 41.4em; a
#: slide's frame, measured on the 28:17 canvas, is 381.6pt, which at 11pt is
#: 34.7em. ``figure_face`` is a key of ``FACES``; changing a target's face is
#: this one line. Both targets use serif, chosen by eye at review.
#: ``text_width`` is the same block in inches, the width of a full figure, and
#: ``max_height`` the tallest figure that fits: on a slide, what fits under
#: the frame title.
TARGETS = {
    'document': {'table_width_em': 41.4, 'figure_face': 'serif',
                 'text_width': 6.3, 'max_height': 8.0},
    'slide': {'table_width_em': 34.7, 'figure_face': 'serif',
              'text_width': 5.3, 'max_height': 2.4},
}

#: The house figure widths, as fractions of the text width. Half is 48%, so
#: two halves leave a 4% gap, as two 48% columns do on a slide; in a document
#: it is 3.0in (77mm), close to a journal's one-column figure (Springer 84mm).
#: A single figure is half width unless it needs more room.
WIDTHS = {'half': 0.48, 'three-quarter': 0.75, 'full': 1.0}

#: The faces figure text can be set in, all shipped inside matplotlib so a
#: figure looks the same on any machine. ``serif`` matches STIX Two in a
#: document: STIXGeneral is the STIX fonts' first release, and the stix
#: mathtext set draws from it. ``sans`` is kept for a figure that wants it.
FACES = {
    'serif': {'font.family': 'serif', 'font.serif': ['STIXGeneral'],
              'mathtext.fontset': 'stix'},
    'sans': {'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
             'mathtext.fontset': 'dejavusans'},
}

#: The print style, beside this module in the installed extension.
MPLSTYLE = Path(__file__).resolve().parent / 'house.mplstyle'

_target = 'document'


def configure(target='document'):
    """Choose the target every later table (and figure) is set for.

    Parameters
    ----------
    target : str
        A key of ``TARGETS``: ``'document'`` for an article, report or book,
        the default, or ``'slide'`` for a beamer deck.

    Raises
    ------
    ValueError
        If the target is unknown.
    """
    global _target
    if target not in TARGETS:
        raise ValueError(f'target {target!r} is not one of {sorted(TARGETS)}')
    _target = target
    _fit_library_figures()


def _fit_library_figures():
    """Tell aggregate the current target's page, when aggregate can be told.

    The kit does not require aggregate, and an aggregate before 1.6.0 has no
    ``fit_page``; either way its figures keep their own size, as before.
    """
    try:
        from aggregate.plots import fit_page
    except ImportError:
        return
    target = TARGETS[_target]
    fit_page(width=target['text_width'], max_height=target['max_height'])


def figures(face=None):
    """Apply the house figure style for the current target.

    Call once in a project's setup, in place of ``aggregate.style.use()``. It
    applies ``house.mplstyle`` and then the face the target names, and fits
    aggregate's own figures to the target's page.

    Parameters
    ----------
    face : str, optional
        A key of ``FACES``, overriding the target's face. The gallery uses it
        to show one figure in both faces; a project should not need it.

    Raises
    ------
    ValueError
        If the face is unknown.

    Notes
    -----
    The style carries no ``figure.figsize``. Quarto sets the size from each
    format's ``fig-width`` and ``fig-height`` when the kernel starts, and the
    style leaves it alone.

    A library figure is fitted, not scaled: aggregate shrinks one wider than
    the text to the text width, never below 2in high nor above the target's
    cap, and draws it at that size, so its type is the style's 9 and 8pt. A
    narrower one keeps its size. A two-panel ``Aggregate.plot()`` is 6.3 ×
    2.2in in a document and 5.3 × 2.0in on a slide.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    face = face or TARGETS[_target]['figure_face']
    if face not in FACES:
        raise ValueError(f'face {face!r} is not one of {sorted(FACES)}')
    plt.style.use(MPLSTYLE)
    mpl.rcParams.update(FACES[face])
    _fit_library_figures()


def figsize(width='half', rows=1, cols=1, height=None):
    """The size, in inches, of a house figure for the current target.

    Parameters
    ----------
    width : str
        A key of ``WIDTHS``: ``'half'``, the default for a single figure,
        ``'three-quarter'`` for one that needs more room, or ``'full'`` for a
        row of panels.
    rows, cols : int
        The grid of panels the figure holds.
    height : float, optional
        The height in inches, overriding the rule below.

    Returns
    -------
    tuple of float
        ``(width, height)``, ready for ``figsize=``.

    Raises
    ------
    ValueError
        If the width is unknown.

    Notes
    -----
    A figure is drawn at the size it is printed, so its type prints at the
    size the style names. Under Jupyter a cell's ``fig-width`` does not reach
    matplotlib: it only scales the finished image, and the type with it. A
    figure that is not the format's default size therefore takes its size
    here, in code.

    Each row of panels is as tall as a panel is wide divided by an aspect of
    1.4, or 1.6 for a panel wider than 4in, and never less than 2in, so the
    panels of a row of three still have room for their labels. The whole is
    capped at the target's ``max_height``. In a document a half-width figure
    is 3.0 × 2.2in, a full row of two 6.3 × 2.25in, and a full row of three
    6.3 × 2.0in.
    """
    if width not in WIDTHS:
        raise ValueError(f'width {width!r} is not one of {sorted(WIDTHS)}')
    target = TARGETS[_target]
    w = WIDTHS[width] * target['text_width']
    if height is None:
        panel = w / cols
        height = rows * max(panel / (1.6 if panel > 4 else 1.4), 2.0)
    return round(w, 2), round(min(height, target['max_height']), 2)


def figure(width='half', rows=1, cols=1, height=None, **kwargs):
    """A house figure and its axes, drawn at the size it is printed.

    ``plt.subplots`` with ``figsize`` taken from ``figsize()``. Everything
    else passes through, such as ``sharey=True``.

    Parameters
    ----------
    width, rows, cols, height
        As for ``figsize()``.
    **kwargs
        Passed to ``matplotlib.pyplot.subplots``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : matplotlib.axes.Axes or ndarray of Axes
    """
    import matplotlib.pyplot as plt
    return plt.subplots(rows, cols, figsize=figsize(width, rows, cols, height),
                        **kwargs)


def house_style():
    """The table style for the current target: its width budget, and a scale ceiling.

    Returns
    -------
    greater_tables.HouseStyle

    Notes
    -----
    ``tikz_scale`` is a ceiling rather than a size. greater_tables takes the
    smaller of it and its own step, and steps further down when columns cannot
    be narrowed enough to fit, so 0.9 means at most nine tenths, and less if
    the table needs it. The budget is the full text width: set short of it,
    the shrink pass narrows columns and wraps cells, which is the wrong
    failure.

    ``tikz_font`` names the face the table is set in, STIX Two Text in a
    document and on a slide, so that greater_tables (7.2.0 or later)
    measures column widths in its glyphs. Left at its default, Times, a
    label sized to its Times width breaks in STIX Two, which is a little
    wider: "Figure 6" set on two lines.
    """
    from greater_tables import HouseStyle
    return HouseStyle(max_table_width_em=TARGETS[_target]['table_width_em'],
                      tikz_scale=0.9, tikz_font='stix-two')


#: Column names that mark an error or residual (rule 4).
_ERROR_COLUMN = re.compile(r'err|resid', re.IGNORECASE)


def _integer_floats(df):
    """A copy with every integer-valued float column and index level as int.

    A grid point, a bucket size or a layer count that pandas happens to hold
    as float otherwise prints as ``1.000``. Only levels with no missing values
    convert, so a dash for a missing value survives. Only magnitudes below
    2**53, where a float holds every integer exactly, convert: ``inf`` equals
    its own rounding, and it and any float past the int64 range would
    otherwise print as -9,223,372,036,854,775,808.
    """
    def whole(s):
        return (pd.api.types.is_float_dtype(s) and len(s)
                and (s.abs() < 2**53).all() and (s == s.round()).all())

    df = df.copy()
    for c in df.columns:
        if whole(df[c]):
            df[c] = df[c].astype('int64')
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.set_levels(
            [lv.astype('int64') if whole(lv.to_series()) else lv
             for lv in df.index.levels])
    elif whole(df.index.to_series()):
        df.index = df.index.astype('int64')
    return df


def GT(data=None, *args, **kwargs):
    """``greater_tables.GT`` carrying the house style.

    The defaults implement the mechanical house rules listed in the module
    docstring. ``TableSpec`` options (formatters, alignment, ``ratio_cols``
    and the rest) pass straight through, and an explicit entry beats a
    default. An explicit ``style=`` or ``mimes=`` wins, for the rare table
    that needs its own.

    Parameters
    ----------
    data : DataFrame or other
        The table. A DataFrame gets rules 3 and 4; anything else passes
        through untouched.
    *args, **kwargs
        Passed to ``greater_tables.GT``.

    Returns
    -------
    greater_tables.GT

    Notes
    -----
    The engineering fallback greater_tables uses for other floats is left
    alone. Switching it to exponents turns ``15.035M`` into ``15.035e+06`` and
    ``1.5`` into ``1.500e+00``, which is worse.
    """
    from greater_tables import GT as _GT
    kwargs.setdefault('style', house_style())
    kwargs.setdefault('mimes', MIMES)
    if isinstance(data, pd.DataFrame):
        data = _integer_floats(data)
        errors = {c: '{x:.2e}' for c in data.columns
                  if pd.api.types.is_float_dtype(data[c])
                  and _ERROR_COLUMN.search(str(c))}
        kwargs['formatters'] = {**errors, **(kwargs.get('formatters') or {})}
    return _GT(data, *args, **kwargs)


def receipt(obj, index_name, sig=6):
    """A mixed-type receipt, formatted for display.

    Several of the library's audit records are one labelled column holding
    strings, integers and floats together, such as ``RuinResult.ruin_df`` and
    ``AggregateNTT.ntt_audit``. Pandas gives such a column ``object`` dtype,
    which defeats every numeric formatter greater_tables would otherwise
    apply, so a probability arrives on the page as ``0.009997890257070097``.
    Here the floats are rendered to ``sig`` significant figures and everything
    else is left as it stands. The index of these frames is also unnamed,
    which prints as ``level_0``, so it takes a name.

    Parameters
    ----------
    obj : Series or DataFrame
        The receipt. A Series becomes a one-column frame named ``value``.
    index_name : str
        Heading for the column of labels.
    sig : int, default 6
        Significant figures for the floats.

    Returns
    -------
    DataFrame
        A display copy. The original is untouched.
    """
    df = obj.to_frame('value') if hasattr(obj, 'to_frame') else obj.copy()
    df = df.copy().rename_axis(index_name)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(
                lambda v: f'{v:,.{sig}g}' if isinstance(v, float) else v)
    return df


def show(program, label=None):
    """Display a DecL program as a ``decl`` code block, colored as aLL colors it.

    The program reaches the page as a fenced ```` ```decl ```` block, published
    through ``display(Markdown(...))``. Quarto then colors it with the kit's
    ``decl.xml`` and ``house.theme``, exactly as it colors a fence typed into
    the document, in PDF, on a slide and on the web. No Pygments, and no ``\\PY``
    macros in the preamble.

    Parameters
    ----------
    program : str or object
        A program string, shown exactly as written, or an object built from
        DecL (``Aggregate``, ``Portfolio``, ``Distortion`` and the rest), whose
        canonical program is shown together with its ``hints{}`` clause.
    label : str, optional
        A short plain-text caption, set bold above the program.

    Raises
    ------
    ValueError
        If the program is empty, which for an object means it was constructed
        through its class rather than built from DecL.

    Notes
    -----
    A string is shown as authored, without a parse. The canonical form a parse
    writes back differs in places (``reinstatements 3 free`` comes back as a
    vector of charges, percentages resolve to decimals), and where the
    language as written is the point, canonicalizing it loses the point. The
    object form shows what the parser understood. Its ``trailer`` keeps
    ``hints{}``, which the library's default drops, so a page that says the
    grid inputs ride inside the program shows them doing so.
    """
    if isinstance(program, str):
        text = program.strip('\n').rstrip()
    else:
        text = program.format_program(fmt='text', trailer=('hints',)).strip('\n').rstrip()
    if not text.strip():
        name = getattr(program, 'name', '')
        raise ValueError(
            f'{type(program).__name__} {name!r} has no program to show; it was not '
            'built from DecL. Pass the program text, or drop the display.')
    head = f'**{label}**\n\n' if label else ''
    display(Markdown(f'{head}```decl\n{text}\n```'))

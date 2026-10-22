"""The style layer for the eltq reproductions: the house kit, and `build`.

The look is the house's. The `house` module ships inside the installed
extension, `_extensions/house/` (from `D:\\Projects\\house-style`), and this
module puts that directory on `sys.path`, derived from its own location, so
the pages build wherever the repository is cloned. Importing this module also
applies the house figure style.

The names match the `bookstyle` module of the aggregate Reproductions book,
so a page here can move into that book unchanged: `GT`, the house table,
`show`, a DecL program as a colored block, and `build`, aggregate's own. That
book's `build` also records each program for its appendix; this one does not,
since nothing here collects them. `figure`, the house figure at its printed
size, is the one addition, and moving a page that uses it means adding the
same name to that book's module.
"""
import sys
from pathlib import Path

#: The directory holding this module, which is also Quarto's working
#: directory during a render.
BOOK_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(BOOK_ROOT / '_extensions' / 'house'))

from aggregate import build                                     # noqa: E402
from house import GT, figure, figures, show                     # noqa: E402

figures()

__all__ = ['BOOK_ROOT', 'GT', 'build', 'figure', 'show']

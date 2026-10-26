"""The pure-Python SciPy fallback backend.

Runs a subprocess with an import hook that blocks ``eltq._eltq``, so the
package must serve its API through ``eltq.reference`` — as it does for users
without the compiled extension.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from conftest import FIXTURES_DIR

FALLBACK_SCRIPT = textwrap.dedent(
    """
    import sys

    class BlockRustExtension:
        def find_spec(self, name, path=None, target=None):
            if name == "eltq._eltq":
                raise ImportError("blocked for fallback test")
            return None

    sys.meta_path.insert(0, BlockRustExtension())

    import math
    import numpy as np
    import eltq

    assert eltq.BACKEND == "reference", eltq.BACKEND
    assert eltq.__rust_version__ is None

    raw, secondary, meta = eltq.quantize_arrays(
        [0.01, 0.02], [100.0, 50.0], [40.0, 20.0], [15.0, 5.0], [0.0, 0.0],
        bucket_size=0.5, event_ids=["a", "b"])
    assert math.isclose(raw.sum(), 0.03, rel_tol=1e-12)
    assert math.isclose(secondary.sum(), 0.03, rel_tol=1e-12)
    assert meta.grid_length == 201

    # The model moments (spec section 4B) from the reference.
    moments = eltq.moments_arrays(
        [0.1], [100.0], [50.0], [10.0], [10.0], event_ids=["a"])
    assert math.isclose(moments.raw_moments[1], 290.0, rel_tol=1e-14)
    assert moments.method == "closed_form"

    # CSV fallback route with the canonical fixture.
    raw_c, secondary_c, meta_c = eltq.quantize_csv(
        sys.argv[1], bucket_size=0.5, quadrature_points=8)
    assert meta_c.valid_rows == 16
    assert meta_c.diagnostics["zero_rate"].examples == ["zero_rate"]

    # Typed errors come from the reference hierarchy in this mode.
    try:
        eltq.quantize_arrays([0.01], [100.0], [30.0], [50.0], [0.0],
                             bucket_size=0.5, variance_policy="error")
    except eltq.VarianceError:
        pass
    else:
        raise AssertionError("expected VarianceError")
    try:
        eltq.quantize_csv(sys.argv[1], bucket_size=0.5,
                          columns={"nope": "x"})
    except eltq.EltqError:
        pass
    else:
        raise AssertionError("expected EltqError for unknown column key")
    print("fallback-ok")
    """
)


def test_fallback_backend_serves_the_api():
    result = subprocess.run(
        [sys.executable, "-c", FALLBACK_SCRIPT,
         str(FIXTURES_DIR / "small_elt.csv")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "fallback-ok" in result.stdout

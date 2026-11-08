"""Every number in the text of the paper equals the value recomputed from results/v6.6 (convoy.claims)."""
import os
from pathlib import Path

import pytest

from convoy import claims


@pytest.mark.parametrize("claim", claims.CLAIMS, ids=lambda c: f"{c.where}: {c.value}"[:80])
def test_claim(claim):
    assert claim.compute(claims.Results()) == claim.value, claim.quote


def test_claims_are_distinct():
    assert len({c.quote for c in claims.CLAIMS}) == len(claims.CLAIMS)
    assert len(claims.CLAIMS) >= 140


@pytest.mark.skipif(not os.environ.get("CONVOY_PAPER"), reason="set CONVOY_PAPER=path/to/main.tex to check the quotes")
def test_every_quote_is_in_the_paper():
    tex = Path(os.environ["CONVOY_PAPER"]).read_text(encoding="utf-8")
    assert [c.quote for c in claims.quotes_in(tex)] == []

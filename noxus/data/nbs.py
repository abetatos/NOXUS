"""NBS (National Bureau of Statistics of China) feasibility probe for the Arm-B benchmark (NOX-016).

Arm A of the scale contrast scores the NO2 features against CREA's national steel series, which are
already committed. That isolates spatial SCALE cleanly -- same variable, same publisher, same
cadence as the Tangshan target -- but it leaves every target sourced from CREA/WIND, so it cannot
break the feature-versus-target ambiguity NOX-011 failed to resolve.

Arm B would add an INDEPENDENT publisher: NBS monthly crude-steel output for Hebei and for the
nation. REQ-030 requires a live feasibility probe before any fetcher is built, and REQ-031 forbids
substituting a licence- or pay-gated source if the free route is unavailable -- the same discipline
that dropped VIIRS Nightfire in NOX-011.

This module implements the probe only. There is no fetcher, because the probe returns NO-GO: see
:func:`probe_nbs`.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# The classic NBS query endpoint (``easyquery.htm``) and the query API behind the redesigned
# data portal SPA. Static assets on the same host serve normally, so a failure here is an access
# policy on the DATA endpoints specifically, not an outage.
NBS_ROUTES: tuple[tuple[str, str], ...] = (
    (
        "easyquery-national-monthly",
        "https://data.stats.gov.cn/easyquery.htm?m=QueryData&dbcode=hgyd&rowcode=zb&colcode=sj"
        '&wds=[]&dfwds=[{"wdcode":"sj","valuecode":"LAST13"}]',
    ),
    (
        "easyquery-province-monthly",
        "https://data.stats.gov.cn/easyquery.htm?m=QueryData&dbcode=fsyd&rowcode=zb&colcode=sj"
        '&wds=[{"wdcode":"reg","valuecode":"130000"}]'
        '&dfwds=[{"wdcode":"sj","valuecode":"LAST13"}]',
    ),
    ("portal-query-api", "https://data.stats.gov.cn/dg/api/query/data"),
)
CONTROL_URL = "https://data.stats.gov.cn/dg/website/page.html"  # static asset: host reachability

USER_AGENT = "Mozilla/5.0 (compatible; NOXUS/1.0; +https://github.com/abetatos/NOXUS)"


@dataclass(frozen=True)
class NbsProbeResult:
    """Outcome of the Arm-B feasibility gate (REQ-030)."""

    go: bool
    host_reachable: bool
    routes: tuple[dict, ...] = field(default_factory=tuple)
    note: str = ""

    def render(self) -> str:
        lines = [
            "NOX-016 Arm B - NBS feasibility probe (REQ-030)",
            "=" * 78,
            f"VERDICT: {'GO' if self.go else 'NO-GO'}",
            f"  host reachable (static asset): {self.host_reachable}",
            "",
            "Routes tried",
            "-" * 78,
        ]
        for r in self.routes:
            lines.append(f"  [{r['status']}] {r['name']}")
            lines.append(f"        {r['url'][:110]}")
            if r.get("detail"):
                lines.append(f"        {r['detail']}")
        lines += ["", "Interpretation", "-" * 78, "  " + self.note]
        return "\n".join(lines) + "\n"


def _try(url: str, timeout: float = 20.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https hosts
            body = resp.read(4096)
            ok = resp.status == 200 and body.lstrip()[:1] in (b"{", b"[")
            return {
                "status": resp.status,
                "json": ok,
                "detail": "" if ok else "200 but the body is not JSON (SPA shell or error page)",
            }
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "json": False, "detail": f"HTTP error: {exc.reason}"}
    except Exception as exc:  # noqa: BLE001 - a probe reports failures, it does not raise
        return {"status": "ERR", "json": False, "detail": f"{type(exc).__name__}: {exc}"}


def probe_nbs(out_path: Path | None = None, timeout: float = 20.0) -> NbsProbeResult:
    """Resolve LIVE whether NBS steel output is reachable free and programmatically (REQ-030).

    Returns GO only if some route returns parseable JSON. A NO-GO is a valid, designed-for outcome
    (ERR-003): the caller reports Arm A alone and records the provenance gap. It must NOT fall back
    to a licence- or pay-gated provider (REQ-031).
    """
    control = _try(CONTROL_URL, timeout)
    host_reachable = control["status"] == 200

    routes = []
    for name, url in NBS_ROUTES:
        res = _try(url, timeout)
        routes.append({"name": name, "url": url, **res})

    go = any(r["json"] for r in routes)
    if go:
        note = (
            "A free programmatic NBS route responded with JSON. Arm B may proceed: build the "
            "fetcher, ingest Hebei-provincial and national monthly crude steel, and fold those "
            "pairings into the same BH family (REQ-032)."
        )
    elif host_reachable:
        note = (
            "NO-GO. The NBS host serves static assets normally but every DATA query endpoint is "
            "refused, so this is an access policy on the query API (observed: HTTP 403 from the "
            "site WAF), not an outage or a transient failure. There is no free programmatic route "
            "to NBS series from this environment. Per REQ-031 no paywalled substitute is used: the "
            "scale contrast reports Arm A alone and records the shared-publisher provenance gap as "
            "an UNRESOLVED limitation. Re-running this probe from a permitted network may return "
            "GO, in which case Arm B can be added without changing anything else."
        )
    else:
        note = (
            "NO-GO. The NBS host itself was unreachable, so this probe cannot distinguish an "
            "access policy from an outage. Re-run before concluding anything about availability."
        )

    result = NbsProbeResult(go=go, host_reachable=host_reachable, routes=tuple(routes), note=note)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.render(), encoding="utf-8")
    return result


def fetch_nbs_steel(*_args, **_kwargs):
    """Arm-B fetcher. Not implemented: the REQ-030 probe returned NO-GO (see :func:`probe_nbs`).

    Deliberately raises rather than silently returning an empty frame, so a future caller cannot
    mistake "no data" for "no signal" -- and so nobody wires in a paid provider to make it pass.
    """
    raise NotImplementedError(
        "NBS Arm-B ingest is not built: the NOX-016 T1 feasibility probe returned NO-GO (NBS data "
        "endpoints refuse programmatic access from this environment). Run 'noxus scale-contrast "
        "--probe' to re-check; if it returns GO, implement this against the route it reports. Do "
        "not substitute a licence- or pay-gated provider (REQ-031)."
    )

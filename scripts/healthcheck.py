"""Container healthcheck: GET a URL and require an expected HTTP status.

Used instead of curl so the image stays slim (python:3.12-slim ships no curl).

The MCP server's healthy answer is 401, not 200: an unauthenticated request to a
bearer-protected endpoint SHOULD be rejected, and requiring exactly 401 proves an
authenticating MCP server is listening rather than something else that happens to
have taken the port. That distinction is not theoretical - a QuestDB container
from an unrelated project once occupied this stack's port and answered 404, and a
plain "is anything listening?" check reported it as healthy.

Usage:
    python scripts/healthcheck.py <url> [expected_status]
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: healthcheck.py <url> [expected_status]", file=sys.stderr)
        return 2

    url = argv[0]
    expected = int(argv[1]) if len(argv) > 1 else 200

    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except Exception as exc:  # noqa: BLE001 - any failure is an unhealthy container
        print(f"unreachable: {exc}", file=sys.stderr)
        return 1

    if status == expected:
        return 0
    print(f"expected HTTP {expected}, got {status}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

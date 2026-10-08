"""Safety guardrails for the DBOps investigation tools."""

READ_ONLY_HTTP_METHODS = frozenset({"GET", "HEAD"})
FORBIDDEN_TOKENS = frozenset({"delete", "_delete_by_query", "put", "post", "patch", "update", "bulk", "sql"})


def validate_read_only_request(method: str, path: str) -> None:
    """Reject anything that could mutate a production database/search domain."""
    normalized_method = method.upper().strip()
    normalized_path = path.lower().strip()
    if normalized_method not in READ_ONLY_HTTP_METHODS:
        raise PermissionError(f"DBOps is read-only: HTTP method {normalized_method} is not permitted")
    if any(token in normalized_path for token in FORBIDDEN_TOKENS):
        raise PermissionError(f"DBOps is read-only: path {path!r} is not permitted")

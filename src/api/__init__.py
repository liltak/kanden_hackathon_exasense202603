"""ExaSense API package."""

# Legacy in-memory store kept for backward compatibility during migration.
# New code should use database.py (PostgreSQL) and storage.py (MinIO).
# This will be removed once all routes are fully migrated.
_sim_store: dict = {}

"""Dependency injection helpers for FastAPI."""

from pathlib import Path

from fastapi import HTTPException, status

from spindle.api.session import Session, session_manager


def get_session(session_id: str) -> Session:
    """Get a session by ID or raise 404."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )
    return session


def verify_directory_access(dir_path: str, create: bool = False) -> Path:
    """Verify that a directory path is accessible."""
    try:
        path = Path(dir_path).resolve()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid directory path: {str(e)}"
        )

    if not path.exists():
        if create:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to create directory: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Directory not found: {dir_path}"
            )

    if not path.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path is not a directory: {dir_path}"
        )

    return path


def get_temp_dir() -> Path:
    """Get or create a temporary directory for API operations."""
    import tempfile
    temp_dir = Path(tempfile.gettempdir()) / "spindle_api"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir

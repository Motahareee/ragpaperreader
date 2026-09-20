"""PyInstaller entrypoint. Kept at the repo root so the frozen executable's
module name doesn't collide with the `paperchat` package."""
from paperchat.main import main

if __name__ == "__main__":
    main()

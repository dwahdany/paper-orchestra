"""``python -m paper_orchestra``: the same entry point as the ``paper-orchestra`` script."""

from paper_orchestra.cli import main

# The guard matters: batch workers are spawned processes that import this module again.
if __name__ == "__main__":
    main()

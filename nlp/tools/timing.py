"""Platform-dependent time measurement."""

try:
    # Use resource module if available.
    import resource

    def cputime():
        """Return the user CPU time since the start of the process."""
        return resource.getrusage(resource.RUSAGE_SELF)[0]

except:
    # Fall back on time module.
    import sys
    import time

    if sys.platform == "win32":
        # On Windows, the best timer is time.clock()
        def cputime():
            """Return the current processor time."""
            return time.clock()

    else:
        # On most other platforms the best timer is time.time()
        def cputime():
            """Return the current processor time."""
            return time.time()

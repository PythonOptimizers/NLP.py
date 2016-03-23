"""Platform-dependent time measurement."""

try:
    # Use resource module if available.
    import resource

    def cputime():
        """Return the user CPU time since the start of the process."""
        return resource.getrusage(resource.RUSAGE_SELF)[0]
except:
    # Fall back on time.clock().
    import time

    def cputime():
        """Return the current processor time."""
        return time.clock()

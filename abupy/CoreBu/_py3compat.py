"""Python 3.13+ compatibility shim for removed stdlib modules."""
import sys
import os

# imghdr — removed in 3.13
try:
    import imghdr
except ImportError:
    class _ImghdrShim:
        """Minimal imghdr replacement for abu's DLBu usage."""
        def what(self, file, h=None):
            return None
    imghdr = _ImghdrShim()
    sys.modules['imghdr'] = imghdr

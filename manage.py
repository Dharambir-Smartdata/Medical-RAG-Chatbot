#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    # Force IPv4 preference for Linux systems without IPv6 internet routes
    import socket
    try:
        _orig_gai = socket.getaddrinfo
        def _gai(host, port, family=0, type=0, proto=0, flags=0):
            if family == 0:
                family = socket.AF_INET
            return _orig_gai(host, port, family, type, proto, flags)
        socket.getaddrinfo = _gai
    except Exception:
        pass

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()

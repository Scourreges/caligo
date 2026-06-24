"""
User-facing Caligo interface.

This module intentionally mirrors the feel of picaso.justdoit:

    import caligo.justdoit as cdi

    opacity_ck = cdi.opannection(...)
    case = cdi.inputs(calculation="planet", climate=True)
"""

import picaso.justdoit as jdi

from .case import CaligoCase


def opannection(*args, **kwargs):
    """
    Thin wrapper around picaso.justdoit.opannection.

    This lets users do:

        opacity_ck = cdi.opannection(...)

    instead of importing picaso.justdoit separately.
    """
    return jdi.opannection(*args, **kwargs)


def inputs(*args, **kwargs):
    """
    Create a Caligo case object.

    This behaves like a haze-enabled version of picaso.justdoit.inputs.
    """
    return CaligoCase(*args, **kwargs)
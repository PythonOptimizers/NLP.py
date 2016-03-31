"""Metaclasses useful in the context of NLP.py."""

import types
from nlp.tools.decorators import counter


class CountCalls(type):
    """Metaclass that introduces call counters for relevant methods.

    Classes intrumented with this metaclass will see their constructor
    overridden so specified methods (obj, grad, hess, etc.) are wrapped with
    a call counter decorator. After instantiation, the number of calls issued
    so far to ``obj`` is available as ``obj.ncalls``.
    """

    def __new__(mcs, cls, bases, attrs):
        orig_init = attrs.get("__init__", None)

        meths = ["obj", "grad", "hess", "cons", "icons", "igrad", "sigrad",
                 "jac", "jprod", "jtprod", "hprod", "hiprod", "ghivprod"]
        to_count = [meth for meth in meths if meth in attrs.keys()]

        def init_wrapper(self, *args, **kwargs):
            if orig_init:
                orig_init(self, *args, **kwargs)

            for meth in to_count:
                new_meth = counter(types.MethodType(attrs[meth], self, cls))
                setattr(self, meth, new_meth)

        if orig_init:
            init_wrapper.__doc__ = orig_init.__doc__

        attrs["__init__"] = init_wrapper
        return super(CountCalls, mcs).__new__(mcs, cls, bases, attrs)

# License: MIT

import sys
import pkgutil
import inspect
import importlib
from collections import OrderedDict


def find_components(package, directory, base_class):
    components = OrderedDict()

    for module_loader, module_name, ispkg in pkgutil.iter_modules([directory]):
        full_module_name = "%s.%s" % (package, module_name)
        if full_module_name not in sys.modules and not ispkg:
            module = importlib.import_module(full_module_name)

            for member_name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, base_class) and \
                        obj != base_class:
                    # Keep in mind that this only instantiates the ensemble_wrapper,
                    # but not the real target classifier
                    classifier = obj
                    components[obj.__name__] = classifier

    return components

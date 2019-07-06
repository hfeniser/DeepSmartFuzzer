import importlib, os

spec = importlib.util.find_spec("pyflann")
pyflann_dir = os.path.dirname(spec.origin)

os.system("2to3 -w " + pyflann_dir)
from setuptools import setup, Extension
import numpy

pym4ri_src = "../cpp/src"   # Path to pym4ri source
m4ri_include_dir = "../m4ri"  # Path to m4ri includes
m4ri_lib_dir = "../m4ri/.libs"  # Path to m4ri libraries

module = Extension(
    "pym4ri",
    sources=[f"{pym4ri_src}/pym4ri.cpp"],
    include_dirs=[m4ri_include_dir, numpy.get_include()],
    library_dirs=[m4ri_lib_dir],
    libraries=["m4ri"],
    extra_compile_args=["-O3"],  
    extra_link_args=[f"-Wl,-rpath,{m4ri_lib_dir}"],
)

setup(
    name="pym4ri",
    version="1.0",
    ext_modules=[module],
)

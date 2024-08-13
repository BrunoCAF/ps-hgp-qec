# Compiling

In order to compile the ```pym4ri.cpp``` source file, run the command

- ```g++ -I../m4ri -I$(python -c "import numpy; print(numpy.get_include())") $(python3-config --cflags --ldflags) -L../m4ri/.libs -Wl,-rpath,../m4ri/.libs -o build/pym4ri.o src/pym4ri.cpp -lm4ri```

when inside the ```ps-hgp-qec/cpp``` directory. For the compilation to be successful, you need to have the original m4ri repo installed on the same level, like:

```bash
ps-hgp-qec/
|-- cpp/
|-- |-- src/
...
|-- m4ri/
...
```

It also works if you just copy all the header files and the .libs directories.

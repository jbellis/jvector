#!/bin/bash

gcc -fPIC -O3 -march=native -shared -o libjvector.so jvector_simd.c

# Generate Java source code
# Should only be run when c header changes
jextract --source \
  --output ../java \
  -t io.github.jbellis.jvector.vector.cnative \
  -I . \
  -l jvector \
  --header-class-name NativeSimdOps \
  jvector_simd.h
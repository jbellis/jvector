# JMH Benchmarks
Micro benchmarks for jVector. While {@link Bench.java} is about recall, the JMH benchmarks
are mostly targeting scalability and latency aspects.

## Building and running the benchmark

1. You can build and then run
```shell
mvn clean install
java --enable-native-access=ALL-UNNAMED \
  --add-modules=jdk.incubator.vector \
  -XX:+HeapDumpOnOutOfMemoryError \
  -Xmx14G -Djvector.experimental.enable_native_vectorization=true \
  -jar benchmarks-jmh/target/benchmarks-jmh-4.0.0-beta.2-SNAPSHOT.jar 
```
2. Build and run the benchmarks directly
```shell
mvn clean package -pl :benchmarks-jmh -am -DskipTests=true exec:exec@benchmark -Pjmh
```
3. Build and run a specific benchmark or use command line arguments
```shell
# Run only benchmarks containing "Sample" in their name
mvn clean package -pl :benchmarks-jmh -am -DskipTests=true exec:exec@benchmark -Pjmh -DargLine=".*Sample.*"

# Run with different parameters
mvn clean package -pl :benchmarks-jmh -am -DskipTests=true exec:exec@benchmark -Pjmh -DargLine="-f 1 -wi 5 -i 3 .*Sample.*"
```

Common JMH command line options you can use in the configuration or command line:
- `-f <num>` - Number of forks
- `-wi <num>` - Number of warmup iterations
- `-i <num>` - Number of measurement iterations
- `-w <time>` - Warmup time per iteration
- `-r <time>` - Measurement time per iteration
- `-t <num>` - Number of threads
- `-p <param>=<value>` - Benchmark parameters
- `-prof <profiler>` - Add profiler

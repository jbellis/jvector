# JVector 
JVector is a pure Java, zero dependency, embedded vector search engine, used by DataStax Astra DB and Apache Cassandra.

What is JVector?
- Algorithmic-fast. JVector uses dstate of the art graph algorithms inspired by DiskANN and related research that offer high recall and low latency.
- Implementation-fast. JVector uses the Panama SIMD API to accelerate index build and queries.
- Memory efficient. JVector compresses vectors using product quantization so they can stay in memory during searches.  (As part of our PQ implementation, our SIMD-accelerated kmeans implementation is 3x faster than Apache Commons Math.)
- Disk-aware. JVector’s disk layout is designed to do the minimum necessary iops at query time.
- Concurrent.  Index builds scale linearly to at least 32 threads.  Double the threads, half the build time.
- Incremental. Query your index as you build it.  No delay between adding a vector and being able to find it in search results.
- Easy to embed. API designed for easy embedding, by people using it in production.

Just add org.github.jbellis.jvector as a dependency and you’re off to the races.

## Developing and Testing

You can run SiftSmall and Bench directly to get an idea of what all is going on here. Bench 
requires some datasets to be downloaded from https://github.com/erikbern/ann-benchmarks. The files used by SiftSmall can be found in the siftsmall directory in the project root. 

To run either class, you can use the Maven exec-plugin via the following incantations:
```mvn clean install exec:exec@bench``` 
or for Sift:
```mvn clean install exec:exec@sift```

To compile for a specific JDK, you can use the targeted execution defined in the pom:
- `mvn clean compiler:compile@jdk11` for JDK 11
- `mvn clean compiler:compile@jdk20` for JDK 20

Similar to the compile executions, a JAR file can be generated via:
- `mvn jar:jar@jar-jdk11` for JDK 11
- `mvn jar:jar@jar-jdk20` for JDK 20

In both cases, you must have invoked the compile target for that specific JDK, or the resulting jar file will be empty.  
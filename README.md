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
```mvn exec:exec@bench``` 
or for Sift:
```mvn exec:exec@sift```
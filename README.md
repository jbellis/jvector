# JVector 
JVector is a pure Java, zero dependency, embedded vector search engine, used by DataStax Astra DB and (soon) Apache Cassandra.

What is JVector?
- Algorithmic-fast. JVector uses dstate of the art graph algorithms inspired by DiskANN and related research that offer high recall and low latency.
- Implementation-fast. JVector uses the Panama SIMD API to accelerate index build and queries.
- Memory efficient. JVector compresses vectors using product quantization so they can stay in memory during searches.  (As part of our PQ implementation, our SIMD-accelerated kmeans implementation is 3x faster than Apache Commons Math.)
- Disk-aware. JVector’s disk layout is designed to do the minimum necessary iops at query time.
- Concurrent.  Index builds scale linearly to at least 32 threads.  Double the threads, half the build time.
- Incremental. Query your index as you build it.  No delay between adding a vector and being able to find it in search results.
- Easy to embed. API designed for easy embedding, by people using it in production.


## JVector performance, visualized
JVector vs Lucene searching the Deep100M dataset (about 35GB of vectors and 25GB index):
![Screenshot from 2023-09-14 18-06-26](https://github.com/jbellis/jvector/assets/42158/217f43aa-9a7e-4f77-b32d-9b9d736af179)

JVector scales updates linearly to at least 32 threads:
![Screenshot from 2023-09-14 18-05-15](https://github.com/jbellis/jvector/assets/42158/f0127bfc-6c45-48b9-96ea-95b2120da0d9)

## JVector basics
Adding to your project. Replace `${latest-version}` with ![Maven Central](https://img.shields.io/maven-central/v/io.github.jbellis/jvector?color=green). Example `<version>1.0.0</version>`:
```
<dependency>        
    <groupId>io.github.jbellis</groupId>          
    <artifactId>jvector</artifactId>
    <!-- Use the latest version from https://central.sonatype.com/artifact/io.github.jbellis/jvector -->
    <version>${latest-version}</version>
</dependency>
```
Building the index:
- [`GraphIndexBuilder`](./jvector-base/src/main/java/io/github/jbellis/jvector/graph/GraphIndexBuilder.java) is the entry point for building a graph.  You will need to implement
  [`RandomAccessVectorValues`](./jvector-base/src/main/java/io/github/jbellis/jvector/graph/RandomAccessVectorValues.java) to provide vectors to the builder;
  [`ListRandomAccessVectorValues`](./jvector-base/src/main/java/io/github/jbellis/jvector/graph/ListRandomAccessVectorValues.java) is a good starting point.
- If all your vectors
  are in the provider
  up front, you can just call `build()` and it will parallelize the build across
  all available cores.  Otherwise you can call `addGraphNode` as you add vectors; 
  this is non-blocking and can be called concurrently from multiple threads.
- Call `GraphIndexBuilder.complete` when you are done adding vectors.  This will
  optimize the index and make it ready to write to disk.  (Graphs that are
  in the process of being built can be searched at any time; you do not have to call
  *complete* first.)
Searching the index:
- [`GraphSearcher`](./jvector-base/src/main/java/io/github/jbellis/jvector/graph/GraphSearcher.java) is the entry point for searching.  Results come back as a [`SearchResult`](./jvector-base/src/main/java/io/github/jbellis/jvector/graph/SearchResult.java) object that contains node IDs and scores, in
  descending order of similarity to the query vector.  `GraphSearcher` objects are re-usable,
  so unless you have a very simple use case you should use `GraphSearcher.Builder` to
  create them; `GraphSearcher::search` is also available with simple defaults, but calling it
  will instantiate a new `GraphSearcher` every time so performance will be worse.
- JVector represents vectors in the index as the ordinal (int) corresponding to their
  index in the `RandomAccessVectorValues` you provided.  You can get the original vector
  back with `GraphIndex.getVector`, if necessary, but since this is a disk-backed index
  you should design your application to avoid doing so if possible.

## DiskANN and Product Quantization 
JVector implements [DiskANN](https://suhasjs.github.io/files/diskann_neurips19.pdf)-style 
search, meaning that vectors can be compressed using product quantization so that searches
can be performed using the compressed representation that is kept in memory.  You can enable
this with the following steps:
- Create a [`ProductQuantization`](./jvector-base/src/main/java/io/github/jbellis/jvector/pq/ProductQuantization.java) object with your vectors.  This will take some time
  to compute the codebooks.
- Use `ProductQuantization.encode` or `encodeAll` to encode your vectors.
- Create a [`CompressedVectors`](./jvector-base/src/main/java/io/github/jbellis/jvector/disk/CompressedVectors.java) object from the encoded vectors.
- Create a [`NeighborSimilarity.ApproximateScoreFunction`](./jvector-base/src/main/java/io/github/jbellis/jvector/graph/NeighborSimilarity.java) for your query that uses the
  `ProductQuantization` object and `CompressedVectors` to compute scores, and pass this
  to the `GraphSearcher.search` method.

## Saving and loading indexes
- [`OnDiskGraphIndex`](./jvector-base/src/main/java/io/github/jbellis/jvector/disk/OnDiskGraphIndex.java) and [`CompressedVectors`](./jvector-base/src/main/java/io/github/jbellis/jvector/disk/CompressedVectors.java) have `write()` methods to save state to disk.
  They initialize from disk using their constructor and `load()` methods, respectively.
  Writing just requires a DataOutput, but reading requires an 
  implementation of [`RandomAccessReader`](./jvector-base/src/main/java/io/github/jbellis/jvector/disk/RandomAccessReader.java) to wrap your
  preferred i/o class for best performance. See MappedRandomAccessReader for an example.
- Building a graph does not technically require your RandomAccessVectorValues object
  to live in memory, but it will perform much better if it does.  OnDiskGraphIndex,
  by contrast, is designed to live on disk and use minimal memory otherwise.
- You can optionally wrap `OnDiskGraphIndex` in a [`CachingGraphIndex`](./jvector-base/src/main/java/io/github/jbellis/jvector/disk/CachingGraphIndex.java) to keep the most commonly accessed
  nodes (the ones nearest to the graph entry point) in memory.

## Sample code
- The [`SiftSmall`](./jvector-examples/src/main/java/io/github/jbellis/jvector/example/SiftSmall.java) class demonstrates how to put all of the above together to index and search the
  "small" SIFT dataset of 10,000 vectors.
- The [`Bench`](./jvector-examples/src/main/java/io/github/jbellis/jvector/example/Bench.java) class performs grid search across the `GraphIndexBuilder` parameter space to find
  the best tradeoffs between recall and throughput.  You can use [`plot_output.py`](./plot_output.py) to graph the [pareto-optimal
  points](https://en.wikipedia.org/wiki/Pareto_efficiency) found by `Bench`.

## Developing and Testing
This project is organized as a [multimodule Maven build](https://maven.apache.org/guides/mini/guide-multiple-modules.html). The intent is to produce a multirelease jar suitable for use as
a dependency from any Java 11 code. When run on a Java 20+ JVM with the Vector module enabled, optimized vector 
providers will be used. In general, the project is structured to be built with JDK 20+, but when `JAVA_HOME` is set to
Java 11 -> Java 19, certain build features will still be available.

Base code is in [jvector-base](./jvector-base) and will be built for Java 11 releases, restricting language features and APIs
appropriately. Code in [jvector-twenty](./jvector-twenty) will be compiled for Java 20 language features/APIs and included in the final
multirelease jar targetting supported JVMs. [jvector-multirelease](./jvector-multirelease) packages [jvector-base](./jvector-base) and [jvector-twenty](./jvector-twenty) as a
multirelease jar for release. [jvector-examples](./jvector-examples) is an additional sibling module that uses the reactor-representation of
jvector-base/jvector-twenty to run example code.

You can run `SiftSmall` and `Bench` directly to get an idea of what all is going on here. `Bench`
requires some datasets to be downloaded from https://github.com/erikbern/ann-benchmarks. The files used by `SiftSmall`
can be found in the [siftsmall directory](./siftsmall) in the project root.

To run either class, you can use the Maven exec-plugin via the following incantations:

> `mvn compile exec:exec@bench`

or for Sift:

> `mvn compile exec:exec@sift`

To run Sift/Bench without the JVM vector module available, you can use the following invocations:

> `mvn -Pjdk11 compile exec:exec@bench`

> `mvn -Pjdk11 compile exec:exec@sift`

The `... -Pjdk11` invocations will also work with `JAVA_HOME` pointing at a Java 11 installation.

To release, configure `~/.m2/settings.xml` to point to OSSRH and run `mvn -Prelease clean deploy`.

---
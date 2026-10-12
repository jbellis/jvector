# Deferred batch serialization

The parallel graph writer can serialize complete layer-zero records into bounded
worker buffers. Each record contains its ordinal, inline features, neighbor count,
and adjacency list. By default, a worker writes up to 64 complete records together.
The final partial batch is written without adding records or padding to the format.
This preserves the V6 layout and existing readers.

## Supplying features at final write

Provide the final feature suppliers to `write` after graph construction and cleanup.
Feature encoding then happens during serialization, rather than through a
synchronized `writeFeaturesInline` call on each insertion. This applies to NVQ,
full-precision inline vectors, fused PQ, and other features implementing the existing
`Feature` interface. The core writer does not choose a quantizer or retain input
vectors on the application's behalf.

```java
try (var writer = new OnDiskParallelGraphIndexWriter.Builder(graph, outputPath)
        .withParallelWorkerThreads(workerCount)
        .with(new InlineVectors(vectors.dimension()))
        .build()) {
    var localVectors = vectors.threadLocalSupplier();
    writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
            ordinal -> new InlineVectors.State(localVectors.get().getVector(ordinal))));
}
```

The source must remain available until `write` finishes. Suppliers must support
concurrent calls; a source that reuses a scratch vector needs an independent view
per worker. Feature state is consumed while serializing its record.

Applications can continue using `writeFeaturesInline` before the final write.
If a feature supplier is absent, its existing bytes are preserved and that write
uses the component-write path. Complete-record batching requires all inline feature
suppliers; it does not overwrite prewritten features to enable batching.

## Batch size and memory

The default is 64 records. To override it, call `withParallelBatchSize(records)`
before `with(...)` on either `OnDiskParallelGraphIndexWriter.Builder` or
`RandomAccessOnDiskGraphIndexWriter.Builder`. The value must be positive. Each active
worker holds one buffer of `recordSize * records` bytes; feature encoders and vector
sources have their own memory costs. No dataset-sized output buffer is allocated.

The unified `RandomAccessOnDiskGraphIndexWriter.Builder` retains its existing
parallel/sequential selection. Setting the batch size does not enable parallel
writing, and is ignored by the sequential writer. A caller-supplied executor
remains caller-owned and must outlive the write.

## Dataset prefetching in Grid

Grid defers its inline NVQ or FP feature serialization until after cleanup. Its final
serialization suppliers consume worker-local ranges through `VectorAccess.openRange`.
Sources implementing the buffered-access capability use that implementation;
ordinary sources retain the ordinary-access fallback. Borrowed vectors are consumed
before advancing a cursor, and ranges are released after the workers finish.

This integrates with #744's optional `useDatasetPrefetching` setting. It does not
change construction scoring, quantization parameters, or the query path. Production
integrations can opt into deferred serialization by supplying their features at the
final write; their existing vector-source and incremental-write APIs remain valid.

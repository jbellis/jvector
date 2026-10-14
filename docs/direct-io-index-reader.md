# Direct I/O for immutable disk indexes

JVector can read a published on-disk index through Linux `O_DIRECT` instead of
mapping it through the OS page cache. This is an explicit reader choice; the
existing mapped reader remains the default.

## Open an index

Use JDK 22 or newer with the JVector native module available and
`--enable-native-access=ALL-UNNAMED`.

```java
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import java.nio.file.Path;

// Eight concurrent reads per view; no shared block cache.
try (var readers = ReaderSupplierFactory.openDirect(Path.of("index.jvector"), 8);
     var index = OnDiskGraphIndex.load(readers)) {
    // Use the index with the usual GraphSearcher and scoring APIs.
}
```

The reader is independent of the SIMD provider. Choosing direct I/O does not
select Google Highway or Panama.

## Batching and memory

`ioDepth` accepts 0 for synchronous reads or 1–32 for Linux kernel AIO. The
single-argument factory defaults to 8. Each view has a 64 KiB aligned read
buffer; AIO adds up to `ioDepth * 64 KiB` of native buffers. A batch submits
known candidate records concurrently and waits for completion. Search visits,
score calculations, candidate ordering, filters and resume semantics are
unchanged.

Disk records keep their existing layout. Reads expand to 4 KiB boundaries in
memory; padding is not added to the file. Records larger than the bounded
buffer are read in chunks. Existing V2–V6 formats, inline and separated vectors,
NVQ and FusedPQ use the same reader.

Upper-layer adjacency and the primary FusedPQ codes associated with those
layers retain their existing resident representation. Loading that data
bypasses the optional shared block cache.

## Optional shared block cache

The third factory argument specifies a per-supplier cache budget:

```java
var readers = ReaderSupplierFactory.openDirect(path, 8, 64L << 20);
```

Zero disables the cache. Nonzero budgets must be at least 32,896 bytes. The
budget includes block payload and a conservative metadata allowance; it does
not include view buffers, kernel AIO contexts or resident graph layers. Cache
payload is allocated outside the Java heap.

This bounded cache stores immutable 4 KiB index blocks, with replacement on
demand. It admits reads consumed by a view, rather than unused speculative
batches. It caches neither query results nor scores across queries.
`LinuxDirectReader.Supplier.statistics()` reports physical read requests,
submissions, cache copy hits/misses, evictions and allocated buffer budgets.
Live counters are approximate; read them after workers stop for an exact
snapshot.

## File lifetime and platform support

Use one supplier per immutable published index. All its views share a pinned
file descriptor and separate cursors/buffers. Replacing the pathname creates a
new generation: open a new supplier for it. Do not truncate or modify a file
while a supplier is open. Mutable compaction output should be finalized before
opening a cached supplier.

Each reader/view belongs to one thread. Stop readers before closing the
supplier; closing it releases its outstanding views, descriptor and cache.

Direct I/O requires little-endian x86_64 or AArch64 Linux and a filesystem that
supports aligned `O_DIRECT` reads. Unsupported configurations fail explicitly;
there is no silent mapped fallback. The implementation uses Java's foreign
function API and Linux kernel AIO directly, without a libaio dependency.

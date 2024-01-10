# Upgrading from 2.0.x to 3.0.x

## Primary API changes

- `GraphIndexBuilder` `M` parameter now represents the maximum degree of the graph,
  instead of half the maximum degree.  (The former behavior was motivated by making
  it easy to make apples-to-apples comparisons with Lucene HNSW graphs.)  So,
  if you were building a graph of M=16 with JVector2, you should build it with M=32
  with JVector3.
- `NodeSimilarity.ReRanker` renamed to `Reranker`
- `NodeSimilarity.Reranker` api has changed.  The interface is no longer parameterized,
  and the `similarityTo` method no longer takes a Map parameter (provided by `search` with
  the full vectors associated with the nodes returned).  This is because we discovered that
  (in contrast with the original DiskANN design) it is more performant to read vectors lazily 
  from disk at reranking time, since this will only have to fetch vectors for the topK nodes 
  instead of all nodes visited.

## Other changes to public classes

- `OnHeapGraphIndex::ramBytesUsedOneNode` no longer takes an `int nodeLevel` parameter

# Upgrading from 1.0.x to 2.0.x

## New features

- In-graph deletes are supported through `GraphIndexBuilder.markNodeDeleted`.  Deleted nodes
  are removed when `GraphIndexBuilder.cleanup` is called (which is not threadsafe wrt other concurrent changes).
  To write a graph with deleted nodes to disk, a `Map` must be supplied indicating what ordinals
  to change the remaining node ids to -- on-disk graphs may not contain "holes" in the ordinal sequence.
- `GraphSearcher.search` now has an experimental overload that takes a
  `float threshold` parameter that may be used instead of topK; (approximately) all the nodes with simlarities greater than the given threshold will be returned.
- Binary Quantization is available as an alternative to Product Quantization. Our tests show that it's primarily suitable for ada002 embedding vectors and loses too much accuracy with smaller embeddings.

## Primary API changes

- `GraphIndexBuilder.complete` is now `cleanup`.
- The `Bits` parameter to `GraphSearcher.search` is no longer nullable;
  pass `Bits.ALL` instead of `null` to indicate that all ordinals are acceptable.

## Other changes to public classes

- `NeighborQueue`, `NeighborArray`, and `NeighborSimilarity` have been renamed to
  `NodeQueue`, `NodeArray`, and `NodeSimilarity`, respectively.

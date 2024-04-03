package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.io.UncheckedIOException;

public abstract class OnDiskView<T extends GraphCache.CachedNode> implements GraphIndex.RerankingView {
    private final OnDiskGraphIndex index;
    protected final RandomAccessReader reader;
    private final int[] neighbors;

    public OnDiskView(RandomAccessReader reader, OnDiskGraphIndex index) {
        this.index = index;
        this.reader = reader;
        this.neighbors = new int[index.maxDegree];
    }

    protected abstract long neighborsOffsetFor(int node);

    public NodesIterator getNeighborsIterator(int node) {
        try {
            reader.seek(neighborsOffsetFor(node));
            int neighborCount = reader.readInt();
            assert neighborCount <= index.maxDegree : String.format("Node %d neighborCount %d > M %d", node, neighborCount, index.maxDegree);
            reader.read(neighbors, 0, neighborCount);
            return new NodesIterator.ArrayNodesIterator(neighbors, neighborCount);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public int size() {
        return index.size();
    }

    @Override
    public int entryNode() {
        return index.entryNode;
    }

    @Override
    public Bits liveNodes() {
        return Bits.ALL;
    }

    @Override
    public void close() throws IOException {
        reader.close();
    }

    abstract T loadCachedNode(int node, int[] neighbors);

    public abstract ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector, VectorSimilarityFunction similarity);

    abstract GraphIndex.RerankingView cachedWith(GraphCache<T> cache);
}

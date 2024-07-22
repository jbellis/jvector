package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.graph.AcceleratedIndex;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.io.UncheckedIOException;

import static io.github.jbellis.jvector.pq.KMeansPlusPlusClusterer.UNWEIGHTED;

public class CagraIndex implements AcceleratedIndex.ExternalIndex {
    GraphIndex index;
    PQVectors pqv;
    ExplicitThreadLocal<GraphSearcher> searchers = ExplicitThreadLocal.withInitial(() -> new GraphSearcher(index));

    public CagraIndex(DataSet dataset) {
        // Euclidean is the only distance metric supported by CAGRA so that's hardcoded here
        var ravv = dataset.getBaseRavv();
        try (var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.EUCLIDEAN, 32, 100, 1.2f, 1.2f)) {
            index = builder.build(ravv);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }

        var pq = (ProductQuantization) Grid.getCompressor(ds -> new CompressorParameters.PQParameters(ds.getDimension() / 16, 256, true, UNWEIGHTED), dataset);
        var encoded = pq.encodeAll(ravv);
        pqv = new PQVectors(pq, encoded);
    }

    public int size() {
        return index.size();
    }

    @Override
    public NodesIterator search(VectorFloat<?> query, int rerankK) {
        var searcher = searchers.get();
        var asf = pqv.precomputedScoreFunctionFor(query, VectorSimilarityFunction.EUCLIDEAN);
        var ssp = new SearchScoreProvider(asf);
        var sr = searcher.search(ssp, rerankK, Bits.ALL);
        return new NodesIterator(sr.getNodes().length) {
            private int index = 0;

            @Override
            public int nextInt() {
                return sr.getNodes()[index++].node;
            }

            @Override
            public boolean hasNext() {
                return index < size;
            }
        };
    }
}

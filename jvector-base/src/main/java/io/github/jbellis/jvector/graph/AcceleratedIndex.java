package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.Arrays;
import java.util.function.Function;

public class AcceleratedIndex {
    ExternalIndex index;
    ExternalReranker reranker;

    public AcceleratedIndex(ExternalIndex index, Function<VectorFloat<?>, ScoreFunction.ExactScoreFunction> rerankerProvider) {
        this.index = index;
        this.reranker = new ExternalReranker(rerankerProvider);
    }

    public SearchResult search(VectorFloat<?> query, int topK, int rerankK) {
        var candidates = index.search(query, rerankK);
        return reranker.rerank(query, candidates, topK);
    }

    private static class ExternalReranker {
        private final Function<VectorFloat<?>, ScoreFunction.ExactScoreFunction> rerankerProvider;

        public ExternalReranker(Function<VectorFloat<?>, ScoreFunction.ExactScoreFunction> rerankerProvider) {
            this.rerankerProvider = rerankerProvider;
        }

        public SearchResult rerank(VectorFloat<?> query, NodesIterator candidates, int topK) {
            SearchResult.NodeScore[] scored = new SearchResult.NodeScore[candidates.size()];
            var rr = rerankerProvider.apply(query);
            for (int i = 0; i < candidates.size(); i++) {
                var node = candidates.next();
                scored[i] = new SearchResult.NodeScore(node, rr.similarityTo(node));
            }
            Arrays.sort(scored, (a, b) -> Float.compare(b.score, a.score));
            var topKScored = Arrays.copyOf(scored, Math.min(topK, scored.length));
            return new SearchResult(topKScored, 0, scored.length, Float.POSITIVE_INFINITY);
        }
    }

    public int size() {
        return index.size();
    }

    public interface ExternalIndex {
        NodesIterator search(VectorFloat<?> query, int rerankK);

        // useful for sanity checks
        int size();

        void save(String filename);
    }
}

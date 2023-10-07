package io.github.jbellis.jvector.microbench;


import java.io.IOException;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.pq.CompressedVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

@Warmup(iterations = 1, time = 5)
@Measurement(iterations = 1, time = 10)
@Fork(warmups = 0, value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})//"-XX:+UnlockDiagnosticVMOptions", "--enable-preview", "-XX:+PreserveFramePointer", "-XX:+DebugNonSafepoints", "-XX:+AlwaysPreTouch", "-Xmx14G", "-Xms14G"})
public class GraphBuildBench {

    private static DataSet loadWikipediaData() throws IOException
    {
        var baseVectors = SiftLoader.readFvecs("fvec/pages_ada_002_100k_base_vectors.fvec");
        var queryVectors = SiftLoader.readFvecs("fvec/pages_ada_002_100k_query_vectors_10k.fvec").subList(0, 10_000);
        var gt = SiftLoader.readIvecs("fvec/pages_ada_002_100k_indices_query_vectors_10k.ivec").subList(0, 10_000);
        var ds = new DataSet("wikipedia",
                VectorSimilarityFunction.DOT_PRODUCT,
                baseVectors,
                queryVectors,
                gt);
        System.out.format("%nWikipedia: %d base and %d query vectors loaded, dimensions %d%n",
                baseVectors.size(), queryVectors.size(), baseVectors.get(0).length);
        return ds;
    }

    @State(Scope.Benchmark)
    public static class Parameters {
        final DataSet ds;
        final ListRandomAccessVectorValues ravv;

        public Parameters() {
            try {
                this.ds = loadWikipediaData();
                this.ravv = new ListRandomAccessVectorValues(ds.baseVectors, ds.baseVectors.get(0).length);
            }
            catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @OutputTimeUnit(TimeUnit.SECONDS)
    public void testGraphBuild(Blackhole bh, Parameters p) {

        int pqDims = p.ravv.dimension() / 4;
        long start = System.nanoTime();
        var pq = ProductQuantization.compute(p.ravv, pqDims, p.ds.similarityFunction == VectorSimilarityFunction.EUCLIDEAN);
        System.out.format("PQ@%s build %.2fs,%n", pqDims, (System.nanoTime() - start) / 1_000_000_000.0);

        start = System.nanoTime();
        var quantizedVectors = pq.encodeAll(p.ds.baseVectors);
        var compressedVectors = new CompressedVectors(pq, quantizedVectors);
        System.out.format("PQ encoded %d vectors [%.2f MB] in %.2fs,%n", p.ds.baseVectors.size(), (compressedVectors.memorySize()/1024f/1024f) , (System.nanoTime() - start) / 1_000_000_000.0);

        start = System.nanoTime();
        GraphIndexBuilder<float[]> graphIndexBuilder =  new GraphIndexBuilder<>(p.ravv, VectorEncoding.FLOAT32, p.ds.similarityFunction, 8, 40, 1.2f, 1.4f);
        var onHeapGraph = graphIndexBuilder.build();
        var avgShortEdges = IntStream.range(0, onHeapGraph.size()).mapToDouble(i -> onHeapGraph.getNeighbors(i).getShortEdges()).average().orElseThrow();
        System.out.format("Build M=%d ef=%d in %.2fs with %.2f short edges%n",
                8, 60, (System.nanoTime() - start) / 1_000_000_000.0, avgShortEdges);
    }
}


/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.example;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOError;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.disk.CachingGraphIndex;
import io.github.jbellis.jvector.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.example.util.MMapRandomAccessVectorValues;
import io.github.jbellis.jvector.example.util.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.util.UpdatableRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.NodeSimilarity;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.pq.CompressedVectors;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.util.PoolingSupport;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.newsclub.net.unix.AFUNIXServerSocket;
import org.newsclub.net.unix.AFUNIXSocket;
import org.newsclub.net.unix.AFUNIXSocketAddress;

/**
 * Simple local service to use for interaction with JVector over IPC.
 * Only handles a single connection at a time.
 */
public class IPCService
{
    // How each command message is marked as finished
    private static final String DELIM = "\n";

    class SessionContext {
        boolean isBulkLoad = false;

        int dimension;
        int M;
        int efConstruction;
        VectorSimilarityFunction similarityFunction;
        RandomAccessVectorValues<float[]> ravv;
        CompressedVectors cv;
        GraphIndexBuilder<float[]> indexBuilder;
        GraphIndex<float[]> index;
        GraphSearcher<float[]> searcher;
        StringBuffer result = new StringBuffer(1024);
    }

    enum Command {
        CREATE,  //DIMENSIONS SIMILARITY_TYPE M EF\n
        WRITE,  //[N,N,N] [N,N,N]...\n
        BULKLOAD, // /path/to/local/file
        OPTIMIZE, //Run once finished writing
        SEARCH, //EF-search limit [N,N,N] [N,N,N]...\n
        MEMORY, // Memory usage in kb
    }

    enum Response {
        OK,
        ERROR,
        RESULT
    }

    final Path socketFile;
    final AFUNIXServerSocket unixSocket;
    IPCService(Path socketFile) throws IOException {
        this.socketFile = socketFile;
        this.unixSocket = AFUNIXServerSocket.newInstance();
        this.unixSocket.bind(AFUNIXSocketAddress.of(socketFile));
    }

    void create(String input, SessionContext ctx) {
        String[] args = input.split("\\s+");

        if (args.length != 4)
            throw new IllegalArgumentException("Illegal CREATE statement. Expecting 'CREATE [DIMENSIONS] [SIMILARITY_TYPE] [M] [EF]'");

        int dimensions = Integer.valueOf(args[0]);
        VectorSimilarityFunction sim = VectorSimilarityFunction.valueOf(args[1]);
        int M = Integer.valueOf(args[2]);
        int efConstruction = Integer.valueOf(args[3]);

        ctx.ravv = new UpdatableRandomAccessVectorValues(dimensions);
        ctx.indexBuilder =  new GraphIndexBuilder<>(ctx.ravv, VectorEncoding.FLOAT32, sim, M, efConstruction, 1.2f, 1.4f);
        ctx.M = M;
        ctx.dimension = dimensions;
        ctx.efConstruction = efConstruction;
        ctx.similarityFunction = sim;
        ctx.isBulkLoad = false;
    }

    void write(String input, SessionContext ctx) {
        if (ctx.isBulkLoad)
            throw new IllegalStateException("Session is for bulk loading.  To reset call CREATE again");

        String[] args = input.split("\\s+");
        for (int i = 0; i < args.length; i++) {
            String vStr = args[i];
            if (!vStr.startsWith("[") || !vStr.endsWith("]"))
                throw new IllegalArgumentException("Invalid vector encountered. Expecting 'WRITE [F1,F2...] [F3,F4...] ...' but got " + vStr);

            String[] values = vStr.substring(1, vStr.length() - 1).split(",");
            if (values.length != ctx.ravv.dimension())
                throw new IllegalArgumentException(String.format("Invalid vector dimension: %d %d!=%d", i, values.length, ctx.ravv.dimension()));

            float[] vector = new float[ctx.ravv.dimension()];
            for (int k = 0; k < vector.length; k++)
                vector[k] = Float.parseFloat(values[k]);

            ((UpdatableRandomAccessVectorValues)ctx.ravv).add(vector);
            ctx.indexBuilder.addGraphNode(ctx.ravv.size() - 1, ctx.ravv);
        }
    }

    void bulkLoad(String input, SessionContext ctx) {
        String[] args = input.split("\\s+");
        if (args.length != 1)
            throw new IllegalArgumentException("Invalid arguments. Expecting 'BULKLOAD /path/to/local/file'. got " + input);

        File f = new File(args[0]);
        if (!f.exists())
            throw new IllegalArgumentException("No file at: " + f);

        long length = f.length();
        if (length % ((long) ctx.dimension * Float.BYTES) != 0)
            throw new IllegalArgumentException("File is not encoded correctly");

        ctx.index = null;
        ctx.ravv = null;
        ctx.indexBuilder = null;
        ctx.isBulkLoad = true;

        var ravv = new MMapRandomAccessVectorValues(f, ctx.dimension);
        var indexBuilder = new GraphIndexBuilder<>(ravv, VectorEncoding.FLOAT32, ctx.similarityFunction, ctx.M, ctx.efConstruction, 1.2f, 1.4f);
        System.out.println("BulkIndexing " + ravv.size());
        ctx.index = flushGraphIndex(indexBuilder.build(), ravv);
        ctx.cv = pqIndex(ravv, ctx);

        //Finished with raw data we can close/cleanup
        ravv.close();
        ctx.searcher = new GraphSearcher.Builder<>(ctx.index.getView()).build();
    }

    private CompressedVectors pqIndex(RandomAccessVectorValues<float[]> ravv, SessionContext ctx) {
        var pqDims = ctx.dimension > 10 ? Math.max(ctx.dimension / 4, 10) : ctx.dimension;
        long start = System.nanoTime();
        ProductQuantization pq = ProductQuantization.compute(ravv, pqDims, ctx.similarityFunction == VectorSimilarityFunction.EUCLIDEAN);
        System.out.format("PQ@%s build %.2fs,%n", pqDims, (System.nanoTime() - start) / 1_000_000_000.0);
        start = System.nanoTime();

        byte[][] quantizedVectors = new byte[ravv.size()][];
        var ravvCopy = ravv.isValueShared() ? PoolingSupport.newThreadBased(ravv::copy) : PoolingSupport.newNoPooling(ravv);
        PhysicalCoreExecutor.instance.execute(() ->
            IntStream.range(0, quantizedVectors.length).parallel().forEach(i -> {
                try (var pooledRavv = ravvCopy.get()) {
                    var localRavv = pooledRavv.get();
                    quantizedVectors[i] = pq.encode(localRavv.vectorValue(i));
                }
            })
        );

        var cv = new PQVectors(pq, quantizedVectors);
        System.out.format("PQ encoded %d vectors [%.2f MB] in %.2fs,%n", ravv.size(), (cv.ramBytesUsed()/1024f/1024f) , (System.nanoTime() - start) / 1_000_000_000.0);
        return cv;
    }

    private CachingGraphIndex flushGraphIndex(OnHeapGraphIndex<float[]> onHeapIndex, RandomAccessVectorValues<float[]> ravv) {
        try {
            var testDirectory = Files.createTempDirectory("BenchGraphDir");
            var graphPath = testDirectory.resolve("graph.bin");

            try (var outputStream = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(graphPath)))) {
                OnDiskGraphIndex.write(onHeapIndex, ravv, outputStream);
            }
            return new CachingGraphIndex(new OnDiskGraphIndex<>(ReaderSupplierFactory.open(graphPath), 0));
        } catch (IOException e) {
            throw new IOError(e);
        }
    }

    void optimize(SessionContext ctx) {
        if (!ctx.isBulkLoad) {
            if (ctx.ravv.size() > 256) {
                ctx.indexBuilder.cleanup();
                ctx.index = flushGraphIndex(ctx.indexBuilder.getGraph(), ctx.ravv);
                ctx.cv = pqIndex(ctx.ravv, ctx);
                ctx.indexBuilder = null;
                ctx.ravv = null;
            } else { //Not enough data for PQ
                ctx.indexBuilder.cleanup();
                ctx.index = ctx.indexBuilder.getGraph();
                ctx.cv = null;
            }

        }
    }

    String memory(SessionContext ctx) {
        long kb = 0;
        if (ctx.cv != null)
            kb = ctx.cv.ramBytesUsed() / 1024;
        return String.format("%s %d\n", Response.RESULT, kb);
    }

    String search(String input, SessionContext ctx) {
        String[] args = input.split("\\s+");

        if (args.length < 3)
            throw new IllegalArgumentException("Invalid arguments search-ef top-k [vector1] [vector2]...");

        int searchEf = Integer.parseInt(args[0]);
        int topK = Integer.parseInt(args[1]);

        float[] queryVector = new float[ctx.dimension];
        ctx.result.setLength(0);
        ctx.result.append(Response.RESULT);
        int[][] results = new int[args.length - 2][];
        IntStream loopStream = IntStream.range(0, args.length-2);

        //Only use parallel path if we have > 1 core
        if (ForkJoinPool.commonPool().getPoolSize() > 1)
            loopStream = loopStream.parallel();

        loopStream.forEach(i -> {
            String vStr = args[i + 2]; //Skipping first 2 args which are not vectors
            if (!vStr.startsWith("[") || !vStr.endsWith("]"))
                throw new IllegalArgumentException("Invalid query vector encountered:" + vStr);

            String[] values = vStr.substring(1, vStr.length() - 1).split(",");
            if (values.length != ctx.dimension)
                throw new IllegalArgumentException(String.format("Invalid vector dimension: %d!=%d", values.length, ctx.dimension));

            for (int k = 0; k < queryVector.length; k++)
                queryVector[k] = Float.parseFloat(values[k]);

            SearchResult r;
            if (ctx.cv != null) {
                NodeSimilarity.ApproximateScoreFunction sf = ctx.cv.approximateScoreFunctionFor(queryVector, ctx.similarityFunction);
                NodeSimilarity.ReRanker<float[]> rr = (j, vectors) -> ctx.similarityFunction.compare(queryVector, vectors.get(j));
                r = new GraphSearcher.Builder<>(ctx.index.getView()).build().search(sf, rr, searchEf, Bits.ALL);
            } else {
                r = GraphSearcher.search(queryVector, topK, ctx.ravv, VectorEncoding.FLOAT32, ctx.similarityFunction, ctx.index, Bits.ALL);
            }

            var resultNodes = r.getNodes();
            int count = Math.min(resultNodes.length, topK);

            results[i] = new int[count];
            for (int k = 0; k < count; k++)
                results[i][k] = resultNodes[k].node;
        });

        //Format Response
        for (int i = 0; i < results.length; i++) {
            ctx.result.append(" [");
            for (int k = 0; k < results[i].length; k++) {
                if (k > 0) ctx.result.append(",");
                ctx.result.append(results[i][k]);
            }
            ctx.result.append("]");
        }
        ctx.result.append("\n");
        return ctx.result.toString();
    }

    String process(String input, SessionContext ctx) {
        int delim = input.indexOf(" ");
        String command = delim < 1 ? input : input.substring(0, delim);
        String commandArgs = input.substring(delim + 1);
        String response = Response.OK.name();
        switch (Command.valueOf(command)) {
            case CREATE: create(commandArgs, ctx); break;
            case WRITE: write(commandArgs, ctx); break;
            case BULKLOAD: bulkLoad(commandArgs, ctx); break;
            case OPTIMIZE: optimize(ctx); break;
            case SEARCH: response = search(commandArgs, ctx); break;
            case MEMORY: response = memory(ctx); break;
            default: throw new UnsupportedOperationException("No support for: '" + command + "'");
        }
        return response;
    }

    void serve() throws IOException {
        int bufferSize = unixSocket.getReceiveBufferSize();
        byte[] buffer = new byte[bufferSize];
        StringBuffer sb = new StringBuffer(1024);
        System.out.println("Service listening on " + socketFile);
        while (true) {
            AFUNIXSocket connection = unixSocket.accept();
            System.out.println("new connection!");
            SessionContext context = new SessionContext();

            try (InputStream is = connection.getInputStream();
                 OutputStream os = connection.getOutputStream()) {
                int read;
                while ((read = is.read(buffer)) != -1) {
                    String s = new String(buffer, 0, read, StandardCharsets.UTF_8);
                    if (s.contains(DELIM)) {
                        int doffset;
                        while ((doffset = s.indexOf(DELIM)) != -1) {
                            sb.append(s, 0, doffset);

                            //Save tail for next loop
                            s = s.substring(doffset + 1);
                            try {
                                String cmd = sb.toString();
                                if (!cmd.trim().isEmpty()) {
                                    String response = process(cmd, context);
                                    os.write(response.getBytes(StandardCharsets.UTF_8));
                                }
                            } catch (Throwable t) {
                                String response = String.format("%s %s\n", Response.ERROR, t.getMessage());
                                os.write(response.getBytes(StandardCharsets.UTF_8));
                                t.printStackTrace();
                            }

                            //Reset buffer
                            sb.setLength(0);
                        }
                    } else {
                        sb.append(s);
                    }
                }
            }
        }
    }

    static void help() {
        System.out.println("Usage: ipcservice.jar [/unix/socket/path.sock]");
        System.exit(1);
    }

    public static void main(String[] args) {
        String socketFile = System.getProperty("java.io.tmpdir") + "/jvector.sock";
        if (args.length > 1)
            help();

        if (args.length == 1)
            socketFile = args[0];

        try {
            IPCService service = new IPCService(Path.of(socketFile));
            service.serve();
        } catch (Throwable t) {
            t.printStackTrace();
            System.exit(1);
        }

        System.exit(0);
    }
}

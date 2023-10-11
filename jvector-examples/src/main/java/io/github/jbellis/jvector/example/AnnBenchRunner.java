package io.github.jbellis.jvector.example;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import io.github.jbellis.jvector.example.util.MMapRandomAccessVectorValues;
import io.github.jbellis.jvector.example.util.UpdatableRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.newsclub.net.unix.AFUNIXServerSocket;
import org.newsclub.net.unix.AFUNIXSocket;
import org.newsclub.net.unix.AFUNIXSocketAddress;

/**
 * Simple local service to use for ann-benchmarks runner.
 * Only handles a single connection at a time.
 */
public class AnnBenchRunner {

    // How each command message is marked as finished
    private static final String DELIM = "\n";

    class SessionContext {
        boolean isBulkLoad = false;
        int dimension;
        int M;
        int efConstruction;
        VectorSimilarityFunction similarityFunction;
        RandomAccessVectorValues<float[]> ravv;
        GraphIndexBuilder<float[]> indexBuilder;

        StringBuffer result = new StringBuffer(1024);
    }


    enum Command {
        CREATE,  //DIMENSIONS SIMILARITY_TYPE M EF\n
        WRITE,  //[N,N,N] [N,N,N]\n
        BULKLOAD, // #ROWS #COLUMNS /path/to/local/file
        OPTIMIZE,
        SEARCH, //[N,N,N] EF-search limit\n
        MEMORY, // Memory usage in kb
    }

    enum Response {
        OK,
        ERROR,
        RESULT
    }

    final Path socketFile;
    final AFUNIXServerSocket unixSocket;
    AnnBenchRunner(Path socketFile) throws IOException {
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

            String[] values = vStr.substring(1, vStr.length() - 1).split("\\s*,\\s*");
            if (values.length != ctx.ravv.dimension())
                throw new IllegalArgumentException(String.format("Invalid vector dimension: %d %d!=%d", i, values.length, ctx.ravv.dimension()));

            float[] vector = new float[ctx.ravv.dimension()];
            for (int k = 0; k < vector.length; k++)
                vector[k] = Float.parseFloat(values[k]);

            ((UpdatableRandomAccessVectorValues)ctx.ravv).add(vector);
            ctx.indexBuilder.addGraphNode(ctx.ravv.size() - 1, ctx.ravv);
        }
    }

    void load(String input, SessionContext ctx) {
        String[] args = input.split("\\s+");
        if (args.length != 1)
            throw new IllegalArgumentException("Invalid arguments. Expecting 'BULKLOAD /path/to/local/file'. got " + input);

        File f = new File(args[0]);
        if (!f.exists())
            throw new IllegalArgumentException("No file at: " + f);

        long length = f.length();
        if (length % ((long) ctx.dimension * Float.BYTES) != 0)
            throw new IllegalArgumentException("File is not encoded correctly");
        
        ctx.ravv = new MMapRandomAccessVectorValues(f, ctx.dimension);
        ctx.indexBuilder = new GraphIndexBuilder<>(ctx.ravv, VectorEncoding.FLOAT32, ctx.similarityFunction, ctx.M, ctx.efConstruction, 1.2f, 1.4f);
        ctx.indexBuilder.build();
    }

    void optimize(SessionContext ctx) {
        if (!ctx.isBulkLoad)
            ctx.indexBuilder.complete();
    }

    String memory(SessionContext ctx) {
        return String.format("%s %d\n", Response.RESULT, 1L);
    }

    String search(String input, SessionContext ctx) {
        String[] args = input.split("\\s+");

        if (args.length != 3)
            throw new IllegalArgumentException("Invalid arguments [vector] search-ef limit");

        String vStr = args[0];
        if (!vStr.startsWith("[") || !vStr.endsWith("]"))
            throw new IllegalArgumentException("Invalid query vector encountered:" + vStr);

        String[] values = vStr.substring(1, vStr.length() - 1).split("\\s*,\\s*");
        if (values.length != ctx.ravv.dimension())
            throw new IllegalArgumentException(String.format("Invalid vector dimension: %d!=%d", values.length, ctx.ravv.dimension()));

        float[] vector = new float[ctx.dimension];
        for (int k = 0; k < vector.length; k++)
            vector[k] = Float.parseFloat(values[k]);

        int searchEf = Integer.parseInt(args[1]);
        int topK = Integer.parseInt(args[2]);

        SearchResult r = GraphSearcher.search(vector, searchEf, ctx.ravv, VectorEncoding.FLOAT32, ctx.similarityFunction, ctx.indexBuilder.getGraph(), null);

        var resultNodes = r.getNodes();
        int count = Math.min(resultNodes.length, topK);
        ctx.result.setLength(0);
        ctx.result.append(Response.RESULT).append(" ");
        for (int i = 0; i < count; i++) {
            if (i > 0) ctx.result.append(",");
            ctx.result.append(resultNodes[i].node);
        }
        ctx.result.append("\n");
        return ctx.result.toString();
    }

    String process(String input, SessionContext ctx) {
        int delim = input.indexOf(" ");
        String command = delim < 1 ? input : input.substring(0, delim);
        String commandArgs = input.substring(delim + 1);
        String response = Response.OK.name();
        System.err.println(command);
        switch (Command.valueOf(command)) {
            case CREATE: create(commandArgs, ctx); break;
            case WRITE: write(commandArgs, ctx); break;
            case BULKLOAD: load(commandArgs, ctx); break;
            case OPTIMIZE: optimize(ctx); break;
            case SEARCH: response = search(commandArgs, ctx); break;
            case MEMORY: response = memory(ctx); break;
            default: throw new UnsupportedOperationException("No support for: '" + command + "'");
        }
        return response;
    }

    void serve() throws IOException
    {
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
        System.out.println("Usage: annbench.jar [/unix/socket/path.sock]");
        System.exit(1);
    }

    public static void main(String[] args) {
        String socketFile = "/tmp/jvector.sock";
        if (args.length > 1)
            help();

        if (args.length == 1)
            socketFile = args[0];

        try {
            AnnBenchRunner service = new AnnBenchRunner(Path.of(socketFile));
            service.serve();
        } catch (Throwable t) {
            t.printStackTrace();
            System.exit(1);
        }

        System.exit(0);
    }
}

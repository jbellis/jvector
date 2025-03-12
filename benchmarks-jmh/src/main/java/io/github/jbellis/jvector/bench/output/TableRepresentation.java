package io.github.jbellis.jvector.bench.output;

public interface TableRepresentation {
    void addEntry(long elapsedSeconds, long qps, double meanLatency, double p999Latency, double meanVisited);
    void print();
}

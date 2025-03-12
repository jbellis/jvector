package io.github.jbellis.jvector.bench.output;

import java.util.ArrayList;
import java.util.List;

public class TextTable implements TableRepresentation {
    private final List<String> resultsTable = new ArrayList<>();

    public TextTable() {
        resultsTable.add("\n");
        resultsTable.add(String.format("%-10s | %-10s | %-15s | %-15s | %-15s", "Elapsed(s)", "QPS", "Mean Latency (µs)", "P99.9 Latency (µs)", "Mean Visited"));
        resultsTable.add("-----------------------------------------------------------------------");
    }

    @Override
    public void addEntry(long elapsedSeconds, long qps, double meanLatency, double p999Latency, double meanVisited) {
        resultsTable.add(String.format("%-10d | %-10d | %-15.3f | %-15.3f | %-15.3f", elapsedSeconds, qps, meanLatency, p999Latency, meanVisited));
    }

    @Override
    public void print() {
        for (String line : resultsTable) {
            System.out.println(line);
        }
    }
}

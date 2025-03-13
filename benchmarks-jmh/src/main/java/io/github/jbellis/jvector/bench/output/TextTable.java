package io.github.jbellis.jvector.bench.output;

import java.util.ArrayList;
import java.util.List;

public class TextTable implements TableRepresentation {
    private final List<String> resultsTable = new ArrayList<>();

    public TextTable() {
        resultsTable.add("\n");
        resultsTable.add(String.format(
                "%-12s | %-12s | %-18s | %-18s | %-15s | %-12s",
                "Elapsed(s)", "QPS", "Mean Latency (µs)", "P99.9 Latency (µs)", "Mean Visited", "Recall (%)"
        ));
        resultsTable.add("-----------------------------------------------------------------------------------------------");
    }

    @Override
    public void addEntry(long elapsedSeconds, long qps, double meanLatency, double p999Latency, double meanVisited, double recallPercentage) {
        resultsTable.add(String.format(
                "%-12d | %-12d | %-18.3f | %-18.3f | %-15.3f | %10.2f%%",
                elapsedSeconds, qps, meanLatency, p999Latency, meanVisited, recallPercentage*100
        ));
    }

    @Override
    public void print() {
        for (String line : resultsTable) {
            System.out.println(line);
        }
    }
}

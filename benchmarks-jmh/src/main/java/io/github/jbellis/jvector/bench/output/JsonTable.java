package io.github.jbellis.jvector.bench.output;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

public class JsonTable implements TableRepresentation {
    private final List<Map<String, Object>> results = new ArrayList<>();
    private final Gson gson = new GsonBuilder().setPrettyPrinting().create();

    @Override
    public void addEntry(long elapsedSeconds, long qps, double meanLatency, double p999Latency, double meanVisited, double recallPercentage) {
        Map<String, Object> entry = new HashMap<>();
        entry.put("elapsed_seconds", elapsedSeconds);
        entry.put("qps", qps);
        entry.put("mean_latency_us", meanLatency);
        entry.put("p999_latency_us", p999Latency);
        entry.put("mean_visited", meanVisited);
        entry.put("recall", String.format("%.2f%%", recallPercentage)); // Store as formatted string
        results.add(entry);
    }

    @Override
    public void print() {
        System.out.println(gson.toJson(results));
    }
}

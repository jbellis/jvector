package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.disk.RandomAccessReader;

import java.io.Closeable;
import java.io.IOException;

public interface FeatureSource extends Closeable {
    RandomAccessReader inlineReaderForNode(int node, FeatureId featureId) throws IOException;
}

package io.github.jbellis.jvector.graph;

public interface MultiAdcQuery extends AutoCloseable {
    void setNodeId(int queryIdx, int offset, int nodeId);

    void computeSimilarities();

    void close();

    int getNodeId(int queryIdx, int i);

    float getScore(int queryIdx, int i);
}

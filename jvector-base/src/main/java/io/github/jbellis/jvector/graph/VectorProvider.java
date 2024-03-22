package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

public abstract class VectorProvider {
    VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final int dimension;

    public VectorProvider(int dimension) {
        this.dimension = dimension;
    }

    public static VectorProvider from(GraphIndex.View view, int dimension) {
        return new VectorProvider(dimension) {
            @Override
            public void getInto(int nodeId, VectorFloat<?> result, int offset) {
                view.getVectorInto(nodeId, result, offset);
            }
        };
    }

    public static VectorProvider from(RandomAccessVectorValues ravv) {
        return new VectorProvider(ravv.dimension()) {
            @Override
            public void getInto(int nodeId, VectorFloat<?> result, int offset) {
                result.copyFrom(ravv.vectorValue(nodeId), 0, offset, ravv.dimension());
            }
        };
    }

    public abstract void getInto(int nodeId, VectorFloat<?> result, int offset);

    public VectorFloat<?> get(int nodeId) {
        var result = vts.createFloatVector(dimension);
        getInto(nodeId, result, 0);
        return result;
    }
}

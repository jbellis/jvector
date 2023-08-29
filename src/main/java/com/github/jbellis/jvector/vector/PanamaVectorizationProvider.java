package com.github.jbellis.jvector.vector;

public class PanamaVectorizationProvider extends VectorizationProvider {

    private final VectorUtilSupport vectorUtilSupport;

    public PanamaVectorizationProvider() {
        this.vectorUtilSupport = new PanamaVectorUtilSupport();
    }

    @Override
    public VectorUtilSupport getVectorUtilSupport()
    {
        return vectorUtilSupport;
    }
}

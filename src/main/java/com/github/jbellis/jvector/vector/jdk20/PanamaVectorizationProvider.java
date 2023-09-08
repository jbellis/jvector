package com.github.jbellis.jvector.vector.jdk20;

import com.github.jbellis.jvector.vector.VectorUtilSupport;
import com.github.jbellis.jvector.vector.VectorizationProvider;

public class PanamaVectorizationProvider extends VectorizationProvider
{

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

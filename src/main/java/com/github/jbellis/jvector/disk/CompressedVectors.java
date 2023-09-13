package com.github.jbellis.jvector.disk;

import com.github.jbellis.jvector.pq.ProductQuantization;
import com.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CompressedVectors
{
    private final ProductQuantization pq;
    private final List<byte[]> compressedVectors;

    public CompressedVectors(ProductQuantization pq, List<byte[]> compressedVectors)
    {
        this.pq = pq;
        this.compressedVectors = compressedVectors;
    }

    public void write(DataOutput out) throws IOException
    {
        // pq codebooks
        pq.write(out);

        // compressed vectors
        out.writeInt(compressedVectors.size());
        out.writeInt(pq.getSubspaceCount());
        for (var v : compressedVectors) {
            out.write(v);
        }
    }

    public static CompressedVectors load(RandomAccessReader in, long offset) throws IOException
    {
        in.seek(offset);

        // pq codebooks
        var pq = ProductQuantization.load(in);

        // read the vectors
        int size = in.readInt();
        var compressedVectors = new ArrayList<byte[]>(size);
        int compressedDimension = in.readInt();
        for (int i = 0; i < size; i++)
        {
            byte[] vector = new byte[compressedDimension];
            in.readFully(vector);
            compressedVectors.add(vector);
        }

        return new CompressedVectors(pq, compressedVectors);
    }

    public float decodedSimilarity(int ordinal, float[] v, VectorSimilarityFunction similarityFunction)
    {
        switch (similarityFunction)
        {
            case DOT_PRODUCT:
                return (1 + pq.decodedDotProduct(compressedVectors.get(ordinal), v)) / 2;
            default:
                // TODO implement other similarity functions efficiently
                var decoded = new float[pq.getOriginalDimension()];
                pq.decode(compressedVectors.get(ordinal), decoded);
                return similarityFunction.compare(decoded, v);
        }
    }
}

package com.github.jbellis.jvector.pq;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import com.github.jbellis.jvector.example.util.MappedRandomAccessReader;
import org.junit.Test;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestProductQuantization extends RandomizedTest {
    @Test
    public void testSaveLoad() throws IOException {
        // Generate a PQ for random 2D vectors
        var vectors = IntStream.range(0, 512).mapToObj(i -> new float[]{getRandom().nextFloat(), getRandom().nextFloat()}).collect(Collectors.toList());
        var pq = new ProductQuantization(vectors, 1, false);

        // Write the pq object
        File tempFile = File.createTempFile("pqtest", ".bin");
        tempFile.deleteOnExit();
        try (var out = new DataOutputStream(new FileOutputStream(tempFile))) {
            pq.write(out);
        }

        // Read the pq object
        try (var in = new MappedRandomAccessReader(tempFile.getAbsolutePath())) {
            var pq2 = ProductQuantization.load(in);
            assertEquals(pq, pq2);
        }
    }
}

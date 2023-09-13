package com.github.jbellis.jvector.disk;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class Io {
    public static void writeFloats(DataOutput out, float[] v) throws IOException {
        for (var a : v) {
            out.writeFloat(a);
        }
    }

    public static float[] readFloats(DataInput in, int size) throws IOException {
        var v = new float[size];
        for (int i = 0; i < size; i++) {
            v[i] = in.readFloat();
        }
        return v;
    }
}

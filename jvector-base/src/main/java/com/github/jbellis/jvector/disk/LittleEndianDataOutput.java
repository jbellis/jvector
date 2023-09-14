package com.github.jbellis.jvector.disk;

import java.io.DataOutput;
import java.io.IOException;

public class LittleEndianDataOutput implements DataOutput {

    private final DataOutput out;

    /**
     * Constructs a LittleEndianDataOutput that writes to the specified DataOutput in little-endian order.
     *
     * @param out the DataOutput stream
     */
    public LittleEndianDataOutput(DataOutput out) {
        this.out = out;
    }

    @Override
    public void write(int b) throws IOException {
        out.write(b);
    }

    @Override
    public void write(byte[] b) throws IOException {
        out.write(b);
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
        out.write(b, off, len);
    }

    @Override
    public void writeBoolean(boolean v) throws IOException {
        out.writeBoolean(v);
    }

    @Override
    public void writeByte(int v) throws IOException {
        out.writeByte(v);
    }

    @Override
    public void writeShort(int v) throws IOException {
        out.writeByte(v & 0xFF);
        out.writeByte((v >> 8) & 0xFF);
    }

    @Override
    public void writeChar(int v) throws IOException {
        out.writeByte(v & 0xFF);
        out.writeByte((v >> 8) & 0xFF);
    }

    @Override
    public void writeInt(int v) throws IOException {
        out.writeByte(v & 0xFF);
        out.writeByte((v >> 8) & 0xFF);
        out.writeByte((v >> 16) & 0xFF);
        out.writeByte((v >> 24) & 0xFF);
    }

    @Override
    public void writeLong(long v) throws IOException {
        out.writeByte((int) (v & 0xFF));
        out.writeByte((int) ((v >> 8) & 0xFF));
        out.writeByte((int) ((v >> 16) & 0xFF));
        out.writeByte((int) ((v >> 24) & 0xFF));
        out.writeByte((int) ((v >> 32) & 0xFF));
        out.writeByte((int) ((v >> 40) & 0xFF));
        out.writeByte((int) ((v >> 48) & 0xFF));
        out.writeByte((int) ((v >> 56) & 0xFF));
    }

    @Override
    public void writeFloat(float v) throws IOException {
        int intBits = Float.floatToIntBits(v);
        writeInt(intBits);
    }

    @Override
    public void writeDouble(double v) throws IOException {
        long longBits = Double.doubleToLongBits(v);
        writeLong(longBits);
    }

    @Override
    public void writeBytes(String s) throws IOException {
        out.writeBytes(s);
    }

    @Override
    public void writeChars(String s) throws IOException {
        for (int i = 0; i < s.length(); i++) {
            writeChar(s.charAt(i));
        }
    }

    @Override
    public void writeUTF(String s) throws IOException {
        out.writeUTF(s);
    }
}



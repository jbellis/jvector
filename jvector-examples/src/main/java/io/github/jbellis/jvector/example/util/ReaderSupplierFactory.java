/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.SimpleMappedReaderSupplier;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class ReaderSupplierFactory {
    public static ReaderSupplier open(Path path) throws IOException {
        try {
            return new MMapReaderSupplier(path);
        } catch (UnsatisfiedLinkError|NoClassDefFoundError e) {
            if (Files.size(path) > Integer.MAX_VALUE) {
                throw new RuntimeException("File sizes greater than 2GB are not supported on Windows--contributions welcome");
            }

            return new SimpleMappedReaderSupplier(path);
        }
    }
}

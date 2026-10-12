/*
 * Copyright DataStax, Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 */
package io.github.jbellis.jvector.example;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import static org.junit.Assert.*;

public class AutoBenchYAMLTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();

    @Test(timeout = 60000)
    public void invalidConfigurationDoesNotAbortRemainingDatasets() throws Exception {
        Path work = temporary.newFolder("work").toPath();
        Path yaml = work.resolve("jvector-examples/yaml-configs");
        Files.createDirectories(yaml);
        Files.writeString(yaml.resolve("datasets.yml"), "regression-tests:\n  - first\n  - second\n");
        Path config = work.resolve("invalid.yml");
        Files.writeString(config, "unknownConfigurationProperty: true\n");
        String classpath = Arrays.stream(System.getProperty("surefire.test.class.path",
                System.getProperty("java.class.path")).split(java.io.File.pathSeparator))
                .map(p -> Path.of(p).toAbsolutePath().toString())
                .collect(Collectors.joining(java.io.File.pathSeparator));
        Path log = work.resolve("console.log");
        Process process = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED",
                "-cp", classpath, AutoBenchYAML.class.getName(), "--output", work.resolve("results.json").toString(),
                "--config", config.toString()).directory(work.toFile()).redirectErrorStream(true)
                .redirectOutput(log.toFile()).start();
        try {
            assertTrue("AutoBenchYAML did not exit", process.waitFor(30, TimeUnit.SECONDS));
            String output = Files.readString(log);
            assertEquals(output, 0, process.exitValue());
            assertTrue(output, output.contains("Exception while processing dataset first"));
            assertTrue(output, output.contains("Exception while processing dataset second"));
        } finally {
            if (process.isAlive()) { process.destroyForcibly(); process.waitFor(); }
        }
    }
}

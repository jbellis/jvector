package io.github.jbellis.jvector.example.util;

import software.amazon.awssdk.auth.credentials.AnonymousCredentialsProvider;
import software.amazon.awssdk.http.crt.AwsCrtAsyncHttpClient;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3AsyncClient;
import software.amazon.awssdk.services.s3.S3AsyncClientBuilder;
import software.amazon.awssdk.transfer.s3.S3TransferManager;
import software.amazon.awssdk.transfer.s3.model.CompletedFileDownload;
import software.amazon.awssdk.transfer.s3.model.DownloadFileRequest;
import software.amazon.awssdk.transfer.s3.model.FileDownload;
import software.amazon.awssdk.transfer.s3.progress.LoggingTransferListener;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class DownloadHelper {
    public static void maybeDownloadData() {
        // TODO how to detect and recover from incomplete downloads?
        String[] keys = {
                "wikipedia_squad/100k/ada_002_100000_base_vectors.fvec",
                "wikipedia_squad/100k/ada_002_100000_query_vectors_10000.fvec",
                "wikipedia_squad/100k/ada_002_100000_indices_query_10000.ivec"
        };

        String bucketName = "astra-vector";

        S3AsyncClientBuilder s3ClientBuilder = S3AsyncClient.builder()
                .region(Region.of("us-east-1"))
                .httpClient(AwsCrtAsyncHttpClient.builder()
                                    .maxConcurrency(1)
                                    .build())
                .credentialsProvider(AnonymousCredentialsProvider.create());

        // get directory from paths in keys
        List<String> dirs = Arrays.stream(keys).map(key -> key.substring(0, key.lastIndexOf("/"))).distinct().collect(Collectors.toList());
        for (String dir : dirs) {
            try {
                dir = "fvec/" + dir;
                Files.createDirectories(Paths.get(dir));
            } catch (IOException e) {
                System.err.println("Failed to create directory: " + e.getMessage());
            }
        }

        try (S3AsyncClient s3Client = s3ClientBuilder.build()) {
            S3TransferManager tm = S3TransferManager.builder().s3Client(s3Client).build();
            for (String key : keys) {
                Path path = Paths.get("fvec", key);
                if (Files.exists(path)) {
                    continue;
                }

                System.out.println("Downloading: " + key);
                DownloadFileRequest downloadFileRequest =
                        DownloadFileRequest.builder()
                                .getObjectRequest(b -> b.bucket(bucketName).key(key))
                                .addTransferListener(LoggingTransferListener.create())
                                .destination(Paths.get(path.toString()))
                                .build();

                FileDownload downloadFile = tm.downloadFile(downloadFileRequest);

                CompletedFileDownload downloadResult = downloadFile.completionFuture().join();
                System.out.println("Downloaded file of length " + downloadResult.response().contentLength());

            }
            tm.close();
        } catch (Exception e) {
            System.out.println("Error downloading data from S3: " + e.getMessage());
            System.exit(1);
        }
    }
}

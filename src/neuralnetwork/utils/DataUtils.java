package neuralnetwork.utils;

import java.util.*;

public class DataUtils {

    public static Map<String, double[][][]> trainTestSplit(double[][] X, double[][] y, double testSize) {
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have same number of samples");
        }

        int totalSamples = X.length;
        int testSamples = (int) (totalSamples * testSize);
        int trainSamples = totalSamples - testSamples;

        if (testSamples == 0 || trainSamples == 0) {
            throw new IllegalArgumentException("Test size too small or too large");
        }

        // Shuffle indices
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalSamples; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices);

        double[][] X_train = new double[trainSamples][X[0].length];
        double[][] y_train = new double[trainSamples][y[0].length];
        double[][] X_test = new double[testSamples][X[0].length];
        double[][] y_test = new double[testSamples][y[0].length];

        for (int i = 0; i < trainSamples; i++) {
            int idx = indices.get(i);
            X_train[i] = X[idx].clone();
            y_train[i] = y[idx].clone();
        }

        for (int i = 0; i < testSamples; i++) {
            int idx = indices.get(trainSamples + i);
            X_test[i] = X[idx].clone();
            y_test[i] = y[idx].clone();
        }

        Map<String, double[][][]> result = new HashMap<>();
        result.put("train", new double[][][]{X_train, y_train});
        result.put("test", new double[][][]{X_test, y_test});

        return result;
    }

    public static double[][] normalize(double[][] data) {
        if (data.length == 0) return data;

        int cols = data[0].length;
        double[][] normalized = new double[data.length][cols];

        for (int col = 0; col < cols; col++) {
            // Calculate mean and std for column
            double sum = 0;
            for (int row = 0; row < data.length; row++) {
                sum += data[row][col];
            }
            double mean = sum / data.length;

            double sumSq = 0;
            for (int row = 0; row < data.length; row++) {
                double diff = data[row][col] - mean;
                sumSq += diff * diff;
            }
            double std = Math.sqrt(sumSq / data.length);

            // Normalize column (avoid division by zero)
            if (std < 1e-8) std = 1;

            for (int row = 0; row < data.length; row++) {
                normalized[row][col] = (data[row][col] - mean) / std;
            }
        }

        return normalized;
    }

    public static double[][] minMaxScale(double[][] data) {
        if (data.length == 0) return data;

        int cols = data[0].length;
        double[][] scaled = new double[data.length][cols];

        for (int col = 0; col < cols; col++) {
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;

            // Find min and max
            for (int row = 0; row < data.length; row++) {
                min = Math.min(min, data[row][col]);
                max = Math.max(max, data[row][col]);
            }

            double range = max - min;
            if (range < 1e-8) range = 1; // Avoid division by zero

            // Scale to [0, 1]
            for (int row = 0; row < data.length; row++) {
                scaled[row][col] = (data[row][col] - min) / range;
            }
        }

        return scaled;
    }

    public static void printDatasetInfo(double[][] X, double[][] y, String name) {
        System.out.println("\n=== " + name + " Dataset Info ===");
        System.out.println("Samples: " + X.length);
        System.out.println("Features: " + X[0].length);
        System.out.println("Targets: " + y[0].length);

        // Show class distribution for binary classification
        if (y[0].length == 1) {
            int class0 = 0, class1 = 0;
            for (int i = 0; i < y.length; i++) {
                if (y[i][0] < 0.5) class0++;
                else class1++;
            }
            System.out.println("Class 0: " + class0 + " (" +
                    String.format("%.1f%%", (double)class0/y.length*100) + ")");
            System.out.println("Class 1: " + class1 + " (" +
                    String.format("%.1f%%", (double)class1/y.length*100) + ")");
        }
    }
}
package neuralnetwork.utils;

import java.util.Random;

public class MathUtils {
    private static final Random random = new Random();

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double relu(double x) {
        return Math.max(0, x);
    }

    public static double tanh(double x) {
        return Math.tanh(x);
    }

    public static double mean(double[] values) {
        double sum = 0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.length;
    }

    public static double std(double[] values) {
        double mean = mean(values);
        double sumSq = 0;
        for (double v : values) {
            sumSq += (v - mean) * (v - mean);
        }
        return Math.sqrt(sumSq / values.length);
    }

    public static double[][] randomMatrix(int rows, int cols) {
        double[][] matrix = new double[rows][cols];
        double scale = Math.sqrt(2.0 / (rows + cols)); // Xavier initialization

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = (random.nextDouble() * 2 - 1) * scale;
            }
        }
        return matrix;
    }
}
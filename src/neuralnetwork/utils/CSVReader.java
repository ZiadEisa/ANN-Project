package neuralnetwork.utils;

import java.io.*;
import java.util.*;

public class CSVReader {

    public static double[][] readCSV(String filename) throws IOException {
        return readCSV(filename, ",");
    }

    public static double[][] readCSV(String filename, String delimiter) throws IOException {
        List<double[]> rows = new ArrayList<>();
        BufferedReader reader = null;

        try {
            File file = new File(filename);
            if (!file.exists()) {
                throw new FileNotFoundException("File not found: " + filename);
            }

            reader = new BufferedReader(new FileReader(file));
            String line;
            boolean firstLine = true;
            int numColumns = -1;

            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                if (line.startsWith("#")) continue;

                String[] parts = line.split(delimiter);

                // Skip header if first line contains non-numeric values
                if (firstLine) {
                    firstLine = false;
                    boolean allNumeric = true;
                    for (String part : parts) {
                        if (!isNumeric(part.trim())) {
                            allNumeric = false;
                            break;
                        }
                    }
                    if (!allNumeric) {
                        System.out.println("Skipping header line...");
                        continue;
                    }
                }

                double[] row = new double[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    try {
                        row[i] = Double.parseDouble(parts[i].trim());
                    } catch (NumberFormatException e) {
                        row[i] = 0.0;
                    }
                }

                if (numColumns == -1) {
                    numColumns = row.length;
                } else if (numColumns != row.length) {
                    throw new IOException("Inconsistent column count in CSV file");
                }

                rows.add(row);
            }

            if (rows.isEmpty()) {
                throw new IOException("No data found in CSV file");
            }

            // Convert to 2D array
            double[][] data = new double[rows.size()][numColumns];
            for (int i = 0; i < rows.size(); i++) {
                System.arraycopy(rows.get(i), 0, data[i], 0, numColumns);
            }

            return data;

        } finally {
            if (reader != null) {
                reader.close();
            }
        }
    }

    public static double[][] extractFeatures(double[][] data, int featureCount) {
        double[][] features = new double[data.length][featureCount];
        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, features[i], 0, featureCount);
        }
        return features;
    }

    public static double[][] extractLabels(double[][] data, int featureCount) {
        int labelCount = data[0].length - featureCount;
        double[][] labels = new double[data.length][labelCount];
        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], featureCount, labels[i], 0, labelCount);
        }
        return labels;
    }

    private static boolean isNumeric(String str) {
        if (str == null || str.isEmpty()) return false;
        try {
            Double.parseDouble(str);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }
}
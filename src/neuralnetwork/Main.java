import neuralnetwork.*;
import neuralnetwork.activations.*;
import neuralnetwork.losses.*;
import neuralnetwork.utils.*;
import neuralnetwork.training.Trainer;

public class Main {
    public static void main(String[] args) {
        System.out.println("=== Heart Disease Prediction Neural Network ===");

        try {
            // 1. Load heart.csv data
            System.out.println("Loading heart.csv data...");
            String csvFile = "data/heart.csv";
            double[][] csvData = CSVReader.readCSV(csvFile);

            System.out.println("Data loaded: " + csvData.length + " samples, " +
                    csvData[0].length + " features");

            // 2. Preprocess data
            System.out.println("\nPreprocessing data...");

            // Assuming last column is target (0/1 for no heart disease/heart disease)
            int featureCount = csvData[0].length - 1;
            double[][] X = CSVReader.extractFeatures(csvData, featureCount);
            double[][] y = CSVReader.extractLabels(csvData, featureCount);

            // Normalize features
            X = DataUtils.minMaxScale(X);

            // Train-test split (80% train, 20% test)
            var split = DataUtils.trainTestSplit(X, y, 0.2);
            double[][] X_train = split.get("train")[0];
            double[][] y_train = split.get("train")[1];
            double[][] X_test = split.get("test")[0];
            double[][] y_test = split.get("test")[1];

            System.out.println("Training set: " + X_train.length + " samples");
            System.out.println("Test set: " + X_test.length + " samples");
            System.out.println("Features per sample: " + X_train[0].length);

            // 3. Create neural network
            System.out.println("\nCreating neural network...");
            NeuralNetwork nn = new NeuralNetwork();

            // Input layer (same as number of features)
            nn.addLayer(X_train[0].length, new Tanh());

            // Hidden layers
            nn.addLayer(16, new ReLU());
            nn.addLayer(8, new ReLU());

            // Output layer (binary classification)
            nn.addLayer(1, new Sigmoid());

            // Set loss function for binary classification
            nn.setLossFunction(new CrossEntropy());

            // 4. Train the network
            System.out.println("\nTraining network...");

            Trainer.TrainingConfig config = new Trainer.TrainingConfig()
                    .setEpochs(1000)
                    .setLearningRate(0.01)
                    .setBatchSize(32)
                    .setVerbose(true);

            Trainer.train(nn, X_train, y_train, config);

            // 5. Evaluate on test set
            System.out.println("\n=== Evaluation on Test Set ===");
            double[][] predictions = nn.predict(X_test);

            // Calculate accuracy
            int correct = 0;
            for (int i = 0; i < predictions.length; i++) {
                double predicted = predictions[i][0] > 0.5 ? 1 : 0;
                double actual = y_test[i][0];
                if (Math.abs(predicted - actual) < 0.5) {
                    correct++;
                }
            }

            double accuracy = (double) correct / predictions.length * 100;
            System.out.printf("Test Accuracy: %.2f%% (%d/%d)%n",
                    accuracy, correct, predictions.length);

            // 6. Show some predictions
            System.out.println("\n=== Sample Predictions ===");
            System.out.println("Index | Predicted | Actual | Correct?");
            System.out.println("------------------------------------");
            for (int i = 0; i < Math.min(10, predictions.length); i++) {
                double predicted = predictions[i][0];
                double predictedClass = predicted > 0.5 ? 1 : 0;
                double actual = y_test[i][0];
                String correctStr = Math.abs(predictedClass - actual) < 0.5 ? "✓" : "✗";
                System.out.printf("%5d | %8.4f | %6.0f | %s%n",
                        i, predicted, actual, correctStr);
            }

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();

            // Run XOR example if CSV fails
            System.out.println("\nRunning XOR example instead...");
            runXORExample();
        }
    }

    private static void runXORExample() {
        // XOR dataset
        double[][] X = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };

        double[][] y = {
                {0},
                {1},
                {1},
                {0}
        };

        NeuralNetwork nn = new NeuralNetwork();
        nn.addLayer(2, new Tanh());
        nn.addLayer(4, new ReLU());
        nn.addLayer(1, new Sigmoid());
        nn.setLossFunction(new MSE());

        nn.train(X, y, 5000, 0.1, 4);

        System.out.println("\nXOR Results:");
        double[][] predictions = nn.predict(X);
        for (int i = 0; i < X.length; i++) {
            System.out.printf("[%.0f, %.0f] -> %.4f (expected: %.0f)%n",
                    X[i][0], X[i][1], predictions[i][0], y[i][0]);
        }
    }
}
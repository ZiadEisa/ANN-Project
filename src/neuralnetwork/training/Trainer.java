package neuralnetwork.training;

import neuralnetwork.model.NeuralNetwork;

public class Trainer {

    public static class TrainingConfig {
        private int epochs = 100;
        private double learningRate = 0.01;
        private int batchSize = 32;
        private boolean verbose = true;

        public TrainingConfig() {}

        public TrainingConfig setEpochs(int epochs) {
            this.epochs = epochs;
            return this;
        }

        public TrainingConfig setLearningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public TrainingConfig setBatchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public TrainingConfig setVerbose(boolean verbose) {
            this.verbose = verbose;
            return this;
        }

        public int getEpochs() { return epochs; }
        public double getLearningRate() { return learningRate; }
        public int getBatchSize() { return batchSize; }
        public boolean isVerbose() { return verbose; }
    }

    public static void train(NeuralNetwork network, double[][] X, double[][] y, TrainingConfig config) {
        if (config.isVerbose()) {
            System.out.println("\n=== Training Configuration ===");
            System.out.println("Epochs: " + config.getEpochs());
            System.out.println("Learning Rate: " + config.getLearningRate());
            System.out.println("Batch Size: " + config.getBatchSize());
            System.out.println("Training Samples: " + X.length);
            System.out.println("=============================\n");
        }

        network.train(X, y, config.getEpochs(), config.getLearningRate(), config.getBatchSize());

        if (config.isVerbose()) {
            System.out.println("\nTraining completed!");
        }
    }
}
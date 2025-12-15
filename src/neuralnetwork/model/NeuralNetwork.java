package neuralnetwork.model;

import neuralnetwork.activations.Activation;
import neuralnetwork.losses.Loss;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private List<Layer> layers;
    private Loss lossFunction;
    private List<Double> trainingLossHistory;

    public NeuralNetwork() {
        this.layers = new ArrayList<>();
        this.trainingLossHistory = new ArrayList<>();
        this.lossFunction = new neuralnetwork.losses.MSE();
    }

    public void addLayer(int neurons, Activation activation) {
        int inputSize = layers.isEmpty() ? neurons : layers.get(layers.size() - 1).getNeurons();
        Layer layer = new DenseLayer(inputSize, neurons, activation);
        layers.add(layer);
    }

    public void setLossFunction(Loss lossFunction) {
        this.lossFunction = lossFunction;
    }

    public double[][] forward(double[][] input) {
        double[][] output = input;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    public void train(double[][] X, double[][] y, int epochs, double learningRate, int batchSize) {
        trainingLossHistory.clear();

        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochLoss = 0;
            int batchCount = 0;

            // Process in batches
            for (int batchStart = 0; batchStart < X.length; batchStart += batchSize) {
                int batchEnd = Math.min(batchStart + batchSize, X.length);
                int actualBatchSize = batchEnd - batchStart;

                double[][] batchX = new double[actualBatchSize][];
                double[][] batchY = new double[actualBatchSize][];

                // Copy batch data
                for (int i = 0; i < actualBatchSize; i++) {
                    batchX[i] = X[batchStart + i].clone();
                    batchY[i] = y[batchStart + i].clone();
                }

                // Forward pass
                double[][] predictions = forward(batchX);

                // Calculate loss
                double loss = lossFunction.calculate(predictions, batchY);
                epochLoss += loss;

                // Backward pass
                double[][] grad = lossFunction.derivative(predictions, batchY);

                for (int i = layers.size() - 1; i >= 0; i--) {
                    grad = layers.get(i).backward(grad, learningRate);
                }

                batchCount++;
            }

            double avgLoss = batchCount > 0 ? epochLoss / batchCount : 0;
            trainingLossHistory.add(avgLoss);

            if (epoch % 100 == 0 && epoch > 0) {
                System.out.printf("Epoch %4d, Loss: %.6f%n", epoch, avgLoss);
            }
        }
    }

    public double[][] predict(double[][] X) {
        return forward(X);
    }

    public List<Double> getTrainingLossHistory() {
        return trainingLossHistory;
    }
}
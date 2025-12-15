package neuralnetwork.model;

import neuralnetwork.activations.Activation;

public abstract class Layer {
    protected int inputSize;
    protected int neurons;
    protected Activation activation;
    protected double[][] weights;
    protected double[] bias;

    public Layer(int inputSize, int neurons, Activation activation) {
        this.inputSize = inputSize;
        this.neurons = neurons;
        this.activation = activation;
        initializeWeights();
    }

    protected void initializeWeights() {
        // Xavier/Glorot initialization
        double scale = Math.sqrt(2.0 / (inputSize + neurons));
        weights = new double[inputSize][neurons];
        bias = new double[neurons];

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < neurons; j++) {
                weights[i][j] = (Math.random() * 2 - 1) * scale;
            }
        }

        for (int i = 0; i < neurons; i++) {
            bias[i] = 0.01;
        }
    }

    public abstract double[][] forward(double[][] input);
    public abstract double[][] backward(double[][] grad, double learningRate);

    public int getNeurons() {
        return neurons;
    }
}
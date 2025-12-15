package neuralnetwork.model;

import neuralnetwork.activations.Activation;
import neuralnetwork.utils.Matrix;

public class DenseLayer extends Layer {
    private double[][] lastInput;
    private double[][] lastPreActivation;

    public DenseLayer(int inputSize, int neurons, Activation activation) {
        super(inputSize, neurons, activation);
    }

    @Override
    public double[][] forward(double[][] input) {
        this.lastInput = Matrix.copy(input);

        // Linear transformation: input * weights + bias
        double[][] weightedSum = Matrix.multiply(input, weights);

        // Add bias
        for (int i = 0; i < weightedSum.length; i++) {
            for (int j = 0; j < weightedSum[0].length; j++) {
                weightedSum[i][j] += bias[j];
            }
        }

        this.lastPreActivation = Matrix.copy(weightedSum);

        // Apply activation function
        double[][] output = new double[weightedSum.length][weightedSum[0].length];
        for (int i = 0; i < weightedSum.length; i++) {
            for (int j = 0; j < weightedSum[0].length; j++) {
                output[i][j] = activation.activate(weightedSum[i][j]);
            }
        }

        return output;
    }

    @Override
    public double[][] backward(double[][] grad, double learningRate) {
        // grad is derivative of loss w.r.t. this layer's output

        // Apply derivative of activation function
        double[][] activationGrad = new double[lastPreActivation.length][lastPreActivation[0].length];
        for (int i = 0; i < lastPreActivation.length; i++) {
            for (int j = 0; j < lastPreActivation[0].length; j++) {
                activationGrad[i][j] = activation.derivative(lastPreActivation[i][j]);
            }
        }

        // Element-wise multiplication
        double[][] delta = Matrix.elementwiseMultiply(grad, activationGrad);

        // Calculate weight gradients
        double[][] weightGrad = Matrix.multiply(Matrix.transpose(lastInput), delta);

        // Calculate bias gradients
        double[] biasGrad = new double[neurons];
        for (int i = 0; i < delta.length; i++) {
            for (int j = 0; j < delta[0].length; j++) {
                biasGrad[j] += delta[i][j];
            }
        }

        // Update weights and biases
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] -= learningRate * weightGrad[i][j];
            }
        }

        for (int i = 0; i < bias.length; i++) {
            bias[i] -= learningRate * biasGrad[i];
        }

        // Calculate gradient for previous layer
        double[][] prevGrad = Matrix.multiply(delta, Matrix.transpose(weights));

        return prevGrad;
    }
}
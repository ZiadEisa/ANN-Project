package neuralnetwork.activations;

public class Tanh implements Activation {
    @Override
    public double activate(double x) {
        return Math.tanh(x);
    }

    @Override
    public double derivative(double x) {
        double tanh = activate(x);
        return 1 - tanh * tanh;
    }
}
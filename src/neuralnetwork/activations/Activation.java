package neuralnetwork.activations;

public interface Activation {
    double activate(double x);
    double derivative(double x);
}
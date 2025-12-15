package neuralnetwork.losses;

public interface Loss {
    double calculate(double[][] predictions, double[][] targets);
    double[][] derivative(double[][] predictions, double[][] targets);
}
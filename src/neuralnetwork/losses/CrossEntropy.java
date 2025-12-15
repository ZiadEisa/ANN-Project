package neuralnetwork.losses;

public class CrossEntropy implements Loss {
    private static final double EPSILON = 1e-12;

    @Override
    public double calculate(double[][] predictions, double[][] targets) {
        double loss = 0;
        int samples = predictions.length;

        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < predictions[i].length; j++) {
                double p = Math.max(EPSILON, Math.min(1 - EPSILON, predictions[i][j]));
                loss += targets[i][j] * Math.log(p);
            }
        }

        return -loss / samples;
    }

    @Override
    public double[][] derivative(double[][] predictions, double[][] targets) {
        int samples = predictions.length;
        int outputs = predictions[0].length;
        double[][] grad = new double[samples][outputs];

        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < outputs; j++) {
                double p = Math.max(EPSILON, Math.min(1 - EPSILON, predictions[i][j]));
                grad[i][j] = (p - targets[i][j]) / (samples * p * (1 - p));
            }
        }

        return grad;
    }
}
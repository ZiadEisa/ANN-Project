package neuralnetwork.losses;

public class MSE implements Loss {
    @Override
    public double calculate(double[][] predictions, double[][] targets) {
        double sum = 0;
        int samples = predictions.length;
        int outputs = predictions[0].length;

        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < outputs; j++) {
                double error = predictions[i][j] - targets[i][j];
                sum += error * error;
            }
        }

        return sum / (samples * outputs);
    }

    @Override
    public double[][] derivative(double[][] predictions, double[][] targets) {
        int samples = predictions.length;
        int outputs = predictions[0].length;
        double[][] grad = new double[samples][outputs];

        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < outputs; j++) {
                grad[i][j] = 2 * (predictions[i][j] - targets[i][j]) / (samples * outputs);
            }
        }

        return grad;
    }
}
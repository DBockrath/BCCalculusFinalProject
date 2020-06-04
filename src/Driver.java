public class Driver {

    static final int INPUT_SIZE = 6;
    static final int HIDDEN_NODES = 4;
    static final int OUTPUT_SIZE = 1;
    static final double LEARNING_RATE = 0.2;
    static final int TRAINING_ITERATIONS = 500;

    private static double[][] layer1_synaptic_weights = new double[HIDDEN_NODES][INPUT_SIZE];
    private static double[][] layer2_synaptic_weights = new double[OUTPUT_SIZE][HIDDEN_NODES];
    private static int count = 1;

    public static void main(String[] args) {
        for (int i = 0; i < HIDDEN_NODES; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) layer1_synaptic_weights[i][j] = Math.random();
            for (int k = 0; k < OUTPUT_SIZE; k++) layer2_synaptic_weights[k][i] = Math.random();
        }
        double[] in = new double[INPUT_SIZE];
        double[] desired_out = new double[OUTPUT_SIZE];
        for (int i = 0; i < INPUT_SIZE; i++) in[i] = Math.random();
        for (int i = 0; i < OUTPUT_SIZE; i++) desired_out[i] = 0.6;
        System.out.println("Training to " + desired_out[0]);
        train(in, desired_out);
    }

    private static double dot(double[] a1, double[] a2) {
        double result = 0;
        for (int i = 0; i < a1.length; i++) result += (a1[i] * a2[i]);
        return result;
    }

    private static double dot(double x, double[] a1) {
        double result = 0;
        for (double v : a1) result += (x * v);
        return result;
    }

    private static double sigmoid(double x) {
        return 1 / (1 + Math.pow(Math.E, -x));
    }

    private static double sigmoid_derivative(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    private static void think(double[] in, double[] layer1_out, double[] layer2_out) {
        for (int i = 0; i < HIDDEN_NODES; i++) layer1_out[i] = sigmoid(dot(in, layer1_synaptic_weights[i]));
        for (int i = 0; i < OUTPUT_SIZE; i++) layer2_out[i] = sigmoid(dot(layer1_out, layer2_synaptic_weights[i]));
        System.out.println("Epoch " + count++ + " - Output from network: " + layer2_out[0]);
    }

    private static void train(double[] in, double[] desired_out) {
        count = 1;
        for (int e = 1; e <= TRAINING_ITERATIONS; e++) {
            double[] layer1_out = new double[HIDDEN_NODES];
            double[] layer1_error = new double[HIDDEN_NODES];
            double[][] layer1_delta = new double[HIDDEN_NODES][INPUT_SIZE];
            double[] layer2_out = new double[OUTPUT_SIZE];
            double[] layer2_error = new double[OUTPUT_SIZE];
            double[][] layer2_delta = new double[OUTPUT_SIZE][HIDDEN_NODES];
            think(in, layer1_out, layer2_out);
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                layer2_error[i] = (desired_out[i] - layer2_out[i]) * sigmoid_derivative(layer2_out[i]);
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    layer2_delta[i][j] = LEARNING_RATE * layer2_error[i] * layer1_out[j];
                    layer1_error[j] = dot(layer2_error[i], layer2_synaptic_weights[i]) * sigmoid_derivative(layer1_out[j]);
                    for (int k = 0; k < INPUT_SIZE; k++) {
                        layer1_delta[j][k] = LEARNING_RATE * layer1_error[j] * in[k];
                    }
                }
            }
            for (int i = 0; i < HIDDEN_NODES; i++) {
                for (int j = 0; j < INPUT_SIZE; j++) layer1_synaptic_weights[i][j] += layer1_delta[i][j];
                for (int k = 0; k < OUTPUT_SIZE; k++) layer2_synaptic_weights[k][i] += layer2_delta[k][i];
            }
        }
    }

}

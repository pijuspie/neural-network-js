import Matrix from "./matrix.js";

class NeuralNetwork {
    constructor(layers) {
        this.layers = layers;
        this.weights = [];
        this.biases = [];

        for (let i = 0; i < layers.length - 1; i++) {
            const wi = new Matrix(layers[i + 1], layers[i]).map(() => Math.random() * 2 - 1);
            const bi = new Matrix(layers[i + 1], 1).map(() => Math.random() * 2 - 1);
            this.weights.push(wi);
            this.biases.push(bi);
        }
    }

    static import(data) {
        const nn = new NeuralNetwork(data.layers);
        nn.weights = data.weights.map(x => Matrix.from2DArray(x));
        nn.biases = data.biases.map(x => Matrix.from2DArray(x));
        return nn;
    }

    export() {
        return {
            layers: this.layers,
            weights: this.weights.map(x => x.data),
            biases: this.biases.map(x => x.data),
        }
    }

    activationFunction(x) {
        return 1 / (1 + Math.exp(-x));
    }

    activationFunctionD(x) {
        return x * (1 - x);
    }

    feedForward(input) {
        if (input.length !== this.layers[0]) {
            throw new Error("Input length does not match.");
        }

        let currentInput = Matrix.fromArray(input);
        this.outputs = [currentInput];

        for (let i = 0; i < this.layers.length - 1; i++) {
            const weightedSum = Matrix.multiply(this.weights[i], currentInput);
            currentInput = weightedSum.map((x, row) => this.activationFunction(x + this.biases[i].data[row][0]));
            this.outputs.push(currentInput);
        }

        return currentInput.transpose().data[0];
    }

    train(inputs, targets, learningRate = 1) {
        if (inputs.length !== this.layers[0]) {
            throw new Error("Input length does not match.");
        }

        if (targets.length !== this.layers[this.layers.length - 1]) {
            throw new Error("Output length does not match.");
        }

        this.feedForward(inputs);
        const error = this.outputs[this.outputs.length - 1].map((x, row) => 2 * (x - targets[row]));

        let loss = 0;
        for (let i = 0; i < error.rows; i++) {
            loss += Math.pow(error.data[i][0] / 2, 2);
        }

        const gradients = this.backpropagate(error);
        for (let i = 0; i < this.weights.length; i++) {
            this.biases[i] = this.biases[i].map((x, row) => x - gradients[i].data[row][0] * learningRate);
            this.weights[i] = this.weights[i].map((x, wr, wc) => {
                return x - gradients[i].data[wr][0] * this.outputs[i].data[wc][0] * learningRate;
            });
        }

        return loss;
    }

    backpropagate(error) {
        const gradients = [];
        let currentError = error;

        for (let i = this.layers.length - 2; i >= 0; i--) {
            const weightedSums = this.outputs[i + 1].map(this.activationFunctionD);
            gradients.unshift(currentError.map((x, row) => x * weightedSums.data[row][0]));
            if (i > 0) {
                currentError = Matrix.multiply(this.weights[i].transpose(), currentError);
            }
        }

        return gradients;
    }
}

export default NeuralNetwork;
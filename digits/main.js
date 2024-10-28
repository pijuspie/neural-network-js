import NeuralNetwork from "../src/neural.js";
import { readFileSync, writeFileSync } from "fs";

const directory = "digits/";

function getNeuralNetwork(create = false) {
    if (create) {
        return new NeuralNetwork([784, 80, 10]);
    }

    const networkData = JSON.parse(readFileSync(directory + "network.json"));
    return NeuralNetwork.import(networkData);
}

function getData(testing = false) {
    let dataFileBuffer, labelFileBuffer, length;
    if (testing) {
        dataFileBuffer = readFileSync(directory + "t10k-images.idx3-ubyte");
        labelFileBuffer = readFileSync(directory + "t10k-labels.idx1-ubyte");
        length = 10000;
    } else {
        dataFileBuffer = readFileSync(directory + "train-images.idx3-ubyte");
        labelFileBuffer = readFileSync(directory + "train-labels.idx1-ubyte");
        length = 60000;
    }
    const data = [];

    for (let image = 0; image < length; image++) {
        var pixels = [];

        for (var x = 0; x <= 27; x++) {
            for (var y = 0; y <= 27; y++) {
                pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 15] / 255);
            }
        }

        const targets = [];
        for (let i = 0; i < 10; i++) {
            if (i === labelFileBuffer[image + 8]) {
                targets.push(1);
            } else {
                targets.push(0);
            }
        }

        data.push({ inputs: pixels, targets: targets });
    }

    return data;
}

function train(nn) {
    const data = getData().map(x => [x, Math.random()]).sort((a, b) => a[1] > b[1]).map(x => x[0]);
    let learningRate = 0.1;
    let reportPeriod = 1000;

    let loss = 0, max = 0;
    for (let i = 0; i < data.length; i++) {
        const li = nn.train(data[i].inputs, data[i].targets, learningRate);
        loss += li;
        max = Math.max(max, li);

        if (i % reportPeriod === 0) {
            console.log(`${Math.round(i / data.length * 100 * 100) / 100}%`, loss / reportPeriod, max);
            loss = 0;
            max = 0;
        }
    }

    writeFileSync(directory + "/network.json", JSON.stringify(nn.export()));
}

function test(nn) {
    const tests = getData(true);
    let mistakes = 0;
    for (let i = 0; i < tests.length; i++) {
        const outputs = nn.feedForward(tests[i].inputs);
        let answer = 0;
        for (let j = 1; j < 10; j++) {
            if (outputs[j] > outputs[answer]) answer = j;
        }
        let correct;
        for (let j = 0; j < 10; j++) {
            if (tests[i].targets[j] === 1) {
                correct = j;
                break;
            }
        }

        if (correct !== answer) {
            mistakes++;
        }
    }

    console.log(`Mistakes: ${mistakes} in ${tests.length}, ${100 - mistakes / tests.length * 100}% accuracy`);
}

function main() {
    const nn = getNeuralNetwork(true);
    train(nn);
    test(nn);
}

main();
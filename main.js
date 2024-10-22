import NeuralNetwork from "./src/neural.js";
import { readFileSync, writeFileSync } from "fs";

// const nn = new NeuralNetwork([784, 128, 10]);
const networkData = JSON.parse(readFileSync("models/digits.json"));
const nn = NeuralNetwork.import(networkData);

function getData(testing = false) {
    let dataFileBuffer, labelFileBuffer, length;
    if (testing) {
        dataFileBuffer = readFileSync('data/t10k-images.idx3-ubyte');
        labelFileBuffer = readFileSync('data/t10k-labels.idx1-ubyte');
        length = 10000;
    } else {
        dataFileBuffer = readFileSync('data/train-images.idx3-ubyte');
        labelFileBuffer = readFileSync('data/train-labels.idx1-ubyte');
        length = 60000;
    }
    const data = [];

    for (let image = 0; image < length; image++) {
        var pixels = [];

        for (var x = 0; x <= 27; x++) {
            for (var y = 0; y <= 27; y++) {
                pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 15]);
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

function printImage(data, image) {
    for (let i = 0; i < 28; i++) {
        const line = [];
        for (let j = 0; j < 28; j++) {
            line.push(data[image].inputs[j * 28 + i]);
        }
        console.log(line.map(x => x > 0 ? "X" : "-").join(" "));
    }
}

const data = getData();
const learningRate = 0.1;
const reportPeriod = 150;
const images = 500;

let loss = 0, max = 0;
for (let i = 0; i < 1000; i++) {
    for (let j = 0; j < reportPeriod; j++) {
        const li = nn.train(data[i % images].inputs, data[i % images].targets, learningRate);
        loss += li;
        max = Math.max(max, li);
    }

    console.log(loss / reportPeriod, max);
    loss = 0;
    max = 0;
}

writeFileSync("models/digits.json", JSON.stringify(nn.export()));

const tests = data; //getData(true);
let mistakes = 0;
for (let i = 0; i < images; i++) {
    const outputs = nn.feedForward(tests[i].inputs);
    let answer = 0;
    for (let j = 1; j < 10; j++) {
        if (outputs[j] > outputs[answer]) answer = j;
    }
    let correct;
    for (let j = 0; j < 10; j++) {
        if (data[i].targets[j] === 1) {
            correct = j;
            break;
        }
    }

    if (correct !== answer) {
        mistakes++;
    }
}

console.log(`Mistakes: ${mistakes} in ${images}, ${100 - mistakes / images * 100}% correct`);
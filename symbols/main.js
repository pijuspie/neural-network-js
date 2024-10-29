import NeuralNetwork from "../src/neural.js";
import { readFileSync, writeFileSync } from "fs";

const directory = "symbols/";
const dataLength = 442450;
const dataBuffer = readFileSync(directory + "numbers_and_letters.dataset");

const frequency = [
    6896, 7862, 6979, 7135, 6818,
    6306, 6865, 7283, 6817, 6951,
    13858, 8661, 23395, 10125, 11435,
    1160, 5759, 7208, 1119, 8488,
    5601, 11572, 12323, 18996, 57769,
    19318, 5807, 11557, 48356, 22470,
    28978, 4180, 10767, 6265, 10850,
    6072
];

function getData(image) {
    const pixels = dataBuffer.subarray(1 + 785 * image, 785 * (image + 1));

    const inputs = [];
    pixels.forEach(x => inputs.push(x / 255));

    const target = dataBuffer[image * 785];
    return { inputs, target };
}

function shuffle(array) {
    const withRandom = array.map(x => [x, Math.random()]);
    const sorted = withRandom.sort((a, b) => a[1] - b[1]);
    const result = sorted.map(x => x[0]);
    return result;
}

function train(nn) {
    const indexes = shuffle(Array.from(Array(dataLength).keys()));
    const learningRate = 0.1;
    const epochs = dataLength;
    const batchSize = 100;

    for (let i = 0; i < epochs; i += batchSize) {
        const batch = [];
        for (let j = 0; j < batchSize; j++) {
            const data = getData(indexes[i + j]);
            const targets = [];
            for (let i = 0; i < 36; i++) {
                targets.push(data.target === i ? 1 : 0);
            }
            batch.push({ inputs: data.inputs, targets });
        }

        nn.trainBatch(batch, learningRate);

        if (i % 1000 === 0) {
            const progress = i / epochs * 100;
            const rounded = Math.round(progress * 100) / 100;
            console.log(`${rounded.toFixed(2)}%`);
        }
    }

    writeFileSync(directory + "/network.json", JSON.stringify(nn.export()));
}

function test(nn) {
    const indexes = shuffle(Array.from(Array(dataLength).keys()));
    const epochs = dataLength;
    let mistakes = 0;

    for (let i = 0; i < epochs; i++) {
        const test = getData(indexes[i]);
        const outputs = nn.feedForward(test.inputs);

        let answer = 0;
        for (let j = 1; j < 36; j++) {
            if (outputs[j] > outputs[answer]) answer = j;
        }

        if (answer !== test.target) {
            mistakes++;
        }

        if (i % 1000 === 0) {
            console.log(frequency);
        }
    }

    console.log(`${100 - mistakes / epochs * 100}% accuracy`);
}

function readBMP() {
    const buffer = readFileSync(directory + "input.bmp");

    if (buffer.readUInt16LE(28) !== 8) {
        throw new Error("Not a 256-color (8-bit) BMP file.");
    }

    if (buffer.readUInt32LE(18) !== 28) {
        throw new Error("Image width should be 28px");
    }

    if (buffer.readUInt32LE(22) !== 28) {
        throw new Error("Image height should be 28px");
    }

    const pixelDataOffset = buffer.readUInt32LE(10);

    const palette = [];
    const paletteOffset = 54;
    for (let i = 0; i < 256; i++) {
        const blue = buffer[paletteOffset + i * 4];
        const green = buffer[paletteOffset + i * 4 + 1];
        const red = buffer[paletteOffset + i * 4 + 2];
        palette.push({ red, green, blue });
    }

    const data = [];
    for (let y = 27; y >= 0; y--) {
        for (let x = 0; x < 28; x++) {
            const pixelIndex = buffer[pixelDataOffset + y * 28 + x];
            const color = palette[pixelIndex];
            const grayscale = 1 - (color.red + color.green + color.blue) / 3 / 255;
            data.push(grayscale);
        }
    }
    return data;
}

function input(nn) {
    const data = readBMP();
    // const data = getData(22000).inputs;

    for (let i = 0; i < 28; i++) {
        const line = [];
        for (let j = 0; j < 28; j++) {
            line.push(data[i * 28 + j]);
        }
        console.log(line.map(x => x > 0 ? "X" : ".").join(" "));
    }

    const outputs = nn.feedForward(data);
    for (let i = 0; i < 10; i++) {
        console.log(`${i}: ${outputs[i] * 100}%`);
    }

    for (let i = 0; i < 26; i++) {
        const letter = String.fromCharCode("A".charCodeAt(0) + i);
        console.log(`${letter}: ${outputs[i + 10] * 100}%`);
    }

    let answer = 0;
    for (let j = 1; j < 36; j++) {
        if (outputs[j] > outputs[answer]) answer = j;
    }

    if (answer > 9) {
        answer = String.fromCharCode("A".charCodeAt(0) + answer - 10);
    }

    console.log(`Answer: ${answer}`);
}


// const nn = new NeuralNetwork([784, 200, 36]);
const networkData = JSON.parse(readFileSync(directory + "network.json"));
const nn = NeuralNetwork.import(networkData);

// for (let i = 0; i < 5; i++) {
//     train(nn);
//     test(nn);
// }

// test(nn);
input(nn);

class Matrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new Array(rows);
        for (let i = 0; i < rows; i++) {
            this.data[i] = new Array(cols).fill(0);
        }
    }

    static fromArray(array) {
        const matrix = new Matrix(array.length, 1);
        for (let i = 0; i < array.length; i++) {
            matrix.data[i][0] = array[i];
        }
        return matrix;
    }

    static from2DArray(array) {
        const matrix = new Matrix(array.length, array[0].length);
        for (let i = 0; i < array.length; i++) {
            for (let j = 0; j < array[0].length; j++) {
                matrix.data[i][j] = array[i][j];
            }
        }
        return matrix;
    }

    static multiply(matrixA, matrixB) {
        const rowsA = matrixA.rows;
        const colsA = matrixA.cols;
        const rowsB = matrixB.rows;
        const colsB = matrixB.cols;

        if (colsA !== rowsB) {
            throw new Error("Matrix dimensions must match for multiplication, got: colsA " + colsA + " rowsB " + rowsB);
        }

        const result = new Matrix(rowsA, colsB);
        for (let i = 0; i < rowsA; i++) {
            for (let j = 0; j < colsB; j++) {
                let sum = 0;
                for (let k = 0; k < colsA; k++) {
                    sum += matrixA.data[i][k] * matrixB.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    map(fn) {
        const result = new Matrix(this.rows, this.cols);
        for (let row = 0; row < this.rows; row++) {
            for (let col = 0; col < this.cols; col++) {
                result.data[row][col] = fn(this.data[row][col], row, col);
            }
        }
        return result;
    }

    transpose() {
        const result = new Matrix(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }
        return result;
    }
}

export default Matrix;
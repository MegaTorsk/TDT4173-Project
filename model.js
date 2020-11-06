var tf = require('@tensorflow/tfjs');
var tfn = require("@tensorflow/tfjs-node");
const { preprocess } = require('./preprocessing.js');
var preprocesser = require('./preprocessing.js');

class Model {

    async init() {
        var handler = tfn.io.fileSystem("./model/model.json")
        this.model = await tf.loadLayersModel(handler);
    }


    async run(input) {
        const processed = preprocesser.preprocess(input);
        const inputData = tf.tensor2d([processed], [1, 200]);
        const prediction = await this.model.predict(inputData);
        const values = prediction.dataSync();
        const arr = Array.from(values);  
        const result = arr.map(x => x*100); 
        return result;
    }
}

const model = new Model();
module.exports = model;
import React, { useState, useEffect } from 'react';
import './App.css';
import { Button, Fade, Form, Spinner } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import BarChart from './components/barChart.js';
import * as tf from '@tensorflow/tfjs';


function App() {
  const tokenizer = require('./tokenizer.json');
  const [showResult, toggleShowResult] = useState(false);
  const [textInput, setTextInput] = useState("");
  const [resultList, setResultList] = useState([0,0,0]);
  const [isLoading, setIsLoading] = useState(false);
  //const [model, setModel] = useState();
/*   useEffect(() => {
    async function fetchModel(){
      const model = await tf.loadLayersModel('model/model.json');
      setModel(cls);
      console.log("loaded");
    }
    fetchModel();
    
    console.log("Loading AI...")
  },[]); */




  const submitButtonClick = async function(event) {
    setIsLoading(true);
    setResultList(await evaluateInput(textInput));
    toggleShowResult(true);
    setIsLoading(false);
  }

  const handleChange = e => {
    setTextInput(e.target.value);
  }

/*   const inputPreprocessing = input => {
    
  } */

  const evaluateInput = async function(input) { //Sett inn AI her
    //const model = await tf.loadLayersModel('model/model.json');
    /*const inputData = tf.tensor2d([[   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
      0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
      0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
      0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
      0,    0,    0,    1,    2,   10,  112,    1,   37,   67,  280,    4,  677,    5,
    547,    9,   12, 1417,   11,  851, 1185,    8,   11,    3,  257,   89,   17,   39,
      9,   12, 2834,   26, 7256,    6,    3,  969,   14,    3, 7256, 1185, 1301,   94,
     52,   12,    2,  123,    5,  492,  190,  242,  501,  969,   30,    3, 7256,   11,
      3,   53,    1,  467,    9,    3, 2095, 1345,  408,   12,  466,   31, 2095,  123,
      9,   48, 2614,    7,   56, 1593,   14,    3, 2614,    8,  969,  366,   73,  358,
      9,  480, 1801,    3,   91, 7483,    1,    2,   28,   87,  775,    3,   91,  386,
      8, 3145,   22,    1,  311,  969,    7,   13,   10,   66,   56,    5,  120,  462,
   6703,    1,    5, 3455,  756,    4,   37,  510,    1,   13,   10,   78,    4,  677,
      3,  514,  408,    4,  239, 2095, 1345,    4, 1939,    3,    8,    3,    1, 1118,
     63,    2, 3098,    6]], [1, 200]); */
    var words = input.split(" ");
    var inputData = new Array(200);
    var wordIndex = words.length - 1;
    for(var i = 199; i >= 0; i--){
        if(wordIndex >= 0){
            inputData[i] = tokenizer[words[wordIndex--]];
        }else{
            inputData[i] = 0;
        }
    }

    //const prediction = model.predict(tf.tensor2d([inputData], [1, 200]));
    //const values = prediction.dataSync();
    //const arr = Array.from(values);

    //gang verdiene med 100
    //const result = arr.map(x => x*100);

    //returner verdier i form av liste
    //console.log(arr);
    return [10,10,10];//result;
  }

  return (
    <div className="App">
      <Fade in={!showResult} timeout={1000}>
        <div className={showResult ? 'hidden' : 'InputSection'}>
          <h3 className="mt-5">Input a comment</h3>
          <Form>
            <Form.Control as="textarea" rows="4" placeholder='Write your comment here...' onChange={handleChange} />
          </Form>
          <Button variant="outline-dark" style={{margin: 20}} onClick={submitButtonClick}>{!isLoading ? "Submit" : <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true"/>}</Button>
        </div>
      </Fade>
      <Fade in={showResult} timeout={1000}>
        <div className={showResult ? 'ResultSection' : 'hidden'}>
          {showResult ? <BarChart resultingValues={resultList} /> : null}
          {showResult ? <Button variant="outline-dark" style={{margin: 20}} onClick={() => toggleShowResult(!showResult)}>Try again</Button> : null}
        </div>
      </Fade>
    </div>
  );
}

export default App;

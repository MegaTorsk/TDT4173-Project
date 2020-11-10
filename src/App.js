import React, { useState } from 'react';
import './App.css';
import { Button, Fade, Form, Spinner } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import BarChart from './components/barChart.js';
//import { Text } from 'react-native-elements';
import axios from 'axios';


function App() {
  const [showResult, toggleShowResult] = useState(false);
  const [textInput, setTextInput] = useState("");
  const [resultList, setResultList] = useState([0,0,0]);
  const [isLoading, setIsLoading] = useState(false);
  const [showAbout, toggleAbout] = useState(false);




  const submitButtonClick = async function(event) {
    setIsLoading(true);
    setResultList(await evaluateInput(textInput));
    toggleShowResult(true);
    setIsLoading(false);
  }

  const handleChange = e => {
    setTextInput(e.target.value);
  }

  const evaluateInput = async function(input) {
    var axiosResult = null;
    await axios.post('https://api-commentclassifier.herokuapp.com/textSubmit/', { input })
      .then(res => {
        axiosResult = res;
      })
    return axiosResult.data;
  }

  return (
    <div className="App-wrapper">
      <div className={showAbout ? "hidden" : "App"}>
        <Fade in={(!showResult && !showAbout)} timeout={1000}>
          <div className={showResult ? 'hidden' : 'InputSection'}>
            <h3 className="mt-5">Input a comment</h3>
            <Form>
              <Form.Control as="textarea" rows="4" style={{marginTop: "3%"}} placeholder='Write your comment here...' onChange={handleChange} />
            </Form>
            <Button variant="outline-dark" style={{margin: "4%"}} onClick={submitButtonClick}>{!isLoading ? "Submit" : <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true"/>}</Button>
          </div>
        </Fade>
        <Fade in={(showResult && !showAbout)} timeout={1000}>
          <div className={showResult ? 'ResultSection' : 'hidden'}>
            {showResult ? <BarChart resultingValues={resultList}/> : null}
            {showResult ? <Button variant="outline-dark" style={{margin: "4%"}} onClick={() => toggleShowResult(!showResult)}>Try again</Button> : null}
          </div>
        </Fade> 
      </div>
      <Fade in={showAbout} timeout={1000}>
          <div className={showAbout ? "about-wrapper" : "hidden"}>
            <h1>The Comment Classifier</h1>
            <p style={{fontSize: "80%", fontStyle: "italic"}}>Created by Ole Fredrik Berg, Alexander Szewczyk and Albert Johannessen</p>
            <p style={{paddingTop: "1%"}}>Ever wonder where your comment writing style fits in on the internet?</p>
            <p>With the help from The Comment Classifier, you'll finally be able to get to know if you best fit in the YouTube, Hacker News or Reddit comment sections.</p>
            <p>Simply enter one of the comments you would usually write in the input field, and press submit!</p>
            <p>
              Our advanced bidirectional LSTM machine learning model used approximately 12 million samples from the comment sections of YouTube, Hacker News and Reddit
              during its learning phase to ble able to classify comments with ~80% accuracy.
             </p>
             <button className={showAbout ? "button" : "hidden"} onClick={() => toggleAbout(false)}> Close </button>
          </div>
        </Fade>
      <Fade in={!showAbout} timeout={1000}>
        <button className={!showAbout ? "button about-button" : "hidden"} onClick={() => toggleAbout(true)}> About</button>
      </Fade>
    </div>
  );
}

export default App;

import React, { useState } from 'react';
import './App.css';
import { Button, Fade, Form, Spinner } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import BarChart from './components/barChart.js';
import axios from 'axios';


function App() {
  const [showResult, toggleShowResult] = useState(false);
  const [textInput, setTextInput] = useState("");
  const [resultList, setResultList] = useState([0,0,0]);
  const [isLoading, setIsLoading] = useState(false);




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
    console.log('before')
    await axios.post('https://api-commentclassifier.herokuapp.com/textSubmit/', { input })
      .then(res => {
        axiosResult = res;
      })
    console.log(axiosResult.data);
    return axiosResult.data;
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

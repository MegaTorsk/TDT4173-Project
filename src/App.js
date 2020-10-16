import React, { useState } from 'react';
import './App.css';
import { Button, Fade, Form } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import BarChart from './components/barChart.js';


function App() {

  const [showResult, toggleShowResult] = useState(false);
  const [textInput, setTextInput] = useState("");
  const [resultList, setResultList] = useState([0,0,0]);

  const submitButtonClick = event => {
    setResultList(evaluateInput(textInput));
    toggleShowResult(true);
    console.log(textInput);
  }

  const handleChange = e => {
    setTextInput(e.target.value);
  }

  const evaluateInput = input => { //Sett inn AI her
    return [22, 68, 10];
  }

  return (
    <div className="App">
      <Fade in={!showResult} timeout={10000}>
        <div className={showResult ? 'hidden' : 'InputSection'}>
          <h3 className="mt-5">Input a comment</h3>
          <Form>
            <Form.Control as="textarea" rows="4" placeholder='Write your comment here...' onChange={handleChange} />
          </Form>
          <Button variant="outline-dark" style={{margin: 20}} onClick={submitButtonClick}>Submit</Button>
        </div>
      </Fade>
      <Fade in={showResult} timeout={10000}>
        <div className={showResult ? 'ResultSection' : 'hidden'}>
          {showResult ? <BarChart resultingValues={resultList} /> : null}
          {showResult ? <Button variant="outline-dark" style={{margin: 20}} onClick={() => toggleShowResult(!showResult)}>Try again</Button> : null}
        </div>
      </Fade>
    </div>
  );
}

export default App;

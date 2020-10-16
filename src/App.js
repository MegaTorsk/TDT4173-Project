import React, { useState } from 'react';
import './App.css';
import { Form } from 'react-bootstrap';
import Button from 'react-bootstrap/Button';
import 'bootstrap/dist/css/bootstrap.min.css';
import BarChart from './components/barChart.js';


function App() {

  const [showResult, toggleShowResult] = useState(false);
  const [textInput, setTextInput] = useState("")

  const submitButtonClick = event => {
    toggleShowResult(true);
    setTextInput(event.target.value);
    console.log(textInput);
  }

  return (
    <div className="App">
      <div className={showResult ? 'hidden' : 'InputSection'}>
        {/*<div className='InputSectionTitle'>Input comment</div>*/}
        <h3 className="mt-5">Input a comment</h3>
        <Form>
          <Form.Control as="textarea" rows="4" placeholder='Write your comment here...' />
        </Form>
        <Button variant="outline-dark" onClick={submitButtonClick} style={{margin: 20}}>Submit</Button>
      </div>
      <div className={showResult ? 'ResultSection' : 'hidden'}>
        {showResult ? <BarChart/> : null}
        {showResult ? <Button variant="outline-dark" onClick={() => toggleShowResult(!showResult)}>Try again</Button> : null}
      </div>
    </div>
  );
}

export default App;

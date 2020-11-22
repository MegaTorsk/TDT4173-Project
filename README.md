# TDT4173-Project
This is a the server side code for a comment classifier project. The server code is based Node.js, Express.js and uses TensorFlow.js for the loading and use of a comment classifier model.

## Code Structure
The server initializes at `app.js`. `app.js` will instantiate the model and load the TensorFlow.js model so that it can be used in the `/textSubmit` route. The model uses the preprocesser object as defined in `preprocessing.js`. When the server receives a post request on the `/textSubmit` route,  the input payload will be handled by the model's run function, as defined in `model.js`.

## Pre-requisits
Make sure to have `npm` and `nodejs` installed. Node.js version should be >= 14.15.1. An installation guide for Node.js and npm can be found on the [npm website](https://www.npmjs.com/get-npm).

## Running Locally
Firstly, clone the repository
```
git clone https://github.com/MegaTorsk/TDT4173-Project.git
```
Make sure to be on the server branch, and run
```
npm install
```
Run the server locally using the command
```
npm start
```

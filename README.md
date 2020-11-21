# TDT4173-Project
This is a the frontend code for a comment classifier project. The website was built using Node.js and React.js.

## Pre-requisits
Make sure to have `npm` and `nodejs` installed. Node.js version should be >= 14.15.1. For how to do this visit the [npm website](https://www.npmjs.com/get-npm).

## Running Locally
Firstly, clone the repository
```
git clone https://github.com/MegaTorsk/TDT4173-Project.git
```
Make sure to be on the website branch, and run
```
npm install
```
After the installation is finished, simply use the command
```
npm start
```
The website will then run locally on http://localhost:3000/ if nothing else is specified.

## Code Structure
App.js renders 4 main bootstrap Fade elements, 3 of which contain the three main pages of the website. The fourth Fade element contains the "About button". The three pages are:

- Submit page, contains a title, text area and submit button.
- Result page, contains a title, the bootstrap barchart showing the results, and a "Try again" button.
- About page, show information about the project and website.

By pressing the submit button, an axios call is made to the backend. A TensorFlow.js model on the backend-side will then evaluate the value that was in the text area prior to the submit button being clicked. The response from the backend is then displayed on the barchart in the Result page.

var express = require('express');
var router = express.Router();
var model = require("../model.js");


/* GET users listing. */

router.post('/', async function(req, res, next) {
  var result;
  try {
    result = await model.run(req.body.input);
  } catch (error) {
    console.log(error);
    result = [0,0,0];
  }
  
  res.send(result);
});

module.exports = router;

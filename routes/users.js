var express = require('express');
var router = express.Router();


/* GET users listing. */
router.get('/', function(req, res, next) {
  res.json({"1": 11, "2": 11, "3": 11});
});

module.exports = router;

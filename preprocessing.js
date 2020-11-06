const tokenizer = require('./tokenizer.json');
const lemmatizer = require('./lemmatizer.json');
const emojiUnicode = require('emoji-dictionary');
const emojis = emojiUnicode.unicode;
const alphabet = 'abcdefghijklmnopqrstuvwxyz'.split('');

class PreProcesser {

    

    tokenize(input) {
        var words = input.split(" ");
        var inputData = new Array(200);
        var wordIndex = words.length - 1;
        for(var i = 199; i >= 0; i--){
            if (!tokenizer.hasOwnProperty(words[wordIndex])){
                inputData[i] = 0;
                wordIndex--;
                continue
            }
            if(wordIndex >= 0){
                inputData[i] = tokenizer[words[wordIndex--]];
            }else{
                inputData[i] = 0;
            }
        }
        return inputData;
    }

    preprocess(input) {
        input = input.toLowerCase();
        input = input.split("<i>").join(" ");
        input = input.split("\\n").join(" ");
        input = input.split("<p>").join(" ");
        input = input.split("</i>").join(" ");
        input = input.split("</p>").join(" ");
        const htmlRegex = /https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)/gi;
        input = input.replace(htmlRegex, "-url-");
        const specCharRegex = /\&.{0,4}\;/gi;
        input = input.replace(specCharRegex, "");
        const emojiRegex = /\ud83d[\ude00-\ude4f]/gi;
        input = input.replace(emojiRegex, " -emoji- ");
        //Remove disallowed characters
        var allowedChars = "";
        input = input.split(" ");
        for (let i=0; i<input.length; i++){
            if (!["-emoji-","-url-"].includes(input[i])) {
                allowedChars = ""
                for (let j=0; j<input[i].length; j++) {
                    if (alphabet.includes(input[i][j])){
                        allowedChars = allowedChars + input[i][j];
                    }
                }
                input[i] = allowedChars;
            }
        }
        for (let i = 0; i<input.length; i++){
            if(lemmatizer.hasOwnProperty(input[i])){
                input[i] = lemmatizer[input[i]];
            }
        }
        input = input.join(" ");
        input = input.replace(/\ +/gi, " ");
        input = input.trim();
        return this.tokenize(input);
    }
}

const preProcesser = new PreProcesser();
module.exports = preProcesser;
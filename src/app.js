var express =require('express');
var http = require('http');
var path = require('path');
var skeleton = require('./modules/extract-skeleton.js');
var cookieParser = require('cookie-parser');
var expressSession = require('express-session');
var fs = require('fs');
var url = require('url');
var qs = require('querystring');
var mysql = require('mysql');
var sanitizeHtml = require('sanitize-html');

var app = express();
var router = express.Router();
var static = require('serve-static');
var multer = require('multer');
var cors = require('cors');

const {spawn} = require('child_process');
const INPUT_VIDEO_DIRECTORY = 'uploads';
const INPUT_VIDEO_NAME = 'input.mp4';

var db = mysql.createConnection({
    host:'localhost',
    user:'root',
    password:'dancearch',
    database:'dancearch'
});


app.use('/views', static(path.join(__dirname, 'views')));
app.use(cors());
app.use(cookieParser());

app.use(expressSession({
    secret: 'password',
    resave: true,
    saveUninitialized: true
}));


var storage = multer.diskStorage({
    destination: function(req, file, callback){
        callback(null, INPUT_VIDEO_DIRECTORY);
    },
    filename: function(req, file, callback) {
        callback(null, INPUT_VIDEO_NAME);
    }
})

var upload = multer({
    storage: storage,
    limits: {
        files: 1,
    }
})

router.route('/').get(function(req, res, next){
    res.redirect('/views/home.html');
})

router.route('/process/videoUpload').post(upload.array('video', 1), function(req, res){

    console.log('/process/videoUpload 호출');

    try {
        var files = req.files;
        console.dir(req.files[0]);

        var originalname = '';
        var filename = '';
        var mimetype = '';
        var size = 0;

        originalname = files[0].originalname;
        filename = files[0].filename;
        mimetype = files[0].mimetype;
        size = files[0].size;


        res.redirect('/views/loading.html');

        return;

    } catch(err) {
        console.dir(err.stack);
    }
}) // /process/photo

router.route('/process/runModel').get(async function(req, res) {
        console.log('run model');
        

        skeleton.run(function(list){
            datas = list.slice(9, -3).split(',').map(function(item){return parseInt(item, 10)}); 
            console.log(datas);
            res.cookie('rank', {
                rank: datas
            })
            res.end();
        })   
        
            /*
            db.query(`SELECT * FROM Song WHERE id=`+datas[0],  await function(err, data){
                console.log('data:', data);
                res.cookie('first', {
                    title: data[0].title,
                    singer: data[0].singer,
                    album: data[0].img,
                    link: data[0].link
                })
            });

            db.query(`SELECT * FROM Song WHERE id=`+datas[1],  await function(err, data){
                console.log('data:', data);
                res.cookie('second', {
                    title: data[0].title,
                    singer: data[0].singer,
                    album: data[0].img,
                    link: data[0].link
                })
            });

            db.query(`SELECT * FROM Song WHERE id=`+datas[2],  await function(err, data){
                console.log('data:', data[0]);

                console.log('data:', data[0].title);
                res.cookie('third', {
                    title: data[0].title,
                    singer: data[0].singer,
                    album: data[0].img,
                    link: data[0].link
                })
                res.end();     
            });
            */
});

router.route('/process/toResult').post(function(req, res, next){

    data = req.cookies.rank.rank


    db.query(`SELECT * FROM Song WHERE id=`+datas[0], function(err, data){
        console.log('data:', data);
        res.cookie('first', {
            title: data[0].title,
            singer: data[0].singer,
            album: data[0].img,
            link: data[0].link
        })
    });

    db.query(`SELECT * FROM Song WHERE id=`+datas[1], function(err, data){
        console.log('data:', data);
        res.cookie('second', {
            title: data[0].title,
            singer: data[0].singer,
            album: data[0].img,
            link: data[0].link
        })
    });

    db.query(`SELECT * FROM Song WHERE id=`+datas[2], function(err, data){
        console.log('data:', data[0]);

        console.log('data:', data[0].title);
        res.cookie('third', {
            title: data[0].title,
            singer: data[0].singer,
            album: data[0].img,
            link: data[0].link
        })
        res.redirect('/views/result.html')

        res.end();     
    });
    
})



app.use('/', router);


http.createServer(app).listen('3000',
function(){

    console.log('Express 서버가 3000번 포트에서 시작됨.');
    db.connect();
});
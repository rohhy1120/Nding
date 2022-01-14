var http = require('http');
var fs = require('fs');
var url = require('url');
var qs = require('querystring');
var template = require('./template.js');
var path = require('path');
var sanitizeHtml = require('sanitize-html');

var mysql = require('mysql');
var express = require('express') // npm install express --save
var app = express()


var db = mysql.createConnection({
  host:'localhost',
  user:'dancearch',
  password:'dancearch',
  database:'dancearch'
});

db.connect();



app.use(express.static('src'));

var app = http.createServer(function(request,response){
    var _url = request.url;
    var queryData = url.parse(_url, true).query;
    var pathname = url.parse(_url, true).pathname;
    
    if(pathname === '/'){
      if(queryData.id === undefined){
        db.query(`SELECT * FROM Song`, function(error,Song){
          var title = 'Dancearh';
          var description = '관리자 화면';
          var list = template.list(Song);
          var html = template.HTML(title, list,
            `<h2>${title}</h2>${description}`,
            `<a href="/create">안무 등록</a>`
          );
          response.writeHead(200);
          response.end(html);
        });
      } else {
        db.query(`SELECT * FROM Song`, function(error,Songs){
         if(error){
           throw error;
         }
         db.query(`SELECT * FROM Song WHERE id=?`,[queryData.id], function(error2, Song){
           if(error2){
             throw error2;
           }
          var title = Song[0].title;
          var singer = Song[0].singer;
          var list = template.list(Songs);
          var html = template.HTML(title, list,
            `<h2>${title}</h2>
            <h3>${singer}</h3>
            <h3>${Song[0].album} 앨범</h3>
            <h3>${Song[0].link} 링크</h3>`,
            ` <a href="/create">안무 등록</a>
                <a href="/update?id=${queryData.id}">안무 수정</a>
                <form action="delete_process" method="post">
                  <input type="hidden" name="id" value="${queryData.id}">
                  <input type="submit" value="안무 삭제">
                </form>`
          );
          response.writeHead(200);
          response.end(html);
         })
      });
      }
    } else if(pathname === '/create'){
      db.query(`SELECT * FROM Song`, function(error,Songs){
        var title = 'Create';
        var list = template.list(Songs);
        var html = template.HTML(title, list,
          `
          <form action="/create_process" method="post">
            <p><input type="text" name="title" placeholder="제목"></p>
            <p>
              <input type="text" name="singer" placeholder="가수">
            </p>
            <p>
              <input type="text" name="album" placeholder="앨범">
            </p>
            <p>
              <input type="text" name="link" placeholder="링크">
            </p>
            <p>
              <input type="submit">
            </p>
          </form>
          `,
          `<a href="/create">create</a>`
        );
        response.writeHead(200);
        response.end(html);
      });
    } else if(pathname === '/create_process'){
      var body = '';
      request.on('data', function(data){
          body = body + data;
      });
      request.on('end', function(){
          var post = qs.parse(body);
          db.query(`
            INSERT INTO Song (title, singer, album, link)
              VALUES(?, ?, ?, ?)`,
            [post.title, post.singer,post.album,post.link],
            function(error, Song){
              if(error){
                throw error;
              }
              response.writeHead(302, {Location: `/?id=${Song.insertId}`});
              response.end();
            }
          )
      });
    } else if(pathname === '/update'){
      db.query('SELECT * FROM Song', function(error, Songs){
        if(error){
          throw error;
        }
        db.query(`SELECT * FROM Song WHERE id=?`,[queryData.id], function(error2, Song){
          if(error2){
            throw error2;
          }
          var list = template.list(Songs);
          var html = template.HTML(Song[0].title, list,
            `
            <form action="/update_process" method="post">
              <input type="hidden" name="id" value="${Song[0].id}">
              <p><input type="text" name="title" placeholder="제목" value="${Song[0].title}"></p>
              <p>
                <input type="text" name="singer" placeholder="가수" value="${Song[0].singer}">
              </p>
              <p>
                <input type="text" name="album" placeholder="앨범" value="${Song[0].album}">
              </p>
              <p>
                <input type="text" name="link" placeholder="링크" value="${Song[0].link}">
              </p>
              <p>
                <input type="submit">
              </p>
            </form>
            `,
            `<a href="/create">create</a> <a href="/update?id=${Song[0].id}">update</a>`
          );
          response.writeHead(200);
          response.end(html);
        });
      });
    } else if(pathname === '/update_process'){
      var body = '';
      request.on('data', function(data){
          body = body + data;
      });
      request.on('end', function(){
          var post = qs.parse(body);
          db.query('UPDATE Song SET title=?, singer=?, album=?, link=?  WHERE id=?', [post.title, post.singer, post.album,post.link,post.id], function(error, result){
            response.writeHead(302, {Location: `/?id=${post.id}`});
            response.end();
          })
      });
    } else if(pathname === '/delete_process'){
      var body = '';
      request.on('data', function(data){
          body = body + data;
      });
      request.on('end', function(){
          var post = qs.parse(body);
          db.query('DELETE FROM Song WHERE id = ?', [post.id], function(error, result){
            if(error){
              throw error;
            }
            response.writeHead(302, {Location: `/`});
            response.end();
          });
      });
    } else {
      response.writeHead(404);
      response.end('Not found');
    }
});
app.listen(3000);

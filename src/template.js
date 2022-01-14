module.exports = {
  HTML: function(title, list, body, control) {
    return `
    <!doctype html>
    <html>
    <head>
      <title>WEB1 - ${title}</title>
      <meta charset="utf-8">
        <style>
        @import './views/home.css';
        #container{
          background-color:powerblue;
          display:flex;
        }
        #item{
          color:white;
          width:200px;
        }
          div {
            border: 1px solid red;
          }
        </style>
    </head>
    <body>
    <h1><a href="/">관리자</a></h1>
  <div id="container" style="margin-top: 100px;">
    <div id="item">
      ${control}
    </div>
    <div id="item">
      ${list}
    </div>
    <div id="item" style="width:700px; color:Black;">
      ${body}
    </div>
  </div>
    </body>
    </html>
    `;
  },
  list: function(Songs) {
    var list = '<ul>';
    var i = 0;
    while (i < Songs.length) {
      list = list + `<li><a href="/?id=${Songs[i].id}">${Songs[i].title}</a></li>`;
      i = i + 1;
    }
    list = list + '</ul>';
    return list;
  }
}

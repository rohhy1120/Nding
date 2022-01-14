

function getRankData(){
    var first = getCookie('first');
    var second = getCookie('second');
    var third = getCookie('third');

    console.log(first);
    setData(first, 'first');
    setData(second, 'second');
    setData(third, 'third');


}

function getCookie(cname){
    var name = cname+"=";
        var decodedCookie = decodeURIComponent(document.cookie);
        var ca = decodedCookie.split(';');
        for(var i = 0; i <ca.length; i++) {
            var c = ca[i];
            while (c.charAt(0) == ' ') {
                c = c.substring(1);
            }
            if (c.indexOf(name) == 0) {
                var data=c.substring(name.length, c.length).slice(2,);
                return JSON.parse(data);
            }
        }
        return "";
}

function setData(cookie, rank){
    var title = document.getElementById(rank+'Title');
    var singer = document.getElementById(rank+'Singer');
    var img = document.getElementById(rank + 'Img');

    title.innerHTML = cookie.title;
    singer.innerHTML = cookie.singer;
    img.src = 'image/'+cookie.title+'.png';
}
window.onload = function(){
    getRankData();
}

const file = document.getElementById('file');
let video = '';

function checkVideo(){

    const inputFile = document.getElementById('file');

    document.getElementById('videoIcon').style.display = 'none';
    document.getElementById('fileCheck').style.display = 'inline-block';
    // document.getElementById('fileUpload').style.display = 'none';



    const fReader = new FileReader();
    fReader.readAsDataURL(inputFile.files[0]);
    fReader.onloadend = function(event){
        document.getElementById('video').src = event.target.result;
        document.getElementById('nextBtn').value = inputFile.files[0].name;
    }

} // checkVideo()

function extractPoint(){

}
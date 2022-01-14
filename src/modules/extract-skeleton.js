
module.exports = {
    run: function(callback){
        var largeDataSet = [];
        const {spawn} = require('child_process');
        var result = '';

        console.log('skeleton 실행됨');

        var largeDataSet = [];
        // spawn new child process to call the python script
        // const python = spawn('python', ['test_out/script3.py']);
        const python = spawn('python', ['./models/Final/final_code.py']);
        
        // collect data from script
        python.stdout.on('data', function (data) {
            console.log('Pipe data from python script ...');
            largeDataSet.push(data.toString());
            
        });

        // in close event we are sure that stream is from child process is closed
        python.on('close', (code) => {
            console.log(`child process close all stdio with code ${code}`);
            callback(largeDataSet[0]);
        });

    }
}
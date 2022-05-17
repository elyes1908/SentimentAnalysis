$(document).ready(function () {

    $('#dataset_table').hide();
    $('#form').on('submit', function (event) {
        $('#load').show();
        $('#submit').hide();
        $.ajax({
            data: {
                comment_input: $('#comment').val(),

            }, type: 'POST', url: '/process'
        })
            .done(function (data) {
                if (data.error) {
                } else {
                    $('#load').hide()
                    $('#submit').show()
                    $('#result').show();
                    if (data.class_ === 'POSITIVE') {
                        $("#result")[0].innerHTML = '👍 The sentiment of the text is ' + data.class_.fontcolor('green') + ' with a score of ' + data.score_ + '.';
                        $('.gif')[0].src = '../static/images/GIF_HAPPY.gif';
                    } else {
                        $("#result")[0].innerHTML = '👎 The sentiment of the text is ' + data.class_.fontcolor('red') + ' with a score of ' + data.score_ + '.';
                        $('.gif')[0].src = '../static/images/GIF_MAD.gif';
                    }


                }
            });
        event.preventDefault();
    });

    $("#files").change(function () {
        $('#filename')[0].innerHTML = this.files[0].name;
        document.body.style.overflowY = 'scroll';
    });

    $('#dataset').on('submit', function (event) {

        $('.buttonload.upload').hide();
        $('#loadupload').show();
        if ($('#filename')[0].innerHTML===''){
            alert('You have to select excel file');
            document.location.reload();

        }

        $.ajax({
            data: new FormData(this), type: 'POST', url: '/upload', contentType: false, processData: false,
        })
            .done(function (data) {
                if (data.error) {
                } else {
                    $('#dataset_table').show()
                    var table = $('#table');
                    var neg = 0;
                    var pos = 0;
                    for (let i = 0; i < data.len; i++) {
                        var row = table[0].insertRow(i);
                        row.insertCell(0).innerHTML = data.file[i]['Task']
                        row.insertCell(1).innerHTML = data.file[i]['CommentValue'];
                        row.insertCell(2).innerHTML = data.file[i]['prediction'];


                        if (data.file[i]['prediction'] === 'NEGATIVE') {
                            row.style.backgroundColor = "#FFB7B2";
                            row.classList.add('negative')
                            neg += 1

                        } else {
                            row.style.backgroundColor = "#89FEA8";
                            row.classList.add('positive');
                            pos += 1
                        }

                    }

                    chart('Positives', (pos / (neg + pos)) * 100, '#poss')
                    chart('Negatives', (neg / (neg + pos)) * 100, '#negs')
                    $('.buttonload.upload').show();
                    $('#loadupload').hide();

                }
            });
        event.preventDefault();
    });

    $('#pos').click(function (event) {
        $('.negative').hide();
        $('.positive').show();

        event.preventDefault();
    });


    $('#neg').click(function (event) {
        $('.negative').show();
        $('.positive').hide();

        event.preventDefault();
    });
    $('#ref').click(function (event) {
        $('.negative').show();
        $('.positive').show();

        event.preventDefault();
    });

    function chart(label, val, id) {
        let color1;
        if (label === 'Negatives') {
            color1 = "#e62020";
            color2 = "#f98787";


        } else {
            color1 = "#20e647";
            color2 = "#87d4f9";
        }
        var options = {
            chart: {
                height: 200,
                type: "radialBar",
            },

            series: [val],
            colors: [color1],
            plotOptions: {
                radialBar: {
                    hollow: {
                        margin: 0,
                        size: "70%",
                        background: "rgba(62,212,224,0.18)"
                    },
                    track: {
                        dropShadow: {
                            enabled: true,
                            top: 2,
                            left: 0,
                            blur: 4,
                            opacity: 0.15
                        }
                    },
                    dataLabels: {
                        name: {
                            offsetY: -10,
                            color: "#fff",
                            fontSize: "15px"
                        },
                        value: {
                            color: "#fff",
                            fontSize: "30px",
                            show: true
                        }
                    }
                }
            },
            fill: {
                type: "gradient",
                gradient: {
                    shade: "dark",
                    type: "vertical",
                    gradientToColors: [color2],
                    stops: [0, 100]
                }
            },
            stroke: {
                lineCap: "round"
            },
            labels: [label]
        };

        var chart = new ApexCharts(document.querySelector(id), options);

        chart.render();
    }


});
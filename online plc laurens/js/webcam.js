/**
 * web camera js
 * */
var webcamJs = (function () {
    function webcamJs() {
    }

    /** handle size property */
    webcamJs.updateSize = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var w = json.width;
        var h = json.height;
        var obj = $("#" + id);
        obj.css("width", w);
        obj.css("height", h);
    }
    webcamJs.updateVideoType = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var newValue = json.newValue;
        $("#" + id).attr("video_type", newValue);
        var stream = $("#" + id).attr("stream");
        updateCamStream(id, stream);
    }
    /** handle stream  property */
    webcamJs.updateStream = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var newValue = json.newValue;
        updateCamStream(id, newValue);
    }

    function updateCamStream(id, newValue) {
        $("#" + id).attr("stream", newValue);

        //mjpge: 
        $("#" + id + " .cam_img").attr("src", newValue);
        $("#" + id + " .cam_video").html("");
        $("#" + id + " .cam_video").attr("id", "video_" + id);
        if($("#" + id).attr("video_type") === "0"){
            //mjpge: default, use <image> 
            return;
        }
        else if ($("#" + id).attr("video_type") === "4") {
            //mpeg
            $("#" + id + " .cam_video").append($(document.createElement("source")).attr('src', newValue).attr('type', "application/x-mpegURL"));
            $("#" + id + " .cam_img").attr("src", "");
        } 
        else {
            //mp4,webm,ogg
            $("#" + id + " .cam_video").append($(document.createElement("source")).attr('src', newValue).attr('type', "video/mp4"));
            $("#" + id + " .cam_video").append($(document.createElement("source")).attr('src', newValue).attr('type', "video/ogg"));
            $("#" + id + " .cam_video").append($(document.createElement("source")).attr('src', newValue).attr('type', "video/webm"));
            $("#" + id + " .cam_img").attr("src", "");
        }
    }

    /** 
     * run-time init all camera component
     * */
    webcamJs.initCameraInRunTime = function () {
        $(".webcam").each(function (index, element) {
            var id = element.getAttribute("id");
            var status = element.getAttribute("video_type");
            if (status === "0") {
                //mjpeg
                $("#" + id + " .cam_img").attr("onerror", "javascript:webcamJs.onImageError(this)");
                $("#" + id + " .cam_img").css("display", "block");
                window.setInterval(function () {
                    var image = $("#" + id + " .cam_img");
                    var src = $("#" + id).attr("stream");
                    image.attr("src", src + "?" + (new Date()).getTime());
                    $("#" + id).find(".cam_img_x").hide('slow');
                    // console.log(image);
                }, 5000);

            } else {
                $("#" + id + " .cam_video").css("display", "block");
                var videoId = $("#" + id + " .cam_video").attr("id");
                var player = videojs(videoId, {
                    muted: true,
                    controls: false,
                    loop: true,
                    autoplay: true
                });
                player.play();
            }
        });
    }

    /** runtime on video playing
     webcamJs.videoPlaying = function (obj) {
     $(obj).parent().children(".marker").html("");
     }*/
    /** runtime on video error
     webcamJs.videoOnError = function (obj) {
     $(obj).parent().children(".marker").html("");
     }*/


    webcamJs.onImageError = function (obj) {
        if ($(obj).parent().parent().attr("video_type") === "0") {
            $(obj).parent().find(".cam_img_x").css("display", "block");
        }
    }


    /** handle Tooltip property */
    webcamJs.updateTooltip = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var newValue = json.newValue;
        $("#" + id).attr("title", newValue);
    }
    /** handle title property */
    webcamJs.updateTitleText = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var newValue = json.newValue;
        $("#" + id + " .title").html(newValue);
    }
    /** handle title FontSize */
    webcamJs.updateTitleFontSize = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var newValue = json.newValue;
        var obj = $("#" + id + " .title");
        if (!newValue) {
            newValue = "14";/*default font size*/
        }
        obj.css("font-size", parseInt(newValue));
    };
    /** handle title font style property */
    webcamJs.updateTitleFontStyle = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var fontStyle = json.style + "";
        var fontFamily = json.family;
        var fontSize = json.size;
        var inputVal = $("#" + id + " .title");
        if (fontStyle === "0") {//normal
            inputVal.css({"font-style": "normal", "font-weight": ""});
        }
        else if (fontStyle === "1") {   //bold
            inputVal.css({"font-style": "normal", "font-weight": "bold"});
        }
        else if (fontStyle === "2") {   //italic
            inputVal.css({"font-style": "italic", "font-weight": ""});
        }
        else if (fontStyle === "3") {   //italic bold
            inputVal.css({"font-style": "italic", "font-weight": "bold"});
        }
        inputVal.css({"font-family": fontFamily, "font-size": parseInt(fontSize)});
    }
    /** handle title color property */
    webcamJs.updateTitleColor = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var color = json.newValue;
        var inputVal = $("#" + id + " .title");
        inputVal.css("color", color);
    }
    /** handle textAlign property */
    webcamJs.updateTitleTextAlign = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var textAlign = json.newValue;
        $("#" + id).attr("text_align", textAlign + '');
        $("#" + id + " .title").css("text-align", textAlign);
    }


    return webcamJs;
})();

/** 
 * register 'webcamJs.initCameraInRunTime()' inito runtime 'onloadStack' 
 * */
if (renderJs) {
    renderJs.registerCustomizeComponentRuntimeInitMethod("webcamJs.initCameraInRunTime()");
}
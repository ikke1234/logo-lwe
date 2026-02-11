/**
 * analog value js Obj
 * */
var pdfJs = (function () {
    function pdfJs() {
    }
    ;

    /** handle size property */
    pdfJs.updateSize = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var w = json.width;
        var h = json.height;
        var obj = $("#" + id);
        obj.css("width", w);
        obj.css("height", h);
        var newSize = w + "px " + (h) + "px";
        obj.css('background-size', newSize);
        $("#" + id + " p").css("width", w);
        $("#" + id + " p").css("height", h);
    };
    /** handle textAlign property */
    pdfJs.updateTextAlign = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var textAlign = json.newValue;
        $("#" + id).attr("text_align", textAlign + '');
        $("#" + id).css("text-align", textAlign);
        $("#" + id).children(".pdftext").css("text-align", textAlign);
        $("#" + id).children(".progress").css("text-align", textAlign);
    };

    pdfJs.updateFontSize = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var newValue = json.newValue;
        var obj = $("#" + id);
        if (!newValue) {
            fontSize = "14";/*default font size*/
        }
        obj.css("font-size", parseInt(fontSize));
    };
    /**handle font style property*/
    pdfJs.updateFontStyle = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var fontStyle = json.style + "";
        var fontFamily = json.family;
        var fontSize = json.size;
        var inputVal = $("#" + id + " p");
        if (fontStyle === "0") {//normal
            inputVal.css({"font-style": "normal", "font-weight": ""});
        } else if (fontStyle === "1") {   //bold
            inputVal.css({"font-style": "normal", "font-weight": "bold"});
        } else if (fontStyle === "2") {   //italic
            inputVal.css({"font-style": "italic", "font-weight": ""});
        } else if (fontStyle === "3") {   //italic bold
            inputVal.css({"font-style": "italic", "font-weight": "bold"});
        }
        inputVal.css({"font-family": fontFamily, "font-size": parseInt(fontSize)});
    };

    /**handle font color property*/
    pdfJs.updateFontColor = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var color = json.newValue;
        var inputVal = $("#" + id + " p");
        inputVal.css("color", color);
    };

    /**handle font bgImage property*/
    pdfJs.updateBgImage = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var imgPath = json.newValue;
        imgPath = (imgPath === "img/null") ? "img/default.jpg" : imgPath;
        $("#" + id).css('background-image', 'url(' + imgPath + ')');
        $("#" + id).css('background-repeat', "no-repeat");
        $("#" + id).attr("bg_image", imgPath);
    };

    /**handle pdf property*/
    pdfJs.updatePdf = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var obj = $("#" + id);
        obj.attr("link_type", json.link_type);
        obj.attr("unique_name", json.unique_name);
        obj.attr("original_name", json.original_name);
        obj.attr("title", json.original_name);
        if (json.unique_name && json.unique_name.trim().length > 0) {
            obj.css("cursor", "pointer");
        } else {
            obj.css("cursor", "default");
        }
    };

    /** runtime click method to open pdf file */
    pdfJs.openFile = function (div) {
        var obj = $(div);
        if (obj.children(".pdftext").css('display') === 'none') {
            return;
        }
        if (obj.attr("unique_name").trim() === "") {
            return;
        }

        if (renderJs.isBmBaseOnV84()) {
            pdfDownloadForV84(obj);
        } else {
            //bm_v83 and earlier
            pdfDownloadForV83(obj);
        }
    };

    function pdfDownloadForV84(obj) {
        //baseOnV84 
        var folderPath =  "/webroot/file/";
        var filePath = folderPath + obj.attr("unique_name");
        window.open(filePath, 'newwindow', 'height=800,width=800,top=200,left=500,toolbar=no,menubar=no,scrollbars=yes, resizable=yes,location=yes, status=no');
    }

    function pdfDownloadForV83(obj) {
        //bm_v83 and earlier
        if (isHttpsProtocol()) {
            pdfDownloadForV83Https(obj);
        } else {
            //http or cloud
            var folderPath = "file/";
            var filePath = folderPath + obj.attr("unique_name");
            filePath = filePath + "?!App-Language=" + LocalStorage.Instance().Get(LOCAL_STORAGE_PREFIX + "_current_language") + "&Security-Hint=" + LocalStorage.Instance().Get(LOCAL_STORAGE_PREFIX + "_current_login_ref");
            window.open(filePath, 'newwindow', 'height=800,width=800,top=200,left=500,toolbar=no,menubar=no,scrollbars=yes, resizable=yes,location=yes, status=no');
        }
    }

    function pdfDownloadForV83Https(obj) {
        var folderPath = "../file/";
        var filePath = folderPath + obj.attr("unique_name");
        var pdfText = obj.find(".pdftext");
        var progressText = obj.find(".progress");
        pdfText.css('display', 'none');
        progressText.css('display', 'block');
        progressText.html('Downloading ...');
        AjaxRequest.Instance().Download(filePath, function (progress) {
            progressText.html(parseFloat(progress * 100).toFixed(1) + "%");
        }, function (context) {
            pdfText.css('display', 'table-cell');
            progressText.css('display', 'none');
            if (context.result) {
                var blob = new Blob([context.data], {type: 'application/pdf'});

                if (window.navigator && window.navigator.msSaveOrOpenBlob) {
                    window.navigator.msSaveOrOpenBlob(blob, obj.attr("original_name"));
                } else {
                    var objectUrl = URL.createObjectURL(blob);
                    window.open(objectUrl, 'newwindow', 'height=800,width=800,top=200,left=500,toolbar=no,menubar=no,scrollbars=yes, resizable=yes,location=yes, status=no');
                }
            } else {
                alert("Download failed! error code:" + context.responseCode + "filename:" + obj.attr("original_name"));
            }
        });
    }

    pdfJs.getFilePathWithFwVersion = function (filePath) {
        //call fwjs to get FW version
        if (typeof LocalGetVersion === 'function') {
            return filePath;
        }
        //bm<8.4
        return  filePath + "?!App-Language=" + LocalStorage.Instance().Get(LOCAL_STORAGE_PREFIX + "_current_language") + "&Security-Hint=" + LocalStorage.Instance().Get(LOCAL_STORAGE_PREFIX + "_current_login_ref");
    }
    return pdfJs;
})();




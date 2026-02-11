/** 
 * [analog value] Runtime item data callback method 
 *  FW js cannot callback a method like objct.xxx(), so this method not in trendJs Obj.
 * */
_callback_analogValue = function () {
    var id = this.m_sShowParam;
    var blockFormat = $("#" + id).attr("block_format");
    var accessModeType = $("#" + id).attr("type");
    $("#" + id).attr("title", blockFormat);

    var value = renderJs.getValueByBlockFormat(blockFormat, accessModeType, this.m_sValue);
    var actVal = renderJs.getFloatValueByDecimalPlace(id, value);
    $("#" + id).children(".valueText").children(".inputVal").attr("placeholder", actVal);
    $("#" + id).attr("last_placeholder", actVal);
}

/**
 * analog value js Obj
 * */
var analogValueJs = (function () {
    function analogValueJs() {
    }

    /** handle size property */
    analogValueJs.updateSize = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var w = json.width;
        var h = json.height;
        var obj = $("#" + id);
        obj.css("width", w);
        obj.css("height", h);
        var valueText = $("#" + id).children(".valueText");
        if (!valueText) {
            return;
        }
        var unit = valueText.children(".unit");
        if (!unit) {
            return;
        }
        var fontSize = unit.css("font-size");
        if (!fontSize) {
            fontSize = 14;/*default font size*/
        }
        var top = (parseInt(h) - parseInt(fontSize)) / 2;
        unit.css("top", top);
    }

    /** handle stream  property */
    analogValueJs.updateStream = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var newValue = json.newValue;
        $("#" + id).attr("stream", newValue);
        $("#" + id).children().eq(0).html("");
        $("#" + id).children().eq(0).attr("src", newValue);
        $("#" + id).children().eq(0).attr("id", "video_" + id);
        $("#" + id).children().eq(0).append($(document.createElement("source")).attr('src', newValue).attr('type', "video/mp4"));
        $("#" + id).children().eq(0).append($(document.createElement("source")).attr('src', newValue).attr('type', "video/ogg"));
        $("#" + id).children().eq(0).append($(document.createElement("source")).attr('src', newValue).attr('type', "video/webm"));
    }


    /** handle textAlign property */
    analogValueJs.updateAnalogValueTextAlign = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var textAlign = json.newValue;
        $("#" + id).attr("text_align", textAlign + '');
        $("#" + id).css("text-align", 'right'); //for unit
        $("#" + id + " input").css("text-align", textAlign + ''); // for input text
    }
    /**handle font style property*/
    analogValueJs.updateAnalogValueFontStyle = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var fontStyle = json.style;
        var fontFamily = json.family;
        var fontSize = json.size;
        var valueText = $("#" + id).children(".valueText");
        if (!valueText) {
            return;
        }
        var inputVal = valueText.children(".inputVal");
        var notice = $("#" + id).children(".notice");
        var unit = valueText.children(".unit");
        if (!inputVal || !notice) {
            return;
        }
        if (fontStyle == 0) {
            //normal
            inputVal.css({"font-style": "normal", "font-weight": ""});
            notice.css({"font-style": "normal", "font-weight": ""});
            unit.css({"font-style": "normal", "font-weight": ""});
        } else if (fontStyle == 1) {   //bold
            inputVal.css({"font-style": "normal", "font-weight": "bold"});
            notice.css({"font-style": "normal", "font-weight": "bold"});
            unit.css({"font-style": "normal", "font-weight": "bold"});
        } else if (fontStyle == 2) {   //italic
            inputVal.css({"font-style": "italic", "font-weight": ""});
            notice.css({"font-style": "italic", "font-weight": ""});
            unit.css({"font-style": "italic", "font-weight": ""});
        } else if (fontStyle == 3) {   //italic bold
            inputVal.css({"font-style": "italic", "font-weight": "bold"});
            notice.css({"font-style": "italic", "font-weight": "bold"});
            unit.css({"font-style": "italic", "font-weight": "bold"});
        }
        inputVal.css({"font-family": fontFamily, "font-size": parseInt(fontSize)});
        notice.css({"font-family": fontFamily, "font-size": parseInt(fontSize)});
        unit.css({"font-family": fontFamily, "font-size": parseInt(fontSize)});
        unit.css("line-height", parseInt(fontSize) + "px");
        var fontSize = unit.css("font-size");
        if (!fontSize) {
            fontSize = 14;/*default font size*/
        }
        var top = (parseInt($("#" + id).css("height")) - parseInt(fontSize)) / 2;
        unit.css("top", top);
    }
    /**handle font color property*/
    analogValueJs.updateAnalogValueFontColor = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var color = json.newValue;
        var valueText = $("#" + id).children(".valueText");
        if (!valueText) {
            return;
        }
        var inputVal = valueText.children(".inputVal");
        if (inputVal) {
            inputVal.css("color", color);
            if (!inputVal.hasClass("invalid" + id)) {
                inputVal.addClass("invalid" + id);
            }
            var placeholderStyle = $("#" + id).children("style.invalid" + id);
            if (placeholderStyle) {
                placeholderStyle.remove();
            }
            valueText.before('<style class="invalid' + id + '">' +
                    '.invalid' + id + ':-moz-placeholder {color: ' + color + ';}' +
                    '.invalid' + id + '::-moz-placeholder {color: ' + color + ';}' +
                    'input.invalid' + id + ':-ms-input-placeholder {color: ' + color + ';}' +
                    'input.invalid' + id + '::-webkit-input-placeholder {color: ' + color + ';}' +
                    '</style>');
        }
        var unit = valueText.children(".unit");
        if (unit) {
            unit.css("color", color);
        }
    }
    
    /** handle font WriteMode property */
    analogValueJs.updateWriteMode = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var newValue = json.newValue;
        $("#" + id).attr("mode", newValue);

        var valueText = $("#" + id).children(".valueText");
        if (valueText) {
            var inputVal = valueText.children(".inputVal");
            if (inputVal) {
                var cursorStyle = newValue === "true" ? "default" : "not-allowed";
                inputVal.css("cursor", cursorStyle);
            }
        }
    }

    return analogValueJs;
})();




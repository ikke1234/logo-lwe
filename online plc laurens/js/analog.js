/**
 * expand edit.js (obj=>editJs)
 * */
Edit.prototype = {
    constructor: Edit,
    updateAnalogBgColor: function (id, color) {
        $("#" + id).children(".analog_bar").css("background-color", color);
    },
    updateAnalogValBarColor: function (id, color) {
        $("#" + id).children(".analog_bar").children(".analog_bar_val").css("background-color", color);
        $("#" + id).attr("valBarCol", color);
        return;
    },
    updateAnalogMaxVal: function (id, maxVal) {
        $("#" + id).attr("maxval", maxVal.toString());
    },
    updateAnalogMinVal: function (id, minVal) {
        $("#" + id).attr("minval", minVal.toString());
    },
    updateAnalogMode: function (id, attriName, newValue) {
        $("#" + id).attr(attriName, newValue);
    },
    updateAnalogTextDemo: function (id, analogTextDemoStr, symbol, place) {
        var analogTextVal = $("#" + id).children(".valueText").children(".value");
        var analogInputPlaceholder = $("#" + id).children(".valueText").children(".inputVal");
        if (analogTextVal)
            analogTextVal.html(analogTextDemoStr);
        if (analogInputPlaceholder)
            analogInputPlaceholder.attr("placeholder", analogTextDemoStr);
        $("#" + id).attr("symbol", symbol);
        $("#" + id).attr("place", place);
    },
    updateBlockFormat: function (id, blockFormat) {
        $("#" + id).attr("block_format", blockFormat);
    },
    updateAnalogUnit: function (id, unit) {
        $("#" + id).children(".valueText").children(".unit").html(unit);
    },
    updateAnalogGain: function (id, gain) {
        $("#" + id).attr("gain", gain);
    },
    updateAnalogTextOffset: function (id, offset) {
        $("#" + id).attr("offset", offset);
    },
    updateAnalogTextSymbol: function (id, symbol) {
        $("#" + id).attr("symbol", symbol);
    },
    updateAnalogSymbolPlace: function (id, place) {
        $("#" + id).attr("place", place);
    },
    updateTimeFormat: function (id, timeUnit) {
        timeUnit = timeUnit.substring(timeUnit.indexOf("(") + 1, timeUnit.indexOf(")"));
        $("#" + id).children(".valueText").children(".unit").html("");
        if (timeUnit === "h") {
            $("#" + id).children(".valueText").children(".inputVal").attr("placeholder", "0");
            $("#" + id).children(".valueText").children(".unit").html("");
            $("#" + id).attr("title", "hours");
        } else if (timeUnit === "m") {
            $("#" + id).children(".valueText").children(".inputVal").attr("placeholder", "00:00");
            $("#" + id).attr("title", "hh:mm");
        } else if (timeUnit === "s") {
            $("#" + id).children(".valueText").children(".inputVal").attr("placeholder", "00:00:00");
            $("#" + id).attr("title", "hh:mm:ss");
        } else if (timeUnit === "ms") {
            $("#" + id).children(".valueText").children(".inputVal").attr("placeholder", "00:00:00.0");
            $("#" + id).attr("title", "hh:mm:ss.ms");
        } else if (timeUnit === "md") {  //for yearly month-day
            $("#" + id).children(".valueText").children(".inputVal").attr("placeholder", "month - day");
            $("#" + id).attr("title", "Yearly on/off time(month-day)");
        } else if (timeUnit === "hm") {   // for weekly day-time
            $("#" + id).children(".valueText").children(".inputVal").attr("placeholder", "hh:mm");
            $("#" + id).attr("title", "Weekly on/off time(hh:mm)");
        } else if (timeUnit === "dm") {
            //for yearly day-month add in LWE.1.1.1
            $("#" + id).children(".valueText").children(".inputVal").attr("placeholder", "day / month");
            $("#" + id).attr("title", "Yearly on/off time(day/month)");
        }

        $("#" + id).attr("time_format", timeUnit);
    },
    resizeAnalogDiv: function (id, width, height) {
        editJs.resizeDiv(id, width, height);

        var sliderBar = $("#" + id).children(".analog_slider_comp");
        var sliderHdl = $("#" + id).children(".analog_slider_comp").children(".ui-slider-handle");
        var valueText = $("#" + id).children(".valueText");
        if (sliderBar) {
            var hdlWidth = $("#" + id).width();
            var hdlHeight = $("#" + id).height();
            var sliderBarWidth = hdlWidth - 9;
            var sliderBarHeight = hdlHeight - 2;
            var rSize = hdlWidth;
            if (hdlWidth > hdlHeight) {
                sliderBarWidth = hdlWidth - 2;
                sliderBarHeight = hdlHeight - 9;
                rSize = hdlHeight;
                sliderHdl.css("margin-left", (hdlWidth / 2) - (hdlHeight / 2));
                sliderHdl.css("margin-top", -6);
                sliderBar.css("margin-top", 4);
            } else {
                rSize = hdlWidth;
                sliderHdl.css("margin-top", (hdlHeight / 2) - (hdlWidth / 2));
                // delete css
                sliderHdl.css("margin-left", "");
                sliderBar.css("margin-top", "");
            }
            sliderHdl.width(rSize).height(rSize).css("background-size", rSize + "px " + rSize + "px");
            // delete css property 'bottom'
            sliderHdl.css("bottom", "");

            sliderBar.width(sliderBarWidth);
            sliderBar.height(sliderBarHeight);

            sliderBar.css("background-size", sliderBarWidth + "px " + sliderBarHeight + "px");
        }
        if (valueText) {
            $("#" + id).children(".valueText").children(".inputVal").css('width', '100%');
            var value = $("#" + id).children(".valueText").children(".inputVal").attr("placeholder");
            if (value) {
                $("#" + id).children(".valueText").children(".inputVal").attr("placeholder", value);
            }
            $("#" + id).children(".valueText").children(".inputVal").css('height', '100%');
            var unit = $("#" + id).children(".valueText").children(".unit");
            var fontSize = unit.css("font-size");
            if (!fontSize) {
                fontSize = 14;//default font size
            }
            unit.css('left', editJs.parseToInt(width) + 2);
            unit.css('top', (editJs.parseToInt(height) - editJs.parseToInt(fontSize)) / 2);
        }
    },
    updateSliderBgImage: function (id, imgPath) {
        $("#" + id).attr("bar_img", imgPath);
        $("#" + id).children(".analog_slider_comp").css('background-image', 'url(' + imgPath + ')')
                .css("background-size", $("#" + id).children(".analog_slider_comp").width() + "px " + $("#" + id).children(".analog_slider_comp").height() + "px");
    },
    updateSliderHandlerImage: function (id, imgPath) {
        $("#" + id).attr("hdl_img", imgPath);
        var sliderHdl = $("#" + id).children(".analog_slider_comp").children(".ui-slider-handle");
        sliderHdl.css('background-image', 'url(' + imgPath + ')')
                .css("background-size", sliderHdl.width() + "px " + sliderHdl.height() + "px");
    },
    //-----------------AnalogBar--------------------------------------
    resizeAnalogBarDiv: function (id, width, height) {

        $("#" + id).css('height', height - 1 + "px");
        $("#" + id).css('width', width + "px");

        var targetBarWidth = width - 52;
        var targetBarHeight = height - 1;
        var targetBar = $("#" + id).children(".bar");
        targetBar.css('width', targetBarWidth + "px");
        targetBar.css('height', targetBarHeight + "px");
        targetBar.children(".barBg").css('width', targetBarWidth + "px");
        targetBar.children(".whiteBg").css('width', targetBarWidth + "px");
        this.setAnalogBarScale(id, parseFloat($("#" + id + " li:first a").html()));
    },
    setAnalogBarScale: function (id, maxVal, minVal, minScale) {
        var a = $("#" + id + " ul li:first a");
        var font_style = "normal";
        var font_weight = "Arial";
        var font_family = "normal";
        var font_size = 14;
        var color = "rgb(0, 0, 0)";
        if (a) {
            font_style = a.css("font-style");
            font_weight = a.css("font-weight");
            font_family = a.css("font-family");
            font_size = a.css("font-size");
            color = a.css("color");
        }
        $("#" + id + " ul li").remove();

        var decimal = $("#" + id).attr("decimal");
        if ((undefined === decimal) || (!decimal) || "1" === decimal) {
            decimal = "0";
        } else if ("10" === decimal) {
            decimal = "1";
        } else if ("100" === decimal) {
            decimal = "2";
        } else if ("1000" === decimal) {
            decimal = "3";
        }

        maxVal = (!maxVal) ? parseFloat($("#" + id).attr("max_val")) : parseFloat(maxVal);
        minVal = (!minVal) ? parseFloat($("#" + id).attr("min_val")) : parseFloat(minVal);

        var scaleValueFixedNum = this.scaleValueDecimalPlace(maxVal, minVal, decimal);
        var sCount = 10;
        minScale = (maxVal - minVal) * 1000 / 10000; // fix bug of float Accuracy
        var scaleHeight = parseInt($("#" + id).css('height')) / sCount;
        var scaleValue = maxVal + minScale;
        var top = 0;
        //sCount = Math.round(sCount);
        for (var i = 0; i <= sCount; i++) {
            scaleValue = scaleValue - minScale;
            scaleValue = scaleValue.toFixed(scaleValueFixedNum);
            if (scaleValue >= (minScale / 2 + minVal)) {
                $("#" + id + " ul").append("<li style='height:" + scaleHeight + "px;top:" + top + "px'><a>" + scaleValue + "</a></li>");
                top = top + scaleHeight;
            }
        }

        // for bottom scale line
        top = parseInt($("#" + id).css('height'));
        $("#" + id + " ul").append("<li style='height:1px; top:" + top + "px'><a>" + minVal + "</a></li>");
        $("#" + id + " ul li a").css({"font-style": font_style, "font-weight": font_weight, "font-family": font_family, "font-size": font_size});
        $("#" + id + " ul li a").css("color", color);
        $("#" + id).attr("max_val", maxVal);
        $("#" + id).attr("min_val", minVal);
        $("#" + id).attr("min_scale", minScale);
        $("#" + id).attr("scale_height", scaleHeight);
        $("#" + id).attr("bar_value", parseInt((maxVal + minVal) / 2));
        this.updateBarValue(id);
    },
    scaleValueDecimalPlace: function (maxVal, minVal, decimal) {
        var sValue = (maxVal - minVal) * 1000 / 10000;
        var y = String(sValue).indexOf(".") + 1;
        if (y === 0) {
            return decimal;
        }
        var count = String(sValue).length - y;
        if (count < decimal) {
            count = decimal;
        }
        if (count > 4) {
            count = 4;
        }
        return count;
    },
    updateScopeColor: function (id, value) {
        $("#" + id).attr("scope_color", value);
        this.updateBarValue(id);
    },
    updateBarBorderColor: function (id, color) {
        //$("#" + id + " .bar ul > li > a").css('color', color);
        $("#" + id + " .bar ul > li").css('border-color', color);
        $("#" + id + " .barBg").css('border-color', color);
    },
    updateBarValue: function (id) {
        var barHeight = parseInt($("#" + id).css('height'));
        var whiteHeight = barHeight / 2 - 1;
        var barValue = parseInt($("#" + id).attr("bar_value"));
        var color = this.getColorByBarValue(id, barValue);
        if (color) {
            $("#" + id + " .barBg").css('background-color', color);
        }
        $("#" + id + " .whiteBg").css('height', whiteHeight + "px");
        //for borderColor
        var borderColor = $("#" + id + " .barBg").css('border-color');
        this.updateBarBorderColor(id, borderColor);
    },
    getColorByBarValue: function (id, barValue) {
        var scopeColor = $("#" + id).attr("scope_color");
        if (!scopeColor) {
            return null;
        }
        var range = scopeColor.split(",");
        barValue = parseInt(barValue);
        for (var i = 0; i < range.length; i++) {
            var value = range[i].split(":");
            if (value[0] <= barValue && barValue <= value[1]) {
                return this.getColorFromRGB(value[2]);
            }
        }
    },
    getColorFromRGB: function (colorStr) {
        var value = parseInt(colorStr);
        value = Math.abs(value);
        var val = 0xff000000 | value;
        val = val.toString(16).substring(1);
        var str = '';
        if (val.length < 6) {
            var i = 6 - val.length;
            for (var t = 0; t < i; t++) {
                str += '0';
            }
        }
        //console.log('color:' + '#' + str + val);  
        var final = '#' + str + val;
        return (final === "#1000000") ? "#000000" : final;
    },
    /**update scale time font style handle method*/
    updateAnalogScaleTimeFontStyle: function (id, fontFamily, fontStyle, fontSize) {
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
        var unitfontSize = unit.css("font-size");
        if (!unitfontSize) {
            unitfontSize = 14;/*default font size*/
        }
        var top = (parseInt($("#" + id).css("height")) - parseInt(unitfontSize)) / 2;
        unit.css("top", top);
    },
    /**update scale time font color handle method*/
    updateAnalogScaleTimeFontColor: function (id, color) {
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
        /*var notice = $("#" + id).children(".notice");
         if ( notice ) {
         notice.css("color", color);
         }*/
    },
    /**update analog bar font style handle method*/
    updateAnalogBarFontStyle: function (id, fontFamily, fontStyle, fontSize) {
        var font_style = "normal";
        var font_weight = "";
        if (fontStyle == 0) {//normal
            font_style = "normal";
            font_weight = "";
        } else if (fontStyle == 1) {   //bold
            font_style = "normal";
            font_weight = "bold";
        } else if (fontStyle == 2) {   //italic
            font_style = "italic";
            font_weight = "";
        } else if (fontStyle == 3) {   //italic bold
            font_style = "italic";
            font_weight = "bold";
        }
        var a = $("#" + id + " ul li a");
        if (!a) {
            return;
        }
        a.css({"font-style": font_style, "font-weight": font_weight, "font-family": fontFamily, "font-size": parseInt(fontSize)});
    },
    /**update analog bar  font color handle method*/
    updateAnalogBarFontColor: function (id, color) {
        var a = $("#" + id + " ul li a");
        if (!a) {
            return;
        }
        a.css("color", color);
    },
    updateAnalogDateTimeFormat: function (id, format) {
        var comp = $("#" + id);
        comp.attr("time_format", format);

        //upgrade to latest version comp (v1.2)
        comp.html("");
        comp.append("<div style='display: table;height:100%;width:100%'>");
        comp.children("div:first").append("<div style='position:relative;display: table-cell; vertical-align:middle;' class='outer'></div>");
        comp.find(".outer").append("<div class = 'dayTime'>Fri.13:01:59</div>");
        comp.find(".outer").append("<div class = 'date'>2019-01-11</div>");
        switch (format) {
            case "ymd":{
                comp.find(".dayTime").css("display", "none");
                comp.attr("title", "yyyy-MM-dd");
                break;
            }
            case "hms":{
                comp.find(".dayTime").html("13:01:59");
                comp.find(".date").css("display", "none");
                comp.attr("title", "");
                break;
            }
            case "dhms":{
                comp.find(".date").css("display", "none");
                comp.attr("title", "");
                break;
            }
            case"dmy":{//new format in LWE1.1.1
                comp.find(".dayTime").css("display", "none");
                comp.find(".date").html("11/01/2019");
                comp.attr("title", "dd/MM/yyyy");
                break;
            }
            case "ftdmy":{
                comp.find(".date").html("11/01/2019");
                comp.attr("title", "dd/MM/yyyy");
                break;
            }
            case "mdy":{//new format in LWE1.2
                comp.find(".dayTime").css("display", "none");
                comp.find(".date").html("01/11/2019");
                comp.attr("title", "MM/dd/yyyy");
                break;
            }
            case "ftmdy":{
                comp.find(".date").html("01/11/2019");
                comp.attr("title", "MM/dd/yyyy");
                break;
            }
            default: {
                comp.attr("title", "yyyy-MM-dd");   
            }
        }
    }
};



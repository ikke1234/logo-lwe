/**
 * version LOGO!8.3 
 * 
 * $jq-->jquery version=1.11.2
 * 
 * editJs for lwe component basic actions in edit-time
 * */
var _dataMap = new UdwMap();
$(document).ready(function () {
    editJs = new Edit();
    _dataMap.put("_version", "logo_8.4");
    editJs.init();
});

function Edit() {
    this.init = function () {
        this.resetDivImgFromAttr();
        this.initDegreeAndSvg();
        this.fixLogoClockDisplay();
    }

    this.fixLogoClockDisplay = function () {
        $("div[comp_type='analogDateTimeComponent']").each(function () {
            let id = $(this).attr("id");
            let format = $(this).attr("time_format");
            editJs.updateAnalogDateTimeFormat(id, format);
        });
    };

    this.resetDivImgFromAttr = function () {
        $("div[bg_image]").each(function (index, element) {
            var image = element.getAttribute("bg_image");
            if (image && image.length > 0) {
                element.style.backgroundImage = "url(" + image + ")";
            }
        });

        $("div[off_image]").each(function (index, element) {
            var image = element.getAttribute("off_image");
            element.style.backgroundImage = "url(" + image + ")";
        });

        var sliders = $(".analog_slider_comp");
        sliders.each(function (index) {
            var parent = $(this).parent().get(0);
            var hdl_img = $(parent).attr("hdl_img");
            if (hdl_img) {
                $(parent).children(".analog_slider_comp").children(".ui-slider-handle").css('background-image', 'url(' + hdl_img + ')');
            }
            var bar_img = $(parent).attr("bar_img");
            if (bar_img) {
                $(parent).children(".analog_slider_comp").css('background-image', 'url(' + bar_img + ')');
            }
        });
    }

    this.moveDiv = function (id, x, y) {
        var target = $("#" + id);
        target.css('left', this.parseToInt(x));
        target.css('top', this.parseToInt(y));
    }

    this.resizeDiv = function (id, width, height) {
        //var target = $("#" + id).children(":first");
        var target = $("#" + id);
        target.css('width', this.parseToInt(width));
        target.css('height', this.parseToInt(height));
        //target.css('text-align', "center");
        target.css('background-repeat', "no-repeat");
        target.css('background-size', this.parseToInt(width) + "px " + (this.parseToInt(height) + 1) + "px");
        if (target.hasClass("TextComponent")) {
            var elementP = $("#" + id + " p");
            this.addTableCellStyle(elementP, width, height);
        }
        if (target.hasClass("textCenter")) {
            if (target.attr("comp_type") === "CustomizedComponent" || target.attr("comp_type") === "PushButton") {
                if ($("#" + id + " p").length === 0) { //target.children(":first")[0].tagName=='p'
                    var text = target.html();
                    target.html("");
                    var p = $("<p>" + text + "</p>");
                    target.append(p);
                }
                var elementP = $("#" + id + " p");
                this.addTableCellStyle(elementP, width, height);
            }

            if (target.attr("comp_type") !== "CustomizedComponent") {
                target.css('line-height', this.parseToInt(height) + "px");
            }
        }
    }

    this.addTableCellStyle = function (target, width, height) {
        target.css('width', this.parseToInt(width));
        target.css('height', this.parseToInt(height));
        target.css('display', "table-cell");
        target.css('vertical-align', "middle");
        // target.css('text-align', "center");
    }



    this.resizePageDiv = function (id, width, height) {
        var target = $("#main");
        target.css('width', this.parseToInt(width) - 20);
        target.css('height', this.parseToInt(height) - 20);
        target.css('text-align', "center");
        if (target.hasClass("textCenter")) {
            target.css('line-height', this.parseToInt(height) + "px");
        }
        target.css('margin', 0);
        target.css('padding', 0);
        target.css('top', 0);
        target.css('left', 0);
        //alert(target.css('background-image'));
        if (target.css('background-image') && target.css('background-image').indexOf("linear-gradient") != -1) {
            //background grid
            target.css('background', "linear-gradient(60deg, white 0px, black 1px, red 0px, white 0px) 25px -24px");
            target.css('background-repeat', "initial initial");
            target.css('background-size', "");
        } else {
            target.css('background-position', "0px 0px");
            target.css('background-repeat', "no-repeat");
            if (target.css("background-image")) {
                target.css('background-size', this.parseToInt(width) + "px " + (this.parseToInt(height) + 1) + "px");
            } else {
                target.css('background-size', "");
            }
        }
    }

    this.updateBgImg = function (id, imgPath) {
        imgPath = (imgPath == "img/null") ? "img/default.jpg" : imgPath;
        $("#" + id).css('background-image', 'url(' + imgPath + ')');
        $("#" + id).css('background-repeat', "no-repeat");
        //    if ($("#" + id).attr("comp_type") == "ImageComponent") {
        $("#" + id).attr("bg_image", imgPath);
        //    }
    }

    this.updatePageBgImg = function (imgPath) {
        if (imgPath == "img/" || imgPath == "img/null" || "" || imgPath == "null") {
            this.setPageBgAsDefault();
            return;
        }

        var width = $("#main").width();
        var height = $("#main").height();
        var newSize = width + "px " + (height) + "px";
        $("#main").css('background-image', 'url(' + imgPath + ')');
        $("#main").css('background-size', newSize);
        $("#main").css('background-repeat', "no-repeat");
        $("#main").css('background-position', "0px 0px");
        $("#main").attr("bg_image", imgPath);
        $("#main").css('margin', 0);
        $("#main").css('padding', 0);
        $("#main").css('top', 0);
        $("#main").css('left', 0);
    }

    this.setPageBgAsDefault = function () {
        //    $("#main").css('background', '');
        //    $("#main").css('background-color', "white");
        //    $("#main").css('background:linear-gradient', "(60deg, white 0px, black 1px, red 0px, white 0px) 25px - 24px");
        //    $("#main").css('background-size', "10px 10px");
        //    $("#main").css('background-image', '');

        var width = $("#main").css('width');
        var height = $("#main").css('height');
        $("#main").attr("bg_image", "");
        $("#main").attr("style", "");
        $("#main").attr("class", "main");
        $("#main").css('width', this.parseToInt(width));
        $("#main").css('height', this.parseToInt(height));
    }

    this.addDiv = function (id, x, y, htmlStr, script) {
        // If exits, delete the old html. be used for line, and so on.
        if ($("#" + id).length !== 0) {
            $("#" + id).remove();
        }
        // getBody().innerHTML = getBody().innerHTML + htmlStr;
        $jq(htmlStr).appendTo("#main");
        var newNode = this.getBody().childNodes[this.getBody().childNodes.length - 1];
        newNode.id = id;
        var target = $("#" + id);
        target.css('position', "absolute");
        target.css('left', this.parseToInt(x));
        target.css('top', this.parseToInt(y));

        if (target.hasClass("server_binding")) {
            target.attr("show_param", id);
        }
        if (target.attr("comp_type") && target.attr("comp_type").indexOf("AnalogSliderComponent") != -1) {
            var slider_maxVal = parseInt(target.attr("maxval"));
            var slider_minVal = parseInt(target.attr("minval"));
            var midVal = (slider_maxVal - slider_minVal) * 0.5 + slider_minVal;
            target.attr("show_param", id);
            var sliderBar = target.children(".analog_slider_comp");
            sliderBar.slider({
                orientation: "vertical",
                max: slider_maxVal,
                min: slider_minVal,
                value: midVal
            });
            sliderBar.css("background-size", sliderBar.width() + "px " + sliderBar.height() + "px");
        }
        //$(script).appendTo("#main");
        //$jq-->jquery version=1.11.2
        $jq(script).attr("id", "script_" + id).appendTo("#main");
    }

    this.cloneDiv = function (clonedId, newId, x, y) {
        var target = $("#" + clonedId);
        var newDiv = target.clone();
        newDiv.attr("id", newId);
        $("#main").append(newDiv);
        newDiv.css('position', "absolute");
        newDiv.css('left', this.parseToInt(x));
        newDiv.css('top', this.parseToInt(y));
        if (target.hasClass("server_binding")) {
            newDiv.attr("show_param", newId);
        }
    }

    this.insertFromCopiedDiv = function (htmlStr, newId, x, y, script) {
        var newDiv = $(htmlStr);
        newDiv.attr("id", newId);
        $("#main").append(newDiv);
        newDiv.css('position', "absolute");
        newDiv.css('left', this.parseToInt(x));
        newDiv.css('top', this.parseToInt(y));
        if (newDiv.hasClass("server_binding")) {
            newDiv.attr("show_param", newId);
        }
        $(script).appendTo("#main");
    }

    this.getWidgetHtmlString = function (id) {
        // return $("#" + id).prop('outerHTML');
        var main = $("#" + id);
        if (main && main[0]) {
            return main[0].outerHTML;
        } else {
            return "";
        }
    }

    this.getRuntimeBody = function () {
        var main = $("#main");
        return main[0].outerHTML;
    }

    this.getCustomzieImportScripts = function () {
        var script = "";
        $("SCRIPT[id]").each(function () {
            if ($(this).attr("src")) {
                //add for ignore broswer cache
                var newStr = $(this).attr("id") + "?rdm=" + Date.now();
                $(this).attr("src", newStr);

                //alert($(this)[0].outerHTML);
                script += $(this)[0].outerHTML;
            }
        });
        //alert(script);
        return script;
    }

    this.getCustomzieImportCss = function () {
        var link = "";
        $("LINK[id]").each(function () {
            if ($(this).attr("href")) {
                //alert($(this)[0].outerHTML);
                link += $(this)[0].outerHTML;
            }
        });
        //alert(script);
        return link;
    }

    this.deleteDiv = function (id) {
        var target = $("#" + id);
        if (target) {
            target.remove();
            var script = $("#script_" + id);
            if (script) {
                script.remove();
            }
        }
    }

    this.updateText = function (id, newValue) {
        if ($("#" + id + " a").length > 0) {
            $("#" + id + " a").html(newValue);
        } else if ($("#" + id + " p").length > 0) {
            $("#" + id + " p").html(newValue);
            // updateTextHeightInDiv($("#" + id));
        } else {
            $("#" + id).html(newValue);
        }

    }

    this.updateBgColor = function (id, color) {
        $("#" + id).css("background-color", color);
    }

    //analogValue, analogScaleTime
    this.updateAnalogInputBgColor = function (id, color) {
        $("#" + id).css("background-color", color);
        var valueText = $("#" + id).children(".valueText");
        if (valueText) {
            var inputVal = valueText.children(".inputVal");
            if (inputVal) {
                inputVal.css("background-color", color);
            }
        }
    }

    this.updateFontColor = function (id, color) {
        $("#" + id).css("color", color);
        if ($("#" + id).has("a")) {
            $("#" + id + " a").css("color", color);
        }
    }

    this.updateFont = function (id, fontFamily, fontStyle, fontSize) {
        this.setFontStyle(id, parseInt(fontStyle));
        $("#" + id).css({"font-family": fontFamily, "font-size": parseInt(fontSize)});
    }

    this.setFontStyle = function (id, fontStyle) {
        if (fontStyle == 0) {
            //normal
            $("#" + id).css({"font-style": "normal", "font-weight": ""});
        } else if (fontStyle == 1) {   //bold
            $("#" + id).css({"font-style": "normal", "font-weight": "bold"});
        } else if (fontStyle == 2) {   //italic
            $("#" + id).css({"font-style": "italic", "font-weight": ""});
        } else if (fontStyle == 3) {   //italic bold
            $("#" + id).css({"font-style": "italic", "font-weight": "bold"});
        }
    }

    this.updateTextAlign = function (id, textAlign) {
        this.updateAttribute(id, "text_align", textAlign);
        if ($("#" + id).attr("comp_type") === "TextComponent") {
            $("#" + id + " p").css("text-align", textAlign);
            this.updatePaddingRight($("#" + id + " p"), textAlign);
        } else if ($("#" + id).attr("comp_type") === "CustomizedComponent") {
            $("#" + id + " p").css("text-align", textAlign);
            this.updatePaddingRight($("#" + id + " p"), textAlign);
        } else if ($("#" + id).attr("comp_type") === "analogDateTimeComponent") {
            $("#" + id).css("text-align", textAlign);
        } else if ($("#" + id).attr("comp_type") === "AnalogTimeComponent") {
            $("#" + id + " input").css("text-align", textAlign);
        } else if ($("#" + id).attr("comp_type") === "AnalogValueComponent") {
            $("#" + id + " input").css("text-align", textAlign);
        } else if ($("#" + id).attr("comp_type") === "linkComponent") {
            $("#" + id).css("text-align", textAlign);
            this.updatePaddingRight($("#" + id + " a"), textAlign);
        } else if ($("#" + id).attr("comp_type") === "PushButton") {
            $("#" + id + " p").css("text-align", textAlign);
            this.updatePaddingRight($("#" + id), textAlign);
            if (textAlign === "right") {
                $("#" + id + " p").css("box-sizing", "border-box");
            } else {
                $("#" + id + " p").css("box-sizing", "");
            }
        }
    }

    this.updatePaddingRight = function (component, textAlign) {
        if (textAlign === "right") {
            component.css("padding-right", 7);
        } else {
            component.css("padding-right", "");
        }
    }

    this.updateOnImage = function (id, newValue) {
        this.updateAttribute(id, "on_image", newValue);
    }

    this.updateOffImage = function (id, newValue) {
        this.updateAttribute(id, "off_image", newValue);
        this.updateBgImg(id, newValue);
        //$("#" + id).css('background-image', 'url(' + newValue + ')');
    }

    this.updateOnText = function (id, newValue) {
        this.updateAttribute(id, "on_text", newValue);
    }

    this.updateOffText = function (id, newValue) {
        this.updateAttribute(id, "off_text", newValue);
    }

    this.updateCustomOnOffText = function (id, newValue) {
        $("#" + id + " p").html(newValue);
    }

    this.SwitchButtonOffText = function (id, newValue) {
        if ($("#" + id + " p").length !== 0) {
            $("#" + id + " p").html(newValue);
        } else {
            $("#" + id).html(newValue);
        }
    }

    this.updateTagID = function (id, newValue) {
        this.updateAttribute(id, "tag_id", newValue);
    }

    this.updateBlockType = function (id, range) {
        this.updateAttribute(id, "range", range);
    }

    this.updateBlockNumber = function (id, address) {
        this.updateAttribute(id, "address", address);
    }

    this.updateAccessMode = function (id, type) {
        this.updateAttribute(id, "type", type);
    }

    this.updateFrequency = function (id, frequency) {
        this.updateAttribute(id, "frequency", frequency);
    }

    this.updateMouseMode = function (id, mouse_mode) {
        this.updateAttribute(id, "mouse_mode", mouse_mode);
    }

    this.updateLength = function (id, length) {
        this.updateAttribute(id, "length", length);
    }

    this.updateAttribute = function (id, attriName, newValue) {
        $("#" + id).attr(attriName, newValue);
    }

    this.updateZIndex = function (id, newValue) {
        $("#" + id).css("zIndex", newValue);
    }

    this.updateMode = function (id, newValue) {
        $("#" + id).attr("mode", newValue);
        this.updateCursorStyleByModeValue(id, newValue);
    }

    this.updateCursorStyleByModeValue = function (id, modeValue) {
        if ($("#" + id).attr("comp_type") === "CustomizedComponent") {//SwitchButton
            if (modeValue === "true") {
                $("#" + id).attr("onclick", "renderJs.setDigitalData(this)");
                $("#" + id).css("cursor", "pointer");
            } else {
                $("#" + id).removeAttr("onclick");
                $("#" + id).css("cursor", "not-allowed");
            }
        } else if ($("#" + id).attr("comp_type") === "AnalogSliderComponent") {
            var cursorStyle = modeValue === "true" ? "pointer" : "not-allowed";
            $("#" + id + " .analog_slider_comp span").css("cursor", cursorStyle);
        } else if ($("#" + id).attr("comp_type") === "AnalogTimeComponent") {
            if ($("#" + id).children(".valueText")) {
                var inputVal = $("#" + id).children(".valueText").children(".inputVal");
                if (inputVal) {
                    var cursorStyle = modeValue === "true" ? "default" : "not-allowed";
                    inputVal.css("cursor", cursorStyle);
                }
            }
        }
    }

    this.updateUrl = function (id, url) {
        $("#" + id + " a").attr("href", url);
    }

    this.updateLinkType = function (id, type) {
        $("#" + id).attr("link_type", type);
    }

    this.updateDeviceName = function (id, deviceName) {
        $("#" + id).attr("device_name", deviceName);
    }

    this.updateDecimalPlaces = function (id, decimal) {
        $("#" + id).attr("decimal", decimal);
    }

    this.updateAnalgoBarDecimalPlaces = function (id, decimal) {
        $("#" + id).attr("decimal", decimal);
        this.setAnalogBarScale(id, parseFloat($("#" + id + " li:first a").html()));
    }

    this.updateAnalgSliderDecimalPlaces = function (id, decimal) {
        $("#" + id).attr("decimal", decimal);
        // TODO init slider step by decimal [0-0.001]
    }

    this.getBody = function () {
        return document.getElementById("main");
    }

    this.parseToInt = function (value) {
        if (typeof (value) === "string") {
            if (value.indexOf(",") > 0) {
                value = value.replace(",", "");
            }
            return parseInt(value);
        } else {
            return value;
        }
    }

    //called by init logoBrowser
    this.resizeMain = function (width, height) {
        var target = $("#main");
        target.css('width', parseInt(width));
        target.css('height', parseInt(height));
    }

    //-----------nav-----------------------------------------------

    this.addNavItem = function (id, html) {
        $("#logOff").before(html);
        $("#newItem").attr('id', id);
        var color = $(".menuTitle").css("color");
        $("#" + id + " a").css("color", color);
    }

    this.insertNavItem = function (newId, position, html) {
        var newDiv = $(html);
        newDiv.attr("id", newId);
        $('.items').children().eq(position).before(newDiv);
        var color = $(".menuTitle").css("color");
        $("#" + newId + " a").css("color", color);
        //$(' .menu').css("top", "11px");
    }

    this.insertNavItemFromCopiedItem = function (newId, htmlStr) {
        var newDiv = $(htmlStr);
        newDiv.attr("id", newId);
        $("#logOff").before(newDiv);

    }

    this.sortNavItems = function (id, position, isMoveDown) {
        var target = $("#" + id);
        var newDiv = target.clone();
        if (isMoveDown) {
            $('.items').children().eq(position).after(newDiv);
        } else {
            $('.items').children().eq(position).before(newDiv);
        }
        target.remove();
        //$(' .menu').css("top", "11px");
    }

    this.updateNavItemLink = function (id, link, text) {
        this.updateNavItmeText(id, text);
        link = (link == "Home Page") ? "main" : link;
        $("#" + id + " a").attr("nav_link", link);
    }

    this.updateNavItmeText = function (id, text) {
        //text = text.replace(/&/g, '&amp;');
        //text = text.replace(/</g, '&lt;');
        //text = text.replace(/>/g, '&gt;');
        //text = text.replace(/"/g, '&quot;');
        //text = text.replace(/'/g, '&#039;');
        //text = text.replace(/\\/g, '\\\\');
        $("#" + id + " a").html(text);
    }

    this.updateNavItemFontColor = function (id, color) {
        $("#" + id + " a").css("color", color);
    }

    this.getRuntimeNavBody = function () {
        var main = $("#menu");
        return main[0].outerHTML;
    }

    this.updateNavDisplayStyle = function (isRight) {
        $("#menu").attr("is_right", isRight);
    }

    this.updateNavFontColor = function (id, color) {
        $(".item").css("color", color);
        $(".menuTitle").css("color", color);
    }
    //--------nav end----------------------------------

    /**
     * border-color
     */
    this.updateBorderColor = function (id, value) {
        this.updateDivCss(id, 'border-color', value);
    }

    /**
     * border-style
     */
    this.updateBorderStyle = function (id, value) {
        this.updateDivCss(id, 'border-style', value);
    }

    /**
     * border-width
     */
    this.updateBorderWidth = function (id, value, width, height) {
        // update border width
        this.updateDivCss(id, 'border-width', value);
        // update div size
        this.resizeDiv(id, width, height);
    }

    /**
     * common method: update div css by styleName.
     */
    this.updateDivCss = function (id, cssName, value) {
        var divObj = $("#" + id);
        if (divObj) {
            divObj.css(cssName, value);
        }
    }

    /**
     * alpha
     */
    this.updateAlpha = function (id, value) {
        // filter: Alpha(opacity=50);zoom:1;opacity: 0.5;/*background-color:rgba(0,0,0,0.1);*/
        var divObj = $("#" + id);
        if (divObj) {
            value = 100 - parseInt(value);
            var tempValue = value / 100;
            divObj.css("filter", "Alpha(opacity=" + value + ")");
            divObj.css("zoom", "1");
            divObj.css("opacity", tempValue);
        }
    }

    //--------------------File-----------------------------------------
    this.removeEditReference = function () {
        $('script').remove();
    }
    this.setEditReference = function () {
        $(document.createElement("script")).attr('type', "text/javascript").attr('src', "res/js/jquery-1.11.2.min.js").appendTo("head").after("\n");
        $(document.createElement("script")).attr('type', "text/javascript").attr('src', "res/js/jquery-ui/jquery-ui.min.js").appendTo("head").after("\n");
        $(document.createElement("script")).attr('type', "text/javascript").attr('src', "res/js/edit.js").appendTo("head").after("\n");
        $(document.createElement("script")).attr('type', "text/javascript").attr('src', "res/js/bootstrap2-toggle.js").appendTo("head").after("\n");
        $(document.createElement("script")).attr('type', "text/javascript").attr('src', "res/js/bootstrap.min.js").appendTo("head").after("\n");
        $('body').attr('onload', "");
    }
    this.setRuntimeReference = function () {
        $(document.createElement("script")).attr('type', "text/javascript").attr('src', "/js/storage.js").appendTo("head").after("\n");
        $(document.createElement("script")).attr('type', "text/javascript").attr('src', "/js/utility.js").appendTo("head").after("\n");
        $(document.createElement("script")).attr('type', "text/javascript").attr('src', "/js/bigint.js").appendTo("head").after("\n");
        $(document.createElement("script")).attr('type', "text/javascript").attr('src', "/js/encrypt.js").appendTo("head").after("\n");
        $('body').attr('onload', "DBInit()");
    }


    //-----------------rotation--------------------------
    /**
     * browser compatibility: when open the page, Processing the special page properties.
     */
    this.initDegreeAndSvg = function () {
        var main = $("#main");
        if (main && main.children && main.children().length > 0) {
            // control the div rotate.
            $("div[degree]").each(function () {
                var degree = $(this).attr("degree");
                if (undefined === degree) {
                    degree = 0;
                }
                editJs.updateRotateCss($(this), parseFloat(degree));
            });
            // control the svg tags show.
            $("svg").each(function () {
                $(this).css("display", "");
            });
        }
    }

    /**
     * the function of common rotate.
     */
    this.rotate = function (id, rotation) {
        var divObj = $("#" + id);
        if (divObj.length !== 0) {
            if (undefined === rotation) {
                rotation = 0;
            }
            var degree = parseFloat(rotation);
            this.updateRotateCss(divObj, degree);
            // save the degree.
            divObj.attr("degree", degree);
        }
    }

    /**
     * the function of line rotate.
     */
    this.rotateLine = function (id, rotation, x, y) {
        var svg = $("#" + id);
        if (svg.length !== 0) {
            var g = svg.find("g");
            if (g.length !== 0) {
                // delete the separator of the micrometer level.(eg. ',')
                x = x.replace(/,/gi, '');
                y = y.replace(/,/gi, '');
                var str = "rotate(" + rotation + " " + x + "," + y + ")";
                g.attr("transform", str);
            }
        }
    }

    /**
     * update rotate css.
     */
    this.updateRotateCss = function (obj, degree) {
        var cValue = "rotate(" + degree + "deg)";
        // Other Browser, including IE > 9
        obj.css("transform", cValue);
        obj.css("-webkit-transform", cValue);
        obj.css("-moz-transform", cValue);
        obj.css("-o-transform", cValue);
        obj.css("-ms-transform", cValue);
    }

    //-----------------SVG graphics--------------------------
    /**
     * update the weight of line.
     * @param {type} id
     * @param {type} value
     */
    this.updateLineWeight = function (id, value) {
        this.updateLineCss(id, "stroke-width", value);
    }

    /**
     * update the color of line.
     * @param {type} id
     * @param {type} value
     */
    this.updateLineColor = function (id, value) {
        this.updateLineCss(id, "stroke", value);
    }

    /**
     * update the dashed of line.
     * @param {type} id
     * @param {type} value
     */
    this.updateLineDash = function (id, value1, value2) {
        this.updateLineCss(id, "stroke-dasharray", value1);
        this.updateLineCss(id, "stroke-linecap", value2);
    }


    /**
     * common method: update the style of line.
     * @param {type} id
     * @param {type} cssName
     * @param {type} value
     */
    this.updateLineCss = function (id, cssName, value) {
        var svg = $("#" + id);
        if (svg.length !== 0) {
            var g = svg.find("g");
            if (g.length !== 0) {
                var leng = g.children().length;
                for (var i = 0; i < leng; i++) {
                    var line = g.children()[i];
                    line.setAttributeNS(null, cssName, value);
                }
            }
        }
    }

    /**
     * update the alpha of line.
     * @param {type} id
     * @param {type} value
     */
    this.updateLineAlpha = function (id, value) {
        var svg = $("#" + id);
        if (svg.length !== 0) {
            var g = svg.find("g");
            if (g.length !== 0) {
                var leng = g.children().length;
                var opacity = 100 - parseInt(value);
                var tempValue = opacity / 100;
                for (var i = 0; i < leng; i++) {
                    var line = g.children()[i];
                    line.setAttributeNS(null, "filter", "Alpha(opacity=" + opacity + ")");
                    line.setAttributeNS(null, "opacity", tempValue);
                }
            }
        }
    }

    this.updateLinkTargetInfo = function (id, linkType, url, uniqueName, originalName) {
        this.dealWithOldVersionLinkComp(id);
        var obj = $("#" + id);
        obj.attr("link_type", linkType);
        obj.attr("url", url);
        obj.attr("unique_name", uniqueName);
        obj.attr("original_name", originalName);
        if (linkType === "1" && originalName.length > 3) {
            obj.attr("title", originalName);
        } else if (linkType === "0" && url.length > 7) {
            obj.attr("title", url);
        }

        $("#" + id + " a").attr("href", "#");
    }

    var RESPONSE_MODE_NEW_PAGE = "0";
    this.updateLinkResponseMode = function (id, responseMode) {
        this.dealWithOldVersionLinkComp(id);
        var obj = $("#" + id);
        obj.attr("response_mode", responseMode);
        $("#" + id + " a").attr("href", "#");
        if (responseMode === RESPONSE_MODE_NEW_PAGE) {
            $("#" + id + " a").attr("onclick", "renderJs.openLinkInNewPage(this)");
        } else {
            $("#" + id + " a").attr("onclick", "renderJs.openLinkInNewWindow(this)");
        }
    }

    this.updateLinkType = function (id, linkType) {
        this.dealWithOldVersionLinkComp(id);
        var obj = $("#" + id);
        obj.attr("link_type", linkType);
    }

    /** Compatiblity for Link component, comvert old LinkComp into new version */
    this.dealWithOldVersionLinkComp = function (id) {
        var obj = $("#" + id);
        if (!obj.attr("version")) {
            var href = $("#" + id + " a").attr("href");
            $("#" + id + " a").attr("href", "#");
            $("#" + id + " a").removeAttr("target");
            $("#" + id + " a").attr("onclick", "renderJs.openLinkInNewPage(this)");
            if (!obj.attr("response_mode")) {
                obj.attr("response_mode", "0");
            }
            if (!obj.attr("link_type")) {
                obj.attr("link_type", "0");
            }
            if (!obj.attr("url")) {
                obj.attr("url", href);
            }
            obj.attr("version", "83");
        }
    }


    //-----------------load customize files--------------------------
    this.loadUserJSFiles = function (files) {
        //files = ["js/jq164.js","js/jquery.jqplot.min.js","js/jqplot.meterGaugeRenderer.min.js","js/myPlot.js"]
        var loadFilesId = "";
        files = files.split(",");
        var filesArr = this.collateLoadJsArr(files);
        if (filesArr.length > 0) {
            loadFilesId = filesArr.toString();
            loadjs(filesArr, loadFilesId, {
                success: function () {
                    $("SCRIPT[async]").each(function () {
                        // load success update <script> tag
                        $(this).removeAttr("async");
                        $(this).attr("type", "text/javascript");
                        // $(this).attr("id", id);
                        // alert($(this)[0].outerHTML);
                    });
                },
                async: false
            });
        }
        return loadFilesId;
    }

    this.loadSingleJSFile = function (fileName) {
        var id = fileName;
        var src = "js/" + fileName;
        loadjs(src, function () {
            $("SCRIPT[async]").each(function () {
                // load success update <script> tag
                $(this).removeAttr("async");
                $(this).attr("type", "text/javascript");
                $(this).attr("id", id);
                // alert($(this)[0].outerHTML);
            });
        });
    }

//    this.loadCSS = function (cssFileName) {
//        var id = cssFileName; //cssFileName.substring(0,cssFileName.indexOf('.')); 
//        var file = "css/" + cssFileName;
//        $("<link>").attr({rel: "stylesheet", type: "text/css", href: file, id: id}).appendTo("head");
//    }

    this.updateCompanyInfo = function (company) {
        $("#main").attr("prefix", company);
    }

    this.collateLoadJsArr = function (filesArr) {
        var newArr = [];
        var count = 0;
        for (var i = 0; i < filesArr.length; i++) {
            if (!this.isJsFilesExist(filesArr[i])) {
                newArr[count] = filesArr[i];
                count++;
            }
        }
        return newArr;
    }
    this.isJsFilesExist = function (filePath) {
        $("SCRIPT[id]").each(function () {
            if ($(this).attr("src")) {
                if (filePath === $(this).attr("id")) {
                    return true;
                }
            }
        });
        return false;
    }


    /**
     * test only 
     * */
    this.showTestMessage = function (message) {
        $("#testMessage").remove();
        $("#main").append("<div id='testMessage' style='position: absolute; left:0px; top: 0px; '>" + message + "</div>");
    }
}

/** 
 * Empty onload method, used for old version.
 * Any init actions please add in $(document).ready() 
 * */
function loadEditPage() {
}
function loadNavPage() {
}

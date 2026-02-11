/** 
 *  =============== json data structure ============================
 *  [xxx(id,jsonData)] JS function: was bindinged in component.xml; 
 *					   User can get the property response via the JS method.
 *	parameter [id]: id of target component(DIV).
 *	parameter [jsonData]: a json object, the structure was defined by your property type. and you can use it like below:
 *  (if your property type was size, you can get the value by json.width and json.height. To get other property type's value by json.newValue)
 *
 *		  var json = $.parseJSON(jsonData);	
 *		  json.width;						
 *		  json.height;						
 * 		  json.newValue;						
 *
 *   ============== Global variable: _dataMap ========================
 *   [_dataMap] was used to transfer/synchcronize data in global JS context.
 *   _dataMap.put(key,value);	
 *   _dataMap.get(key);
 *
 *
 *   ============== property bindinged jsMethod example ==============
 *   function updateMySize(id,jsonData){
 *		var json = $.parseJSON(jsonData);
 *		var width = json.width;
 *		var height = json.height;
 *		$("#"+id).css("width",parseInt(width)-2);
 *		$("#"+id).css("height",parseInt(height)-2);
 *   }	
 * 
 */

/** 
 * Runtime rainbow callback method 
 * - FW js cannot callback a method like objct.xxx(), so this method not in rainbow Obj.
 * */
_callback_rainbowOnChange = function () {
    //get callback data
    var id = this.m_sShowParam;
    var blockFormat = $("#" + id).attr("block_format");
    var accessModeType = $("#" + id).attr("type");
    var actVal = renderJs.getValueByBlockFormat(blockFormat, accessModeType, this.m_sValue);
    actVal = renderJs.getFloatValueByDecimalPlace(id, actVal);
    $("#" + id).attr("c_value", actVal);
    //udpate value
    var myChart = _dataMap.get(id + "obj");
    var option = _dataMap.get(id + "option");
    var maxVal = parseFloat($("#" + id).attr("d_max"));
    var minVal = parseFloat($("#" + id).attr("d_min"));
    if (actVal > maxVal) {
        actVal = maxVal;
    }
    if (actVal < minVal) {
        actVal = minVal;
    }
    option.series[0].data[0].value = actVal;
    myChart.setOption(option, true);
    //synchcronize data
    _dataMap.put(id + "obj", myChart);
    _dataMap.put(id + "option", option);
}

/**
 *  trendJs Object
 */
var rainbowJs = (function () {
    function rainbowJs() {
    }
    /**init rainbow chart*/
    rainbowJs.initChart = function (compID) {
        /** remove old instance when SVG */
        if ($("#" + compID)) {
            $("#" + compID).empty();
        }
        /**init rainbow gauge only in creat*/
        var myChart;
        if ($('#wrap').length === 0) {
            //editTime
            myChart = echarts.init(document.getElementById(compID), null, {renderer: 'svg'});
        } else {
            //runTime
            myChart = echarts.init(document.getElementById(compID));
        }
        var option = {series: [{name: 'data', type: 'gauge', min: 0, max: 1000,	radius:'99%',center:["51%","55%"], axisLine: {lineStyle: {color: [[0.20, '#66CDAA'], [0.80, '#4682B4'], [1, '#A52A2A']], width: 6 }}, axisTick: {length: 10, lineStyle: {color: 'auto', shadowBlur: 0}}, splitLine: {length: 15, lineStyle: {width: 2, color: 'auto', shadowBlur: 0}}, data: [{value: 0, name: 'Analogdata'}]}]};
        option.series[0].max = parseFloat($("#" + compID).attr("d_max"));
        option.series[0].min = parseFloat($("#" + compID).attr("d_min"));
        option.series[0].data[0].name = $("#" + compID).attr("d_title");
        myChart.setOption(option, true);
        /**save chart into global _dataMap */
        _dataMap.put(compID + "obj", myChart);
        _dataMap.put(compID + "option", option);
        rainbowJs.initAnalogRainbowFont(compID);
        rainbowJs.setDecimalPlaces(compID);
    }

    /**size property handling method*/
    rainbowJs.updateRainbowSize = function (id, jsonData) {
        //get new value from property sheet
        var json = $.parseJSON(jsonData);
        var width = json.width;
        var height = json.height;
        //update UI and synchcronize data in _dataMap
        if (_dataMap.get(id + "obj")) {
            $("#" + id).css("width", parseInt(width) - 2);
            $("#" + id).css("height", parseInt(height) - 2);
            _dataMap.get(id + "obj").resize();
        }
    }

    /**max property handling method*/
    rainbowJs.setRainbowMax = function (id, jsonData) {
        //get new value from property sheet
        var json = $.parseJSON(jsonData);
        var max = json.newValue;
        $("#" + id).attr("d_max", max);
        //update chart
        var myChart = _dataMap.get(id + "obj", myChart);
        var option = _dataMap.get(id + "option", option);
        option.series[0].max = parseFloat(max);
        myChart.setOption(option, true);
        //synchcronize data
        _dataMap.put(id + "obj", myChart);
        _dataMap.put(id + "option", option);
        this.setDecimalPlaces(id);
    }

    /**min property handling method*/
    rainbowJs.setRainbowMin = function (id, jsonData) {
        //get new value from property sheet
        var json = $.parseJSON(jsonData);
        var min = json.newValue;
        $("#" + id).attr("d_min", min);
        //update chart
        var myChart = _dataMap.get(id + "obj", myChart);
        var option = _dataMap.get(id + "option", option);
        option.series[0].min = parseFloat(min);
        myChart.setOption(option, true);
        //synchcronize data
        _dataMap.put(id + "obj", myChart);
        _dataMap.put(id + "option", option);
        this.setDecimalPlaces(id);
    }

    /**title property handling method*/
    rainbowJs.updateRainbowTitle = function (id, jsonData) {
        //get new value from property sheet
        var json = $.parseJSON(jsonData);
        var newValue = json.newValue;
        //udpate 
        $("#" + id).attr("d_title", newValue);
        var myChart = _dataMap.get(id + "obj", myChart);
        var option = _dataMap.get(id + "option", option);
        option.series[0].data[0].name = newValue
        myChart.setOption(option, true);
        //synchcronize data
        _dataMap.put(id + "obj", myChart);
        _dataMap.put(id + "option", option);
    }
    /**
     * init rainbow font style and font color
     * @param {type} compID
     * @returns {undefined}
     */
    rainbowJs.initAnalogRainbowFont = function (compID) {
        var detail_fontStyle = $("#" + compID).attr("detail_fontStyle");
        var detail_fontSize = $("#" + compID).attr("detail_fontSize");
        var detail_fontFamily = $("#" + compID).attr("detail_fontFamily");
        var detail_color = $("#" + compID).attr("detail_color");
        var detail_fontWeight = $("#" + compID).attr("detail_fontWeight");
        var axisLabel_fontStyle = $("#" + compID).attr("axisLabel_fontStyle");
        var axisLabel_fontSize = $("#" + compID).attr("axisLabel_fontSize");
        var axisLabel_fontFamily = $("#" + compID).attr("axisLabel_fontFamily");
        var axisLabel_fontWeight = $("#" + compID).attr("axisLabel_fontWeight");
        var myChart = _dataMap.get(compID + "obj", myChart);
        var option = _dataMap.get(compID + "option", option);
        if (!option.series[0].axisLabel) {
            option.series[0].axisLabel = {};
        }
        if (!option.series[0].detail) {
            option.series[0].detail = {};
        }
        if (axisLabel_fontSize) {
            option.series[0].axisLabel.fontStyle = axisLabel_fontStyle;
            option.series[0].axisLabel.fontSize = axisLabel_fontSize;
            option.series[0].axisLabel.fontFamily = axisLabel_fontFamily;
            option.series[0].axisLabel.fontWeight = axisLabel_fontWeight;
        }

        if (detail_fontSize) {
            option.series[0].detail.fontStyle = detail_fontStyle;
            option.series[0].detail.fontSize = detail_fontSize;
            option.series[0].detail.fontFamily = detail_fontFamily;
            option.series[0].detail.fontWeight = detail_fontWeight;
        }

        if (detail_color) {
            var color = detail_color.split(",");
            var colors = [];
            var length = color.length / 2;
            for (var i = 0; i < length; i++) {
                colors[i] = [];
                colors[i][0] = color[i * 2];
                colors[i][1] = color[i * 2 + 1];
            }
            option.series[0].axisLine.lineStyle.color = colors;
        }

        myChart.setOption(option, true);
        _dataMap.put(compID + "obj", myChart);
        _dataMap.put(compID + "option", option);
    }
    /**font style property handling method*/
    rainbowJs.updateAnalogRainbowFontStyle = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var fontStyle = json.style;
        var fontFamily = json.family;
        var fontSize = json.size;
        var style = "";
        var fontWeight = "";
        if (fontStyle == 0) {
            //normal
            style = "normal";
            fontWeight = "";
        }
        else if (fontStyle == 1) {   //bold
            style = "normal";
            fontWeight = "bold";
        }
        else if (fontStyle == 2) {   //italic
            style = "italic";
            fontWeight = "";
        }
        else if (fontStyle == 3) {   //italic bold
            style = "italic";
            fontWeight = "bold";
        }
        var myChart = _dataMap.get(id + "obj", myChart);
        var option = _dataMap.get(id + "option", option);
        if (!option.series[0].axisLabel) {
            option.series[0].axisLabel = {};
        }
        if (!option.series[0].detail) {
            option.series[0].detail = {};
        }
        option.series[0].axisLabel.fontStyle = style;
        option.series[0].axisLabel.fontSize = Math.max(fontSize, 1);
        option.series[0].axisLabel.fontFamily = fontFamily;
        option.series[0].axisLabel.fontWeight = fontWeight;
        option.series[0].detail.fontStyle = style;
        option.series[0].detail.fontSize = Math.min(parseInt(fontSize) + 16, 72);
        option.series[0].detail.fontFamily = fontFamily;
        option.series[0].detail.fontWeight = fontWeight;
        myChart.setOption(option, true);
        $("#" + id).attr("detail_fontStyle", style);
        $("#" + id).attr("detail_fontSize", option.series[0].detail.fontSize);
        $("#" + id).attr("detail_fontFamily", fontFamily);
        $("#" + id).attr("detail_fontWeight", fontWeight);
        $("#" + id).attr("axisLabel_fontStyle", style);
        $("#" + id).attr("axisLabel_fontSize", option.series[0].axisLabel.fontSize);
        $("#" + id).attr("axisLabel_fontFamily", fontFamily);
        $("#" + id).attr("axisLabel_fontWeight", fontWeight);
        //synchcronize data
        _dataMap.put(id + "obj", myChart);
        _dataMap.put(id + "option", option);
        /*_dataMap.get(id + "obj").setOption({
         series: [{
         axisLabel: {
         fontStyle: style,
         fontSize: fontSize,
         fontFamily: fontFamily,
         fontWeight: fontWeight
         },
         detail: {
         fontStyle: style,
         fontSize: fontSize,
         fontFamily: fontFamily,
         fontWeight: fontWeight
         }
         }]
         });*/
    }
    rainbowJs.updateAnalogRainbowLineStyle = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        if (!json.scopeColorList) {
            return;
        }
        var myChart = _dataMap.get(id + "obj", myChart);
        var option = _dataMap.get(id + "option", option);
        /*axisLine: {lineStyle: {color: [[0.20, '#66CDAA'], [0.80, '#4682B4'], [1, '#A52A2A']], width: 6, }}*/
        if (!option.series[0].axisLine) {
            option.series[0].axisLine = {};
        }
        if (!option.series[0].axisLine.lineStyle) {
            option.series[0].axisLine.lineStyle = {};
        }
        var colors = [];
        for (var i = 0; i < json.scopeColorList.length; i++) {
            colors[i] = [];
            colors[i][0] = parseFloat(json.scopeColorList[i].endPercentage) / 100;
            colors[i][1] = this.calculaterJavaColorToJsColor(json.scopeColorList[i].color.value);
        }
        option.series[0].axisLine.lineStyle.color = colors;
        myChart.setOption(option, true);
        $("#" + id).attr("detail_color", colors);
        _dataMap.put(id + "obj", myChart);
        _dataMap.put(id + "option", option);
    }
    rainbowJs.calculaterJavaColorToJsColor = function (color) {
        /*var rgbs = [];
         rgbs[0] = ((color>>16)&0xFF).toString(16);
         rgbs[1] = ((color>>8)&0xFF).toString(16);
         rgbs[2] = ((color)&0xFF).toString(16);
         return "#"+ rgbs[0]+ rgbs[1]+ rgbs[2];*/
        var rgb = (color & 0xFFFFFF).toString(16);
        var addNum = 6 - rgb.length;
        for (var i = 0; i < addNum; i++) {
            rgb = "0" + rgb;
        }
        rgb = "#" + rgb;
        return rgb;
    }

    rainbowJs.updateDecimalPlaces = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var decimal = json.decimal;
        if (!decimal) {
            return;
        }
        $("#" + id).attr("decimal", decimal);
        this.setDecimalPlaces(id);
    }

    rainbowJs.setDecimalPlaces = function (id) {
        var myChart = _dataMap.get(id + "obj", myChart);
        var option = _dataMap.get(id + "option", option);
        var decimalNum = this.getDecimalNum(id);
        var scaleDecimalNum = this.scaleValueDecimalPlace(id, decimalNum);
        if (!option.series[0].axisLabel) {
            option.series[0].axisLabel = {};
        }
        if (!option.series[0].detail) {
            option.series[0].detail = {};
        }
        option.series[0].axisLabel.formatter = function (value) {
            return value.toFixed(scaleDecimalNum);
        }
        option.series[0].detail.formatter = function (value) {
            return value.toFixed(decimalNum);
        }
        myChart.setOption(option, true);
        _dataMap.put(id + "obj", myChart);
        _dataMap.put(id + "option", option);
    }

    rainbowJs.getDecimalNum = function (id) {
        var decimal = $("#" + id).attr("decimal");
        if ((undefined === decimal) || (!decimal) || "1" === decimal) {
            return 0;
        } else if ("10" === decimal) {
            return 1;
        } else if ("100" === decimal) {
            return 2;
        } else if ("1000" === decimal) {
            return 3;
        }
        return 0;
    }
    rainbowJs.scaleValueDecimalPlace = function (id, decimal) {
        var maxVal = parseFloat($("#" + id).attr("d_max"));
        var minVal = parseFloat($("#" + id).attr("d_min"));
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
    }
    
    rainbowJs.updateBackgroundColor = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var color = this.calculaterJavaColorToJsColor(json.value);
        var opacity = $("#" + id).attr("record_opacity");
        if (opacity > 0) {
            $("#" + id).css("background-color", color);
        }
        $("#" + id).attr("last_background-color", color);
    }

    rainbowJs.updateBackgroundOpacity = function (id, opacity) {
        var value = parseFloat(opacity);
        if (value > 0) {
            var color = $("#" + id).attr("last_background-color");
            $("#" + id).css("background-color", color);
        } else {
            $("#" + id).css("background-color", "");
        }
        $("#" + id).attr("record_opacity", value);
    }
    return rainbowJs;
})();
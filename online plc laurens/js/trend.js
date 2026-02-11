/**
 * trend view js
 * */

/** 
 * Runtime item data callback method (refresh trend data) 
 * - FW js cannot callback a method like objct.xxx(), so this method not in trendJs Obj.
 * */
_callback_trendRefreshSelfData = function () {
    var id = this.m_sShowParam;
    var type = $("#" + id).attr("type");
    var blockFormat = $("#" + id).attr("block_format");
    var value = renderJs.getValueByBlockFormat(blockFormat, type, this.m_sValue);
    value = renderJs.getFloatValueByDecimalPlace(id, value);
    $("#" + id).attr("m_value", value);
}

/**
 * OBJECT trendJs 
 */
var trendJs = (function () {
    function trendJs() {
    }

    /**init trendView chart*/
    trendJs.initChart = function (compID) {
        var viewDiv = $("#" + compID + " #view")[0];
        if (viewDiv) {
            $("#" + compID + " #view").empty();
        }
        var myChart;
        if ($('#wrap').length === 0) {
            // editTime
            myChart = echarts.init(viewDiv, null, {renderer: 'svg'});
        } else {
            // runTime
            myChart = echarts.init(viewDiv);
        }
        var option = trendJs.initTrendViewsOption();
        myChart.setOption(option);
        _dataMap.put(compID + "obj", myChart);
        _dataMap.put(compID + "option", option);
        /** load chart setting in edit time*/
        trendJs.loadTrendViewInEditTime(compID);
    }

    /** handle size property */
    trendJs.setTrendViewSize = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var w = json.width;
        var h = json.height;
        var obj = $("#" + id);
        obj.css("width", w);
        obj.css("height", h);

        if (_dataMap.get(id + "obj")) {
            _dataMap.get(id + "obj").resize();
        }
    }

    /** handle title property */
    trendJs.updateTrendViewTitleText = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var newValue = json.newValue;
        $("#" + id).attr("title_text", newValue);
        _dataMap.get(id + "obj").setOption({
            title: {
                text: newValue
            }
        });
    }

    /** handle title font style property */
    trendJs.updateTrendViewTitleFontStyle = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var family = json.family;
        var fontStyle = json.style;
        var size = json.size;
        var fontWeight = "";
        var style = "";
        if (fontStyle === 0) {
            //normal
            style = "normal";
            fontWeight = "";
        } else if (fontStyle === 1) {   //bold
            style = "normal";
            fontWeight = "bold";
        } else if (fontStyle === 2) {   //italic
            style = "italic";
            fontWeight = "";
        } else if (fontStyle === 3) {   //italic bold
            style = "italic";
            fontWeight = "bold";
        }
        $("#" + id).attr("title_family", family);
        $("#" + id).attr("title_style", style);
        $("#" + id).attr("title_size", size);
        $("#" + id).attr("title_weight", fontWeight);

        var legendSize = Math.max(size - 20, 12);
        $("#" + id).attr("legend_size", legendSize);

        _dataMap.get(id + "obj").setOption({
            title: {
                textStyle: {
                    fontStyle: style,
                    fontSize: size,
                    fontFamily: family,
                    fontWieght: fontWeight
                }
            },
            legend: {
                textStyle: {
                    fontStyle: style,
                    fontSize: legendSize,
                    fontFamily: family,
                    fontWieght: fontWeight
                }
            },
            xAxis: {
                axisLabel: {
                    fontStyle: style,
                    fontSize: legendSize,
                    fontFamily: family,
                    fontWieght: fontWeight
                }
            },
            yAxis: {
                axisLabel: {
                    fontStyle: style,
                    fontSize: legendSize,
                    fontFamily: family,
                    fontWieght: fontWeight
                }
            }
        });
    }

    /** handle title color property */
    trendJs.updateTrendViewTitleColor = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var newValue = json.newValue;
        $("#" + id).attr("title_color", newValue);
        _dataMap.get(id + "obj").setOption({
            title: {
                textStyle: {
                    color: newValue
                }
            }
            /*legend: {
             textStyle:{
             color: newValue
             }
             },
             xAxis: {
             axisLabel:{
             color: newValue
             }
             },
             yAxis: {
             axisLabel:{
             color: newValue
             }
             }*/
        });
    }

    /** handle textAlign property */
    trendJs.updateTrendViewTextAlign = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var textAlign = json.newValue;
        $("#" + id).attr("text_align", textAlign + '');

        _dataMap.get(id + "obj").setOption({
            title: {
                left: textAlign
            }
        });
    }

    /** handle intervalTime property */
    trendJs.updateTrendViewIntervalTime = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var intervalTime = json.newValue;
        $("#" + id).attr("interval_time", intervalTime + '');
        updateEditTimeTrendViewLegend(id);
    }

    /** handle TrendViewTags property */
    trendJs.updateTrendViewTags = function (id, jsonData) {
        $('#' + id + ' .data').empty();
        if (jsonData) {
            for (var i = 0; i < jsonData.length; i++) {
                addDivForTrendViewItemData(id, jsonData[i]);
            }
        }
        //update chart legend
        updateEditTimeTrendViewLegend(id);
    }
    /** update editTime TrendView Legend */
    function updateEditTimeTrendViewLegend(id) {
        var chart = _dataMap.get(id + "obj");
        var op = _dataMap.get(id + "option");
        chart.setOption({
            legend: {data: getTreddViewLegendDataArr(id)},
            xAxis: {data: getTrendViewXAxisData(op, 1, id)},
            series: createTrendViewSeries(id)
        });
    }

    /** init Option  */
    trendJs.initTrendViewsOption = function () {
        var option = {
            title: {text: 'Trend View', bottom: 10},
            tooltip: {trigger: 'axis', hideDelay: 2200}, /** alwaysShowContent:true */
            legend: {data: []}, /** , type: 'scroll' */
            grid: {top: '25%', left: '2%', right: '8%', containLabel: true},
            xAxis: {name: "", boundaryGap: false, data: []},
            yAxis: {name: ""},
            series: [{name: 'value', type: 'line', symbol: 'emptycircle', data: []}]
        };
        return option;
    }

    /** runtime page onload method, init trend view.  to be registered at end of this js file. */
    trendJs.initTrendViews = function () {
        //console.log('--initTrendViews--');
        $('.trend').each(function () {
            var id = this.id;
            //init
            initTrendViewToolbox(id);

            //wait 5s for connetion setup and refresh chart for 1st data display
            setTimeout(function () {
                refreshTrendViewData(id);
            }, 5000);

            //refresh chart by cutomize Interval time
            var intervalTime = parseInt($(this).attr('interval_time'));
            setInterval(function () {
                refreshTrendViewData(id);
            }, intervalTime * 1000);
        });
    }

    /** edit-time page onload method, init trend view*/
    trendJs.loadTrendViewInEditTime = function (compID) {
        //only works in edit pages
        if ($('#wrap').length === 0) {
            initTrendViewToolbox(compID);
            refreshTrendViewData(compID);
        }
    }


    trendJs.zoomTrendView = function (id) {
        if (!$("#" + id).attr("max_size")) {
            setTrendView2MaxSize(id);
        } else {
            setTrendView2DefaultSize(id);
        }
        _dataMap.get(id + "obj").resize();
    }

    /** create div and insert data from table */
    function addDivForTrendViewItemData(id, jsonData) {
        var div = $('<div></div>');
        var tagCallBackId = generateId();
        div.attr('id', tagCallBackId);
        div.attr('tag_id', jsonData.id);
        div.attr('show_function', '_callback_trendRefreshSelfData');
        div.addClass('server_binding');
        div.addClass('trend_data');
        div.attr('show_param', tagCallBackId);
        div.attr('range', jsonData.blockType.realvalue);
        div.attr('address', jsonData.blockNum.realvalue);
        div.attr('type', jsonData.type.realvalue);
        div.attr('length', jsonData.length.realvalue);
        div.attr('vm_name', jsonData.name);
        div.attr('device_name', jsonData.deviceName);
        div.attr('block_number', jsonData.blockNum.value);
        div.attr('block_type_name', jsonData.blockType.value);
        div.attr('block_format', jsonData.blockFormat);
        div.attr('decimal', getDecimalValueFromIndex(jsonData.decimalPlaces));
        if (jsonData.color) {
            div.attr('item_color', jsonData.color.realvalue);
        }

        $('#' + id + ' .data').append(div);
    }
    function getDecimalValueFromIndex(str) {
        if (str === "1") {
            return "10";
        } else if (str === "2") {
            return "100";
        } else if (str === "3") {
            return "1000";
        }
        return "1";
    }

    function getDecimalIndexFromValue(str) {
        if (str === "10") {
            return "1";
        } else if (str === "100") {
            return "2";
        } else if (str === "1000") {
            return "3";
        }
        return "0";
    }

    function generateId() {
        return Number(Math.random().toString().substr(3, 6) + Date.now()).toString(36);
    }

    trendJs.updateTrendViewLegendByAddItemData = function (id) {
        var chart = _dataMap.get(id + "obj");
        var op = _dataMap.get(id + "option");
        chart.setOption({
            legend: {data: getTreddViewLegendDataArr(id)}
        });
        _dataMap.put(id + "option", chart.getOption());
    }

    function getVrendViewItemDataName(obj) {
        var itemName = "";
        if (obj.attr("block_type_name").indexOf("V") !== -1) {
            var blockNameWithNumber = obj.attr("block_type_name") + obj.attr("block_Number");
            itemName = obj.attr("vm_name") + " (" + blockNameWithNumber + obj.attr("block_format").substring(0, 1).toLowerCase() + getDecimalIndexFromValue(obj.attr("decimal")) + ")";
        } else {
            itemName = obj.attr("vm_name") + " (" + obj.attr("block_Number") + obj.attr("block_format").substring(0, 1).toLowerCase() + getDecimalIndexFromValue(obj.attr("decimal")) + ")";
        }
        return itemName;
    }

    function getTreddViewLegendDataArr(id) {
        var arr = [];
        $('#' + id + ' .data div').each(function (i) {
            var dataName = getVrendViewItemDataName($(this));
            arr.push(dataName);
        });
        //console.log(arr)
        return arr;
    }

    function initTrendViewToolbox(id) {
        var chart = _dataMap.get(id + "obj");
        var title = $("#" + id).attr("title_text");
        var color = $("#" + id).attr("title_color");
        var textAlign = $("#" + id).attr("text_align");

        var family = $("#" + id).attr("title_family");
        var style = $("#" + id).attr("title_style");
        var size = $("#" + id).attr("title_size");
        var legendSize = $("#" + id).attr("legend_size");
        var fontWeight = $("#" + id).attr("title_weight");
        chart.setOption({
            title: {
                text: title,
                left: textAlign,
                textStyle: {
                    color: color,
                    fontStyle: style,
                    fontFamily: family,
                    fontSize: size,
                    fontWieght: fontWeight
                }
            },
            legend: {
                textStyle: {
                    fontStyle: style,
                    fontSize: legendSize,
                    fontFamily: family,
                    fontWieght: fontWeight
                }
            },
            xAxis: {
                axisLabel: {
                    fontStyle: style,
                    fontSize: legendSize,
                    fontFamily: family,
                    fontWieght: fontWeight
                }
            },
            yAxis: {
                axisLabel: {
                    fontStyle: style,
                    fontSize: legendSize,
                    fontFamily: family,
                    fontWieght: fontWeight
                }
            },
            toolbox: {
                show: true, bottom: 10, right: 30,
                feature: {
                    myTool: {
                        show: true,
                        title: 'zoom',
                        onclick: function (opt) {
                            trendJs.zoomTrendView(id);
                        },
                        icon: 'path://M525.4 721.2H330.9c-9 0-18.5-7.7-18.5-18.1V311c0-9 9.3-18.1 18.5-18.1h336.6c9.3 0 18.5 9.1 18.5 18.1v232.7c0 6 8.8 12.1 15 12.1 6.2 0 15-6 15-12.1V311c0-25.6-25.3-48.9-50.1-48.9h-335c-26.2 0-50.1 23.3-50.1 48.9v389.1c0 36.3 20 51.5 50.1 51.5h197.6c6.2 0 9.3-7.5 9.3-15.1 0-6-6.2-15.3-12.4-15.3zM378.8 580.6c-6.2 0-12.3 8.6-12.3 14.6s6.2 14.6 12.3 14.6h141.4c6.2 0 12.3-5.8 12.3-13.4 0.3-9.5-6.2-15.9-12.3-15.9H378.8z m251.6-91.2c0-6-6.2-14.6-12.3-14.6H375.7c-6.2 0-12.4 8.6-12.4 14.6s6.2 14.6 12.4 14.6h240.8c6.2 0.1 13.9-8.5 13.9-14.6z m-9.2-120.5H378.8c-6.2 0-12.3 8.6-12.3 14.6s6.2 14.6 12.3 14.6h240.8c7.7 0 13.9-8.6 13.9-14.6s-6.2-14.6-12.3-14.6z m119.4 376.6L709 714.1c9.2-12 14.6-27 14.6-43.2 0-39.4-32.1-71.4-71.8-71.4-39.7 0-71.8 32-71.8 71.4s32.1 71.4 71.8 71.4c16.3 0 31.3-5.4 43.4-14.5l31.6 31.5c3.8 3.8 10 3.8 13.8 0 3.8-3.8 3.8-10 0-13.8z m-88.8-23.6c-28.3 0-51.3-22.8-51.3-51s23-51 51.3-51c28.3 0 51.3 22.8 51.3 51s-23 51-51.3 51z'
                    },
                    dataView: {show: false, readOnly: true, title: 'data view'},
                    saveAsImage: {show: false, title: 'image'}
                }
            }
        });
        _dataMap.put(id + "option", chart.getOption());
    }
    /*update trend view total step size handle method*/
    trendJs.updateTrendViewTotalStepSize = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var stepSize = json.stepSize;
        $("#" + id).attr("max_count", stepSize);
    }
    function refreshTrendViewData(id) {
        var maxDataCount = parseInt($("#" + id).attr("max_count"));
        var chart = _dataMap.get(id + "obj");
        var op = _dataMap.get(id + "option");
        var seriesData = [];
        if (op.series.length && op.series[0].data.length === 0) {
            //first time init
            seriesData = createTrendViewSeries(id);
        } else {
            //update
            seriesData = op.series;
            updateTrendViewSeries(id, seriesData, maxDataCount);
        }
        //chart.setOption(option, true);
        chart.setOption({
            legend: {data: getTreddViewLegendDataArr(id)},
            xAxis: {data: getTrendViewXAxisData(op, maxDataCount, id)},
            series: seriesData
        });
        _dataMap.put(id + "option", chart.getOption());
    }

    function getTrendViewXAxisData(op, maxDataCount, id) {
        var xArr = op.xAxis.data;
        if (!xArr) {
            xArr = op.xAxis[0].data;
        }

        var xData = getCurrentTime(id);
        xArr.push(xData);
        if (xArr.length > maxDataCount) {
            xArr.shift();
        }

        xArr = xArr.map(function (str) {
            return str.replace(' ', '\n');
        });
        return xArr;
    }

    function updateTrendViewSeries(id, series, dataCount) {
        $('#' + id + ' .data div').each(function (i) {
            var data = getTrendViewDataFromDiv($(this));
            for (var i = 0; i < series.length; i++) {
                if (data.name === series[i].name) {
                    series[i].data.push(data.value);
                    if (series[i].data.length > dataCount) {
                        series[i].data.shift();
                    }
                    break;
                }
            }
        });
    }

    function createTrendViewSeries(id) {
        var seriesData = [];
        //get values by id ...
        $('#' + id + ' .data div').each(function (i) {
            var divData = getTrendViewDataFromDiv($(this));
            var data = createTrendViewItemData(divData);
            seriesData.push(data);
        });
        // console.log(seriesData);
        return seriesData;
    }

    function createTrendViewItemData(divData) {
        var values = [];
        values.push(divData.value);
        var obj = {"name": divData.name, "type": "line", symbol: 'emptycircle', data: values, 'itemStyle': divData.itemStyle};
        return obj;
    }

    function getTrendViewDataFromDiv(obj) {
        var name = getVrendViewItemDataName(obj);
        var value = obj.attr("m_value");
        var color = obj.attr("item_color");
        var colorObj = {"name": name, "value": value};
        if (color && color !== "auto" && color !== "-1") {
            //console.log("color:"+color);
            var colorRGB = getColorFromRGB(color);
            //console.log("color:"+color+"  colorRGB:"+colorRGB);
            colorObj.itemStyle = {"color": colorRGB};
        }
        return colorObj;
    }

    function getCurrentTime(id) {
        var oDate = new Date();
        var oYear = oDate.getFullYear();
        var oMonth = oDate.getMonth() + 1;
        oMonth = oMonth < 10 ? "0" + oMonth : oMonth;
        var oDay = oDate.getDate() < 10 ? "0" + oDate.getDate() : oDate.getDate();
        var oHour = oDate.getHours() < 10 ? "0" + oDate.getHours() : oDate.getHours();
        var oMin = oDate.getMinutes() < 10 ? "0" + oDate.getMinutes() : oDate.getMinutes();
        var oSen = oDate.getSeconds() < 10 ? "0" + oDate.getSeconds() : oDate.getSeconds();
        var oTime = oYear + '-' + oMonth + '-' + oDay + ' ' + oHour + ':' + oMin + ':' + oSen;
        var intervalTime = parseInt($("#" + id).attr('interval_time'));
        if (intervalTime < 60) {
            oTime = oHour + ':' + oMin + ':' + oSen;
        }
        return oTime;
    }
    function getColorFromRGB(colorStr) {
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
        var final = '#' + str + val;
        return (final === "#1000000") ? "#000000" : final;
    }

    function setTrendView2MaxSize(id) {
        var obj = $("#" + id);
        obj.attr("_width", obj.css("width"));
        obj.attr("_height", obj.css("height"));
        obj.attr("_top", obj.css("top"));
        obj.attr("_left", obj.css("left"));
        obj.attr("_z-index", obj.css("z-index"));
        obj.attr("max_size", true);

        //obj.css("width", $(document).width() * 0.74);
        //obj.css("height", $(window).height() * 0.84);
        obj.css("width", $("#main").width());
        obj.css("height", $("#main").height());
        obj.css("top", 0);
        obj.css("left", 0);
        obj.css("background-color", "silver");
        obj.css("opacity", 0.95);
        obj.css("z-index", 999);
        _dataMap.get(id + "obj").resize();
    }
    function setTrendView2DefaultSize(id) {
        var obj = $("#" + id);
        obj.css("width", obj.attr("_width"));
        obj.css("height", obj.attr("_height"));
        obj.css("top", obj.attr("_top"));
        obj.css("left", obj.attr("_left"));
        obj.css("background-color", "white");
        obj.css("opacity", "");
        obj.css("z-index", obj.attr("_z-index"));
        obj.css("background-color", "");

        obj.attr("_width", obj.css("width"));
        obj.attr("_height", obj.css("width"));
        obj.attr("_top", obj.css("top"));
        obj.attr("_left", obj.css("left"));
        obj.attr("_z-index", obj.css("z-index"));
        obj.attr("max_size", '');

        _dataMap.get(id + "obj").resize();
    }

    return trendJs;
})();

/** 
 * register 'trendJs.initTrendViews()' inito runtime 'onloadStack' after trendJsFile loaded 
 * */
if (renderJs) {
    renderJs.registerCustomizeComponentRuntimeInitMethod("trendJs.initTrendViews()");
}

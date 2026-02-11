/**login js*/

/**--runtime js--*/
function _initLoginPage() {
    Initialize();
    $("#input_password").val("");
}

function OnPasswordFocus(oPassword) {
    if (oPassword && oPassword.value) {
        oPassword.select();
    }
}
function LoginPageLogin() {
    var sPublicKey1 = document.body.getAttribute("logo_public_key1");
    var sPublicKey2 = document.body.getAttribute("logo_public_key2");
    var oInputPassword = document.getElementById("input_password");
    var oButtonLogin = document.getElementById("button_login");
    var sPassword = oInputPassword.value;
    var sUsername = document.getElementById("input_username").value;
    var bKeepSignIn = document.getElementById("check_keepsignin").checked;
    var bLogToCustomizedSite = document.getElementById("check_logoncustomized").checked;
    //this new interface required the latest FW version 
    if (renderJs.isBm83HttpsMode()) {
        //for bm 8.3 https only
        LocalLogin_v83(sPublicKey1, sPublicKey2, sUsername, sPassword, bKeepSignIn, bLogToCustomizedSite, oButtonLogin, oInputPassword);
    } else {
        LocalLogin(sPublicKey1, sPublicKey2, sUsername, sPassword, bKeepSignIn, bLogToCustomizedSite, oButtonLogin, oInputPassword);
    }
    return false;
}
function Initialize() {
    //initialize:
    //input password:
    var oInputPassword = document.getElementById("input_password");
    var sPassword = LocalStorage.Instance().Get("logo_current_password");
    if (sPassword)
        oInputPassword.value = sPassword;
    //select language:
    var oSelectLanguage = document.getElementById("select_language");
    var iCurLanguage = LocalStorage.Instance().Get("logo_current_language");
    var iIndex;
    for (iIndex = 0; iIndex < oSelectLanguage.options.length; iIndex++) {
        if (oSelectLanguage.options[iIndex].value == iCurLanguage)
            break;
    }
    if (iIndex < oSelectLanguage.options.length) {
        //found the save language:
        oSelectLanguage.selectedIndex = iIndex;
    } else {
        //use english as default:
        LocalStorage.Instance().Set("logo_current_language", 1);
        oSelectLanguage.selectedIndex = 1;
        iCurLanguage = 1;
    }
    //button login:
    document.getElementById("button_login").disabled = "";
    //check logto customized site:
    if (LocalStorage.Instance().Get("logo_current_logto_customized_site")) {
        document.getElementById("check_logoncustomized").checked = true;
    }
    //check keep sign in
    if (LocalStorage.Instance().Get("logo_current_autologin")) {
        document.getElementById("check_keepsignin").checked = true;
    } else {
        oInputPassword.focus();
    }

    // bind check_keepsignin
    $("#check_keepsignin").bind("click", function () {
        handleKeepSigninMsg();
    });
    //show msg when page loaded
    handleKeepSigninMsg();
}
function removeKeepSigninMsg() {
    $("#keepSigninMsg").remove();
}
function handleKeepSigninMsg() {
    if (!$("#check_keepsignin").is(':checked') || $("#keepMeLoggedOn").css("display") === 'none') {
        $("#keepSigninMsg").remove();
        return;
    }
    let msgDiv = "<div id='keepSigninMsg' style='background:#ffffe0;text-align:center;' onclick='removeKeepSigninMsg()'>"
            + "<div style='width:25px;height:25px;background-color:#e17100;border-radius:25px;margin-top:10px;margin-left:10px;float:left'><span style='color:white; text-align:center;font-size:18px'>!</span></div>"
            + "<div style='color:#e17100;'>[keep me logged on] is enabled <br>As the user information is stored in cookies, there is a risk that the information might be downloaded and used by malicious websites.</div>"
            + "</div>";
    $jq(msgDiv).appendTo($("#login").parent());
    $("#login").parent().css("z-index", 99);
}

/**--runtime js end--*/

/** loginJS obj */
var loginJS = (function () {
    function loginJS() {
    }

    /** handle size property */
    loginJS.setLoginSize = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var w = json.width;
        var h = json.height;
        var obj = $("#" + id);
        obj.css("width", w);
        obj.css("height", h);
    }

    /** handle font property */
    loginJS.updateLoginTextStyle = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var family = json.family;
        var size = json.size;
        var style = json.style;
        $(".loginText").css({"font-family": family, "font-size": parseInt(size)});
        setFontStyle("", parseInt(style));
    }
    function setFontStyle(obj, fontStyle) {
        if (fontStyle == 0) {
            //normal
            $(".loginText").css({"font-style": "normal", "font-weight": ""});
        } else if (fontStyle == 1) {   //bold
            $(".loginText").css({"font-style": "normal", "font-weight": "bold"});
        } else if (fontStyle == 2) {   //italic
            $(".loginText").css({"font-style": "italic", "font-weight": ""});
        } else if (fontStyle == 3) {   //italic bold
            $(".loginText").css({"font-style": "italic", "font-weight": "bold"});
        }
    }

    /** handle TextColor property */
    loginJS.updateLoginTextColor = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var newValue = json.newValue;
        $(".loginText").css("color", newValue);
    }

    /** handle TextColor property */
    loginJS.updateKeepLoggedOn = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var newValue = json.newValue;
        newValue = newValue === "1" ? "none" : "";
        $("#keepMeLoggedOn").css("display", newValue);
    }

    /** handle TextColor property */
    loginJS.updateToCustomizedSite = function (id, jsonData) {
        var json = $.parseJSON(jsonData);
        var newValue = json.newValue;
        newValue = newValue === "1" ? "none" : ""
        $("#toCustomizedSite").css("display", newValue);
    }

    return loginJS;
})();

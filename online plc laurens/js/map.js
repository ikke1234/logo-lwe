// version 1.1 ( LWE1.1.1 )
/* 
 * update content:
 *   var map = new Map(); --> var map = new UdwMap();  
 *   
 * =============================   
 * version 1.0
 * 
 * size()     
 * isEmpty()     
 * clear()   
 * put(key, value)   
 * remove(key)    
 * get(key)   
 * element(index)   
 * containsKey(key)
 * containsValue(value) 
 * values()    
 * keys()       
 *  
 * var map = new Map();  
 * map.put("key", "value");  
 * var val = map.get("key")  
 *
 */  
function UdwMap() {  
    this.elements = new Array();  

    this.size = function() {  
        return this.elements.length;  
    };  
  
    this.isEmpty = function() {  
        return (this.elements.length < 1);  
    };  
  
    this.clear = function() {  
        this.elements = new Array();  
    };  
  
    this.put = function(_key, _value) {  
        this.removeByKey(_key);
        this.elements.push( {  
            key : _key,  
            value : _value  
        });  
    };  
  
    this.remove = function(_key) {  
        var bln = false;  
        try {  
            for (i = 0; i < this.elements.length; i++) {  
                if (this.elements[i].key == _key) {  
                    this.elements.splice(i, 1);  
                    return true;  
                }  
            }  
        } catch (e) {  
            bln = false;  
        }  
        return bln;  
    };  
  
  
    this.removeByKey = function(_key) {  
        var bln = false;  
        try {  
            for (i = 0; i < this.elements.length; i++) {  
                if (this.elements[i].key == _key) {  
                    this.elements.splice(i, 1);  
                    return true;  
                }  
            }  
        } catch (e) {  
            bln = false;  
        }  
        return bln;  
    };  
      
    this.removeByValue = function(_value) {//removeByValueAndKey  
        var bln = false;  
        try {  
            for (i = 0; i < this.elements.length; i++) {  
                if (this.elements[i].value == _value) {  
                    this.elements.splice(i, 1);  
                    return true;  
                }  
            }  
        } catch (e) {  
            bln = false;  
        }  
        return bln;  
    };  
      
    this.removeByValueAndKey = function(_key,_value) {  
        var bln = false;  
        try {  
            for (i = 0; i < this.elements.length; i++) {  
                if (this.elements[i].value == _value && this.elements[i].key == _key) {  
                    this.elements.splice(i, 1);  
                    return true;  
                }  
            }  
        } catch (e) {  
            bln = false;  
        }  
        return bln;  
    };  
  
    this.get = function(_key) {  
        try {  
            for (i = 0; i < this.elements.length; i++) {  
                if (this.elements[i].key == _key) {  
                    return this.elements[i].value;  
                }  
            }  
        } catch (e) {  
            return false;  
        }  
        return false;  
    };  
  
    this.element = function(_index) {  
        if (_index < 0 || _index >= this.elements.length) {  
            return null;  
        }  
        return this.elements[_index];  
    };  
 
    this.containsKey = function(_key) {  
        var bln = false;  
        try {  
            for (i = 0; i < this.elements.length; i++) {  
                if (this.elements[i].key == _key) {  
                    bln = true;  
                }  
            }  
        } catch (e) {  
            bln = false;  
        }  
        return bln;  
    };  

    this.containsValue = function(_value) {  
        var bln = false;  
        try {  
            for (i = 0; i < this.elements.length; i++) {  
                if (this.elements[i].value == _value) {  
                    bln = true;  
                }  
            }  
        } catch (e) {  
            bln = false;  
        }  
        return bln;  
    };  
      
    this.containsObj = function(_key,_value) {  
        var bln = false;  
        try {  
            for (i = 0; i < this.elements.length; i++) {  
                if (this.elements[i].value == _value && this.elements[i].key == _key) {  
                    bln = true;  
                }  
            }  
        } catch (e) {  
            bln = false;  
        }  
        return bln;  
    };  
  
    this.values = function() {  
        var arr = new Array();  
        for (i = 0; i < this.elements.length; i++) {  
            arr.push(this.elements[i].value);  
        }  
        return arr;  
    };  
      
    this.valuesByKey = function(_key) {  
        var arr = new Array();  
        for (i = 0; i < this.elements.length; i++) {  
            if (this.elements[i].key == _key) {  
                arr.push(this.elements[i].value);  
            }  
        }  
        return arr;  
    }; 

    this.keys = function() {  
        var arr = new Array();  
        for (i = 0; i < this.elements.length; i++) {  
            arr.push(this.elements[i].key);  
        }  
        return arr;  
    };  
      
    this.keysByValue = function(_value) {  
        var arr = new Array();  
        for (i = 0; i < this.elements.length; i++) {  
            if(_value == this.elements[i].value){  
                arr.push(this.elements[i].key);  
            }  
        }  
        return arr;  
    };  

    this.keysRemoveDuplicate = function() {  
        var arr = new Array();  
        for (i = 0; i < this.elements.length; i++) {  
            var flag = true;  
            for(var j=0;j<arr.length;j++){  
                if(arr[j] == this.elements[i].key){  
                    flag = false;  
                    break;  
                }   
            }  
            if(flag){  
                arr.push(this.elements[i].key);  
            }  
        }  
        return arr;  
    };  
}  
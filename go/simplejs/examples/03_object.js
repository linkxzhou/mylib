// ES5: Object literal and property access
var obj = {a: 1, b: 2};
if (obj.a !== 1 || obj.b !== 2) throw 'fail';
obj.c = 3;
if (obj.c !== 3) throw 'fail';
